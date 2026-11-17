"""Export one folder to an archive, and import such an archive as a new folder.

Archive layout (``folio-folder-<name>-<UTC time>.tar``, see :data:`FOLDER_EXPORT`)::

    folder.json      the folder, its Files rows and FileIndices rows, the vector table's shape
    chunks.jsonl     one line per row of the folder's vector table: text, metadata,
                     node id, embedding and full-text search vector, exactly as stored
    files/<file id>  the stored files
    manifest.json    written last: versions, embedding model and dimension, segmenter,
                     counts, and a SHA-256 for every entry

The export is read in one REPEATABLE READ transaction, so the rows, the chunks and
the file list agree; it is refused while an index job runs on the folder.

An import always creates a **new** folder, owned by the owner key it is given
(``sub:<sub>`` for a person, ``console`` in open mode), named like the exported one
or ``<name>-2``, ``-3``, … when that owner already has one of that name. Every file
gets a new id (so a copy can sit next to the original in the same Folio), and the
chunks' metadata is rewritten to the new file ids, paths, folder id and name; node
ids and the parent/child links between chunks are kept. The database part runs in one
transaction and the files are moved into place before it commits, so a failed import
leaves no folder behind.

The vectors are only usable when this Folio embeds queries with the same model and
dimension. Otherwise the import is refused, unless ``files_only``: then only the
folder and its files are imported, and the folder is indexed again here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from sqlalchemy import text

from src.backup import service as S
from src.backup.archive import (ArchiveError, ArchiveWriter, Layout, read_manifest,
                                verify_and_extract)

FORMAT = "folio-folder"
FORMAT_VERSION = 1
FOLDER_JSON = "folder.json"
CHUNKS_NAME = "chunks.jsonl"
FILES_PREFIX = "files/"
FOLDER_EXPORT = Layout(FORMAT, FORMAT_VERSION, frozenset({FOLDER_JSON, CHUNKS_NAME}), (FILES_PREFIX,),
                       frozenset({FOLDER_JSON, CHUNKS_NAME}), "Folio folder export")

# Columns copied from an export; anything else in the archive is ignored. Ids, folder
# ids and paths are always assigned by the import.
FILE_COLUMNS = ("file_name", "file_size", "mime_type", "description", "tags", "content_hash",
                "upload_time", "updated_time")
INDEX_COLUMNS = ("index_id", "chunk_size", "chunk_overlap", "embedding_model", "num_chunks", "status",
                 "error_message", "content_hash", "index_warning", "pipeline_version",
                 "embedding_dimension", "indexed_at", "updated_at")
# Chunk metadata keys that name the file / folder (rewritten on import).
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_INSERT_BATCH = 500


class FolderRefused(S.BackupRefused):
    """A precondition of an export or import failed.

    ``code`` says which one, for callers that answer differently per case (the
    console): ``not_found``, ``newer_version``, ``embedding_mismatch``, or ``refused``.
    ``reasons`` are the problems without the command-line hints.
    """

    def __init__(self, message: str, *, code: str = "refused", reasons: Optional[List[str]] = None):
        super().__init__(message)
        self.code = code
        self.reasons = list(reasons) if reasons else [message]


# Added to an embedding mismatch on the command line (the console offers its own option).
FILES_ONLY_HINT = "pass --files-only to import the files without the index and index the folder here"


@dataclass
class ExportResult:
    path: Path
    size: int
    manifest: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


@dataclass
class ImportResult:
    manifest: Dict[str, Any]
    imported: bool
    warnings: List[str] = field(default_factory=list)
    folder_id: Optional[int] = None
    name: Optional[str] = None
    owner: Optional[str] = None
    files: int = 0
    chunks: int = 0
    indexed: bool = False


# ---------------------------------------------------------------- helpers

def export_name(folder_name: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    safe = re.sub(r"[^\w.-]+", "_", folder_name, flags=re.UNICODE).strip("._") or "folder"
    return f"folio-folder-{safe[:80]}-{now.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.tar"


def resolve_owner(value: str) -> str:
    """The folder owner key for ``--owner``: ``console``, ``sub:<sub>`` or a bare sub.

    Raises FolderRefused when no storage directory can be derived from it.
    """
    from src.auth.owner_key import OWNER_PREFIX, owner_key_for_sub
    from src.domain.exceptions import InvalidTokenError
    from src.storage.file_storage import owner_dir_name
    value = (value or "").strip()
    if not value:
        raise FolderRefused("an owner is required: the person's MCP Gate user id (sub), or console")
    try:
        key = value if value == "console" else owner_key_for_sub(
            value[len(OWNER_PREFIX):] if value.startswith(OWNER_PREFIX) else value)
        owner_dir_name(key)
    except (ValueError, InvalidTokenError) as e:
        raise FolderRefused(f"{value!r} is not a usable owner: {e}") from e
    return key


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Decimal):
        return int(value) if value == int(value) else float(value)
    return value


def _row(mapping) -> Dict[str, Any]:
    return {k: _jsonable(v) for k, v in dict(mapping).items()}


def _vector_dimension(conn, table: str) -> Optional[int]:
    dim = conn.execute(text(
        "SELECT a.atttypmod FROM pg_attribute a WHERE a.attrelid = to_regclass(:t) "
        "AND a.attname = 'embedding' AND NOT a.attisdropped"), {"t": f'public."{table}"'}).scalar()
    return int(dim) if dim and dim > 0 else None


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _find_folder(conn, folder_id: Optional[int], owner_key: Optional[str],
                 name: Optional[str]) -> Dict[str, Any]:
    if folder_id is not None:
        row = conn.execute(text('SELECT * FROM "Folders" WHERE id = :i'), {"i": folder_id}).mappings().first()
        if row is None:
            raise FolderRefused(f"there is no folder {folder_id}", code="not_found")
        return dict(row)
    if not (owner_key and name):
        raise FolderRefused("name the folder: --id, or --owner with --name")
    row = conn.execute(text('SELECT * FROM "Folders" WHERE user_token = :o AND name = :n'),
                       {"o": owner_key, "n": name}).mappings().first()
    if row is None:
        raise FolderRefused(f"{owner_key!r} has no folder named {name!r}", code="not_found")
    return dict(row)


def _stored_file(file_path: str, storage_root: Path) -> Optional[Path]:
    """The stored file a Files.file_path points to, when it lies inside storage/."""
    arc = S._arcname_for_db_path(file_path, storage_root)
    if arc is None:
        return None
    full = (storage_root / arc[len("storage/"):]).resolve()
    root = storage_root.resolve()
    return full if root in full.parents else None


# ---------------------------------------------------------------- export

def export_folder(*, db_url: str, storage_root: Path, folder_id: Optional[int] = None,
                  owner: Optional[str] = None, name: Optional[str] = None,
                  out_dir: Optional[Path] = None, out_path: Optional[Path] = None, rag_conf=None,
                  require_owner: Optional[str] = None,
                  now: Optional[datetime] = None, log: Callable[[str], None] = lambda m: None
                  ) -> ExportResult:
    """Write one folder to an archive; raise FolderRefused / ArchiveError.

    ``require_owner`` (an owner key) refuses a folder someone else owns as if it did
    not exist; it is checked in the export's own transaction.
    """
    now = now or datetime.now(timezone.utc)
    owner_key = resolve_owner(owner) if owner else None
    warnings: List[str] = []
    engine = S._engine(db_url)
    tmp = tempfile.TemporaryDirectory(prefix=".folio-export-")
    try:
        chunks_file = Path(tmp.name) / CHUNKS_NAME
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="REPEATABLE READ")
            with conn.begin():
                if not S._has_table(conn, "Folders"):
                    raise FolderRefused("this database has no Folio tables (no \"Folders\" table)")
                folder = _find_folder(conn, folder_id, owner_key, name)
                if require_owner is not None and folder.get("user_token") != require_owner:
                    raise FolderRefused(f"there is no folder {folder['id']}", code="not_found")
                jobs = [j for j in S.active_jobs(conn, now) if j["folder_id"] == folder["id"]]
                if jobs:
                    raise S.JobsRunning(f"{len(jobs)} index job(s) are pending or running on folder "
                                        f"{folder['name']!r}; export it once they finish")
                files = [_row(r) for r in conn.execute(text(
                    'SELECT * FROM "Files" WHERE folder_id = :f ORDER BY upload_time, id'),
                    {"f": folder["id"]}).mappings()]
                indices = [_row(r) for r in conn.execute(text(
                    'SELECT * FROM "FileIndices" WHERE folder_id = :f ORDER BY file_id'),
                    {"f": folder["id"]}).mappings()]
                table = f"data_{folder['id']}_{folder['vector_table_uuid']}"
                has_table = S._has_table(conn, table)
                dimension = _vector_dimension(conn, table) if has_table else None
                segmenter = None
                n_chunks = 0
                with open(chunks_file, "w", encoding="utf-8") as out:
                    if has_table:
                        from src.domain.rag.vector_store_manager import read_tsv_segmenter
                        segmenter = read_tsv_segmenter(conn, table)
                        log(f"reading the chunks of {folder['name']!r}")
                        rows = conn.execution_options(stream_results=True).execute(text(
                            f"SELECT text, metadata_::text AS metadata, node_id, embedding::text AS embedding, "
                            f"text_search_tsv::text AS tsv FROM {_quote(table)} ORDER BY id"))
                        for r in rows.mappings():
                            out.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
                            n_chunks += 1
                settings = S.effective_settings(conn, rag_conf)
                revision = conn.execute(text("SELECT version_num FROM alembic_version")).scalar() \
                    if S._has_table(conn, "alembic_version") else None
        folder_doc = {
            "folder": {"name": folder["name"], "description": folder.get("description")},
            # Where the files sat here (owner directory included) means nothing elsewhere.
            "files": [{k: v for k, v in f.items() if k != "file_path"} for f in files],
            "file_indices": indices,
            "vector_table": {"exists": has_table, "dimension": dimension, "segmenter": segmenter},
        }
        folder_file = Path(tmp.name) / FOLDER_JSON
        folder_file.write_text(json.dumps(folder_doc, ensure_ascii=False, indent=1), encoding="utf-8")

        if out_path is None:
            if out_dir is None:
                raise ValueError("out_dir or out_path is required")
            out_path = Path(out_dir) / export_name(folder["name"], now)
        out_path = Path(out_path)
        if out_path.exists():
            raise FolderRefused(f"{out_path} already exists")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = ArchiveWriter(out_path, FOLDER_EXPORT)
        try:
            writer.add_file(FOLDER_JSON, folder_file)
            writer.add_file(CHUNKS_NAME, chunks_file)
            missing: List[str] = []
            n_bytes = 0
            for f in files:
                src = _stored_file(f["file_path"], storage_root)
                if src is None or not src.is_file() or src.is_symlink():
                    missing.append(f["file_name"])
                    continue
                with open(src, "rb") as fh:
                    writer.add_open_file(FILES_PREFIX + f["id"], fh)
                n_bytes += writer.sizes[FILES_PREFIX + f["id"]]
            if missing:
                warnings.append(f"{len(missing)} stored file(s) are missing and were not exported "
                                f"(e.g. {missing[0]!r}); their records are, and import as missing files")
            indexed = [i for i in indices if i.get("status") == "indexed"]
            models = sorted({(i.get("embedding_model"), i.get("embedding_dimension")) for i in indexed},
                            key=lambda t: (str(t[0]), t[1] or 0))
            manifest = {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "created_at": now.isoformat(),
                "folio_version": S.current_version(),
                "alembic_revision": revision,
                "embedding": settings["embedding"],
                "segmenter": settings["segmenter"],
                "indexed_embeddings": [{"model": m, "dimension": d,
                                        "files": sum(1 for i in indexed
                                                     if (i.get("embedding_model"), i.get("embedding_dimension"))
                                                     == (m, d))} for m, d in models],
                "vector_table": folder_doc["vector_table"],
                "folder": {"name": folder["name"], "files": len(files), "indexed_files": len(indexed),
                           "chunks": n_chunks, "bytes": n_bytes, "missing_files": len(missing)},
                "warnings": warnings,
                "checksums": dict(writer.checksums),
            }
            writer.add_manifest(manifest)
            path = writer.commit()
        except BaseException:
            writer.abort()
            raise
    finally:
        engine.dispose()
        tmp.cleanup()
    return ExportResult(path=path, size=path.stat().st_size, manifest=manifest, warnings=warnings)


# ---------------------------------------------------------------- import checks

def compatibility(manifest: Dict[str, Any], *, target: Dict[str, Any], revisions: List[str],
                  version: str, files_only: bool) -> Tuple[List[str], List[str]]:
    """(blocking problems, warnings); see :func:`check_compatibility`."""
    blocking, warnings = check_compatibility(manifest, target=target, revisions=revisions,
                                             version=version, files_only=files_only)
    return [msg + ("; " + FILES_ONLY_HINT if code == "embedding_mismatch" else "")
            for code, msg in blocking], warnings


def check_compatibility(manifest: Dict[str, Any], *, target: Dict[str, Any], revisions: List[str],
                        version: str, files_only: bool) -> Tuple[List[Tuple[str, str]], List[str]]:
    """([(code, problem)], warnings) for importing ``manifest`` into this Folio.

    ``target`` is :func:`src.backup.service.effective_settings` of this Folio. The
    chunks carry vectors from the models in ``indexed_embeddings`` with the table's
    dimension; this Folio embeds queries with ``target``'s model, so anything else is
    refused unless ``files_only``.
    """
    blocking: List[Tuple[str, str]] = []
    warnings: List[str] = []
    rev = manifest.get("alembic_revision")
    if rev and rev not in revisions:
        blocking.append(("newer_version", f"the export was made by a newer Folio ({manifest.get('folio_version')}, database "
                        f"revision {rev}) than this one ({version})"))
    elif S._version_tuple(manifest.get("folio_version")) > S._version_tuple(version):
        blocking.append(("newer_version", f"the export was made by Folio {manifest.get('folio_version')}, "
                         f"newer than this Folio ({version})"))
    folder = manifest.get("folder") or {}
    vt = manifest.get("vector_table") or {}
    if folder.get("chunks") and not files_only:
        want_model = (target.get("embedding") or {}).get("model")
        want_dim = (target.get("embedding") or {}).get("dimension")
        models = sorted({str(e.get("model")) for e in manifest.get("indexed_embeddings") or []
                         if e.get("model") != want_model or e.get("dimension") not in (None, want_dim)})
        problems = []
        if models:
            problems.append(f"its vectors were made with {', '.join(models)}")
        if vt.get("dimension") and vt.get("dimension") != want_dim:
            problems.append(f"they have dimension {vt.get('dimension')}")
        if problems:
            blocking.append(("embedding_mismatch",
                             f"{' and '.join(problems)}, but this Folio uses {want_model} ({want_dim})"))
    if files_only and folder.get("chunks"):
        warnings.append("imported without the index (files only): index the folder in the console")
    seg = (manifest.get("vector_table") or {}).get("segmenter")
    here = (target.get("segmenter") or {}).get("name")
    if not files_only and folder.get("chunks") and seg and here and seg != here:
        warnings.append(f"its full-text search was segmented with {seg}, this Folio uses {here}: run "
                        "Rebuild FTS on the folder")
    if folder.get("missing_files"):
        warnings.append(f"{folder['missing_files']} file(s) were already missing when the folder was exported")
    return blocking, warnings


# ---------------------------------------------------------------- import

def _remap_meta(meta: Dict[str, Any], ids: Dict[str, str], new_paths: Dict[str, str],
                folder_id: int, folder_name: str) -> Dict[str, Any]:
    """Point one chunk's metadata at the new file, folder and path (in place)."""
    def _map(value: Any) -> Any:
        if isinstance(value, str):
            if value in ids:
                return ids[value]
            if value.startswith("file_") and value[5:] in ids:
                return "file_" + ids[value[5:]]
        return value

    old_fid = meta.get("file_id")
    if "file_id" in meta:
        meta["file_id"] = _map(old_fid)
    if "file_path" in meta and isinstance(old_fid, str) and old_fid in new_paths:
        meta["file_path"] = new_paths[old_fid]
    if "folder_id" in meta:
        meta["folder_id"] = folder_id
    if "folder_name" in meta:
        meta["folder_name"] = folder_name
    for key in ("doc_id", "ref_doc_id", "document_id"):
        if key in meta:
            meta[key] = _map(meta[key])
    return meta


def remap_chunk_metadata(raw: str, ids: Dict[str, str], new_paths: Dict[str, str],
                         folder_id: int, folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    """(rewritten metadata JSON, new file id); (None, None) when the chunk's file is unknown.

    LlamaIndex keeps a serialized copy of the node in ``_node_content``: its metadata
    and relationships are rewritten the same way.
    """
    meta = json.loads(raw) if raw else {}
    if not isinstance(meta, dict):
        return None, None
    old_fid = meta.get("file_id")
    if not isinstance(old_fid, str) or old_fid not in ids:
        return None, None
    _remap_meta(meta, ids, new_paths, folder_id, folder_name)
    node = meta.get("_node_content")
    if isinstance(node, str) and node:
        try:
            content = json.loads(node)
        except json.JSONDecodeError:
            content = None
        if isinstance(content, dict):
            if isinstance(content.get("metadata"), dict):
                _remap_meta(content["metadata"], ids, new_paths, folder_id, folder_name)
            rels = content.get("relationships")
            if isinstance(rels, dict):
                for rel in rels.values():
                    for r in (rel if isinstance(rel, list) else [rel]):
                        if isinstance(r, dict) and isinstance(r.get("node_id"), str):
                            r["node_id"] = ids.get(r["node_id"], r["node_id"])
            meta["_node_content"] = json.dumps(content, ensure_ascii=False)
    return json.dumps(meta, ensure_ascii=False), ids[old_fid]


def _read_chunks(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ArchiveError(f"{CHUNKS_NAME} line {n} is not valid JSON: {e}") from e
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ArchiveError(f"{CHUNKS_NAME} line {n} is not a chunk")
            yield row


def _load_folder_doc(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ArchiveError(f"{FOLDER_JSON} is not valid JSON: {e}") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("folder"), dict) \
            or not isinstance(doc.get("files"), list) or not isinstance(doc.get("file_indices"), list):
        raise ArchiveError(f"{FOLDER_JSON} does not describe a folder")
    for f in doc["files"]:
        if not isinstance(f, dict) or not isinstance(f.get("id"), str) or not _UUID.match(f["id"]) \
                or not isinstance(f.get("file_name"), str):
            raise ArchiveError(f"{FOLDER_JSON} has a file without a valid id or name")
    if not all(isinstance(i, dict) for i in doc["file_indices"]):
        raise ArchiveError(f"{FOLDER_JSON} has an index record that is not an object")
    return doc


def _create_vector_table(conn, table: str, dimension: int, segmenter: Optional[str]) -> None:
    """The table LlamaIndex's PGVectorStore makes, with the keyword vector as plain data.

    The search vectors are copied as stored, so the column is never a generated one
    here (Folio drops the expression the first time it writes segmented vectors anyway);
    the segmenter stamp comes along so Rebuild FTS knows what built them.
    """
    q = _quote(table)
    short = table[len("data_"):]
    conn.execute(text(
        f"CREATE TABLE {q} (id BIGSERIAL PRIMARY KEY, text VARCHAR NOT NULL, metadata_ JSON, "
        f"node_id VARCHAR, embedding vector({int(dimension)}), text_search_tsv TSVECTOR)"))
    conn.execute(text(f"CREATE INDEX {_quote(short + '_idx')} ON {q} USING gin (text_search_tsv)"))
    conn.execute(text(f"CREATE INDEX {_quote(short + '_idx_1')} ON {q} ((metadata_ ->> 'ref_doc_id'))"))
    if segmenter:
        from src.domain.rag.vector_store_manager import _write_tsv_segmenter
        _write_tsv_segmenter(conn, table, segmenter)


def import_folder(archive: Path, *, db_url: str, storage_root: Path, owner: Optional[str] = None,
                  owner_key: Optional[str] = None,
                  name: Optional[str] = None, files_only: bool = False, check_only: bool = False,
                  rag_conf=None, revisions: Optional[List[str]] = None, version: Optional[str] = None,
                  now: Optional[datetime] = None, log: Callable[[str], None] = lambda m: None
                  ) -> ImportResult:
    """Verify ``archive`` and import it as a new folder of ``owner``.

    ``owner`` is resolved like ``--owner``; ``owner_key`` is an owner key used as it
    is (the console passes its session's). Raises FolderRefused on a version or
    embedding mismatch (see :func:`check_compatibility`), ArchiveError on a damaged
    archive.
    """
    from db.folder_claim_db import lock_owner, owner_names
    from src.adapter.folder_claim import free_name
    from src.storage.file_storage import _validate_safe_name, owner_dir_name
    from src.utils.names import MAX_NAME_LEN, normalize_name

    archive = Path(archive)
    manifest = read_manifest(archive, FOLDER_EXPORT)
    if owner_key is None:
        owner_key = resolve_owner(owner or "")
    else:
        try:
            owner_dir_name(owner_key)
        except ValueError as e:
            raise FolderRefused(f"{owner_key!r} is not a usable owner: {e}") from e
    wanted = normalize_name(name) if name else str((manifest.get("folder") or {}).get("name") or "")
    try:
        _validate_safe_name(wanted, "folder name")
        if len(wanted) > MAX_NAME_LEN:
            raise ValueError(f"folder name is longer than {MAX_NAME_LEN} characters")
    except ValueError as e:
        raise FolderRefused(str(e)) from e

    engine = S._engine(db_url)
    try:
        with engine.connect() as conn:
            if not S._has_table(conn, "Folders"):
                raise FolderRefused("this database has no Folio tables: start Folio once first")
            target = S.effective_settings(conn, rag_conf)
        problems, warnings = check_compatibility(
            manifest, target=target, revisions=revisions if revisions is not None else S.code_revisions(),
            version=version or S.current_version(), files_only=files_only)
        if problems:
            raise FolderRefused(
                "; ".join(m + ("; " + FILES_ONLY_HINT if c == "embedding_mismatch" else "") for c, m in problems),
                code=problems[0][0], reasons=[m for _, m in problems])
        if check_only:
            log("verifying every entry's checksum")
            verify_and_extract(archive, manifest, layout=FOLDER_EXPORT)
            return ImportResult(manifest=manifest, imported=False, warnings=warnings, owner=owner_key)

        now = now or datetime.now(timezone.utc)
        storage_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".folio-import-", dir=storage_root.parent))
        target_dir: Optional[Path] = None
        try:
            log("verifying and unpacking the archive")
            verify_and_extract(archive, manifest, staging, layout=FOLDER_EXPORT)
            doc = _load_folder_doc(staging / FOLDER_JSON)
            vt = doc.get("vector_table") or {}
            owner_root = storage_root / owner_dir_name(owner_key)
            with engine.begin() as conn:
                lock_owner(conn, owner_key)
                final = free_name(wanted, owner_names(conn, owner_key), lambda n: (owner_root / n).exists())
                table_uuid = uuid.uuid4()
                folder_id = conn.execute(text(
                    'INSERT INTO "Folders" (name, description, file_count, total_size, user_token, '
                    "vector_table_uuid, created_at, updated_at) VALUES (:n, :d, :c, :s, :o, :u, :t, :t) "
                    "RETURNING id"),
                    {"n": final, "d": doc["folder"].get("description"), "c": len(doc["files"]),
                     "s": sum(int(f.get("file_size") or 0) for f in doc["files"]), "o": owner_key,
                     "u": table_uuid, "t": now.replace(tzinfo=None)}).scalar()
                prefix = f"storage/{owner_dir_name(owner_key)}/{final}/"
                ids = {f["id"]: str(uuid.uuid4()) for f in doc["files"]}
                new_paths = {old: prefix + new for old, new in ids.items()}
                for f in doc["files"]:
                    row = {c: f.get(c) for c in FILE_COLUMNS}
                    row["tags"] = json.dumps(row["tags"] if isinstance(row["tags"], list) else [])
                    row.update(id=ids[f["id"]], folder_id=folder_id, file_path=new_paths[f["id"]])
                    cols = ", ".join(row)
                    vals = ", ".join(f"CAST(:{c} AS JSONB)" if c == "tags" else f":{c}" for c in row)
                    conn.execute(text(f'INSERT INTO "Files" ({cols}) VALUES ({vals})'), row)

                table = f"data_{folder_id}_{table_uuid}"
                n_chunks = dropped = 0
                indexed = not files_only and bool(vt.get("exists")) and bool(vt.get("dimension"))
                if indexed:
                    for i in doc["file_indices"]:
                        if i.get("file_id") not in ids:
                            continue
                        row = {c: i.get(c) for c in INDEX_COLUMNS}
                        if isinstance(row["index_id"], str) and row["index_id"].startswith("file_") \
                                and row["index_id"][5:] in ids:
                            row["index_id"] = "file_" + ids[row["index_id"][5:]]
                        row.update(id=str(uuid.uuid4()), file_id=ids[i["file_id"]], folder_id=folder_id,
                                   vector_store_table=table)
                        conn.execute(text(f'INSERT INTO "FileIndices" ({", ".join(row)}) VALUES '
                                          f'({", ".join(":" + c for c in row)})'), row)
                    _create_vector_table(conn, table, int(vt["dimension"]), vt.get("segmenter"))
                    log("writing the chunks")
                    batch: List[Dict[str, Any]] = []
                    insert = text(f"INSERT INTO {_quote(table)} (text, metadata_, node_id, embedding, "
                                  f"text_search_tsv) VALUES (:text, CAST(:meta AS JSON), :node_id, "
                                  f"CAST(:embedding AS vector), CAST(:tsv AS tsvector))")
                    for c in _read_chunks(staging / CHUNKS_NAME):
                        meta, _ = remap_chunk_metadata(c.get("metadata") or "", ids, new_paths,
                                                       folder_id, final)
                        if meta is None:
                            dropped += 1
                            continue
                        batch.append({"text": c["text"], "meta": meta, "node_id": c.get("node_id"),
                                      "embedding": c.get("embedding"), "tsv": c.get("tsv")})
                        if len(batch) >= _INSERT_BATCH:
                            conn.execute(insert, batch)
                            n_chunks += len(batch)
                            batch = []
                    if batch:
                        conn.execute(insert, batch)
                        n_chunks += len(batch)
                # Files go into place last, still inside the transaction: if anything
                # fails from here on the folder row is rolled back and the directory removed.
                target_dir = owner_root / final
                target_dir.mkdir(parents=True, exist_ok=False)
                for old, new in ids.items():
                    src = staging / FILES_PREFIX / old
                    if src.is_file():
                        os.replace(src, target_dir / new)
        except BaseException:
            if target_dir is not None:
                shutil.rmtree(target_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        if indexed:
            from src.utils.db_bootstrap import ensure_hnsw_for_table
            ensure_hnsw_for_table(engine, table)
    finally:
        engine.dispose()
    if dropped:
        warnings.append(f"{dropped} chunk(s) belonged to no exported file and were left out")
    if final != wanted:
        warnings.append(f"{owner_key!r} already has a folder named {wanted!r}; imported as {final!r}")
    return ImportResult(manifest=manifest, imported=True, warnings=warnings, folder_id=folder_id, name=final,
                        owner=owner_key, files=len(doc["files"]), chunks=n_chunks, indexed=indexed)


def manifest_summary(manifest: Dict[str, Any]) -> str:
    folder = manifest.get("folder") or {}
    emb = ", ".join(f"{e.get('model')} ({e.get('dimension')})" for e in manifest.get("indexed_embeddings") or [])
    return json.dumps({
        "created_at": manifest.get("created_at"),
        "folio_version": manifest.get("folio_version"),
        "folder": folder.get("name"),
        "files": folder.get("files"),
        "chunks": folder.get("chunks"),
        "embedding": emb or None,
        "segmenter": (manifest.get("vector_table") or {}).get("segmenter"),
    }, ensure_ascii=False)
