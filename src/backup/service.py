"""Make and restore whole-instance backups.

Consistency: the database is dumped from one exported snapshot (``pg_export_snapshot``
in a REPEATABLE READ transaction, handed to ``pg_dump --snapshot``), and the same
snapshot lists the folders, the indexed embedding models and the files the database
refers to. ``storage/`` is archived right after the dump. Files are only ever added
or deleted (never rewritten in place), so the archive can at most hold a few files
uploaded after the snapshot (harmless) or miss a file deleted in between, which the
manifest lists under ``storage.missing``.

Index jobs: a backup is refused while an index job is pending or running (its chunks
and its FileIndices row would not match), unless the caller allows it; the manifest
then records those jobs, and after a restore Folio marks them interrupted at start-up
so the console offers Retry for their files.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from src.backup import pg
from src.backup.archive import (DUMP_NAME, FORMAT, FORMAT_VERSION, STORAGE_PREFIX, ArchiveError,
                                ArchiveWriter, archive_name, archive_time, read_manifest,
                                safe_member_path, verify_and_extract)

# A pending/running job whose row has not been updated for this long belongs to a server
# that is gone (running jobs heartbeat every 60 s); it does not block a backup.
JOB_STALE_AFTER = timedelta(minutes=15)
ACTIVE_JOB_STATUSES = ("pending", "running")
# Files a fresh Folio writes under storage/ on its own; they do not make storage/ "in use".
STORAGE_HOUSEKEEPING = {".console_session_secret", ".console_ended_sessions"}
_LOCK_NAME = ".folio-backup.lock"
_MAX_LISTED = 100


class BackupRefused(Exception):
    """A precondition failed (running jobs, non-empty target, mismatch, a backup already running)."""


class JobsRunning(BackupRefused):
    """Index jobs are pending or running (see the module docstring)."""


@dataclass
class BackupResult:
    path: Path
    size: int
    manifest: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


@dataclass
class RestoreResult:
    manifest: Dict[str, Any]
    restored: bool
    warnings: List[str] = field(default_factory=list)
    moved_aside: Optional[Path] = None
    folders: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------- configuration

def backup_settings():
    """``database.backup`` (defaults when the config has none)."""
    from src.config.config_manager import Config
    from src.config.model import BackupConfig
    try:
        conf = Config.get_database_config().backup
        if isinstance(conf, BackupConfig):
            return conf
    except Exception:  # noqa: BLE001
        pass
    return BackupConfig()


def backup_dir(conf=None) -> Path:
    """The archive directory; a relative ``dir`` is next to config/ and storage/."""
    conf = conf or backup_settings()
    d = Path(conf.dir)
    if d.is_absolute():
        return d
    from src.storage.file_storage import FileStorage
    return FileStorage.STORAGE_FILE_ROOT / d


def default_storage_root() -> Path:
    from src.storage.file_storage import FileStorage
    return FileStorage.STORAGE_ROOT


def code_revisions() -> List[str]:
    """Every migration revision this code knows, base to head."""
    from db.migrate import DatabaseMigrator, get_versions_dir
    return [r for r, _, _ in DatabaseMigrator._discover_revisions(get_versions_dir())]


def current_version() -> str:
    from src.version import get_version
    return get_version()


# ---------------------------------------------------------------- helpers

@contextmanager
def _exclusive(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / _LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise BackupRefused(f"another backup is already being written to {directory}") from e
        yield
    finally:
        os.close(fd)


def _engine(db_url: str):
    return create_engine(db_url, poolclass=NullPool)


def _has_table(conn, name: str) -> bool:
    return bool(conn.execute(text("SELECT to_regclass(:n) IS NOT NULL"), {"n": f'public."{name}"'}).scalar())


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def active_jobs(conn, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Index jobs that are pending/running and were updated recently (live ones)."""
    if not _has_table(conn, "IndexJobs"):
        return []
    now = now or datetime.now(timezone.utc)
    rows = conn.execute(text(
        'SELECT job_id, folder_id, status, kind, last_updated_at FROM "IndexJobs" '
        "WHERE status IN ('pending', 'running')")).mappings().all()
    live = []
    for r in rows:
        updated = _parse_iso(r["last_updated_at"])
        if updated is not None and now - updated > JOB_STALE_AFTER:
            continue
        live.append({"job_id": str(r["job_id"]), "folder_id": r["folder_id"],
                     "status": r["status"], "kind": r["kind"]})
    return live


def _override(overrides: Dict[str, Any], key: str) -> Tuple[bool, Any]:
    if key not in overrides:
        return False, None
    value = overrides[key]
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    return True, value


def _snapshot_info(conn, rag_conf) -> Dict[str, Any]:
    """What the snapshot holds: revision, folders, files, indexed models, settings."""
    info: Dict[str, Any] = {"alembic_revision": None}
    if _has_table(conn, "alembic_version"):
        info["alembic_revision"] = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()

    overrides: Dict[str, Any] = {}
    if _has_table(conn, "RuntimeSettings"):
        for key, value in conn.execute(text(
                'SELECT key, value FROM "RuntimeSettings" WHERE key IN '
                "('rag.embedding.model', 'rag.embedding.dimension', 'rag.retrieval.segmenter')")):
            overrides[key] = value
    emb = getattr(rag_conf, "embedding", None)
    has_m, model = _override(overrides, "rag.embedding.model")
    has_d, dim = _override(overrides, "rag.embedding.dimension")
    info["embedding"] = {
        "model": model if has_m else getattr(emb, "model", None),
        "dimension": int(dim if has_d else getattr(emb, "dimension", 0) or 0),
        "from_runtime_settings": [k for k, has in (("model", has_m), ("dimension", has_d)) if has],
    }
    has_s, seg = _override(overrides, "rag.retrieval.segmenter")
    info["segmenter"] = {"name": seg if has_s else _configured_segmenter(rag_conf),
                         "from_runtime_settings": has_s}

    folders = conn.execute(text(
        'SELECT id, name, vector_table_uuid FROM "Folders" ORDER BY id')).mappings().all()
    file_counts = dict(conn.execute(text(
        'SELECT folder_id, count(*) FROM "Files" GROUP BY folder_id')).all())
    info["folders"] = []
    for f in folders:
        table = f"data_{f['id']}_{f['vector_table_uuid']}"
        chunks = 0
        if _has_table(conn, table):
            chunks = conn.execute(text(f'SELECT count(*) FROM "{table}"')).scalar() or 0
        info["folders"].append({"id": f["id"], "name": f["name"], "files": file_counts.get(f["id"], 0),
                                "chunks": int(chunks), "vector_table": table})
    info["file_paths"] = [r[0] for r in conn.execute(text('SELECT file_path FROM "Files"'))]
    info["indexed_embeddings"] = []
    if _has_table(conn, "FileIndices"):
        for model, dim, n in conn.execute(text(
                'SELECT embedding_model, embedding_dimension, count(*) FROM "FileIndices" '
                "WHERE status = 'indexed' GROUP BY 1, 2 ORDER BY 3 DESC")):
            info["indexed_embeddings"].append({"model": model, "dimension": dim, "files": int(n)})
    info["vector_dimensions"] = sorted({int(r[0]) for r in conn.execute(text(
        "SELECT a.atttypmod FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relname LIKE 'data\\_%' AND a.attname = 'embedding' "
        "AND a.atttypmod > 0"))})
    info["server_version"] = conn.execute(text("SHOW server_version")).scalar()
    return info


def _configured_segmenter(rag_conf) -> Optional[str]:
    try:
        from src.config.roles import effective_segmenter
        return effective_segmenter(rag_conf) if rag_conf is not None else None
    except Exception:  # noqa: BLE001
        return None


def _arcname_for_db_path(file_path: str, storage_root: Path) -> Optional[str]:
    """The archive name a Files.file_path maps to, or None when it lies outside storage/."""
    p = Path(file_path)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(storage_root.resolve())
        except ValueError:
            return None
        return STORAGE_PREFIX + rel.as_posix()
    posix = PurePosixPath(p.as_posix())
    if posix.parts and posix.parts[0] == storage_root.name and len(posix.parts) > 1:
        return STORAGE_PREFIX + "/".join(posix.parts[1:])
    return None


def _add_storage(writer: ArchiveWriter, storage_root: Path, warnings: List[str],
                 exclude: Optional[Path] = None) -> Tuple[int, int]:
    """Archive every regular file under storage_root (``exclude``: the archive directory,
    when someone put it inside storage/)."""
    count = size = 0
    if not storage_root.is_dir():
        return 0, 0
    skip = exclude.resolve() if exclude is not None else None
    for dirpath, dirnames, filenames in os.walk(storage_root, followlinks=False):
        if skip is not None and Path(dirpath).resolve() == skip:
            dirnames[:] = []
            continue
        dirnames.sort()
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(storage_root).as_posix()
            arc = STORAGE_PREFIX + rel
            try:
                safe_member_path(arc)
            except ArchiveError:
                warnings.append(f"skipped a file whose name cannot be archived safely: {rel!r}")
                continue
            if full.is_symlink() or not full.is_file():
                warnings.append(f"skipped {rel!r}: not a regular file")
                continue
            try:
                with open(full, "rb") as f:
                    writer.add_open_file(arc, f)
            except FileNotFoundError:
                warnings.append(f"{rel!r} was deleted while the backup ran")
                continue
            size += writer.sizes[arc]
            count += 1
    return count, size


# ---------------------------------------------------------------- backup

def create_backup(*, db_url: str, storage_root: Path, out_dir: Optional[Path] = None,
                  out_path: Optional[Path] = None, allow_running_jobs: bool = False,
                  bin_dir: Optional[str] = None, rag_conf=None,
                  now: Optional[datetime] = None, log: Callable[[str], None] = lambda m: None
                  ) -> BackupResult:
    """Write one archive; raise BackupRefused / pg.PgToolError / ArchiveError."""
    now = now or datetime.now(timezone.utc)
    if out_path is None:
        if out_dir is None:
            raise ValueError("out_dir or out_path is required")
        out_path = Path(out_dir) / archive_name(now)
    out_path = Path(out_path)
    if out_path.exists():
        raise BackupRefused(f"{out_path} already exists")
    warnings: List[str] = []
    with _exclusive(out_path.parent), tempfile.TemporaryDirectory(
            prefix=".folio-backup-", dir=out_path.parent) as tmp:
        dump_file = Path(tmp) / DUMP_NAME
        engine = _engine(db_url)
        try:
            with engine.connect() as conn:
                conn = conn.execution_options(isolation_level="REPEATABLE READ")
                with conn.begin():
                    if not _has_table(conn, "Folders"):
                        raise BackupRefused("this database has no Folio tables (no \"Folders\" table)")
                    jobs = active_jobs(conn, now)
                    if jobs and not allow_running_jobs:
                        raise JobsRunning(
                            f"{len(jobs)} index job(s) are pending or running (folder "
                            f"{', '.join(str(j['folder_id']) for j in jobs[:5])}); wait for them to "
                            "finish, or pass --allow-running-jobs (those jobs are then marked "
                            "interrupted after a restore and their files can be retried)")
                    snapshot = conn.execute(text("SELECT pg_export_snapshot()")).scalar()
                    info = _snapshot_info(conn, rag_conf)
                    log(f"dumping the database ({len(info['folders'])} folder(s))")
                    pg.dump(db_url, dump_file, snapshot=snapshot, bin_dir=bin_dir)
        finally:
            engine.dispose()

        writer = ArchiveWriter(out_path)
        try:
            writer.add_file(DUMP_NAME, dump_file)
            log("archiving storage/")
            n_files, n_bytes = _add_storage(writer, storage_root, warnings, exclude=out_path.parent)
            archived = set(writer.checksums)
            missing, outside = [], 0
            for fp in info.pop("file_paths"):
                arc = _arcname_for_db_path(fp, storage_root)
                if arc is None:
                    outside += 1
                elif arc not in archived:
                    missing.append(arc)
            if missing:
                warnings.append(f"{len(missing)} file(s) the database refers to are not in storage/ "
                                f"(e.g. {missing[0]})")
            if outside:
                warnings.append(f"{outside} file record(s) point outside storage/ and were not archived")
            manifest = {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "created_at": now.isoformat(),
                "folio_version": current_version(),
                "alembic_revision": info["alembic_revision"],
                "embedding": info["embedding"],
                "segmenter": info["segmenter"],
                "indexed_embeddings": info["indexed_embeddings"],
                "vector_dimensions": info["vector_dimensions"],
                "folders": info["folders"],
                "running_jobs": jobs,
                "database": {"server_version": info["server_version"],
                             "pg_dump": pg.tool_version(pg.find_tool("pg_dump", bin_dir)),
                             "size": writer.sizes[DUMP_NAME]},
                "storage": {"files": n_files, "bytes": n_bytes, "missing": missing[:_MAX_LISTED],
                            "missing_count": len(missing), "outside_storage": outside},
                "warnings": warnings,
                "checksums": dict(writer.checksums),
            }
            writer.add_manifest(manifest)
            path = writer.commit()
        except BaseException:
            writer.abort()
            raise
    return BackupResult(path=path, size=path.stat().st_size, manifest=manifest, warnings=warnings)


def list_archives(directory: Path) -> List[Path]:
    """Scheduled-style archives in ``directory``, oldest first."""
    if not directory.is_dir():
        return []
    named = [(archive_time(p.name), p) for p in directory.iterdir() if p.is_file()]
    return [p for t, p in sorted((t, p) for t, p in named if t is not None)]


def prune(directory: Path, keep: int) -> List[Path]:
    """Delete all but the newest ``keep`` scheduled-style archives; return what was deleted."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    archives = list_archives(directory)
    doomed = archives[:-keep] if len(archives) > keep else []
    for p in doomed:
        p.unlink(missing_ok=True)
    return doomed


# ---------------------------------------------------------------- restore checks

def _version_tuple(v: Any) -> Tuple[int, ...]:
    out = []
    for part in str(v or "0").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def compatibility(manifest: Dict[str, Any], *, revisions: List[str], version: str,
                  rag_conf) -> Tuple[List[str], List[str]]:
    """(blocking problems, warnings) for restoring ``manifest`` into this Folio.

    Embedding settings saved in RuntimeSettings come back with the database, so they
    win after the restore; otherwise this instance's config/models.yaml decides. The
    vectors in the archive were made with the model the manifest records; a different
    model or dimension here would make every search compare incompatible vectors.
    """
    blocking: List[str] = []
    warnings: List[str] = []
    rev = manifest.get("alembic_revision")
    if rev and rev not in revisions:
        blocking.append(
            f"the database was migrated to revision {rev}, which this Folio ({version}) does not "
            f"know: the backup was made by a newer Folio (it says {manifest.get('folio_version')})")
    elif _version_tuple(manifest.get("folio_version")) > _version_tuple(version):
        blocking.append(f"the backup was made by Folio {manifest.get('folio_version')}, newer than this "
                        f"Folio ({version})")

    emb = manifest.get("embedding") or {}
    overridden = set(emb.get("from_runtime_settings") or [])
    conf_emb = getattr(rag_conf, "embedding", None)
    after_model = emb.get("model") if "model" in overridden else getattr(conf_emb, "model", None)
    after_dim = emb.get("dimension") if "dimension" in overridden else getattr(conf_emb, "dimension", None)
    data_models = {e.get("model") for e in manifest.get("indexed_embeddings") or [] if e.get("model")}
    data_dims = {e.get("dimension") for e in manifest.get("indexed_embeddings") or [] if e.get("dimension")}
    data_dims |= set(manifest.get("vector_dimensions") or [])
    if not data_models and emb.get("model"):
        data_models = {emb.get("model")}
    if not data_dims and emb.get("dimension"):
        data_dims = {emb.get("dimension")}
    wrong_models = sorted(m for m in data_models if m != after_model)
    wrong_dims = sorted(d for d in data_dims if d != after_dim)
    if wrong_models:
        blocking.append(f"the archive's vectors were made with embedding model {', '.join(wrong_models)}; "
                        f"this Folio uses {after_model}")
    if wrong_dims:
        blocking.append(f"the archive's vectors have dimension {', '.join(map(str, wrong_dims))}; this "
                        f"Folio uses {after_dim}")

    seg = manifest.get("segmenter") or {}
    after_seg = seg.get("name") if seg.get("from_runtime_settings") else _configured_segmenter(rag_conf)
    if seg.get("name") and after_seg and seg.get("name") != after_seg:
        warnings.append(f"full-text search was segmented with {seg.get('name')}, this Folio uses "
                        f"{after_seg}: run Rebuild FTS on each folder after the restore")
    if manifest.get("running_jobs"):
        warnings.append(f"{len(manifest['running_jobs'])} index job(s) were running when the backup was "
                        "made; Folio marks them interrupted at start-up and their files can be retried")
    if (manifest.get("storage") or {}).get("missing_count"):
        warnings.append(f"{manifest['storage']['missing_count']} file(s) were already missing from "
                        "storage/ when the backup was made")
    return blocking, warnings


# ---------------------------------------------------------------- restore

def _target_state(db_url: str) -> Dict[str, Any]:
    engine = _engine(db_url)
    try:
        with engine.connect() as conn:
            tables = [r[0] for r in conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"))]
            others = conn.execute(text(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                "AND pid <> pg_backend_pid()")).scalar() or 0
        return {"tables": tables, "connections": int(others)}
    finally:
        engine.dispose()


def _ensure_vector(db_url: str) -> None:
    """The ``vector`` extension, created when this role may; else a clear refusal."""
    engine = _engine(db_url)
    try:
        with engine.connect() as conn:
            if conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar():
                return
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            except Exception as e:  # noqa: BLE001
                raise BackupRefused(
                    f"the database {make_url(db_url).database!r} has no pgvector extension and this role "
                    f"cannot create it ({type(e).__name__}); ask the database administrator to run "
                    f"CREATE EXTENSION vector; in it, then restore again") from e
    finally:
        engine.dispose()


def _drop_tables(db_url: str, tables: List[str]) -> None:
    engine = _engine(db_url)
    try:
        with engine.begin() as conn:
            for t in tables:
                conn.execute(text('DROP TABLE IF EXISTS "{}" CASCADE'.format(t.replace('"', '""'))))
    finally:
        engine.dispose()


def _storage_in_use(storage_root: Path) -> List[str]:
    if not storage_root.exists():
        return []
    return sorted(p.name for p in storage_root.iterdir() if p.name not in STORAGE_HOUSEKEEPING)


def _restored_counts(db_url: str) -> List[Dict[str, Any]]:
    engine = _engine(db_url)
    try:
        with engine.connect() as conn:
            out = []
            for f in conn.execute(text('SELECT id, name, vector_table_uuid FROM "Folders" ORDER BY id')).mappings():
                table = f"data_{f['id']}_{f['vector_table_uuid']}"
                chunks = conn.execute(text(f'SELECT count(*) FROM "{table}"')).scalar() \
                    if _has_table(conn, table) else 0
                out.append({"id": f["id"], "name": f["name"], "chunks": int(chunks or 0)})
            return out
    finally:
        engine.dispose()


def restore_backup(archive: Path, *, db_url: str, storage_root: Path, force: bool = False,
                   allow_mismatch: bool = False, check_only: bool = False,
                   bin_dir: Optional[str] = None, rag_conf=None,
                   revisions: Optional[List[str]] = None, version: Optional[str] = None,
                   now: Optional[datetime] = None, log: Callable[[str], None] = lambda m: None
                   ) -> RestoreResult:
    """Verify ``archive`` and restore it into the configured database and storage/.

    Refuses (BackupRefused) on a version/model mismatch unless ``allow_mismatch``, on a
    database that already has tables or a storage/ that already has folders unless
    ``force``, and always while other connections to the database are open (a running Folio). With ``force`` the
    existing tables are dropped and the existing storage/ is moved aside, never deleted.
    """
    archive = Path(archive)
    manifest = read_manifest(archive)
    blocking, warnings = compatibility(
        manifest, revisions=revisions if revisions is not None else code_revisions(),
        version=version or current_version(), rag_conf=rag_conf)
    if blocking and not allow_mismatch:
        raise BackupRefused("; ".join(blocking) + " (pass --allow-mismatch to restore anyway)")
    warnings = [f"ignored: {b}" for b in blocking] + warnings

    if check_only:
        log("verifying every entry's checksum")
        verify_and_extract(archive, manifest)
        return RestoreResult(manifest=manifest, restored=False, warnings=warnings)

    from src.utils.db_bootstrap import _ensure_database_exists
    _ensure_database_exists(db_url, _QuietLogger(log))
    _ensure_vector(db_url)
    state = _target_state(db_url)
    if state["connections"]:
        # Never overridable: a running Folio would keep writing into what is being replaced.
        raise BackupRefused(f"{state['connections']} other connection(s) are open to the database "
                            f"{make_url(db_url).database!r}: stop Folio (and anything else using it) first")
    problems = []
    if state["tables"]:
        problems.append(f"the database {make_url(db_url).database!r} already has {len(state['tables'])} "
                        "table(s)")
    in_use = _storage_in_use(storage_root)
    if in_use:
        problems.append(f"storage/ already holds {len(in_use)} entr{'y' if len(in_use) == 1 else 'ies'}")
    if problems and not force:
        raise BackupRefused("; ".join(problems) + " (pass --force to replace them: tables are dropped, "
                            "storage/ is moved aside)")

    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    storage_root.parent.mkdir(parents=True, exist_ok=True)
    staging = storage_root.parent / f".folio-restore-{stamp}"
    staging.mkdir()
    moved_aside = None
    try:
        log("verifying and unpacking the archive")
        verify_and_extract(archive, manifest, staging)
        if state["tables"]:
            log(f"dropping {len(state['tables'])} existing table(s)")
            _drop_tables(db_url, state["tables"])
        log("restoring the database")
        pg.restore(db_url, staging / DUMP_NAME, staging / "toc.list", bin_dir=bin_dir)
        if storage_root.exists():
            moved_aside = storage_root.with_name(f"{storage_root.name}.before-restore-{stamp}")
            os.replace(storage_root, moved_aside)
        restored_storage = staging / STORAGE_PREFIX.rstrip("/")
        if restored_storage.is_dir():
            os.replace(restored_storage, storage_root)
        else:
            storage_root.mkdir(parents=True, exist_ok=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if moved_aside is not None:
        warnings.append(f"the previous storage/ was moved to {moved_aside}")
    folders = _restored_counts(db_url)
    expected = {f["id"]: f.get("chunks") for f in manifest.get("folders") or []}
    for f in folders:
        if expected.get(f["id"]) is not None and expected[f["id"]] != f["chunks"]:
            warnings.append(f"folder {f['name']!r} has {f['chunks']} chunks, the manifest says "
                            f"{expected[f['id']]}")
    return RestoreResult(manifest=manifest, restored=True, warnings=warnings,
                         moved_aside=moved_aside, folders=folders)


class _QuietLogger:
    """Adapts db_bootstrap's logger calls to the CLI's printer (info goes nowhere)."""

    def __init__(self, log: Callable[[str], None]):
        self._log = log

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        self._log(msg)

    error = warning


def manifest_summary(manifest: Dict[str, Any]) -> str:
    emb = manifest.get("embedding") or {}
    folders = manifest.get("folders") or []
    return json.dumps({
        "created_at": manifest.get("created_at"),
        "folio_version": manifest.get("folio_version"),
        "alembic_revision": manifest.get("alembic_revision"),
        "embedding": f"{emb.get('model')} ({emb.get('dimension')})",
        "segmenter": (manifest.get("segmenter") or {}).get("name"),
        "folders": len(folders),
        "chunks": sum(int(f.get("chunks") or 0) for f in folders),
        "files": (manifest.get("storage") or {}).get("files"),
    }, ensure_ascii=False)
