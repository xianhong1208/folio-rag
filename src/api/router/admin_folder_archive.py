"""Console: export one of your folders to an archive, and import such an archive.

The archive is the one ``main.py export-folder`` writes (src/backup/folder.py), so an
export from the console imports with the command line and the other way round.

- ``POST /api/admin/manage/folders/{id}/export`` writes the archive into a private
  work directory and answers with an ``export_id``; ``GET
  /api/admin/manage/exports/{export_id}`` then downloads it once (the copy is deleted
  when the download ends, or after :data:`EXPORT_TTL_S` if it is never fetched). Two
  steps, so a refusal (an index job running on the folder: 409) is a JSON answer and
  not a broken download, and the browser streams the file instead of holding it.
- ``POST /api/admin/manage/folders/import`` takes the archive as the raw request body
  (``Content-Type: application/x-tar``), streamed to a temporary file and refused with
  413 as soon as it passes ``database.backup.folder_import_max_mb``. Query parameters:
  ``name`` (the new folder's name; default the exported one) and ``files_only``.

Ownership: a folder is exported only when the session owns it (otherwise 404, as on
the rest of the console), and an export is downloaded only by the identity that made
it. An imported folder always belongs to the session's owner key (``console`` in open
mode); nothing in the request can choose another owner.

Refusals match the command line: another embedding model or dimension (409
``embedding_mismatch``; ``files_only=true`` imports the files without the index, to be
indexed here), an export from a newer Folio (409 ``newer_version``), a damaged or
foreign archive (400 ``damaged_archive``).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from src.api.router.admin_scope import owned_folder
from src.auth.console_session import ConsoleIdentity, require_console_session
from src.log import get_api_logger

logger = get_api_logger()

router = APIRouter(tags=["Admin: Folder archives"], dependencies=[Depends(require_console_session)])
_Session = Depends(require_console_session)

# Owner of folders created in open (auth disabled) mode; same as admin_manage.
OPEN_MODE_OWNER = "console"
# An export nobody downloads is deleted after this long (and so is a leftover upload).
EXPORT_TTL_S = 3600
WORK_DIR_NAME = ".folio-console-archives"
_EXPORT_ID = re.compile(r"^[0-9a-f]{32}$")
_META = "export.json"
_WRITE_CHUNK = 1024 * 1024


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, {"error": code, "error_description": message})


def _storage_root() -> Path:
    from src.backup.service import default_storage_root
    return default_storage_root()


def work_dir() -> Path:
    """Private directory for exports waiting to be downloaded and uploads being imported.

    Next to storage/ (the import unpacks there too, so files move into place without
    a copy); never inside it, so a backup or an owner directory never sees it.
    """
    d = _storage_root().resolve().parent / WORK_DIR_NAME
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d


def import_limit_bytes() -> int:
    from src.backup.service import backup_settings
    return int(backup_settings().folder_import_max_mb) * 1024 * 1024


def _newest_mtime(p: Path) -> float:
    try:
        newest = p.lstat().st_mtime
        if p.is_dir() and not p.is_symlink():
            for child in p.iterdir():
                newest = max(newest, child.lstat().st_mtime)
        return newest
    except OSError:
        return 0.0


def sweep(base: Path, now: Optional[float] = None, ttl: float = EXPORT_TTL_S) -> int:
    """Delete exports and uploads untouched for ``ttl`` seconds; returns how many."""
    now = time.time() if now is None else now
    removed = 0
    for entry in list(base.iterdir()) if base.is_dir() else []:
        if now - _newest_mtime(entry) < ttl:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except OSError:
                continue
        removed += 1
    return removed


def _rag_conf():
    from src.config.config_manager import Config
    model = Config.get_config_model()
    return getattr(model, "rag", None) if model is not None else None


def _db_url() -> str:
    from src.config.config_manager import Config
    from src.utils.db_url import resolve_db_url
    return resolve_db_url(Config.get_database_config())


# ---------------------------------------------------------------- export

@router.post("/api/admin/manage/folders/{folder_id}/export")
async def admin_export_folder(folder_id: int, ident: ConsoleIdentity = _Session):
    """Write one of your folders to an archive; download it from ``download_url``."""
    from src.backup import folder as F
    from src.backup.archive import ArchiveError
    from src.backup.service import JobsRunning

    owned_folder(folder_id, ident)  # 404 / 409 like every other folder action
    base = work_dir()
    sweep(base)
    export_id = secrets.token_hex(16)
    out = base / f"export-{export_id}"
    out.mkdir(mode=0o700)
    try:
        result = await run_in_threadpool(
            F.export_folder, db_url=_db_url(), storage_root=_storage_root(), folder_id=folder_id,
            out_dir=out, rag_conf=_rag_conf(),
            # Checked again inside the export's transaction (open mode reaches any folder).
            require_owner=ident.owner_key)
    except JobsRunning as e:
        shutil.rmtree(out, ignore_errors=True)
        raise _error(409, "index_job_running", str(e))
    except F.FolderRefused as e:
        shutil.rmtree(out, ignore_errors=True)
        if e.code == "not_found":
            raise HTTPException(404, f"Folder {folder_id} not found")
        raise _error(400, "export_refused", str(e))
    except ArchiveError as e:  # pragma: no cover - writing does not raise it
        shutil.rmtree(out, ignore_errors=True)
        raise _error(500, "export_failed", str(e))
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    (out / _META).write_text(json.dumps({"owner": ident.owner_key, "file": result.path.name}),
                             encoding="utf-8")
    m = result.manifest.get("folder") or {}
    logger.info(f"[ADMIN] folder {folder_id} exported by {ident.actor} ({result.size} bytes)")
    return {"data": {
        "export_id": export_id,
        "download_url": f"/api/admin/manage/exports/{export_id}",
        "file_name": result.path.name,
        "size": result.size,
        "folder": m.get("name"),
        "files": m.get("files"),
        "chunks": m.get("chunks"),
        "expires_in": EXPORT_TTL_S,
        "warnings": result.warnings,
    }, "message": f"Folder \"{m.get('name')}\" exported"}


@router.get("/api/admin/manage/exports/{export_id}")
async def admin_download_export(export_id: str, ident: ConsoleIdentity = _Session):
    """Download an export once; it is deleted when the download ends."""
    if not _EXPORT_ID.match(export_id):
        raise HTTPException(404, "Export not found")
    base = work_dir()
    src = base / f"export-{export_id}"
    try:
        meta = json.loads((src / _META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(404, "Export not found")
    if meta.get("owner") != ident.owner_key:
        raise HTTPException(404, "Export not found")
    # Claim it: a second download (or a sweep) cannot get the same copy.
    taken = base / f"served-{export_id}-{secrets.token_hex(4)}"
    try:
        os.rename(src, taken)
    except OSError:
        raise HTTPException(404, "Export not found")
    name = Path(str(meta.get("file") or "")).name
    path = taken / name
    if not name or not path.is_file():
        shutil.rmtree(taken, ignore_errors=True)
        raise HTTPException(404, "Export not found")
    return FileResponse(path, filename=name, media_type="application/x-tar",
                        background=BackgroundTask(shutil.rmtree, taken, ignore_errors=True))


# ---------------------------------------------------------------- import

def check_expanded_size(path: Path, limit: int) -> None:
    """Refuse an archive whose entries unpack to more than ``limit`` bytes (sparse
    entries could otherwise unpack far beyond the upload's own size)."""
    from src.backup.archive import ArchiveError
    total = 0
    try:
        with tarfile.open(path, "r:") as tar:
            for member in tar:
                if member.issparse():
                    raise ArchiveError(f"{member.name!r} is a sparse entry; folder exports have none")
                total += max(member.size, 0)
                if total > limit:
                    raise ArchiveError("the archive unpacks to more than the import limit")
    except tarfile.TarError as e:
        raise ArchiveError(f"not a readable tar archive: {e}") from e


def _content_length(request: Request) -> Optional[int]:
    try:
        v = request.headers.get("content-length")
        return int(v) if v is not None else None
    except ValueError:
        return None


def _too_large(limit: int) -> HTTPException:
    return _error(413, "too_large", f"The archive is larger than the import limit of {limit // (1024 * 1024)} MB "
                                    "(database.backup.folder_import_max_mb)")


@router.post("/api/admin/manage/folders/import")
async def admin_import_folder(
    request: Request,
    name: Optional[str] = Query(None, max_length=255, description="name of the new folder"),
    files_only: bool = Query(False, description="import the files without the index; index them here"),
    ident: ConsoleIdentity = _Session,
):
    """Import a folder archive (the raw request body) as a new folder of yours."""
    from src.backup import folder as F
    from src.backup.archive import ArchiveError

    limit = import_limit_bytes()
    declared = _content_length(request)
    if declared is not None and declared > limit:
        raise _too_large(limit)
    owner_key = ident.owner_key or OPEN_MODE_OWNER
    base = work_dir()
    sweep(base)
    fd, tmp_name = tempfile.mkstemp(prefix="upload-", suffix=".tar", dir=base)
    tmp = Path(tmp_name)
    try:
        received = 0
        with os.fdopen(fd, "wb") as out:
            async for chunk in request.stream():
                received += len(chunk)
                if received > limit:
                    raise _too_large(limit)
                out.write(chunk)
        if received == 0:
            raise _error(400, "empty_upload", "Send the archive as the request body")
        await run_in_threadpool(check_expanded_size, tmp, limit)
        result = await run_in_threadpool(
            F.import_folder, tmp, db_url=_db_url(), storage_root=_storage_root(), owner_key=owner_key,
            name=(name or "").strip() or None, files_only=files_only, rag_conf=_rag_conf())
    except ArchiveError as e:
        raise _error(400, "damaged_archive", f"The archive cannot be imported: {e}")
    except F.FolderRefused as e:
        if e.code in ("embedding_mismatch", "newer_version"):
            raise _error(409, e.code, "; ".join(e.reasons))
        raise _error(400, "import_refused", "; ".join(e.reasons))
    finally:
        tmp.unlink(missing_ok=True)
    _forget_cached(owner_key, result.folder_id)
    logger.info(f"[ADMIN] folder {result.folder_id} imported by {ident.actor} "
                f"({result.files} files, {result.chunks} chunks, indexed={result.indexed})")
    data: Dict[str, Any] = {
        "folder_id": result.folder_id, "name": result.name, "files": result.files,
        "chunks": result.chunks, "indexed": result.indexed, "warnings": result.warnings,
    }
    return {"data": data, "message": f"Imported as folder \"{result.name}\""}


def _forget_cached(owner_key: str, folder_id: Optional[int]) -> None:
    """The import wrote the folder with SQL of its own: drop the owner's cached list."""
    try:
        from db.cached_folderdb import CachedFolderDB
        folder = CachedFolderDB.get_by_id(folder_id) if folder_id is not None else None
        if folder is not None:
            CachedFolderDB.invalidate_folder_caches(folder)
        else:
            CachedFolderDB._cache.delete(f"folders:user:{owner_key}")
    except Exception as e:  # pragma: no cover - the cache is an optimisation
        logger.warning(f"[ADMIN] could not refresh the folder cache after an import: {e}")
