"""Startup check that the database and the storage directory belong together.

Folio keeps its state in two places: PostgreSQL (folders, file records, chunks, vectors) and the
storage directory (the uploaded files, ``storage/``; in Docker the ``folio-storage`` volume). Each can
go missing on its own: a volume not mounted, a different database URL, a restore of only one of them.
Starting anyway would look healthy and quietly lose data: an empty database next to old uploads shows
no folders, and a database without its files cannot re-index or download anything. So startup refuses
both cases with a message that says what is missing, instead of carrying on.

- **New database, old uploads**: the database had no schema before this start (no ``alembic_version``
  or ``Files`` table) while the storage directory already holds uploaded files.
- **Database without its files**: the database lists files, and none of the most recent ones exists
  under the storage directory. Some missing (a partial loss) is only logged as a warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from sqlalchemy import create_engine, inspect, text

# How many of the most recently uploaded files are looked for
SAMPLE_SIZE = 200


class StorageMismatchError(RuntimeError):
    """The database and the storage directory do not belong together; startup must stop."""


def uploaded_files(storage_root: Path) -> list[Path]:
    """Uploaded files under ``storage_root`` (``<owner>/<folder>/<file id>``). Files directly in the
    root (the console session secret and similar) and Folio's scratch directory
    (``.folio-work``: exports, uploads, import staging) do not count: a new install has them too."""
    from src.storage.file_storage import WORK_DIR_NAME
    if not storage_root.is_dir():
        return []
    found = []
    for owner in storage_root.iterdir():
        if not owner.is_dir() or owner.name == WORK_DIR_NAME:
            continue
        for path in owner.rglob("*"):
            if path.is_file():
                found.append(path)
                if len(found) >= 5:  # enough to say "there are uploads"
                    return found
    return found


def assess(*, database_was_new: bool, total_files: int, recent_paths: Iterable[str],
           resolve: Callable[[str], Path], storage_root: Path) -> tuple[Optional[str], Optional[str]]:
    """Return ``(error, warning)``: an error means refuse to start."""
    if database_was_new:
        uploads = uploaded_files(storage_root)
        if uploads:
            return (
                f"The database is new (it had no tables before this start), but the storage directory "
                f"{storage_root} already holds uploaded files (for example {uploads[0]}). The database "
                "that belongs to these files is missing: check DATABASE_URL, or that the database volume "
                "is mounted (Docker: folio-db), and restore it from the backup taken with this storage. "
                "To really start over, empty the storage directory first.",
                None,
            )
        return None, None

    recent = list(recent_paths)
    if total_files <= 0 or not recent:
        return None, None
    missing = [p for p in recent if not resolve(p).is_file()]
    if len(missing) == len(recent):
        return (
            f"The database lists {total_files} uploaded file(s), but none of the {len(recent)} most recent "
            f"is in the storage directory {storage_root} (for example {resolve(recent[0])}). The storage "
            "that belongs to this database is missing: check that the storage volume is mounted (Docker: "
            "folio-storage at /app/storage) or FOLIO_STORAGE_ROOT, and restore it from the backup taken "
            "with this database.",
            None,
        )
    if missing:
        return None, (
            f"{len(missing)} of the {len(recent)} most recent uploaded files are missing from "
            f"{storage_root} (for example {resolve(missing[0])}): those files cannot be re-indexed or "
            "downloaded."
        )
    return None, None


def database_is_new(db_url: str) -> bool:
    """True when the database has no schema yet (neither Alembic's version table nor the file
    table, so a pre-Alembic database counts as existing): call it before migrating."""
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            found = inspect(conn)
            return not (found.has_table("alembic_version") or found.has_table("Files"))
    finally:
        engine.dispose()


def check_storage_matches_database(db_url: str, *, database_was_new: bool, storage_root: Path,
                                   resolve: Callable[[str], Path], logger) -> None:
    """Raise :class:`StorageMismatchError` when the database and the storage directory do not
    belong together (see the module docstring). Call it after migrating."""
    total, recent = 0, []
    if not database_was_new:
        engine = create_engine(db_url)
        try:
            with engine.connect() as conn:
                total = conn.execute(text('SELECT count(*) FROM "Files"')).scalar() or 0
                recent = [row[0] for row in conn.execute(
                    text('SELECT file_path FROM "Files" ORDER BY upload_time DESC NULLS LAST LIMIT :n'),
                    {"n": SAMPLE_SIZE})]
        finally:
            engine.dispose()
    error, warning = assess(database_was_new=database_was_new, total_files=total, recent_paths=recent,
                            resolve=resolve, storage_root=storage_root)
    if error:
        raise StorageMismatchError(error)
    if warning:
        logger.warning(f"⚠️ {warning}")
    else:
        logger.info(f"✅ Storage check passed: {storage_root} matches the database ({total} file(s))")
