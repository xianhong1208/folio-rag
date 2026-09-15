
"""IndexJob persistence — the write-through projection of IndexingJobManager.

The DB is a passive projection; the manager remains the source of truth for live state.
Writes are best-effort and DB failures only log a warning (the system runs even if the
table does not exist).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .db import IndexJob
from db.db import Session as DBSession
from .baseDB import BaseDB
from src.log import get_db_logger

log = get_db_logger()


# Known writable fields; any other key in state_dict is dropped automatically (forward-compat)
_PERSISTED_FIELDS = {
    "job_id", "folder_id", "status", "total_files", "processed_files",
    "current_index", "current_file_id", "current_file_name",
    "last_file_status", "last_message", "message", "error",
    "skip_existing", "started_at", "completed_at", "last_updated_at",
    "result_summary", "scope_file_ids", "file_timings",
}


class IndexJobDB(BaseDB):
    """IndexJob CRUD."""

    @classmethod
    def get_orm_class(cls) -> type:
        return IndexJob

    @classmethod
    def get_log(cls):
        return log

    @classmethod
    def ensure_table(cls) -> None:
        """Create the IndexJobs table (IF NOT EXISTS, idempotent).

        The canonical schema is Alembic migration 20260819cafe01 (IndexJobs is in the
        chain and normal deployments create it via startup auto_migrate); this is a
        runtime fallback so the manager can still start in environments that have not
        run Alembic.
        """
        try:
            from db.db import get_engine
            cls.get_orm_class().__table__.create(get_engine(), checkfirst=True)
        except Exception as e:
            cls.get_log().warning(f"ensure_table failed (will fall back to in-memory only): {e}")

    @classmethod
    def upsert(cls, state_dict: Dict[str, Any]) -> None:
        """Upsert IndexJobState.to_dict() to the DB (write-through; failures are logged, not raised).

        Args:
            state_dict: Output of IndexJobState.to_dict(); unrecognized keys are dropped automatically.
        """
        try:
            payload = {k: v for k, v in state_dict.items() if k in _PERSISTED_FIELDS}
            job_id = payload.get("job_id")
            if not job_id:
                return
            with DBSession() as session:
                existing = session.query(cls.get_orm_class()).filter_by(job_id=job_id).first()
                if existing:
                    for k, v in payload.items():
                        setattr(existing, k, v)
                else:
                    obj = cls.get_orm_class()(**payload)
                    session.add(obj)
                session.commit()
        except Exception as e:
            cls.get_log().warning(f"IndexJobDB.upsert failed for job {state_dict.get('job_id')}: {e}")

    @classmethod
    def load_active(cls) -> List[Dict[str, Any]]:
        """Read all PENDING / RUNNING jobs from the DB (for startup cleanup).

        Returns:
            List of row dicts; an empty list on DB failure.
        """
        try:
            with DBSession() as session:
                rows = (
                    session.query(cls.get_orm_class())
                    .filter(cls.get_orm_class().status.in_(["pending", "running"]))
                    .all()
                )
                return [
                    {col.name: getattr(row, col.name) for col in row.__table__.columns}
                    for row in rows
                ]
        except Exception as e:
            cls.get_log().warning(f"IndexJobDB.load_active failed: {e}")
            return []

    @classmethod
    def delete_for_folder(cls, folder_id: int) -> int:
        """Delete all job rows for a folder (best-effort cleanup when a folder is deleted).

        Args:
            folder_id: The id of the deleted folder.

        Returns:
            Number of rows deleted; 0 on DB failure.
        """
        try:
            with DBSession() as session:
                deleted = (
                    session.query(cls.get_orm_class())
                    .filter_by(folder_id=folder_id)
                    .delete(synchronize_session=False)
                )
                session.commit()
                if deleted:
                    cls.get_log().info(
                        f"Deleted {deleted} IndexJob row(s) for folder {folder_id}"
                    )
                return deleted
        except Exception as e:
            cls.get_log().warning(f"IndexJobDB.delete_for_folder({folder_id}) failed: {e}")
            return 0

    @classmethod
    def purge_terminal_older_than(cls, days: int = 30) -> int:
        """Purge terminal job rows older than the retention period (called at startup to prevent unbounded growth).

        Timestamps are ISO-8601 UTC strings; within the same format, lexical order equals
        chronological order, so string comparison is used directly.

        Args:
            days: Retention window; terminal rows with last_updated_at earlier than now-days are deleted.

        Returns:
            Number of rows deleted; 0 on DB failure.
        """
        try:
            from datetime import datetime, timedelta, timezone
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=days)
            ).isoformat()
            orm = cls.get_orm_class()
            with DBSession() as session:
                deleted = (
                    session.query(orm)
                    .filter(orm.status.in_(
                        ["succeeded", "partial_success", "failed", "cancelled"]
                    ))
                    .filter(orm.last_updated_at < cutoff)
                    .delete(synchronize_session=False)
                )
                session.commit()
                if deleted:
                    cls.get_log().info(
                        f"Purged {deleted} terminal IndexJob row(s) older than {days}d"
                    )
                return deleted
        except Exception as e:
            cls.get_log().warning(f"IndexJobDB.purge_terminal_older_than failed: {e}")
            return 0

    @classmethod
    def get_by_id(cls, job_id) -> Optional[Dict[str, Any]]:
        """Fetch a single job row by job_id.

        Args:
            job_id: UUID or str.

        Returns:
            A row dict, or None (not found / DB failure).
        """
        try:
            with DBSession() as session:
                row = session.query(cls.get_orm_class()).filter_by(job_id=job_id).first()
                if not row:
                    return None
                return {col.name: getattr(row, col.name) for col in row.__table__.columns}
        except Exception as e:
            cls.get_log().warning(f"IndexJobDB.get_by_id({job_id}) failed: {e}")
            return None
