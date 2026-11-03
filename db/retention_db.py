"""Retention purges for the append-only tables (QueryLogs, SettingsAudit, IndexJobs).

None of these tables has another reclamation path, so without a purge they grow
for ever. Rows are deleted in batches, each in its own short transaction, so a
large backlog (the first run after an upgrade) never holds one long lock or bloats
a single transaction. Table and column names are module constants, never input.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import text

from src.log import get_db_logger

log = get_db_logger()

# IndexJobs rows that may be purged (an in-flight job is never touched).
TERMINAL_JOB_STATUSES = ("succeeded", "partial_success", "failed", "cancelled", "interrupted")

DEFAULT_BATCH_SIZE = 5000


@dataclass(frozen=True)
class _Target:
    table: str
    pk: str
    ts_column: str
    # IndexJobs stores ISO-8601 UTC strings (lexical order == time order); the
    # others are timestamptz.
    iso_string_ts: bool = False
    extra_where: str = ""


TARGETS: Dict[str, _Target] = {
    "query_logs": _Target("QueryLogs", "id", "created_at"),
    "settings_audit": _Target("SettingsAudit", "id", "changed_at"),
    "index_jobs": _Target(
        "IndexJobs", "job_id", "last_updated_at", iso_string_ts=True,
        extra_where="AND status IN ('" + "', '".join(TERMINAL_JOB_STATUSES) + "')"),
}


def purge_older_than(name: str, days: int, batch_size: int = DEFAULT_BATCH_SIZE,
                     engine=None, now: Optional[datetime] = None) -> int:
    """Delete ``name``'s rows older than ``days`` days, ``batch_size`` rows per transaction.

    ``days`` <= 0 keeps everything. Returns the number of rows deleted. Raises on DB
    errors (run_retention catches them per table).
    """
    if days is None or days <= 0:
        return 0
    target = TARGETS[name]
    if engine is None:
        from db.db import get_engine
        engine = get_engine()
    cutoff_dt = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    cutoff: Any = cutoff_dt.isoformat() if target.iso_string_ts else cutoff_dt
    sql = text(
        f'DELETE FROM "{target.table}" WHERE {target.pk} IN ('
        f'SELECT {target.pk} FROM "{target.table}" '
        f'WHERE {target.ts_column} < :cutoff {target.extra_where} LIMIT :n)'
    )
    total = 0
    while True:
        with engine.begin() as conn:
            deleted = conn.execute(sql, {"cutoff": cutoff, "n": batch_size}).rowcount or 0
        total += deleted
        if deleted < batch_size:
            break
    if total:
        log.info(f"Retention: purged {total} {target.table} row(s) older than {days}d")
    return total


def run_retention(conf=None, engine=None) -> Dict[str, Optional[int]]:
    """Apply every retention window from ``conf`` (database.retention).

    One table failing (missing table, DB hiccup) is logged and reported as None;
    the others still run.
    """
    from src.config.model import RetentionConfig
    conf = conf or RetentionConfig()
    windows = {
        "index_jobs": conf.index_jobs_days,
        "query_logs": conf.query_logs_days,
        "settings_audit": conf.settings_audit_days,
    }
    result: Dict[str, Optional[int]] = {}
    for name, days in windows.items():
        try:
            result[name] = purge_older_than(name, days, conf.batch_size, engine=engine)
        except Exception as e:
            log.warning(f"Retention purge of {TARGETS[name].table} failed: {e}")
            result[name] = None
    return result
