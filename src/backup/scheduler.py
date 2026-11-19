"""The server's backup schedule (``database.backup.enabled``) and its health view.

One background task on the server's event loop. The next backup is due
``interval_hours`` after the newest archive in the backup directory (whoever made it:
this schedule, ``main.py backup`` or a cron job), so restarting the server neither
skips nor repeats one. While an index job is running the round waits and retries every
``BUSY_RETRY`` (an archive made mid-index would not match its own FileIndices rows);
after ``BUSY_MAX_WAIT`` it backs up anyway and records the jobs in the manifest. After
each archive the oldest scheduled archives beyond ``keep`` are deleted. A failed round
is logged, shown in ``/health/detailed`` and retried after ``FAILURE_RETRY``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.log import get_api_logger

logger = get_api_logger()

FIRST_RUN_DELAY = timedelta(minutes=2)    # never compete with start-up itself
BUSY_RETRY = timedelta(minutes=10)
BUSY_MAX_WAIT = timedelta(hours=6)
FAILURE_RETRY = timedelta(hours=1)

_state: Dict[str, Any] = {
    "running": False,
    "last_attempt_at": None,
    "last_result": None,      # ok | failed | waiting
    "last_error": None,
    "next_run_at": None,
    "waiting_since": None,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def latest_archive(conf=None) -> Optional[Dict[str, Any]]:
    """The newest scheduled-style archive in the backup directory, or None."""
    from src.backup.archive import archive_time
    from src.backup.service import backup_dir, list_archives
    archives = list_archives(backup_dir(conf))
    if not archives:
        return None
    p = archives[-1]
    made = archive_time(p.name)
    try:
        size = p.stat().st_size
    except OSError:
        size = None
    return {"name": p.name, "created_at": _iso(made), "size_bytes": size,
            "age_hours": round((_now() - made).total_seconds() / 3600, 2)}


def status(conf=None) -> Dict[str, Any]:
    """Health view: ``{"status": healthy|degraded|disabled, ...}``. Never raises."""
    from src.backup.service import backup_settings
    conf = conf or backup_settings()
    try:
        latest = latest_archive(conf)
    except Exception:  # noqa: BLE001
        latest = None
    view: Dict[str, Any] = {
        "enabled": bool(conf.enabled),
        "interval_hours": conf.interval_hours,
        "keep": conf.keep,
        "latest": latest,
    }
    if not conf.enabled:
        view["status"] = "disabled"
        return view
    view.update({k: _state[k] for k in ("last_attempt_at", "last_result", "last_error", "next_run_at")})
    stale = latest is not None and latest["age_hours"] > 2 * conf.interval_hours + 1
    view["status"] = "degraded" if (_state["last_result"] == "failed" or stale) else "healthy"
    if stale:
        view["message"] = "the newest backup is older than two intervals"
    return view


def due_at(conf, now: Optional[datetime] = None) -> datetime:
    """When the next scheduled backup is due (from the newest archive)."""
    now = now or _now()
    latest = latest_archive(conf)
    earliest = now + FIRST_RUN_DELAY
    if latest is None:
        return earliest
    made = datetime.fromisoformat(latest["created_at"])
    return max(made + timedelta(hours=conf.interval_hours), earliest)


def run_once(conf, *, allow_running_jobs: bool = False) -> Dict[str, Any]:
    """One scheduled backup (blocking): archive, then prune. Returns a summary."""
    from src.backup.service import backup_dir, create_backup, default_storage_root, prune
    from src.config.config_manager import Config
    from src.utils.db_url import resolve_db_url
    out_dir = backup_dir(conf)
    result = create_backup(
        db_url=resolve_db_url(Config.get_database_config()), storage_root=default_storage_root(),
        out_dir=out_dir, allow_running_jobs=allow_running_jobs, bin_dir=conf.pg_bin_dir,
        rag_conf=getattr(Config.get_config_model(), "rag", None))
    removed = prune(out_dir, conf.keep)
    return {"path": result.path, "size": result.size, "removed": len(removed),
            "warnings": result.warnings}


async def backup_loop(conf) -> None:
    from src.backup.service import JobsRunning
    next_run = due_at(conf)
    while True:
        _state["next_run_at"] = _iso(next_run)
        delay = (next_run - _now()).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        now = _now()
        # After waiting BUSY_MAX_WAIT for index jobs, back up anyway (the jobs are recorded).
        allow = bool(_state["waiting_since"]) and now - _state["waiting_since"] >= BUSY_MAX_WAIT
        _state["running"] = True
        _state["last_attempt_at"] = _iso(now)
        try:
            summary = await asyncio.to_thread(run_once, conf, allow_running_jobs=allow)
        except asyncio.CancelledError:
            raise
        except JobsRunning as e:
            _state["waiting_since"] = _state["waiting_since"] or now
            _state["last_result"] = "waiting"
            _state["last_error"] = "waiting for index jobs to finish"
            logger.info(f"Scheduled backup postponed: {e}")
            next_run = _now() + BUSY_RETRY
            continue
        except Exception as e:  # noqa: BLE001
            _fail(e)
            next_run = _now() + FAILURE_RETRY
            continue
        finally:
            _state["running"] = False
        _state.update(last_result="ok", last_error=None, waiting_since=None)
        logger.info(f"Scheduled backup written: {summary['path'].name} ({summary['size']} bytes); "
                    f"{summary['removed']} old archive(s) removed")
        for w in summary["warnings"]:
            logger.warning(f"Scheduled backup: {w}")
        next_run = _now() + timedelta(hours=conf.interval_hours)


def _fail(e: Exception) -> None:
    _state["last_result"] = "failed"
    # The health endpoint is public: only the kind of failure, the details go to the log.
    _state["last_error"] = f"{type(e).__name__} (see the server log)"
    _state["waiting_since"] = None
    logger.error(f"Scheduled backup failed (retrying in {FAILURE_RETRY}): {e}")


def start(conf=None) -> Optional[asyncio.Task]:
    """Start the schedule when ``database.backup.enabled``; None otherwise."""
    from src.backup.service import backup_settings
    conf = conf or backup_settings()
    if not conf.enabled:
        return None
    logger.info(f"Backup schedule: every {conf.interval_hours} h, keeping {conf.keep}")
    return asyncio.get_running_loop().create_task(backup_loop(conf), name="backup-schedule")
