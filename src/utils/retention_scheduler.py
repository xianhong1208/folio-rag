"""Run the table-retention purges at startup and then on a fixed interval.

One background task on the server's event loop: purge immediately, sleep
``database.retention.interval_hours`` (default 24), repeat. The purge itself is
sync SQLAlchemy and runs on a worker thread. A failed round is logged and retried
at the next interval; it never stops the loop or the server.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from src.log import get_api_logger

logger = get_api_logger()


def _retention_conf():
    from src.config.config_manager import Config
    from src.config.model import RetentionConfig
    try:
        conf = Config.get_database_config().retention
        if isinstance(conf, RetentionConfig):
            return conf
    except Exception:
        pass
    return RetentionConfig()


async def run_once(conf=None) -> dict:
    """One retention round off the event loop; returns {table: rows deleted or None}."""
    from db.retention_db import run_retention
    conf = conf or _retention_conf()
    result = await asyncio.to_thread(run_retention, conf)
    purged = {k: v for k, v in result.items() if v}
    if purged:
        logger.info(f"Retention round purged rows: {purged}")
    return result


async def retention_loop(conf=None, interval_seconds: Optional[float] = None) -> None:
    conf = conf or _retention_conf()
    interval = interval_seconds if interval_seconds is not None else conf.interval_hours * 3600
    while True:
        try:
            await run_once(conf)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Retention round failed (retrying next interval): {e}")
        await asyncio.sleep(interval)


def start(conf=None) -> asyncio.Task:
    """Start the retention loop on the running loop; cancel the returned task on shutdown."""
    return asyncio.get_running_loop().create_task(
        retention_loop(conf), name="table-retention")
