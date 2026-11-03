"""Run blocking DB helpers off the event loop.

The DB layer is synchronous SQLAlchemy (psycopg2) with a session per call, so a
helper can run on a worker thread safely. Calling one directly from an ``async``
function blocks the whole server (API, SSE, MCP) for the length of the round trip.

Do not use inside an ``except asyncio.CancelledError:`` block: awaiting during
cancellation re-raises CancelledError.
"""

from __future__ import annotations

import asyncio


async def run_db(fn, *args, **kwargs):
    """``await run_db(FolderDB.get, id=1)`` — run ``fn`` on a worker thread and return its result."""
    return await asyncio.to_thread(fn, *args, **kwargs)
