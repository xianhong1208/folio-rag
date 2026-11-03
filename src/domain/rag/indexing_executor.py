"""A bounded thread pool for indexing work, separate from the query path's.

``asyncio.to_thread`` and ``run_in_executor(None, ...)`` all share the event loop's
default executor (``min(32, cpu + 4)`` threads). Indexing fans out blocking work —
docling parsing, contextual-prefix LLM calls, embedding batches, pgvector writes,
Rebuild-FTS segmentation — so a few large jobs could hold every worker and leave the
query path's ``to_thread`` calls (auth, segmentation, rerank, DB reads) queued
behind them.

``RoutingExecutor`` replaces the loop's default executor. Submissions made while
``indexing_work()`` is active go to a dedicated indexing pool; everything else keeps
a general pool of the usual default size. Routing is by ContextVar, read when the
work is submitted, so the indexing pipeline keeps calling plain
``asyncio.to_thread`` and needs no changes; tasks spawned inside it (gather,
create_task) copy the context and inherit the flag.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional

from src.log import get_api_logger

logger = get_api_logger()

DEFAULT_INDEXING_WORKERS = 16

_INDEXING_WORK: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "folio_indexing_work", default=False)


def in_indexing_work() -> bool:
    """True while the current context is marked as indexing work."""
    return _INDEXING_WORK.get()


@contextlib.contextmanager
def indexing_work() -> Iterator[None]:
    """Mark the current context (and tasks/threads spawned from it) as indexing work.

    Use it around an awaited indexing call inside one coroutine; the flag is reset on
    exit, so the surrounding task is unaffected afterwards.
    """
    token = _INDEXING_WORK.set(True)
    try:
        yield
    finally:
        _INDEXING_WORK.reset(token)


class RoutingExecutor(ThreadPoolExecutor):
    """The loop's default executor: a general pool plus a bounded indexing pool.

    asyncio requires the default executor to be a ThreadPoolExecutor, so this class is
    the general pool itself and forwards indexing submissions to a second pool.
    """

    def __init__(self, indexing_workers: int, max_workers: Optional[int] = None):
        if max_workers is None:
            # The same size asyncio would give its own default executor.
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        super().__init__(max_workers=max_workers, thread_name_prefix="asyncio")
        self.indexing_workers = indexing_workers
        self._indexing_pool = ThreadPoolExecutor(
            max_workers=indexing_workers, thread_name_prefix="indexing")

    def submit(self, fn, /, *args, **kwargs):
        if _INDEXING_WORK.get():
            return self._indexing_pool.submit(fn, *args, **kwargs)
        return super().submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._indexing_pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


def resolve_indexing_workers() -> int:
    """config rag.indexing.executor_workers, else DEFAULT_INDEXING_WORKERS."""
    try:
        from src.config.config_manager import Config
        cfg = Config.get_config_model()
        value = cfg.rag.indexing.executor_workers
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    except Exception:
        pass
    return DEFAULT_INDEXING_WORKERS


def install(loop: Optional[asyncio.AbstractEventLoop] = None,
            indexing_workers: Optional[int] = None) -> RoutingExecutor:
    """Make a RoutingExecutor the default executor of ``loop`` (the running loop by default).

    Idempotent per loop: a loop that already has one keeps it.
    """
    loop = loop or asyncio.get_running_loop()
    current = getattr(loop, "_default_executor", None)
    if isinstance(current, RoutingExecutor):
        return current
    executor = RoutingExecutor(indexing_workers or resolve_indexing_workers())
    loop.set_default_executor(executor)
    logger.info(
        f"[INIT] Indexing thread pool installed (indexing_workers={executor.indexing_workers}, "
        f"general_workers={executor._max_workers})"
    )
    return executor
