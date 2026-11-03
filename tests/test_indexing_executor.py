"""Indexing work runs on its own bounded thread pool, never on the query path's.

Pins the routing contract of src/domain/rag/indexing_executor.py (ContextVar-based,
so the indexing pipeline keeps calling plain asyncio.to_thread), the config bounds
for rag.indexing.max_concurrent_jobs / executor_workers, and the env clamp.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.domain.rag import indexing_executor as ie


def _thread_name():
    return threading.current_thread().name


async def test_to_thread_routes_by_context():
    ex = ie.install(indexing_workers=2)
    try:
        assert ie.install() is ex            # idempotent per loop
        assert (await asyncio.to_thread(_thread_name)).startswith("asyncio")
        with ie.indexing_work():
            assert ie.in_indexing_work()
            name = await asyncio.to_thread(_thread_name)
            # Tasks spawned inside inherit the flag (gather / create_task copy the context).
            child = await asyncio.create_task(asyncio.to_thread(_thread_name))
            raw = await asyncio.get_running_loop().run_in_executor(None, _thread_name)
        assert name.startswith("indexing") and child.startswith("indexing")
        assert raw.startswith("indexing")
        assert not ie.in_indexing_work()     # reset on exit
        assert (await asyncio.to_thread(_thread_name)).startswith("asyncio")
    finally:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor())
        ex.shutdown(wait=False)


def test_routing_executor_bounds_and_shutdown():
    ex = ie.RoutingExecutor(indexing_workers=1, max_workers=2)
    assert ex.indexing_workers == 1 and ex._max_workers == 2
    gate = threading.Event()
    with ie.indexing_work():
        f1 = ex.submit(gate.wait, 5)
        f2 = ex.submit(_thread_name)
    # One indexing worker: the second indexing job waits while the general pool is free.
    assert ex.submit(_thread_name).result(timeout=5).startswith("asyncio")
    assert not f2.done()
    gate.set()
    assert f1.result(timeout=5) is True
    assert f2.result(timeout=5).startswith("indexing")
    ex.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        ex.submit(_thread_name)
    with pytest.raises(RuntimeError), ie.indexing_work():
        ex.submit(_thread_name)


def test_default_general_pool_size_matches_asyncio():
    import os
    ex = ie.RoutingExecutor(indexing_workers=3)
    try:
        assert ex._max_workers == min(32, (os.cpu_count() or 1) + 4)
    finally:
        ex.shutdown(wait=False)


def test_resolve_indexing_workers_from_config_or_default():
    cfg = MagicMock()
    cfg.rag.indexing.executor_workers = 7
    with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
        assert ie.resolve_indexing_workers() == 7
    cfg.rag.indexing.executor_workers = MagicMock()   # not an int -> default
    with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
        assert ie.resolve_indexing_workers() == ie.DEFAULT_INDEXING_WORKERS
    with patch("src.config.config_manager.Config.get_config_model", side_effect=RuntimeError):
        assert ie.resolve_indexing_workers() == ie.DEFAULT_INDEXING_WORKERS


def test_install_on_explicit_loop():
    loop = asyncio.new_event_loop()
    try:
        ex = ie.install(loop, indexing_workers=2)
        assert isinstance(ex, ie.RoutingExecutor) and ex.indexing_workers == 2
        assert ie.install(loop) is ex
    finally:
        loop.close()
        ex.shutdown(wait=False)


class TestIndexingConfigBounds:
    def test_defaults(self):
        from src.config.model import IndexingConfig
        c = IndexingConfig()
        assert c.max_concurrent_jobs == 4 and c.executor_workers == 16

    def test_large_concurrency_is_capped_not_rejected(self, caplog):
        from src.config.model import IndexingConfig, MAX_INDEXING_CONCURRENCY
        with caplog.at_level("WARNING"):
            c = IndexingConfig(max_concurrent_jobs=100)
        assert c.max_concurrent_jobs == MAX_INDEXING_CONCURRENCY == 32
        assert "max_concurrent_jobs=100" in caplog.text
        assert IndexingConfig(max_concurrent_jobs=32).max_concurrent_jobs == 32

    @pytest.mark.parametrize("kwargs", [{"max_concurrent_jobs": 0}, {"executor_workers": 0},
                                        {"executor_workers": 65}])
    def test_out_of_range_rejected(self, kwargs):
        from src.config.model import IndexingConfig
        with pytest.raises(ValidationError):
            IndexingConfig(**kwargs)


class TestEnvConcurrencyClamp:
    @pytest.mark.parametrize("env,expected", [("8", 8), ("100", 32), ("0", 1), ("-3", 1)])
    def test_env_value_clamped(self, env, expected):
        from src.adapter import rag_indexing
        with patch.object(rag_indexing.env_compat, "getenv", return_value=env):
            assert rag_indexing._resolve_indexing_concurrency() == expected


async def test_indexing_slot_marks_work_and_holds_semaphore():
    from src.adapter import rag_indexing
    sem = asyncio.Semaphore(1)
    with patch.object(rag_indexing, "_get_indexing_semaphore", return_value=sem):
        async with rag_indexing._indexing_slot():
            assert ie.in_indexing_work() and sem.locked()
    assert not ie.in_indexing_work() and not sem.locked()
