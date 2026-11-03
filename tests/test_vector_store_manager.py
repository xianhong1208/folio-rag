
"""Regression tests — VectorStoreManager's physical-table deletion API.

The raw SQL for DROP TABLE / DELETE chunk used to be scattered across adapters
(rag_maintenance / folder), duplicating the data_{folder_id}_{uuid} table-name
convention several times. These were consolidated into VectorStoreManager; this
test first locks the behavioral contract (table name, IF EXISTS / CASCADE, bind
params, errors swallowed not raised, cache invalidation) so the refactor has a
safety net.

Pure mock engine, no external dependencies (per the unit-testing principle in
ENVIRONMENTS.md).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.domain.rag.vector_store_manager import VectorStoreManager

_VSM = "src.domain.rag.vector_store_manager"


def _mock_engine():
    """Return (engine, conn): engine.connect() acts as a context manager yielding conn."""
    conn = MagicMock()
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    return engine, conn


def test_physical_table_name_convention():
    """Physical table name = data_{folder_id}_{uuid} — the single source of the table-name convention project-wide."""
    name = VectorStoreManager.physical_table_name(5, "abc-uuid")
    assert name == "data_5_abc-uuid"


@pytest.fixture()
def vsm():
    """Build a VectorStoreManager; __init__ needs config → patch it out."""
    fake_cfg = MagicMock()
    fake_cfg.database.url = "postgresql://x/y"
    fake_cfg.database.port = 5432
    with patch(f"{_VSM}.Config.get_config_model", return_value=fake_cfg):
        return VectorStoreManager(embed_dim=1024)


def test_drop_table_issues_if_exists_cascade(vsm):
    """DROP must carry IF EXISTS (the table may not exist) + CASCADE (drops the HNSW index too) + commit."""
    engine, conn = _mock_engine()
    with patch(f"{_VSM}.get_engine", return_value=engine):
        ok = vsm.drop_table(5, "abc-uuid")
    assert ok is True
    sql = str(conn.execute.call_args[0][0])
    assert 'DROP TABLE IF EXISTS "data_5_abc-uuid" CASCADE' in sql
    conn.commit.assert_called_once()


def test_drop_table_swallows_errors(vsm):
    """A DDL failure only logs and returns False, never propagating to interrupt the deletion flow."""
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("db down")
    with patch(f"{_VSM}.get_engine", return_value=engine):
        ok = vsm.drop_table(5, "abc-uuid")
    assert ok is False


def test_drop_table_invalidates_cache(vsm):
    """After drop, the corresponding cache entry (key = {folder_id}_{uuid}, no data_ prefix) must be cleared, otherwise a later get_or_create returns a stale store pointing at the dropped table."""
    vsm._vector_stores["5_abc-uuid"] = MagicMock()
    engine, _ = _mock_engine()
    with patch(f"{_VSM}.get_engine", return_value=engine):
        vsm.drop_table(5, "abc-uuid")
    assert "5_abc-uuid" not in vsm._vector_stores


def test_delete_chunks_by_file_uses_bind_param():
    """file_id must go through a bind param (:file_id), never f-string interpolation (injection surface)."""
    engine, conn = _mock_engine()
    conn.execute.return_value.rowcount = 3
    with patch(f"{_VSM}.get_engine", return_value=engine):
        n = VectorStoreManager.delete_chunks_by_file("data_5_abc-uuid", 42)
    assert n == 3
    args, kwargs = conn.execute.call_args
    sql = str(args[0])
    params = args[1]
    assert 'DELETE FROM "data_5_abc-uuid"' in sql
    assert ":file_id" in sql
    assert params == {"file_id": "42"}  # compared as a string against the metadata_ JSONB value
    conn.commit.assert_called_once()


def test_delete_chunks_by_file_swallows_errors():
    """A chunk-delete failure returns 0 without raising (consistent with existing adapter behavior)."""
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("boom")
    with patch(f"{_VSM}.get_engine", return_value=engine):
        n = VectorStoreManager.delete_chunks_by_file("data_5_abc-uuid", 42)
    assert n == 0


# ---- LRU cache + engine ownership (connection blow-up regression) -----------------

def _vsm_with(mode="per_folder", cache_size=3):
    fake_cfg = MagicMock()
    fake_cfg.database.url = "postgresql://u:secret@h/db"
    fake_cfg.database.port = 5444
    fake_cfg.database.vector_store_engines = mode
    fake_cfg.database.vector_store_cache_size = cache_size
    fake_cfg.database.pool_timeout = 30
    fake_cfg.database.pool_recycle = 1800
    with patch(f"{_VSM}.Config.get_config_model", return_value=fake_cfg):
        return VectorStoreManager(embed_dim=8)


def _fake_store_factory(created):
    """PGVectorStore stand-in: records kwargs and carries its own mock engines."""
    def _make(**kwargs):
        store = MagicMock(name=f"store-{kwargs['table_name']}")
        store.kwargs = kwargs
        store._engine = MagicMock(name="engine")
        store._async_engine = MagicMock(name="async_engine")
        created.append(store)
        return store
    return _make


def test_lru_evicts_oldest_and_disposes_its_engines():
    """Creating N > cache_size stores keeps only cache_size and disposes every evicted store's engines."""
    vsm = _vsm_with(cache_size=3)
    created = []
    with patch(f"{_VSM}.PGVectorStore", side_effect=_fake_store_factory(created)):
        for i in range(5):
            vsm.get_or_create(i, "u")
    assert list(vsm._vector_stores) == ["2_u", "3_u", "4_u"]
    for evicted in created[:2]:
        evicted._engine.dispose.assert_called_once()
        # No running loop here: the async pool is dropped without awaiting.
        evicted._async_engine.sync_engine.dispose.assert_called_once_with(close=False)
    for kept in created[2:]:
        kept._engine.dispose.assert_not_called()
        kept._async_engine.sync_engine.dispose.assert_not_called()


def test_per_folder_stores_get_small_pools():
    vsm = _vsm_with(cache_size=3)
    created = []
    with patch(f"{_VSM}.PGVectorStore", side_effect=_fake_store_factory(created)):
        vsm.get_or_create(1, "u")
    kw = created[0].kwargs
    assert kw["create_engine_kwargs"]["pool_size"] == 1
    assert kw["create_engine_kwargs"]["max_overflow"] == 2
    assert kw["create_engine_kwargs"]["pool_pre_ping"] is True
    assert kw["create_engine_kwargs"]["pool_recycle"] == 1800
    assert kw["async_connection_string"].startswith("postgresql+asyncpg://")
    assert "engine" not in kw


def test_cache_hit_refreshes_recency():
    vsm = _vsm_with(cache_size=2)
    created = []
    with patch(f"{_VSM}.PGVectorStore", side_effect=_fake_store_factory(created)):
        first = vsm.get_or_create(1, "u")
        vsm.get_or_create(2, "u")
        assert vsm.get_or_create(1, "u") is first      # hit: 1 becomes most recent
        vsm.get_or_create(3, "u")                       # evicts 2, not 1
    assert list(vsm._vector_stores) == ["1_u", "3_u"]
    assert len(created) == 3


async def test_eviction_on_running_loop_awaits_async_dispose():
    """On the event loop the async engine is disposed properly (scheduled), not dropped."""
    import asyncio
    from unittest.mock import AsyncMock
    vsm = _vsm_with(cache_size=1)
    created = []
    with patch(f"{_VSM}.PGVectorStore", side_effect=_fake_store_factory(created)):
        vsm.get_or_create(1, "u")
        created[0]._async_engine.dispose = AsyncMock()
        vsm.get_or_create(2, "u")
    await asyncio.sleep(0)
    created[0]._async_engine.dispose.assert_awaited_once()
    created[0]._async_engine.sync_engine.dispose.assert_not_called()


def test_shared_mode_injects_shared_engines_and_never_disposes():
    vsm = _vsm_with(mode="shared", cache_size=1)
    created = []
    eng, aeng = MagicMock(name="shared"), MagicMock(name="shared_async")
    with patch(f"{_VSM}.PGVectorStore", side_effect=_fake_store_factory(created)), \
         patch(f"{_VSM}.get_engine", return_value=eng), \
         patch(f"{_VSM}.get_async_engine", return_value=aeng):
        vsm.get_or_create(1, "u")
        vsm.get_or_create(2, "u")
        vsm.clear_cache()
    kw = created[0].kwargs
    assert kw["engine"] is eng and kw["async_engine"] is aeng
    assert "secret" not in kw["connection_string"]      # informational string, no password
    assert "create_engine_kwargs" not in kw
    for store in created:
        store._engine.dispose.assert_not_called()
        store._async_engine.sync_engine.dispose.assert_not_called()
    assert vsm._vector_stores == {}


def test_clear_cache_and_drop_table_dispose_owned_engines():
    vsm = _vsm_with(cache_size=5)
    created = []
    with patch(f"{_VSM}.PGVectorStore", side_effect=_fake_store_factory(created)):
        vsm.get_or_create(1, "a")
        vsm.get_or_create(2, "b")
    engine, _ = _mock_engine()
    with patch(f"{_VSM}.get_engine", return_value=engine):
        vsm.drop_table(1, "a")
    created[0]._engine.dispose.assert_called_once()
    created[1]._engine.dispose.assert_not_called()
    vsm.clear_cache()
    created[1]._engine.dispose.assert_called_once()
    assert vsm._vector_stores == {}


def test_release_survives_dispose_errors():
    vsm = _vsm_with(cache_size=1)
    created = []
    with patch(f"{_VSM}.PGVectorStore", side_effect=_fake_store_factory(created)):
        vsm.get_or_create(1, "u")
        created[0]._engine.dispose.side_effect = RuntimeError("x")
        created[0]._async_engine.sync_engine.dispose.side_effect = RuntimeError("y")
        vsm.get_or_create(2, "u")   # must not raise
    assert list(vsm._vector_stores) == ["2_u"]


def test_create_failure_propagates_and_caches_nothing():
    vsm = _vsm_with(cache_size=2)
    with patch(f"{_VSM}.PGVectorStore", side_effect=RuntimeError("bad")):
        with pytest.raises(RuntimeError):
            vsm.get_or_create(1, "u")
    assert vsm._vector_stores == {}


def test_invalid_cache_settings_fall_back_to_defaults(vsm):
    """A config without the new keys (MagicMock here) gets the documented defaults."""
    from src.domain.rag.vector_store_manager import DEFAULT_CACHE_SIZE
    assert vsm.cache_size == DEFAULT_CACHE_SIZE
    assert vsm._engine_mode == "shared"


def test_pool_kwargs_reads_config_and_overrides():
    from types import SimpleNamespace
    from db.db import pool_kwargs
    conf = SimpleNamespace(pool_size=7, max_overflow=3, pool_timeout=11, pool_recycle=99)
    assert pool_kwargs(conf) == {"pool_size": 7, "max_overflow": 3, "pool_timeout": 11,
                                 "pool_recycle": 99, "pool_pre_ping": True}
    assert pool_kwargs(conf, pool_size=1, max_overflow=2)["pool_size"] == 1
    # MagicMock / missing values fall back to SQLAlchemy-like defaults
    assert pool_kwargs(MagicMock()) == {"pool_size": 5, "max_overflow": 10, "pool_timeout": 30,
                                        "pool_recycle": 3600, "pool_pre_ping": True}
