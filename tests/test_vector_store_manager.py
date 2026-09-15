
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
