"""Integration test -- the boot-time HNSW sweep indexes only tables that lack one."""

import logging

from sqlalchemy import text


def _hnsw_indexes(conn, table):
    return conn.execute(text(
        "SELECT ic.relname FROM pg_index i JOIN pg_class ic ON ic.oid = i.indexrelid "
        "JOIN pg_am am ON am.oid = ic.relam "
        "WHERE i.indrelid = to_regclass(:t) AND am.amname = 'hnsw'"), {"t": f'"{table}"'}).scalars().all()


def test_sweep_backfills_missing_hnsw_once(itest_db):
    from db.db import get_engine
    from src.config.config_manager import Config
    from src.utils.db_bootstrap import _TABLES_MISSING_HNSW, _ensure_hnsw_indexes
    from src.utils.db_url import resolve_db_url

    eng = get_engine()
    bare, done, other = "data_990101_sweep-a", "data_990102_sweep-b", "data_990103_sweep-c"
    with eng.begin() as c:
        c.execute(text(f'CREATE TABLE "{bare}" (id int, embedding vector(3))'))
        c.execute(text(f'CREATE TABLE "{done}" (id int, embedding vector(3))'))
        c.execute(text(f'CREATE INDEX "my_custom_hnsw" ON "{done}" '
                       'USING hnsw (embedding vector_cosine_ops)'))
        c.execute(text(f'CREATE TABLE "{other}" (id int, note text)'))   # no embedding column
    try:
        with eng.connect() as c:
            missing = set(c.execute(text(_TABLES_MISSING_HNSW)).scalars().all())
        assert bare in missing and done not in missing and other not in missing

        _ensure_hnsw_indexes(resolve_db_url(Config.get_database_config()), logging.getLogger("t"))
        with eng.connect() as c:
            assert _hnsw_indexes(c, bare) == [f"{bare}_hnsw_idx"]
            assert _hnsw_indexes(c, done) == ["my_custom_hnsw"]     # untouched
            assert bare not in set(c.execute(text(_TABLES_MISSING_HNSW)).scalars().all())
        # Second boot: nothing left to do, still succeeds.
        _ensure_hnsw_indexes(resolve_db_url(Config.get_database_config()), logging.getLogger("t"))
    finally:
        with eng.begin() as c:
            for t in (bare, done, other):
                c.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
