"""Integration test -- vector-store connections stay bounded as folders are touched.

PGVectorStore.from_params used to give every folder its own sync AND async engine
(5 + 10 pool each) and the store cache never released them, so the server's
max_connections ran out after ~80-100 folders. Touches more folders than the LRU
holds -- sync write + async query on each, so both engines really connect -- and
checks pg_stat_activity for this test database stays within the shared pools.
"""

import uuid

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from sqlalchemy import text

_DIM = 4
_FOLDERS = 24


def _backend_count(engine, dbname: str) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT count(*) FROM pg_stat_activity WHERE datname = :d"), {"d": dbname}
        ).scalar()


async def test_touching_many_folders_keeps_connections_bounded(itest_db):
    from db.db import get_async_engine, get_engine
    from src.config.config_manager import Config
    from src.domain.rag.vector_store_manager import VectorStoreManager

    db_conf = Config.get_database_config()
    vsm = VectorStoreManager(embed_dim=_DIM, hybrid_search=False)
    vsm._cache_size = 8  # fewer than _FOLDERS, so the LRU evicts along the way
    tables = []
    try:
        for i in range(_FOLDERS):
            folder_uuid = str(uuid.uuid4())
            store = vsm.get_or_create(900000 + i, folder_uuid)
            tables.append(vsm.physical_table_name(900000 + i, folder_uuid))
            store.add([TextNode(text=f"t{i}", embedding=[0.1 * (i + 1)] * _DIM)])
            res = await store.aquery(VectorStoreQuery(query_embedding=[0.1] * _DIM,
                                                      similarity_top_k=1))
            assert res.ids
        assert len(vsm._vector_stores) == 8

        limit = 2 * (db_conf.pool_size + db_conf.max_overflow)
        used = _backend_count(get_engine(), itest_db)
        # Shared engines: in practice one idle connection per engine here; never one per folder.
        assert used <= limit, f"{used} backends for {_FOLDERS} folders (limit {limit})"
        assert used < _FOLDERS
    finally:
        vsm.clear_cache()
        with get_engine().begin() as conn:
            for t in tables:
                conn.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
        await get_async_engine().dispose()
