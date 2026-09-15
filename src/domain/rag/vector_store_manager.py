
"""Vector Store Manager

Manages the lifecycle and caching of PGVector stores for different folders.
Each folder gets its own isolated vector store identified by vector_table_uuid.
"""

from typing import Dict, Optional
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url, text
from db.db import get_engine
from src.log import get_api_logger
from src.config.config_manager import Config

logger = get_api_logger()

# Column llama-index's PGVectorStore uses for the FTS tsvector when
# hybrid_search=True (llama-index 0.14 builds it as
# to_tsvector(text_search_config, content)). Update if a future version renames it.
_TSV_COLUMN = "text_search_tsv"

# Batch size for the CKIP re-segmentation + UPDATE loop (bounds memory, commits
# incrementally).
_TSV_REBUILD_BATCH = 256


def rebuild_text_search_tsv(engine, table_name: str, segmenter) -> int:
    """Recompute a PGVector table's `text_search_tsv` from CKIP-segmented text.

    llama-index's native `to_tsvector('simple', text)` cannot segment space-less
    Chinese, so we overwrite the column with CKIP-segmented tokens. The displayed
    `text` column is left untouched — only the search vector is CKIP-derived.

    llama-index creates `text_search_tsv` as a STORED GENERATED column
    (`GENERATED ALWAYS AS to_tsvector(config, text)`), which Postgres forbids
    UPDATEing. CKIP runs in Python (Postgres cannot call it, unlike the old pg_jieba
    text-search config), so we first detach the generated expression, turning it into
    a plain tsvector column we own. `DROP EXPRESSION IF EXISTS` is idempotent and a
    no-op once detached; existing values are kept until this UPDATE overwrites them.
    (Rows llama-index inserts afterwards land with NULL tsv until the next rebuild,
    which runs after every add — see hierarchical_indexer.)

    Blocking (sync SQLAlchemy + CKIP inference); callers must run it off the event
    loop. Returns the number of rows updated.
    """
    updated = 0
    # table_name is server-generated (int + UUID), not user input — no injection surface.
    detach_sql = text(
        f'ALTER TABLE "{table_name}" '
        f"ALTER COLUMN {_TSV_COLUMN} DROP EXPRESSION IF EXISTS"
    )
    select_sql = text(f'SELECT id, text FROM "{table_name}"')
    update_sql = text(
        f'UPDATE "{table_name}" '
        f"SET {_TSV_COLUMN} = to_tsvector('simple', :seg) WHERE id = :id"
    )
    with engine.connect() as conn:
        # Detach the generated expression once so the column becomes UPDATEable.
        conn.execute(detach_sql)
        conn.commit()
        rows = conn.execute(select_sql).fetchall()
        if not rows:
            return 0
        for start in range(0, len(rows), _TSV_REBUILD_BATCH):
            batch = rows[start:start + _TSV_REBUILD_BATCH]
            texts = [(r[1] or "") for r in batch]
            segmented = segmenter.segment_batch(texts)
            params = [
                {"id": r[0], "seg": seg}
                for r, seg in zip(batch, segmented)
            ]
            conn.execute(update_sql, params)
            conn.commit()
            updated += len(params)
    logger.info(
        f"Rebuilt {_TSV_COLUMN} from CKIP tokens for {updated} row(s) "
        f"in {table_name}"
    )
    return updated


class VectorStoreManager:
    """Manages PGVector store lifecycle and caching.

    Each folder gets its own store (table data_{folder_id}_{uuid}) configured for
    hybrid (vector + BM25) search; created stores are cached per table name.
    """

    def __init__(self, embed_dim: int, text_search_config: str = "simple", hybrid_search: bool = True):
        """Initialize vector store manager.

        text_search_config is the Postgres config that tokenizes the space-joined
        result of Python-side CKIP segmentation ('simple' default; 'english' for
        English-only corpora).
        """
        self._embed_dim = embed_dim
        self._text_search_config = text_search_config
        self._hybrid_search = hybrid_search
        self._vector_stores: Dict[str, PGVectorStore] = {}

        # Get database URL from config
        config = Config.get_config_model()
        if config is None or config.database is None:
            raise RuntimeError("Database configuration is not loaded")

        self._db_url = config.database.url
        self._db_port = config.database.port

        logger.info(f"VectorStoreManager initialized (embed_dim={embed_dim}, text_search_config={text_search_config}, hybrid_search={hybrid_search})")

    def get_or_create(
        self,
        folder_id: int,
        vector_table_uuid: str
    ) -> PGVectorStore:
        """Get existing vector store or create a new one (cache-aside)."""
        table_name = f"{folder_id}_{vector_table_uuid}"

        if table_name in self._vector_stores:
            logger.debug(f"Vector store cache HIT: {table_name}")
            return self._vector_stores[table_name]

        logger.debug(f"Vector store cache MISS: {table_name}, creating new one")

        try:
            vector_store = self._create_vector_store(table_name)
            self._vector_stores[table_name] = vector_store

            logger.info(
                f"Vector store created successfully: {table_name} "
                f"(folder_id={folder_id}, embed_dim={self._embed_dim})"
            )

            return vector_store

        except Exception as e:
            logger.error(f"Failed to create vector store for {table_name}: {e}")
            raise

    def _create_vector_store(self, table_name: str) -> PGVectorStore:
        """Create a new PGVector store with hybrid search enabled."""
        url = make_url(self._db_url)
        port = url.port or self._db_port or 5432

        # Do not build the HNSW index through LlamaIndex — its internal SQL does not
        # quote identifiers containing UUID hyphens and breaks:
        #     CREATE INDEX IF NOT EXISTS data_2_b09d9acf-d049-...
        #                                              ^ syntax error
        # Instead, src/utils/db_bootstrap.py:_ensure_hnsw_indexes() scans all data_*
        # tables at server boot and adds HNSW indexes (idempotently), covering both
        # existing tables and tables for folders added later.
        vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=int(port),
            user=url.username,
            table_name=table_name,
            embed_dim=self._embed_dim,
            hybrid_search=self._hybrid_search,
            text_search_config=self._text_search_config,
        )

        logger.debug(
            f"PGVector store created: table={table_name}, "
            f"hybrid_search={self._hybrid_search}, embed_dim={self._embed_dim} "
            f"(HNSW auto-managed at boot via db_bootstrap._ensure_hnsw_indexes)"
        )

        return vector_store

    def clear_cache(self):
        """Clear all cached vector stores."""
        count = len(self._vector_stores)
        self._vector_stores.clear()
        logger.info(f"Vector store cache cleared ({count} entries removed)")

    def update_embed_dim(self, embed_dim: int) -> None:
        """Change the embedding dimension at runtime (admin hot-change).

        Cached PGVectorStore instances freeze the old dimension, so the cache is
        cleared. Physical table dimensions are fixed at creation, so folders must be
        reindexed to realign.
        """
        if embed_dim == self._embed_dim:
            return
        old = self._embed_dim
        self._embed_dim = embed_dim
        self.clear_cache()
        logger.info(f"[RUNTIME] embed_dim {old} → {embed_dim} (store cache cleared)")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'cached_stores': len(self._vector_stores),
            'table_names': list(self._vector_stores.keys()),
            'embed_dim': self._embed_dim
        }

    # Physical table deletion
    @staticmethod
    def physical_table_name(folder_id: int, vector_table_uuid) -> str:
        """A folder's pgvector physical table name (the single source of this convention project-wide).

        llama_index PGVectorStore prepends a data_ prefix to its internal table_name,
        so the actual persisted table is data_{folder_id}_{uuid}.
        """
        return f"data_{folder_id}_{vector_table_uuid}"

    @staticmethod
    def drop_table_by_name(table_name: str) -> bool:
        """DROP one pgvector physical table (CASCADE also removes its HNSW index/constraints). No cache side effects.

        For callers without a VSM instance (e.g. folder.py deleting an entire folder);
        callers that have an instance should use drop_table() (which also clears the
        cache). IF EXISTS: the table may not exist. On failure it only logs and returns
        False without raising — a single DDL failure must not interrupt the deletion
        flow. The table name is int + server-generated UUID, not user input, so there
        is no injection surface.
        """
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
                conn.commit()
            logger.info(f"Dropped vector table: {table_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to drop vector table {table_name}: {e}")
            return False

    def drop_table(self, folder_id: int, vector_table_uuid) -> bool:
        """DROP a folder's pgvector table and clear the corresponding cache (use this when you have an instance)."""
        # cache key has no data_ prefix (see get_or_create); clear it to avoid a stale store pointing at the dropped table
        self._vector_stores.pop(f"{folder_id}_{vector_table_uuid}", None)
        return self.drop_table_by_name(self.physical_table_name(folder_id, vector_table_uuid))

    @staticmethod
    def delete_chunks_by_file(table_name: str, file_id) -> int:
        """Delete all chunks of a file from a folder's pgvector table (matching the metadata_ JSONB).

        file_id goes through a bind param (:file_id) to prevent injection; table_name
        is a server-generated table name. Returns the number of rows deleted; on
        failure returns 0 without raising.
        """
        try:
            engine = get_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    text(f'DELETE FROM "{table_name}" WHERE metadata_->>\'file_id\' = :file_id'),
                    {"file_id": str(file_id)},
                )
                conn.commit()
                return result.rowcount or 0
        except Exception as e:
            # Table absent = no chunks to delete (hit by the pre-write purge on a
            # first index) — a normal outcome, so downgrade to debug to avoid noise;
            # other errors still log a warning.
            if "does not exist" in str(e).lower() or "undefinedtable" in type(e).__name__.lower():
                logger.debug(f"delete_chunks: table {table_name} absent (nothing to delete)")
            else:
                logger.warning(
                    f"Failed to delete chunks for file {file_id} from {table_name}: {e}"
                )
            return 0
