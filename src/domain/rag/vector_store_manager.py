
"""Vector Store Manager

Manages the lifecycle and caching of PGVector stores for different folders.
Each folder gets its own isolated vector store identified by vector_table_uuid.
"""

import asyncio
import threading
from collections import OrderedDict
from typing import Optional
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url, text
from db.db import get_async_engine, get_engine, pool_kwargs
from src.utils.db_url import resolve_db_url
from src.log import get_api_logger
from src.config.config_manager import Config

logger = get_api_logger()

# Strong references to scheduled async-engine disposals (a bare create_task result
# can be garbage-collected before it runs).
_PENDING_DISPOSALS: set = set()

# Column llama-index's PGVectorStore uses for the FTS tsvector when
# hybrid_search=True (llama-index 0.14 builds it as
# to_tsvector(text_search_config, content)). Update if a future version renames it.
_TSV_COLUMN = "text_search_tsv"

# Batch size for the CKIP re-segmentation + UPDATE loop (bounds memory, commits
# incrementally).
_TSV_REBUILD_BATCH = 256


# The segmenter that built a table's tsvector is recorded as a comment on the
# column ("segmenter=<name>"). A table without one predates the stamp and was
# built by CKIP, the only segmenter that existed then.
_STAMP_PREFIX = "segmenter="
LEGACY_SEGMENTER = "ckip"


def _regclass(table_name: str) -> str:
    return '"' + table_name.replace('"', '""') + '"'


def read_tsv_segmenter(conn, table_name: str) -> Optional[str]:
    """The segmenter stamped on the table's tsvector column, or None if unstamped."""
    row = conn.execute(text(
        "SELECT col_description(a.attrelid, a.attnum) FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass(:t) AND a.attname = :col"),
        {"t": _regclass(table_name), "col": _TSV_COLUMN}).first()
    comment = (row[0] if row else None) or ""
    return comment[len(_STAMP_PREFIX):] if comment.startswith(_STAMP_PREFIX) else None


def _write_tsv_segmenter(conn, table_name: str, name: str) -> None:
    # COMMENT does not take bind parameters; name is one of SEGMENTERS.
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_")
    conn.execute(text(
        f'COMMENT ON COLUMN "{table_name}".{_TSV_COLUMN} IS \'{_STAMP_PREFIX}{safe}\''))


def stamp_new_table(conn, table_name: str, name: Optional[str], file_id) -> bool:
    """Stamp ``name`` on a table whose only keyword-indexed rows are ``file_id``'s.

    Called after one file's search vectors were written (by the indexer or a per-file
    rebuild). A brand-new folder gets its segmenter recorded; a folder that already
    holds rows from other files keeps its stamp (or lack of one = legacy CKIP), so a
    folder with rows from another segmenter stays flagged for Rebuild FTS. The caller
    commits. Returns True when it stamped.
    """
    if not name or read_tsv_segmenter(conn, table_name) is not None:
        return False
    others = conn.execute(text(
        f'SELECT EXISTS (SELECT 1 FROM "{table_name}" '
        f"WHERE metadata_->>'file_id' IS DISTINCT FROM :fid "
        f"AND {_TSV_COLUMN} IS NOT NULL)"), {"fid": str(file_id)}).scalar()
    if others:
        return False
    _write_tsv_segmenter(conn, table_name, name)
    return True


def stamp_tsv_segmenter(engine, table_name: str, name: str) -> None:
    """Record ``name`` as the segmenter that built the whole table's tsvector.

    Used by the per-file Rebuild-FTS job once every file has been re-segmented
    (a per-file ``rebuild_text_search_tsv`` call never stamps a table that still
    holds rows from other files). No-op for a table that does not exist.
    """
    if not name:
        return
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT to_regclass(:t) IS NOT NULL"),
                              {"t": _regclass(table_name)}).scalar()
        if not exists:
            return
        _write_tsv_segmenter(conn, table_name, name)
        conn.commit()


def fts_status(engine, table_name: str, configured: str) -> dict:
    """Which segmenter built the folder's keyword index vs the configured one.

    ``needs_rebuild`` is True when they differ: queries are segmented with the
    configured segmenter, so a stale index loses keyword recall until Rebuild FTS.
    A table that does not exist yet (never indexed) needs nothing.
    """
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT to_regclass(:t) IS NOT NULL"),
                              {"t": _regclass(table_name)}).scalar()
        if not exists:
            return {"indexed_with": None, "configured": configured, "needs_rebuild": False}
        stamp = read_tsv_segmenter(conn, table_name)
        if stamp is None:
            has_rows = conn.execute(text(
                f'SELECT EXISTS (SELECT 1 FROM "{table_name}" WHERE {_TSV_COLUMN} IS NOT NULL)'
            )).scalar()
            stamp = LEGACY_SEGMENTER if has_rows else None
    return {"indexed_with": stamp, "configured": configured,
            "needs_rebuild": stamp is not None and stamp != configured}


def rebuild_text_search_tsv(
    engine, table_name: str, segmenter, file_id: Optional[str] = None
) -> int:
    """Recompute a PGVector table's `text_search_tsv` from Python-segmented text.

    ``segmenter`` is the configured FTS segmenter (``text_segmenter.get_segmenter()``:
    CKIP, jieba or bigram). A whole-table rebuild stamps the table with its name;
    a per-file rebuild stamps only a table that has no other indexed rows yet
    (a new folder), so a folder with older rows from another segmenter stays
    flagged for Rebuild FTS.

    llama-index's native `to_tsvector('simple', text)` cannot segment space-less
    Chinese, so we overwrite the column with segmented tokens. The displayed
    `text` column is left untouched — only the search vector changes.

    llama-index creates `text_search_tsv` as a STORED GENERATED column
    (`GENERATED ALWAYS AS to_tsvector(config, text)`), which Postgres forbids
    UPDATEing. Segmentation runs in Python (Postgres cannot call CKIP, unlike the old pg_jieba
    text-search config), so we first detach the generated expression, turning it into
    a plain tsvector column we own. `DROP EXPRESSION IF EXISTS` is idempotent and a
    no-op once detached; existing values are kept until this UPDATE overwrites them.
    (The indexer writes the tsv itself in the same transaction as the rows — see
    chunk_store.replace_file_chunks.)

    Blocking (sync SQLAlchemy + segmentation); callers must run it off the event
    loop. Returns the number of leaf rows updated.
    """
    updated = 0
    # table_name is server-generated (int + UUID), not user input — no injection surface.
    detach_sql = text(
        f'ALTER TABLE "{table_name}" '
        f"ALTER COLUMN {_TSV_COLUMN} DROP EXPRESSION IF EXISTS"
    )
    # file_id scopes the rebuild to one file's rows (a per-file rebuild); None
    # re-segments the whole table. Without the scope, every single-file pass re-ran
    # segmentation over the entire folder.
    # Exactly what the indexer writes (chunk_store.replace_file_chunks): leaves are
    # searched by their retrieval text (prefix + breadcrumb + original text, see
    # chunk_identity; SEARCH_TEXT_SQL is plain `text` for rows indexed before that),
    # segmented by the configured segmenter; parents get no search vector.
    from src.domain.rag.chunk_identity import LEAF_ONLY_SQL, SEARCH_TEXT_SQL

    scope = " AND metadata_->>'file_id' = :fid" if file_id else ""
    params = {"fid": str(file_id)} if file_id else {}
    select_sql = text(
        f'SELECT id, {SEARCH_TEXT_SQL} FROM "{table_name}" WHERE {LEAF_ONLY_SQL}{scope}'
    )
    clear_parents_sql = text(
        f'UPDATE "{table_name}" SET {_TSV_COLUMN} = NULL '
        f"WHERE NOT {LEAF_ONLY_SQL} AND {_TSV_COLUMN} IS NOT NULL{scope}"
    )
    update_sql = text(
        f'UPDATE "{table_name}" '
        f"SET {_TSV_COLUMN} = to_tsvector('simple', :seg) WHERE id = :id"
    )
    name = getattr(segmenter, "name", None)
    with engine.connect() as conn:
        # Detach the generated expression once so the column becomes UPDATEable.
        conn.execute(detach_sql)
        conn.commit()
        if file_id and stamp_new_table(conn, table_name, name, file_id):
            conn.commit()
        conn.execute(clear_parents_sql, params)
        conn.commit()
        rows = conn.execute(select_sql, params).fetchall()
        if name and not file_id:
            _write_tsv_segmenter(conn, table_name, name)
            conn.commit()
        if not rows:
            return 0
        for start in range(0, len(rows), _TSV_REBUILD_BATCH):
            batch = rows[start:start + _TSV_REBUILD_BATCH]
            texts = [(r[1] or "") for r in batch]
            segmented = segmenter.segment_batch(texts)
            batch_params = [
                {"id": r[0], "seg": seg}
                for r, seg in zip(batch, segmented)
            ]
            conn.execute(update_sql, batch_params)
            conn.commit()
            updated += len(batch_params)
    logger.info(
        f"Rebuilt {_TSV_COLUMN} from {name or 'segmented'} tokens for {updated} row(s) "
        f"in {table_name}"
    )
    return updated


# Defaults when the database config section does not set them (see DatabaseConfig).
DEFAULT_CACHE_SIZE = 64
# Pool for a store's private engines in "per_folder" mode: one steady connection plus a
# small burst, per engine (sync + async), so 64 cached folders hold at most ~384.
_PER_FOLDER_POOL = {"pool_size": 1, "max_overflow": 2}


def _cache_size_from(db_conf) -> int:
    size = getattr(db_conf, "vector_store_cache_size", None)
    return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 \
        else DEFAULT_CACHE_SIZE


def _engine_mode_from(db_conf) -> str:
    mode = getattr(db_conf, "vector_store_engines", None)
    return mode if mode in ("shared", "per_folder") else "shared"


class VectorStoreManager:
    """Manages PGVector store lifecycle and caching.

    Each folder gets its own store (table data_{folder_id}_{uuid}) configured for
    hybrid (vector + BM25) search. Stores are kept in a bounded LRU cache keyed by
    table name.

    Connections: by default every store is built on the two shared engines
    (db.db.get_engine / get_async_engine), so the connection count is bounded by
    those pools no matter how many folders are touched. Left to itself,
    PGVectorStore.from_params creates a private sync AND async engine per store,
    each with SQLAlchemy's default 5 + 10 pool, and the old unbounded dict never
    released them: ~80-100 folders were enough to exhaust max_connections=200.
    database.vector_store_engines = "per_folder" restores private engines, but with a
    1 + 2 pool each, and they are disposed when the store leaves the cache.
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
        # LRU: most recently used at the end. get_or_create may run on worker
        # threads (adapters offload DB work), hence the lock.
        self._vector_stores: "OrderedDict[str, PGVectorStore]" = OrderedDict()
        self._lock = threading.Lock()

        # Get database URL from config
        config = Config.get_config_model()
        if config is None or config.database is None:
            raise RuntimeError("Database configuration is not loaded")

        self._db_conf = config.database
        # Resolved once (database.port applied), same as every other connection path.
        self._db_url = resolve_db_url(config.database)
        self._cache_size = _cache_size_from(config.database)
        self._engine_mode = _engine_mode_from(config.database)

        logger.info(
            f"VectorStoreManager initialized (embed_dim={embed_dim}, "
            f"text_search_config={text_search_config}, hybrid_search={hybrid_search}, "
            f"engines={self._engine_mode}, cache_size={self._cache_size})"
        )

    @property
    def cache_size(self) -> int:
        return self._cache_size

    def get_or_create(
        self,
        folder_id: int,
        vector_table_uuid: str
    ) -> PGVectorStore:
        """Get existing vector store or create a new one (cache-aside, LRU-bounded)."""
        table_name = f"{folder_id}_{vector_table_uuid}"

        with self._lock:
            store = self._vector_stores.get(table_name)
            if store is not None:
                self._vector_stores.move_to_end(table_name)
                logger.debug(f"Vector store cache HIT: {table_name}")
                return store

        logger.debug(f"Vector store cache MISS: {table_name}, creating new one")

        try:
            vector_store = self._create_vector_store(table_name)
        except Exception as e:
            logger.error(f"Failed to create vector store for {table_name}: {e}")
            raise

        evicted = []
        with self._lock:
            # Another thread may have created the same store meanwhile: keep theirs.
            existing = self._vector_stores.get(table_name)
            if existing is not None:
                self._vector_stores.move_to_end(table_name)
                evicted.append(vector_store)
                vector_store = existing
            else:
                self._vector_stores[table_name] = vector_store
                while len(self._vector_stores) > self._cache_size:
                    old_name, old_store = self._vector_stores.popitem(last=False)
                    logger.debug(f"Vector store cache EVICT: {old_name}")
                    evicted.append(old_store)
        for old_store in evicted:
            self._release(old_store)

        logger.info(
            f"Vector store ready: {table_name} "
            f"(folder_id={folder_id}, embed_dim={self._embed_dim})"
        )
        return vector_store

    def _create_vector_store(self, table_name: str) -> PGVectorStore:
        """Create a new PGVector store with hybrid search enabled."""
        # Do not build the HNSW index through LlamaIndex — its internal SQL does not
        # quote identifiers containing UUID hyphens and breaks:
        #     CREATE INDEX IF NOT EXISTS data_2_b09d9acf-d049-...
        #                                              ^ syntax error
        # Instead, src/utils/db_bootstrap.py:_ensure_hnsw_indexes() scans all data_*
        # tables at server boot and adds HNSW indexes (idempotently), covering both
        # existing tables and tables for folders added later.
        common = dict(
            table_name=table_name,
            embed_dim=self._embed_dim,
            hybrid_search=self._hybrid_search,
            text_search_config=self._text_search_config,
        )
        if self._engine_mode == "per_folder":
            async_url = self._db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            vector_store = PGVectorStore(
                connection_string=self._db_url,
                async_connection_string=async_url,
                create_engine_kwargs=pool_kwargs(self._db_conf, **_PER_FOLDER_POOL),
                **common,
            )
        else:
            # PGVectorStore takes ready-made engines (both or neither); it then never
            # creates its own. The connection strings are informational only, so
            # they carry no password.
            masked = make_url(self._db_url).render_as_string(hide_password=True)
            vector_store = PGVectorStore(
                connection_string=masked,
                async_connection_string=masked,
                engine=get_engine(),
                async_engine=get_async_engine(),
                **common,
            )

        logger.debug(
            f"PGVector store created: table={table_name}, "
            f"hybrid_search={self._hybrid_search}, embed_dim={self._embed_dim}, "
            f"engines={self._engine_mode} "
            f"(HNSW auto-managed at boot via db_bootstrap._ensure_hnsw_indexes)"
        )

        return vector_store

    def _release(self, store) -> None:
        """Dispose the engines a store owns ("per_folder" mode only).

        Shared-mode stores run on db.db's engines, which outlive every store, so
        dropping the reference is all there is to do. A private sync pool is closed
        right away. A private async engine's connections must be closed on the event
        loop that opened them: on a running loop the dispose is scheduled there; from
        a worker thread the pool is dropped without closing (close=False) and asyncpg
        closes the sockets as they are collected.
        """
        if self._engine_mode != "per_folder":
            return
        engine = getattr(store, "_engine", None)
        async_engine = getattr(store, "_async_engine", None)
        try:
            if engine is not None:
                engine.dispose()
        except Exception as e:
            logger.warning(f"Vector store sync engine dispose failed: {e}")
        if async_engine is None:
            return
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                task = loop.create_task(async_engine.dispose())
                _PENDING_DISPOSALS.add(task)
                task.add_done_callback(_PENDING_DISPOSALS.discard)
            else:
                async_engine.sync_engine.dispose(close=False)
        except Exception as e:
            logger.warning(f"Vector store async engine dispose failed: {e}")

    def clear_cache(self):
        """Clear all cached vector stores, disposing any engines they own."""
        with self._lock:
            stores = list(self._vector_stores.values())
            self._vector_stores.clear()
        for store in stores:
            self._release(store)
        logger.info(f"Vector store cache cleared ({len(stores)} entries removed)")

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
        with self._lock:
            store = self._vector_stores.pop(f"{folder_id}_{vector_table_uuid}", None)
        if store is not None:
            self._release(store)
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
