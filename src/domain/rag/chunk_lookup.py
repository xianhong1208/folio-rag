
"""ChunkLookup — query PGVector table metadata directly via SQL, bypassing LlamaIndex retrieval.

Why this layer exists:
- LlamaIndex's retriever is "query → embedding → similarity ranking", which is unsuited to
  exact lookups like "fetch a single node by node_id" or "fetch a chunk_index range by file_id".
- PGVector table schema (auto-created by LlamaIndex):
    data_{folder_id}_{uuid}:
        id           bigserial primary key
        text         text
        metadata_    jsonb        (note the trailing underscore — a LlamaIndex convention)
        node_id      varchar
        embedding    vector(N)
- Querying metadata with the JSONB operator (->>) is 10-100x faster than going through LlamaIndex's abstraction layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import bindparam, text as sql_text

from src.log import get_api_logger

logger = get_api_logger()


class ChunkLookup:
    """Stateless utility class — everything is a staticmethod."""

    @staticmethod
    def _get_table_name(vector_store: PGVectorStore) -> str:
        """Return the full SQL table name (incl. `data_` prefix), tolerant across
        LlamaIndex versions; raises if it can't be determined."""
        name = getattr(vector_store, "table_name", None)
        if name:
            return name if name.startswith("data_") else f"data_{name}"

        # Fallback: read from _table_class
        table_class = getattr(vector_store, "_table_class", None)
        if table_class is not None:
            return table_class.__tablename__

        raise RuntimeError(
            "Cannot determine PGVector table name — LlamaIndex internal API changed?"
        )

    @staticmethod
    def _get_engine(vector_store: PGVectorStore):
        """Return a SQLAlchemy engine. Prefer the one LlamaIndex already built (avoids
        a dual-track pool); otherwise use the shared main engine. A self-built cache
        would have no health check and stay broken after a DB restart.
        """
        existing = getattr(vector_store, "_engine", None)
        if existing is not None:
            return existing

        from db.db import get_engine
        return get_engine()

    # Async batched API for the query hot path: batch N parent counts + M neighbor
    # windows into single queries on the async engine, avoiding N+1 serial round
    # trips. Sync versions below remain for non-hot paths (read mode / older callers).

    @staticmethod
    async def aget_parent_children_counts(
        vector_store: PGVectorStore, parent_ids: List[str],
    ) -> Dict[str, int]:
        """Batch-read chunk_count for multiple parents in one query.

        Returns {parent_id: chunk_count}; missing ids are absent. Fail-open on DB
        error (empty dict → caller treats as count=0, no merge, retrieval still works).
        """
        if not parent_ids:
            return {}
        table = ChunkLookup._get_table_name(vector_store)
        from db.db import get_async_engine

        query = sql_text(f"""
            SELECT node_id, (metadata_->>'chunk_count')::int AS cnt
            FROM "{table}"
            WHERE node_id IN :ids
              AND metadata_->>'node_role' = 'parent'
        """).bindparams(bindparam("ids", expanding=True))
        try:
            async with get_async_engine().connect() as conn:
                rows = (await conn.execute(query, {"ids": list(parent_ids)})).fetchall()
            return {r[0]: int(r[1]) for r in rows if r[1] is not None}
        except Exception as e:
            logger.warning(f"aget_parent_children_counts failed ({len(parent_ids)} ids): {e}")
            return {}

    @staticmethod
    async def afetch_nodes(
        vector_store: PGVectorStore, node_ids: List[str],
    ) -> Dict[str, TextNode]:
        """Batch-fetch the full content of multiple nodes (single query).

        Returns:
            {node_id: TextNode}; nodes not found are absent from the dict. On DB failure returns an empty dict.
        """
        if not node_ids:
            return {}
        table = ChunkLookup._get_table_name(vector_store)
        from db.db import get_async_engine

        query = sql_text(f"""
            SELECT node_id, text, metadata_
            FROM "{table}"
            WHERE node_id IN :ids
        """).bindparams(bindparam("ids", expanding=True))
        try:
            async with get_async_engine().connect() as conn:
                rows = (await conn.execute(query, {"ids": list(node_ids)})).fetchall()
            return {
                r[0]: TextNode(id_=r[0], text=r[1], metadata=r[2] or {})
                for r in rows
            }
        except Exception as e:
            logger.warning(f"afetch_nodes failed ({len(node_ids)} ids): {e}")
            return {}

    @staticmethod
    async def afetch_neighbor_windows(
        vector_store: PGVectorStore,
        requests: List[Tuple[str, int]],
        n: int,
    ) -> List[dict]:
        """Batch-fetch multiple (file_id, center_chunk_index±n) neighbor windows in one query.

        Uses an OR-chain of parameterized range conditions, then splits by window on
        the Python side; overlapping windows each take their own rows. Returns a list
        the same length/order as requests, each in fetch_neighbor_window's format.
        """
        empty = lambda c: {"text": "", "index_range": [c, c], "chunks": []}  # noqa: E731
        if not requests:
            return []
        table = ChunkLookup._get_table_name(vector_store)
        from db.db import get_async_engine

        conditions = []
        params: Dict[str, object] = {}
        for i, (file_id, center) in enumerate(requests):
            conditions.append(
                f"(metadata_->>'file_id' = :fid_{i} "
                f"AND (metadata_->>'chunk_index')::int BETWEEN :s_{i} AND :e_{i})"
            )
            params[f"fid_{i}"] = str(file_id)
            params[f"s_{i}"] = max(center - n, 0)
            params[f"e_{i}"] = center + n

        query = sql_text(f"""
            SELECT
                metadata_->>'file_id' AS fid,
                text,
                (metadata_->>'chunk_index')::int AS idx
            FROM "{table}"
            WHERE COALESCE(metadata_->>'node_role', 'leaf') = 'leaf'
              AND ({" OR ".join(conditions)})
            ORDER BY fid, idx ASC
        """)
        try:
            async with get_async_engine().connect() as conn:
                rows = (await conn.execute(query, params)).fetchall()
        except Exception as e:
            logger.warning(f"afetch_neighbor_windows failed ({len(requests)} windows): {e}")
            return [empty(c) for _, c in requests]

        # Bucket by file_id, then each window takes its own range (an overlapping window can include the same row)
        by_file: Dict[str, List[Tuple[int, str]]] = {}
        for fid, txt, idx in rows:
            by_file.setdefault(fid, []).append((idx, txt))

        results = []
        for file_id, center in requests:
            start, end = max(center - n, 0), center + n
            hits = [
                {"chunk_index": idx, "text": txt}
                for idx, txt in by_file.get(str(file_id), [])
                if start <= idx <= end
            ]
            if not hits:
                results.append(empty(center))
                continue
            results.append({
                "text": "\n\n".join(h["text"] for h in hits),
                "index_range": [hits[0]["chunk_index"], hits[-1]["chunk_index"]],
                "chunks": hits,
            })
        return results

    @staticmethod
    def get_parent_children_count(vector_store: PGVectorStore, parent_id: str) -> int:
        """Read chunk_count from a parent node's metadata (written by hierarchy.build_hierarchy)."""
        table = ChunkLookup._get_table_name(vector_store)
        engine = ChunkLookup._get_engine(vector_store)

        query = sql_text(f"""
            SELECT (metadata_->>'chunk_count')::int AS cnt
            FROM "{table}"
            WHERE node_id = :node_id
              AND metadata_->>'node_role' = 'parent'
            LIMIT 1
        """)
        try:
            with engine.connect() as conn:
                row = conn.execute(query, {"node_id": parent_id}).first()
                return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.warning(f"get_parent_children_count failed for {parent_id[:8]}: {e}")
            return 0

    @staticmethod
    def fetch_node(vector_store: PGVectorStore, node_id: str) -> Optional[TextNode]:
        """Fetch the full content of a single node (leaf or parent)."""
        table = ChunkLookup._get_table_name(vector_store)
        engine = ChunkLookup._get_engine(vector_store)

        query = sql_text(f"""
            SELECT node_id, text, metadata_
            FROM "{table}"
            WHERE node_id = :node_id
            LIMIT 1
        """)
        try:
            with engine.connect() as conn:
                row = conn.execute(query, {"node_id": node_id}).first()
                if not row:
                    return None
                return TextNode(
                    id_=row[0],
                    text=row[1],
                    metadata=row[2] or {},
                )
        except Exception as e:
            logger.warning(f"fetch_node failed for {node_id[:8]}: {e}")
            return None

    @staticmethod
    def fetch_neighbor_window(
        vector_store: PGVectorStore,
        file_id: str,
        center_chunk_index: int,
        n: int,
    ) -> dict:
        """Fetch the (file_id, chunk_index ± n) leaf chunks and join them in order.

        Returns:
            {
                "text": "the joined, expanded passage",
                "index_range": [start, end],
                "chunks": [{"chunk_index": i, "text": "..."}, ...]
            }
        """
        table = ChunkLookup._get_table_name(vector_store)
        engine = ChunkLookup._get_engine(vector_store)

        start = max(center_chunk_index - n, 0)
        end = center_chunk_index + n

        # Backward compatibility: legacy chunks (old flat index) have no node_role; treat them as leaves and fetch too
        query = sql_text(f"""
            SELECT
                node_id,
                text,
                (metadata_->>'chunk_index')::int AS idx
            FROM "{table}"
            WHERE metadata_->>'file_id' = :file_id
              AND COALESCE(metadata_->>'node_role', 'leaf') = 'leaf'
              AND (metadata_->>'chunk_index')::int BETWEEN :start AND :end
            ORDER BY idx ASC
        """)
        try:
            with engine.connect() as conn:
                rows = conn.execute(query, {
                    "file_id": str(file_id),
                    "start": start,
                    "end": end,
                }).fetchall()

            if not rows:
                return {"text": "", "index_range": [center_chunk_index, center_chunk_index], "chunks": []}

            chunks = [{"chunk_index": r[2], "text": r[1]} for r in rows]
            joined = "\n\n".join(c["text"] for c in chunks)
            return {
                "text": joined,
                "index_range": [chunks[0]["chunk_index"], chunks[-1]["chunk_index"]],
                "chunks": chunks,
            }
        except Exception as e:
            logger.warning(f"fetch_neighbor_window failed for file={file_id} center={center_chunk_index}: {e}")
            return {"text": "", "index_range": [center_chunk_index, center_chunk_index], "chunks": []}

    @staticmethod
    def fetch_file_full(vector_store: PGVectorStore, file_id: str, max_tokens: int = 30000) -> dict:
        """Fetch all leaf chunks of an entire file and join them in order into the full text (for mode='read').

        Returns:
            {
                "text": "the full text",
                "tokens_estimated": int,
                "truncated": bool,
                "chunk_count": int,
            }
        """
        table = ChunkLookup._get_table_name(vector_store)
        engine = ChunkLookup._get_engine(vector_store)

        # Backward compatibility: treat legacy chunks as leaves
        query = sql_text(f"""
            SELECT
                text,
                (metadata_->>'chunk_index')::int AS idx
            FROM "{table}"
            WHERE metadata_->>'file_id' = :file_id
              AND COALESCE(metadata_->>'node_role', 'leaf') = 'leaf'
            ORDER BY idx ASC
        """)
        try:
            with engine.connect() as conn:
                rows = conn.execute(query, {"file_id": str(file_id)}).fetchall()

            if not rows:
                return {"text": "", "tokens_estimated": 0, "truncated": False, "chunk_count": 0}

            from src.domain.rag.hierarchy import estimate_tokens

            joined_parts = []
            running_tokens = 0
            truncated = False
            for r in rows:
                t = r[0]
                t_tokens = estimate_tokens(t)
                if running_tokens + t_tokens > max_tokens:
                    truncated = True
                    break
                joined_parts.append(t)
                running_tokens += t_tokens

            return {
                "text": "\n\n".join(joined_parts),
                "tokens_estimated": running_tokens,
                "truncated": truncated,
                "chunk_count": len(joined_parts),
            }
        except Exception as e:
            logger.warning(f"fetch_file_full failed for file={file_id}: {e}")
            return {"text": "", "tokens_estimated": 0, "truncated": False, "chunk_count": 0}
