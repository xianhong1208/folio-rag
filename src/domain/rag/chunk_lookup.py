
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
    async def afetch_rows_in_ranges(
        vector_store: PGVectorStore,
        ranges: List[Tuple[str, int, int]],
    ) -> Dict[str, List[Tuple[int, str, Optional[str]]]]:
        """Leaf rows of several ``(file_id, lo, hi)`` chunk-index ranges, in one query.

        Overlapping ranges of a file are merged before querying, so each row is
        read once. Returns ``{file_id: [(chunk_index, text, parent_node_id), ...]}``
        sorted by chunk index; empty on DB failure (callers then show the hit alone).
        """
        merged: Dict[str, List[List[int]]] = {}
        for fid, lo, hi in ranges:
            merged.setdefault(str(fid), []).append([max(int(lo), 0), int(hi)])
        if not merged:
            return {}
        conditions = []
        params: Dict[str, object] = {}
        i = 0
        for fid, ivs in merged.items():
            ivs.sort()
            union: List[List[int]] = []
            for lo, hi in ivs:
                if union and lo <= union[-1][1] + 1:
                    union[-1][1] = max(union[-1][1], hi)
                else:
                    union.append([lo, hi])
            for lo, hi in union:
                conditions.append(
                    f"(metadata_->>'file_id' = :fid_{i} "
                    f"AND (metadata_->>'chunk_index')::int BETWEEN :s_{i} AND :e_{i})")
                params.update({f"fid_{i}": fid, f"s_{i}": lo, f"e_{i}": hi})
                i += 1

        table = ChunkLookup._get_table_name(vector_store)
        from db.db import get_async_engine

        query = sql_text(f"""
            SELECT
                metadata_->>'file_id' AS fid,
                (metadata_->>'chunk_index')::int AS idx,
                text,
                metadata_->>'parent_node_id' AS pid
            FROM "{table}"
            WHERE COALESCE(metadata_->>'node_role', 'leaf') = 'leaf'
              AND ({" OR ".join(conditions)})
            ORDER BY fid, idx ASC
        """)
        try:
            async with get_async_engine().connect() as conn:
                rows = (await conn.execute(query, params)).fetchall()
        except Exception as e:
            logger.warning(f"afetch_rows_in_ranges failed ({len(conditions)} ranges): {e}")
            return {}
        out: Dict[str, List[Tuple[int, str, Optional[str]]]] = {}
        for fid, idx, txt, pid in rows:
            out.setdefault(fid, []).append((idx, txt, pid))
        return out

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
    def fetch_file_full(vector_store: PGVectorStore, file_id: str, max_tokens: int = 30000,
                        offset: int = 0) -> dict:
        """One page of a file's text (mode='read'): its leaf chunks in order, from
        chunk index ``offset``, joined until ``max_tokens`` (estimated) is reached.

        The first chunk of a page is always included, so paging always advances.

        Returns:
            {
                "text": the page's text,
                "tokens_estimated": int,
                "truncated": bool,          # more chunks follow this page
                "chunk_count": int,         # chunks in this page
                "total_chunks": int,        # leaf chunks of the whole file
                "offset": int,              # chunk index this page starts at
                "next_offset": int | None,  # chunk index of the next page
            }
        """
        from src.domain.rag.token_budget import estimate_tokens

        offset = max(0, int(offset or 0))
        empty = {"text": "", "tokens_estimated": 0, "truncated": False, "chunk_count": 0,
                 "total_chunks": 0, "offset": offset, "next_offset": None}
        table = ChunkLookup._get_table_name(vector_store)
        engine = ChunkLookup._get_engine(vector_store)

        # Backward compatibility: treat legacy chunks as leaves
        query = sql_text(f"""
            SELECT text, idx, total FROM (
                SELECT
                    text,
                    (metadata_->>'chunk_index')::int AS idx,
                    count(*) OVER () AS total
                FROM "{table}"
                WHERE metadata_->>'file_id' = :file_id
                  AND COALESCE(metadata_->>'node_role', 'leaf') = 'leaf'
            ) leaves
            WHERE idx >= :offset
            ORDER BY idx ASC
        """)
        try:
            with engine.connect() as conn:
                rows = conn.execute(query, {"file_id": str(file_id), "offset": offset}).fetchall()
        except Exception as e:
            logger.warning(f"fetch_file_full failed for file={file_id}: {e}")
            return empty
        if not rows:
            return empty

        parts: List[str] = []
        running = 0
        next_offset: Optional[int] = None
        for text, idx, _ in rows:
            t_tokens = estimate_tokens(text)
            if parts and running + t_tokens > max_tokens:
                next_offset = int(idx)
                break
            parts.append(text)
            running += t_tokens
        return {
            "text": "\n\n".join(parts),
            "tokens_estimated": running,
            "truncated": next_offset is not None,
            "chunk_count": len(parts),
            "total_chunks": int(rows[0][2]),
            "offset": offset,
            "next_offset": next_offset,
        }

    @staticmethod
    def file_token_estimates(table_name: str) -> Dict[str, int]:
        """Estimated tokens of every file's leaf text in a folder table (mode='list').

        Computed from the stored text (CJK and other characters counted separately,
        see token_budget.py) in one aggregate query. Empty when the table does not
        exist yet or the query fails — callers fall back to a per-chunk guess.
        """
        from src.domain.rag.token_budget import CJK_TOKENS_PER_CHAR, OTHER_TOKENS_PER_CHAR
        from db.db import get_engine

        query = sql_text(f"""
            SELECT metadata_->>'file_id' AS fid,
                   sum(length(text)) AS chars,
                   sum(length(regexp_replace(text, :non_cjk, '', 'g'))) AS cjk
            FROM "{table_name}"
            WHERE COALESCE(metadata_->>'node_role', 'leaf') = 'leaf'
            GROUP BY 1
        """)
        try:
            with get_engine().connect() as conn:
                exists = conn.execute(sql_text("SELECT to_regclass(:t) IS NOT NULL"),
                                      {"t": '"' + table_name.replace('"', '""') + '"'}).scalar()
                if not exists:
                    return {}
                rows = conn.execute(query, {"non_cjk": _NON_CJK_PATTERN}).fetchall()
        except Exception as e:
            logger.warning(f"file_token_estimates failed for {table_name}: {e}")
            return {}
        out: Dict[str, int] = {}
        for fid, chars, cjk in rows:
            chars, cjk = int(chars or 0), int(cjk or 0)
            out[str(fid)] = int(round(cjk * CJK_TOKENS_PER_CHAR
                                      + (chars - cjk) * OTHER_TOKENS_PER_CHAR))
        return out


# Postgres regex matching every character that is NOT counted as CJK by
# token_budget.estimate_tokens (kept in sync with its character class).
_NON_CJK_PATTERN = ("[^\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
                    "\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")
