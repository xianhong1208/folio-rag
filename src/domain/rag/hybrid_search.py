"""Hybrid search — the single retrieval path behind every entry point.

REST ``/api/rag/query``, the MCP ``search`` mode, the console search playground
(and its trace view) and the evaluation harness all retrieve through
:class:`HybridSearcher`. Before this module each of them carried its own copy of
"dense + BM25 + fusion + rerank", and the copies drifted: the MCP path sent the raw
query to llama-index's native hybrid mode (no Chinese segmentation, no RRF), the
eval harness skipped segmentation for its BM25 mode, and ``concat`` fusion was the
native hybrid mode, which for Chinese behaved as vector-only.

Per-request knobs travel in an immutable :class:`SearchSettings` passed to every
call. The searcher keeps no per-request state, so one cached instance can serve
concurrent requests that ask for different ``top_k`` / cutoff / fusion values.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Sequence, Tuple

from src.log import get_api_logger

logger = get_api_logger()

FUSION_RRF = "rrf"
FUSION_CONCAT = "concat"
FUSIONS = (FUSION_RRF, FUSION_CONCAT)

# When a reranker is present we retrieve this many times top_k as candidates and
# let the cross-encoder pick the final top_k.
RERANK_CANDIDATE_FACTOR = 3


@dataclass(frozen=True)
class SearchSettings:
    """Everything one search needs to know, fixed for the lifetime of the call."""

    top_k: int = 5
    similarity_cutoff: float = 0.0
    sparse_top_k: int = 12
    fusion: str = FUSION_RRF
    rrf_k: int = 60
    text_search_config: str = "simple"

    def __post_init__(self):
        if self.fusion not in FUSIONS:
            object.__setattr__(self, "fusion", FUSION_RRF)

    def with_overrides(self, **overrides: Any) -> "SearchSettings":
        """Copy with every non-None override applied (None means "keep")."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    @classmethod
    def from_config(cls, **overrides: Any) -> "SearchSettings":
        """Settings from ``rag.retrieval`` (read live, so runtime edits apply to the
        next request), with per-request overrides on top. A missing or unreadable
        config falls back to the dataclass defaults rather than failing the query."""
        base = cls()
        try:
            from src.config.config_manager import Config

            ret = getattr(getattr(Config.get_config_model(), "rag", None), "retrieval", None)
        except Exception:  # noqa: BLE001
            ret = None
        if ret is not None:
            def _get(name, default):
                v = getattr(ret, name, None)
                return default if v is None else v

            base = cls(
                top_k=int(_get("default_top_k", base.top_k)),
                similarity_cutoff=float(_get("default_similarity_cutoff", base.similarity_cutoff)),
                sparse_top_k=int(_get("default_sparse_top_k", base.sparse_top_k)),
                fusion=str(_get("hybrid_fusion", base.fusion)),
                rrf_k=int(_get("rrf_k", base.rrf_k)),
                text_search_config=str(_get("text_search_config", base.text_search_config)),
            )
        return base.with_overrides(**overrides)


@dataclass
class SearchOutcome:
    """Result of one :meth:`HybridSearcher.search` call."""

    nodes: List[Any]                     # final leaf NodeWithScore list, best first
    candidates: int = 0                  # fused candidates handed to rerank / cutoff
    reranked: bool = False               # the reranker scored these nodes
    rerank_failed: bool = False          # a reranker was configured but its call failed
    dense: List[Any] = field(default_factory=list)
    sparse: List[Any] = field(default_factory=list)
    fused: List[Any] = field(default_factory=list)


def is_leaf(node_with_score) -> bool:
    """Parents share the table with leaves; legacy rows without node_role count as leaves."""
    return node_with_score.node.metadata.get("node_role", "leaf") == "leaf"


def _node_id(n) -> str:
    return n.node.node_id


def rrf_fuse(ranked_lists: Sequence[Sequence[Any]], k: int, rrf_k: int = 60) -> List[Any]:
    """Reciprocal Rank Fusion: each list contributes 1/(rrf_k + rank).

    Only ranks matter, so cosine similarity and ts_rank (different scales) can be
    combined. The returned nodes carry the RRF score; the input lists' score
    values are left alone on copies the caller already captured.
    """
    scores: dict = {}
    node_map: dict = {}
    for lst in ranked_lists:
        for rank, n in enumerate(lst):
            nid = _node_id(n)
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (rrf_k + rank + 1)
            node_map.setdefault(nid, n)
    ranked = sorted(node_map.values(), key=lambda n: scores[_node_id(n)], reverse=True)
    for n in ranked:
        n.score = scores[_node_id(n)]
    return ranked[:k]


def concat_fuse(dense: Sequence[Any], sparse: Sequence[Any],
                cutoff: Optional[float] = None) -> List[Any]:
    """Dense hits first (optionally above ``cutoff``), then BM25-only hits.

    Not truncated and not re-scored: cosine and ts_rank live on different scales,
    so the order is "dense rank, then keyword rank" and the reranker (when
    enabled) picks from the union. ``cutoff`` filters only the dense half — it is
    a similarity threshold and means nothing for ts_rank.
    """
    if cutoff is not None:
        dense = [n for n in dense if n.score is not None and n.score >= cutoff]
    seen = {_node_id(n) for n in dense}
    return list(dense) + [n for n in sparse if _node_id(n) not in seen]


def _snapshot(nodes: Sequence[Any]) -> List[Tuple[str, Optional[float]]]:
    return [(_node_id(n), float(n.score) if n.score is not None else None) for n in nodes]


class HybridSearcher:
    """Dense + BM25 candidate retrieval, fusion and optional rerank for one folder.

    Holds only the vector store, its index and the reranker; everything that can
    vary per request arrives in :class:`SearchSettings`.
    """

    def __init__(self, vector_store, reranker=None, index=None):
        self._vector_store = vector_store
        self._reranker = reranker
        if index is None:
            from llama_index.core import VectorStoreIndex

            index = VectorStoreIndex.from_vector_store(vector_store)
        self._index = index

    @property
    def vector_store(self):
        return self._vector_store

    @property
    def reranker(self):
        return self._reranker

    # ------------------------------------------------------------------ dense
    async def dense(self, query_text: str, k: int) -> List[Any]:
        """Vector search, leaves only (parents carry a zero vector, but be explicit)."""
        from llama_index.core.retrievers import VectorIndexRetriever

        retriever = VectorIndexRetriever(
            index=self._index, similarity_top_k=k, vector_store_query_mode="default")
        nodes = await retriever.aretrieve(query_text)
        return [n for n in nodes if is_leaf(n)]

    # ----------------------------------------------------------------- sparse
    async def _or_query(self, query_text: str, settings: SearchSettings) -> str:
        """Space-joined lexemes so llama-index's sparse mode builds an OR query.

        llama-index hands the raw string to to_tsquery; space-less Chinese then
        becomes one phrase that never matches the segmented tsvector.
        """
        if settings.text_search_config == "simple":
            try:
                from src.domain.rag.ckip_segmenter import get_segmenter

                seg = await asyncio.to_thread(get_segmenter().segment, query_text)
                return seg or query_text
            except Exception as e:  # noqa: BLE001
                logger.debug(f"query segmentation failed, using raw text: {e}")
                return query_text

        def _db_lexemes():
            from sqlalchemy import text as _sql
            from db.db import get_engine

            with get_engine().connect() as c:
                return c.execute(
                    _sql("SELECT array_to_string(tsvector_to_array(to_tsvector(:cfg, :q)), ' ')"),
                    {"cfg": settings.text_search_config, "q": query_text}).scalar()

        try:
            return (await asyncio.to_thread(_db_lexemes)) or query_text
        except Exception as e:  # noqa: BLE001
            logger.debug(f"query lexeme extraction failed, using raw text: {e}")
            return query_text

    async def sparse(self, query_text: str, k: int, settings: SearchSettings) -> List[Any]:
        """BM25 (Postgres full-text search) over the segmented query, leaves only."""
        from llama_index.core.retrievers import VectorIndexRetriever

        q = await self._or_query(query_text, settings)
        retriever = VectorIndexRetriever(
            index=self._index, similarity_top_k=k, vector_store_query_mode="sparse")
        nodes = await retriever.aretrieve(q)
        return [n for n in nodes if is_leaf(n)]

    # ----------------------------------------------------------------- fusion
    def candidate_depth(self, settings: SearchSettings) -> int:
        """How many fused candidates to produce for this request."""
        factor = RERANK_CANDIDATE_FACTOR if self._reranker else 1
        return max(1, settings.top_k) * factor

    async def candidates(self, query_text: str, settings: SearchSettings,
                         k: Optional[int] = None) -> Tuple[List[Any], List[Any], List[Any]]:
        """Return (fused, dense, sparse) for the query.

        ``dense`` / ``sparse`` are the per-path lists before fusion (their scores are
        the raw cosine / ts_rank values, snapshotted before RRF rewrites them).
        """
        k = k or self.candidate_depth(settings)
        dense = await self.dense(query_text, k)
        sparse = await self.sparse(query_text, k, settings)
        dense_raw, sparse_raw = _snapshot(dense), _snapshot(sparse)
        if settings.fusion == FUSION_CONCAT:
            fused = concat_fuse(dense, sparse, settings.similarity_cutoff)
        else:
            fused = rrf_fuse([dense, sparse], k, settings.rrf_k)
        # Expose the raw per-path scores even though RRF overwrote node.score.
        return fused, _with_scores(dense, dense_raw), _with_scores(sparse, sparse_raw)

    # ----------------------------------------------------------------- search
    async def search(self, query_text: str, settings: SearchSettings) -> SearchOutcome:
        """Candidates → rerank (if configured) → top_k leaves."""
        fused, dense, sparse = await self.candidates(query_text, settings)
        # Snapshot the fused order and scores: rerank rewrites node.score in place.
        outcome = SearchOutcome(nodes=[], candidates=len(fused), dense=dense, sparse=sparse,
                                fused=_with_scores(fused, _snapshot(fused)))
        if self._reranker and fused:
            nodes = await self._reranker.arerank(query_text, list(fused), top_n=settings.top_k)
            outcome.nodes = list(nodes)[: settings.top_k]
            outcome.reranked = True
            return outcome
        # No reranker: RRF scores are rank-based (≈0.03), so the similarity cutoff
        # does not apply; concat already applied it to its dense half.
        outcome.nodes = list(fused)[: settings.top_k]
        return outcome


class _ScoredView:
    """A read-only (node, score) pair, so trace code can show raw per-path scores."""

    __slots__ = ("node", "score")

    def __init__(self, node, score):
        self.node = node
        self.score = score

    @property
    def metadata(self):
        return self.node.metadata

    @property
    def text(self):
        return self.node.text


def _with_scores(nodes: Sequence[Any], snapshot) -> List[Any]:
    return [_ScoredView(n.node, s) for n, (_, s) in zip(nodes, snapshot)]
