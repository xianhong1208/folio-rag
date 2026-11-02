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
import re
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


# Words that carry no search intent. In an AND query they over-constrain; in an
# OR query they match almost every row. Deliberately small: question words,
# particles and English function words only, never content words.
_ZH_STOPWORDS = frozenset({
    "什麼", "甚麼", "什么", "怎麼", "怎么", "怎樣", "怎样", "如何", "為什麼", "为什么",
    "為何", "为何", "哪裡", "哪里", "哪些", "哪個", "哪个", "哪", "多少", "幾", "几",
    "嗎", "吗", "呢", "吧", "啊", "呀", "是否", "請問", "请问", "一下", "的", "了",
    "是", "有", "在", "要", "會", "会", "能", "可以", "我", "你", "我們", "我们",
    "這", "这", "那", "這個", "这个", "那個", "那个", "嗎?", "呢?",
})
_EN_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does",
    "did", "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "of", "to", "in", "on", "at", "for", "by", "with", "from", "about", "as",
    "and", "or", "it", "its", "this", "that", "these", "those", "there", "can",
    "could", "should", "would", "will", "i", "me", "my", "we", "our", "you",
    "your", "please", "tell",
})
QUERY_STOPWORDS = _ZH_STOPWORDS | _EN_STOPWORDS

# A term must contain at least one letter, digit or CJK character.
_WORDLIKE = re.compile(r"[^\W_]")

# Leaves only. IS DISTINCT FROM keeps legacy rows that have no node_role.
_LEAF_ONLY_SQL = "(metadata_->>'node_role' IS DISTINCT FROM 'parent')"


def query_lexemes(segmented: str) -> List[str]:
    """Search terms from a space-separated query: lowercased, de-duplicated,
    punctuation and stop words dropped (all kept if nothing else is left)."""
    terms: List[str] = []
    for tok in (segmented or "").split():
        t = tok.lower()
        if _WORDLIKE.search(t) and t not in terms:
            terms.append(t)
    kept = [t for t in terms if t not in QUERY_STOPWORDS]
    return kept or terms


def to_tsquery_text(lexemes: Sequence[str], op: str) -> str:
    """Quoted terms joined by ``&`` or ``|`` — quoting keeps tsquery operators
    inside a term literal, and to_tsquery still normalises each term the way
    to_tsvector did (lowercase; stemming for language configs)."""
    quoted = ["'" + t.replace("\\", "\\\\").replace("'", "''") + "'" for t in lexemes]
    return f" {op} ".join(quoted)


def _bm25_statement(table: str):
    from sqlalchemy import text as _sql

    # table is server-generated (folder id + uuid), never user input.
    return _sql(
        f'SELECT node_id, text, metadata_, ts_rank(text_search_tsv, q) AS rank '
        f'FROM "{table}", to_tsquery(CAST(:cfg AS regconfig), :q) AS q '
        f'WHERE text_search_tsv @@ q AND {_LEAF_ONLY_SQL} '
        f'ORDER BY rank DESC LIMIT :k')


def _row_to_node(row):
    """(node_id, text, metadata_, rank) → NodeWithScore, as llama-index builds it."""
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.vector_stores.utils import metadata_dict_to_node

    node_id, text, meta, rank = row
    try:
        node = metadata_dict_to_node(meta)
        node.set_content(str(text))
    except Exception:  # noqa: BLE001 — legacy rows without _node_content
        node = TextNode(id_=node_id, text=text or "", metadata=meta or {})
    return NodeWithScore(node=node, score=float(rank) if rank is not None else 0.0)


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
    async def segment_query(self, query_text: str, settings: SearchSettings) -> str:
        """Space-separated query terms, tokenised the way the index was.

        With ``text_search_config == "simple"`` the stored tsvector holds
        Python-segmented tokens, so the query is segmented by the same segmenter.
        Any other config keeps Postgres's own tsvector, whose parser splits on
        whitespace and punctuation — the raw words are what it needs.
        """
        if settings.text_search_config != "simple":
            return query_text
        try:
            from src.domain.rag.ckip_segmenter import get_segmenter

            seg = await asyncio.to_thread(get_segmenter().segment, query_text)
            return seg or query_text
        except Exception as e:  # noqa: BLE001
            logger.debug(f"query segmentation failed, using raw text: {e}")
            return query_text

    async def sparse(self, query_text: str, k: int, settings: SearchSettings) -> List[Any]:
        """BM25 over leaf chunks: all terms (AND) first, then any term (OR) to fill k.

        A row matching every term is usually the best keyword hit, and AND matches
        far fewer rows than OR, so ranking is cheaper. Parents are excluded in SQL:
        their long concatenated text otherwise wins ``ts_rank`` and fills the
        LIMIT. Failure degrades to "no BM25 hits" (dense results still return).
        """
        lexemes = query_lexemes(await self.segment_query(query_text, settings))
        if not lexemes:
            return []
        try:
            from src.domain.rag.chunk_lookup import ChunkLookup
            from db.db import get_async_engine

            table = ChunkLookup._get_table_name(self._vector_store)
            stmt = _bm25_statement(table)
            async with get_async_engine().connect() as conn:
                async def run(op: str):
                    params = {"cfg": settings.text_search_config,
                              "q": to_tsquery_text(lexemes, op), "k": k}
                    return list((await conn.execute(stmt, params)).fetchall())

                rows = await run("&") if len(lexemes) > 1 else []
                if len(rows) < k:
                    seen = {r[0] for r in rows}
                    rows += [r for r in await run("|") if r[0] not in seen]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"BM25 search skipped ({type(e).__name__}: {e}); dense results only")
            return []
        return [_row_to_node(r) for r in rows[:k]]

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
        # Independent paths: run them concurrently (latency = the slower one).
        dense, sparse = await asyncio.gather(
            self.dense(query_text, k),
            self.sparse(query_text, max(k, int(settings.sparse_top_k or 0)), settings))
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
