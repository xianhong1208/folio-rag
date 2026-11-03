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
import contextvars
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.log import get_api_logger

logger = get_api_logger()

FUSION_RRF = "rrf"
FUSION_CONCAT = "concat"
FUSIONS = (FUSION_RRF, FUSION_CONCAT)

# When a reranker is present we retrieve this many times top_k as candidates and
# let the cross-encoder pick the final top_k.
RERANK_CANDIDATE_FACTOR = 3

# pgvector's default hnsw.ef_search. An HNSW scan returns at most ef_search rows,
# so a dense query deeper than this would be silently cut short; the dense query
# raises it (transaction-local) to max(DEFAULT_EF_SEARCH, k).
DEFAULT_EF_SEARCH = 40

# How long a folder's segmenter stamp is trusted before it is read again (a
# Rebuild FTS re-stamps the table; queries follow within this many seconds).
SEGMENTER_STAMP_TTL_S = 60.0

# Per-request stage timings (ms). Set by HybridSearcher.search for the duration of
# one call; the concurrent dense / BM25 tasks inherit the same dict through the
# context, so each records its own stage without shared searcher state.
_STAGE_TIMINGS: contextvars.ContextVar[Optional[Dict[str, float]]] = contextvars.ContextVar(
    "folio_stage_timings", default=None)


def record_stage(stage: str, started: float) -> None:
    """Add the time since ``started`` (perf_counter) to ``stage`` of the current
    request's timings; a no-op outside a timed search."""
    timings = _STAGE_TIMINGS.get()
    if timings is not None:
        timings[stage] = round(timings.get(stage, 0.0) + (time.perf_counter() - started) * 1000, 1)


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

            retrieval_cfg = getattr(getattr(Config.get_config_model(), "rag", None), "retrieval", None)
        except Exception:  # noqa: BLE001
            retrieval_cfg = None
        if retrieval_cfg is not None:
            def _get(name, default):
                v = getattr(retrieval_cfg, name, None)
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


def configured_text_search_config() -> str:
    """``rag.retrieval.text_search_config`` from the live config; 'simple' when it is
    unset or the config is unavailable."""
    try:
        from src.config.config_manager import Config

        retrieval_cfg = getattr(getattr(Config.get_config_model(), "rag", None), "retrieval", None)
        return getattr(retrieval_cfg, "text_search_config", "simple") or "simple"
    except Exception:  # noqa: BLE001
        return "simple"


def cache_fingerprint(query_text: str, settings: SearchSettings, *, reranker=None,
                      merge_threshold: Optional[float] = None) -> str:
    """Hash of everything that changes a search's results, for the result cache.

    Covers the query, every SearchSettings field, the segmenter (simple config
    only), the reranker's model / threshold / prompt templates and the merge
    threshold, so a changed setting can never be answered from a stale entry.
    """
    import hashlib

    parts = [query_text, repr(sorted(settings.__dict__.items()))]
    if settings.text_search_config == "simple":
        from src.domain.rag.text_segmenter import configured_segmenter_name

        parts.append(f"seg:{configured_segmenter_name()}")
    if reranker is not None:
        parts.append("rerank:" + "|".join(str(getattr(reranker, a, None)) for a in (
            "_model", "_score_threshold", "_query_template", "_document_template")))
    else:
        parts.append("no_rerank")
    parts.append(f"merge:{merge_threshold}")
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


@dataclass
class SearchOutcome:
    """Result of one :meth:`HybridSearcher.search` call."""

    nodes: List[Any]                     # final leaf NodeWithScore list, best first
    candidates: int = 0                  # fused candidates handed to rerank / cutoff
    reranked: bool = False               # the reranker scored these nodes
    rerank_failed: bool = False          # a reranker was configured but its call failed
    top_score: Optional[float] = None    # best rerank score before score_threshold
    dense: List[Any] = field(default_factory=list)
    sparse: List[Any] = field(default_factory=list)
    fused: List[Any] = field(default_factory=list)
    fusion: str = FUSION_RRF             # how candidates were fused (see score_type)
    # Per-stage wall time in ms: embed, dense, segment, bm25, fuse, rerank (the
    # MCP retriever adds merge / expand). dense and bm25 run concurrently.
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def score_type(self) -> str:
        """What ``score`` means on the final nodes: ``rerank`` (cross-encoder
        relevance, 0..1), ``rrf`` (reciprocal-rank fusion, ~0.01-0.03, only the
        order is meaningful) or ``vector`` (legacy concat fusion: cosine for dense
        hits, ts_rank for keyword-only hits)."""
        if self.reranked:
            return "rerank"
        return "vector" if self.fusion == FUSION_CONCAT else "rrf"


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
from src.domain.rag.chunk_identity import LEAF_ONLY_SQL as _LEAF_ONLY_SQL  # noqa: E402


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
    """One statement for both keyword passes: rows matching ANY term, rows that
    match ALL terms first, then by ts_rank. (It used to be an AND query plus an
    OR query to fill up to k — two round trips whenever AND came up short.)"""
    from sqlalchemy import text as _sql

    # table is server-generated (folder id + uuid), never user input.
    return _sql(
        f'SELECT node_id, text, metadata_, ts_rank(text_search_tsv, q_any) AS rank '
        f'FROM "{table}", to_tsquery(CAST(:cfg AS regconfig), :q_any) AS q_any, '
        f'to_tsquery(CAST(:cfg AS regconfig), :q_all) AS q_all '
        f'WHERE text_search_tsv @@ q_any AND {_LEAF_ONLY_SQL} '
        f'ORDER BY (text_search_tsv @@ q_all) DESC, rank DESC LIMIT :k')


def _dense_statement(table: str):
    """Cosine top-k over the folder table (parents are dropped in Python, as before:
    a WHERE filter would make the HNSW scan return fewer than k rows)."""
    from sqlalchemy import text as _sql

    return _sql(
        f'SELECT node_id, text, metadata_, '
        f'1 - (embedding <=> CAST(:emb AS vector)) AS similarity '
        f'FROM "{table}" ORDER BY embedding <=> CAST(:emb AS vector) LIMIT :k')


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


# ---------------------------------------------------- bigram query terms
#
# The bigram segmenter pairs every two adjacent CJK characters, so a question
# such as 請問公司的年假規定是什麼 also yields pairs that straddle particles and
# question words (問公, 司的, 的年, 定是, 是什 …). Those pairs are noise in an AND
# query. On the query side we drop every pair that touches a particle — unless
# that would leave a content character with no pair at all (現在 keeps 現在, so
# 現 is still searchable). The index keeps all pairs, so a pruned query is always
# a subset of what was indexed and matches the same rows.
_PARTICLE_WORDS = tuple(sorted({
    "請問", "请问", "什麼", "甚麼", "什么", "為什麼", "为什么", "怎麼", "怎么", "如何",
    "是否", "哪些", "哪裡", "哪里", "一下",
}, key=len, reverse=True))
_PARTICLE_CHARS = frozenset("的了是在和與与及或嗎吗呢吧啊呀")
_CJK_CLASS = "㐀-䶿一-鿿豈-﫿぀-ヿ가-힯"
_QUERY_RUN = re.compile(rf"([{_CJK_CLASS}]+)|([^\W_{_CJK_CLASS}]+)")


def _particle_mask(run: str) -> List[bool]:
    mask = [False] * len(run)
    i = 0
    while i < len(run):
        word = next((w for w in _PARTICLE_WORDS if run.startswith(w, i)), None)
        if word:
            for j in range(i, i + len(word)):
                mask[j] = True
            i += len(word)
            continue
        mask[i] = run[i] in _PARTICLE_CHARS
        i += 1
    return mask


def _run_bigrams(run: str) -> List[str]:
    if len(run) == 1:
        return [run]
    plain = [run[i:i + 2] for i in range(len(run) - 1)]
    mask = _particle_mask(run)
    if not any(mask) or all(mask):
        return plain
    starts = [i for i in range(len(run) - 1) if not mask[i] and not mask[i + 1]]
    covered = {j for i in starts for j in (i, i + 1)}
    for j, is_particle in enumerate(mask):
        if is_particle or j in covered:
            continue
        # A content character with no clean pair keeps the pair it forms with a
        # neighbour: the preceding one (和平與 → 和平, 的確 → 的確), or the
        # following one at the start of the run (現在 → 現在).
        starts.append(j - 1 if j > 0 else j)
        covered.update((starts[-1], starts[-1] + 1))
    return [run[i:i + 2] for i in sorted(set(starts))]


def bigram_query_terms(text: str) -> List[str]:
    """Query-side bigrams: like the bigram segmenter, minus particle-straddling pairs."""
    out: List[str] = []
    for cjk, word in _QUERY_RUN.findall(text or ""):
        out.extend(_run_bigrams(cjk) if cjk else [word])
    return out


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
        # table name -> (segmenter stamp or None, monotonic expiry)
        self._stamp_cache: Dict[str, Tuple[Optional[str], float]] = {}

    @property
    def vector_store(self):
        return self._vector_store

    @property
    def reranker(self):
        return self._reranker

    # ------------------------------------------------------------------ dense
    async def embed_query(self, query_text: str) -> List[float]:
        """The query embedding, from the index's embedding model."""
        t0 = time.perf_counter()
        try:
            return await self._index._embed_model.aget_query_embedding(query_text)
        finally:
            record_stage("embed", t0)

    async def dense(self, query_text: str, k: int) -> List[Any]:
        """Vector search, leaves only (parents carry a zero vector, but be explicit).

        Runs its own cosine query rather than llama-index's: llama-index never
        raises ``hnsw.ef_search`` (and would do it with a session-wide ``SET`` that
        leaks to the next user of the pooled connection), and an HNSW scan returns
        at most ef_search (default 40) rows — a deeper k was silently truncated.
        Here ``SET LOCAL`` scopes the raise to this query's transaction.
        """
        from src.domain.rag.chunk_lookup import ChunkLookup

        embedding = await self.embed_query(query_text)
        table = ChunkLookup._get_table_name(self._vector_store)
        ef = max(DEFAULT_EF_SEARCH, int(k))
        t0 = time.perf_counter()
        async with self._dense_engine().begin() as conn:
            await conn.execute(_sql_text(f"SET LOCAL hnsw.ef_search = {ef}"))
            rows = (await conn.execute(_dense_statement(table), {
                "emb": _vector_literal(embedding), "k": int(k)})).fetchall()
        record_stage("dense", t0)
        nodes = [_row_to_node(r) for r in rows]
        return [n for n in nodes if is_leaf(n)]

    # ----------------------------------------------------------------- sparse
    def _dense_engine(self):
        """llama-index's own async engine for this store when it has built one (the
        dense query has always run on that pool), else the shared async engine."""
        try:
            from sqlalchemy.ext.asyncio import AsyncEngine

            eng = getattr(self._vector_store, "_async_engine", None)
            if isinstance(eng, AsyncEngine):
                return eng
        except Exception:  # noqa: BLE001  # pragma: no cover - import guard
            pass
        from db.db import get_async_engine

        return get_async_engine()

    async def segment_query(self, query_text: str, settings: SearchSettings,
                            segmenter_name: Optional[str] = None) -> str:
        """Space-separated query terms, tokenised the way the index was.

        With ``text_search_config == "simple"`` the stored tsvector holds
        Python-segmented tokens, so the query goes through the segmenter that
        built the folder's index (``segmenter_name``, from the table's stamp) or,
        when the folder has no stamp, the configured one (ckip / jieba / bigram).
        Bigram queries drop particle-straddling pairs (see bigram_query_terms).
        Any other config keeps Postgres's own tsvector, whose parser splits on
        whitespace and punctuation — the raw words are what it needs.
        """
        if settings.text_search_config != "simple":
            return query_text
        t0 = time.perf_counter()
        try:
            from src.domain.rag.text_segmenter import get_segmenter

            seg = get_segmenter(segmenter_name)
            if getattr(seg, "name", None) == "bigram":
                return " ".join(bigram_query_terms(query_text)) or query_text
            out = await asyncio.to_thread(seg.segment, query_text)
            return out or query_text
        except Exception as e:  # noqa: BLE001
            logger.debug(f"query segmentation failed, using raw text: {e}")
            return query_text
        finally:
            record_stage("segment", t0)

    async def folder_segmenter(self, engine, table: str) -> Optional[str]:
        """The segmenter to tokenise this folder's queries with: the one stamped on
        its keyword index, else None (= the configured one). A CKIP stamp under
        ``profile: cloud`` (where the CKIP model may not be loaded) also falls back
        to the configured segmenter. Cached per table for SEGMENTER_STAMP_TTL_S."""
        now = time.monotonic()
        cached = self._stamp_cache.get(table)
        if cached is not None and cached[1] > now:
            stamp = cached[0]
        else:
            try:
                from src.domain.rag.vector_store_manager import read_tsv_segmenter

                async with engine.connect() as conn:
                    stamp = await conn.run_sync(lambda c: read_tsv_segmenter(c, table))
            except Exception as e:  # noqa: BLE001
                logger.debug(f"segmenter stamp unreadable for {table}: {e}")
                stamp = None
            self._stamp_cache[table] = (stamp, now + SEGMENTER_STAMP_TTL_S)
        return _usable_segmenter(stamp)

    async def sparse(self, query_text: str, k: int, settings: SearchSettings) -> List[Any]:
        """BM25 over leaf chunks in one statement: rows matching every term first,
        then rows matching any term, each by ts_rank.

        Parents are excluded in SQL: their long concatenated text otherwise wins
        ``ts_rank`` and fills the LIMIT. Failure degrades to "no BM25 hits"
        (dense results still return).
        """
        try:
            from src.domain.rag.chunk_lookup import ChunkLookup
            from db.db import get_async_engine

            table = ChunkLookup._get_table_name(self._vector_store)
            engine = get_async_engine()
            seg_name = (await self.folder_segmenter(engine, table)
                        if settings.text_search_config == "simple" else None)
            lexemes = query_lexemes(await self.segment_query(query_text, settings, seg_name))
            if not lexemes:
                return []
            t0 = time.perf_counter()
            async with engine.connect() as conn:
                rows = list((await conn.execute(_bm25_statement(table), {
                    "cfg": settings.text_search_config,
                    "q_any": to_tsquery_text(lexemes, "|"),
                    "q_all": to_tsquery_text(lexemes, "&"),
                    "k": k})).fetchall())
                record_stage("bm25", t0)
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
        t0 = time.perf_counter()
        if settings.fusion == FUSION_CONCAT:
            fused = concat_fuse(dense, sparse, settings.similarity_cutoff)
        else:
            fused = rrf_fuse([dense, sparse], k, settings.rrf_k)
        record_stage("fuse", t0)
        # Expose the raw per-path scores even though RRF overwrote node.score.
        return fused, _with_scores(dense, dense_raw), _with_scores(sparse, sparse_raw)

    # ----------------------------------------------------------------- search
    async def search(self, query_text: str, settings: SearchSettings) -> SearchOutcome:
        """Candidates → rerank (if configured) → top_k leaves.

        ``outcome.timings`` holds this call's per-stage milliseconds.
        """
        timings: Dict[str, float] = {}
        token = _STAGE_TIMINGS.set(timings)
        try:
            fused, dense, sparse = await self.candidates(query_text, settings)
            # Snapshot the fused order and scores: rerank rewrites node.score in place.
            outcome = SearchOutcome(nodes=[], candidates=len(fused), dense=dense, sparse=sparse,
                                    fused=_with_scores(fused, _snapshot(fused)),
                                    fusion=settings.fusion, timings=timings)
            if self._reranker and fused:
                from src.domain.rag.rerank_policy import rerank

                t0 = time.perf_counter()
                rr = await rerank(self._reranker, query_text, list(fused), settings.top_k)
                record_stage("rerank", t0)
                outcome.nodes = rr.nodes
                outcome.top_score = rr.top_score
                outcome.reranked = not rr.failed
                outcome.rerank_failed = rr.failed
                return outcome
            # No reranker: RRF scores are rank-based (≈0.03), so the similarity cutoff
            # does not apply; concat already applied it to its dense half.
            outcome.nodes = list(fused)[: settings.top_k]
            return outcome
        finally:
            _STAGE_TIMINGS.reset(token)


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


def _usable_segmenter(stamp: Optional[str]) -> Optional[str]:
    """``stamp`` when queries can use it, else None (the configured segmenter)."""
    from src.domain.rag.text_segmenter import SEGMENTERS, configured_segmenter_name

    if not stamp or stamp not in SEGMENTERS:
        return None
    if stamp == "ckip" and configured_segmenter_name() != "ckip":
        try:
            from src.config.config_manager import Config

            profile = getattr(getattr(Config.get_config_model(), "rag", None), "profile", "local")
        except Exception:  # noqa: BLE001
            profile = "local"
        if profile == "cloud":
            return None
    return stamp


def _sql_text(sql: str):
    from sqlalchemy import text as _sql

    return _sql(sql)


def _with_scores(nodes: Sequence[Any], snapshot) -> List[Any]:
    return [_ScoredView(n.node, s) for n, (_, s) in zip(nodes, snapshot)]
