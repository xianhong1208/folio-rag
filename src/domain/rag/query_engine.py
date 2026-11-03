"""Query Engine — REST / console retrieval over one folder.

A thin layer over :class:`~src.domain.rag.hybrid_search.HybridSearcher` (the
retrieval path shared with MCP and the eval harness) that turns the searcher's
nodes into ``RAGSearchResult`` DTOs, applies the shared auto-merge rule, and
provides the console's retrieval trace.

Instances are cached per folder and shared by concurrent requests, so nothing
request-specific is stored on them: ``top_k``, cutoff, fusion and friends arrive
with every call as a :class:`SearchSettings`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import sqlalchemy.exc

from src.domain.exceptions import FolderNotFoundError
from src.domain.rag.dto import RAGChunkMetadata, RAGSearchResult
from src.domain.rag.hybrid_search import HybridSearcher, SearchOutcome, SearchSettings
from src.domain.rag.merge_plan import MergeGroup, arrange, plan_merges
from src.log import get_api_logger

if TYPE_CHECKING:
    from src.domain.rag.reranker import Reranker

logger = get_api_logger()

_MISSING_TABLE = ("Vector store table missing — folder may have been deleted "
                  "or its index cleared.")


def _missing_table(e: Exception) -> bool:
    err = str(e).lower()
    return "does not exist" in err or "undefined" in err


class QueryEngine:
    """Hybrid search + optional rerank + auto-merge for one folder's vector store."""

    def __init__(self, vector_store, reranker: Optional["Reranker"] = None,
                 searcher: Optional[HybridSearcher] = None):
        # Guard against a dropped pgvector table: translate to FolderNotFoundError
        # instead of surfacing a raw SQL error.
        try:
            self._searcher = searcher or HybridSearcher(vector_store, reranker=reranker)
        except sqlalchemy.exc.ProgrammingError as e:
            if _missing_table(e):
                raise FolderNotFoundError(folder_name=_MISSING_TABLE) from e
            raise
        self._vector_store = vector_store

    @property
    def searcher(self) -> HybridSearcher:
        return self._searcher

    async def asearch(self, query_text: str, settings: SearchSettings,
                      merge_threshold: Optional[float] = None
                      ) -> Tuple[List[RAGSearchResult], SearchOutcome]:
        """Run one search; returns the result DTOs and the searcher's outcome."""
        try:
            outcome = await self._searcher.search(query_text, settings)
            groups = await plan_merges(self._vector_store, outcome.nodes, merge_threshold)
            results = [_to_result(item) for item in arrange(outcome.nodes, groups)]
        except sqlalchemy.exc.ProgrammingError as e:
            # vector_store table was dropped (e.g. after delete_folder_index).
            if _missing_table(e):
                logger.warning(f"Vector store table missing during query: {e}")
                raise FolderNotFoundError(folder_name=_MISSING_TABLE) from e
            logger.error(f"Query execution failed: {e}")
            raise
        logger.info(
            f"Query completed: {len(results)} result(s) from {outcome.candidates} "
            f"candidate(s) (fusion={settings.fusion}, reranked={outcome.reranked})")
        return results, outcome

    async def aquery(self, query_text: str, settings: SearchSettings,
                     merge_threshold: Optional[float] = None) -> List[RAGSearchResult]:
        results, _ = await self.asearch(query_text, settings, merge_threshold)
        return results

    async def atrace(self, query_text: str, settings: SearchSettings) -> dict:
        """Retrieval trace (diagnostic): per-path ranks and scores for every final hit.

        Uses exactly the production candidates (same dense / BM25 lists, same
        fusion, same rerank) and reports where each final hit ranked in each list.
        """
        outcome = await self._searcher.search(query_text, settings)

        def _rankmap(nodes):
            return {n.node.node_id: (float(n.score) if n.score is not None else None, i + 1)
                    for i, n in enumerate(nodes)}

        vmap, bmap = _rankmap(outcome.dense), _rankmap(outcome.sparse)
        hmap = _rankmap(outcome.fused)
        results = []
        for node in outcome.nodes:
            nid = node.node.node_id
            h, v, b = hmap.get(nid), vmap.get(nid), bmap.get(nid)
            score = float(node.score) if node.score is not None else None
            results.append({
                "text": node.text,
                "final_score": score,
                "reranked": outcome.reranked,
                "rerank_score": score if outcome.reranked else None,
                "hybrid_score": h[0] if h else None, "hybrid_rank": h[1] if h else None,
                "vector_score": v[0] if v else None, "vector_rank": v[1] if v else None,
                "bm25_score": b[0] if b else None, "bm25_rank": b[1] if b else None,
                "metadata": {
                    "node_id": nid,
                    "file_name": node.metadata.get("file_name"),
                    "page": node.metadata.get("page_no"),
                    "headings": node.metadata.get("headings"),
                    "content_type": node.metadata.get("content_type"),
                },
            })
        return {"results": results, "reranked": outcome.reranked,
                "candidates": outcome.candidates, "fusion": settings.fusion,
                "score_type": outcome.score_type}


def _to_result(item) -> RAGSearchResult:
    """A searcher leaf (NodeWithScore) or a merged parent → RAGSearchResult."""
    if isinstance(item, MergeGroup):
        parent, best = item.parent, item.best_hit
        md = parent.metadata or {}
        prov = best.node.metadata
        return RAGSearchResult(
            text=parent.text,
            score=float(item.score),
            metadata=RAGChunkMetadata(
                node_id=parent.node_id,
                mcp_file_id=md.get("file_id") or prov.get("file_id"),
                file_name=md.get("file_name") or prov.get("file_name"),
                folder_name=md.get("folder_name") or prov.get("folder_name"),
                page=prov.get("page_no"),
                headings=prov.get("headings"),
                content_type=prov.get("content_type"),
                node_role="parent",
                merged_from=len(item.hits),
            ),
        )
    md = item.node.metadata
    return RAGSearchResult(
        text=item.node.text,
        score=float(item.score) if item.score else 0.0,
        metadata=RAGChunkMetadata(
            node_id=item.node.node_id,
            mcp_file_id=md.get("file_id"),
            file_name=md.get("file_name"),
            folder_name=md.get("folder_name"),
            page=md.get("page_no"),
            headings=md.get("headings"),
            content_type=md.get("content_type"),
            node_role="leaf",
        ),
    )
