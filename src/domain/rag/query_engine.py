
"""Query Engine

Handles RAG queries with hybrid search (vector + BM25) and optional reranking.
"""

import asyncio
from typing import List, Optional, TYPE_CHECKING
import sqlalchemy.exc
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.postgres import PGVectorStore
from src.domain.rag.dto import RAGSearchResult, RAGChunkMetadata
from src.domain.exceptions import FolderNotFoundError
from src.log import get_api_logger

if TYPE_CHECKING:
    from src.domain.rag.reranker import Reranker

logger = get_api_logger()


class QueryEngine:
    """Handles RAG queries with hybrid search (vector + BM25) and optional reranking."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        top_k: int = 5,
        similarity_cutoff: float = 0.7,
        sparse_top_k: int = 10,
        hybrid_alpha: float = 0.5,
        reranker: Optional["Reranker"] = None,
        hybrid_fusion: str = "rrf",
        rrf_k: int = 60,
        text_search_config: str = "simple",
    ):
        """Initialize query engine."""
        self._vector_store = vector_store
        self._top_k = top_k
        self._similarity_cutoff = similarity_cutoff
        self._sparse_top_k = sparse_top_k
        self._hybrid_alpha = hybrid_alpha
        self._reranker = reranker
        self._hybrid_fusion = hybrid_fusion
        self._rrf_k = rrf_k
        self._text_search_config = text_search_config

        # Guard against a dropped pgvector table: translate to FolderNotFoundError
        # instead of surfacing a raw SQL error.
        try:
            self._index = VectorStoreIndex.from_vector_store(vector_store)
        except sqlalchemy.exc.ProgrammingError as e:
            err = str(e).lower()
            if "does not exist" in err or "undefined" in err:
                raise FolderNotFoundError(
                    folder_name="Vector store table missing — folder may have been deleted or its index cleared."
                ) from e
            raise

        rerank_status = "enabled" if reranker else "disabled"
        logger.debug(
            f"QueryEngine initialized (top_k={top_k}, "
            f"similarity_cutoff={similarity_cutoff}, "
            f"sparse_top_k={sparse_top_k}, hybrid_alpha={hybrid_alpha}, "
            f"rerank={rerank_status})"
        )

    async def _or_query(self, query_text: str) -> str:
        """Convert the query into an OR-of-lexemes, fixing llama_index's sparse bug
        for languages without whitespace.

        llama_index hands the whole query to to_tsquery; space-less Chinese becomes
        a phrase query requiring adjacency, giving almost zero hits. Segmenting into
        space-joined lexemes (which llama_index rewrites with |) makes it an OR
        query. On failure, fall back to the raw string rather than block retrieval.

        CKIP mode (text_search_config == 'simple') segments in Python with the same
        segmenter used at index time so query tokens align with the CKIP-derived
        text_search_tsv; any other config falls back to Postgres-side to_tsvector.
        """
        if self._text_search_config == "simple":
            try:
                from src.domain.rag.ckip_segmenter import get_segmenter
                segmenter = get_segmenter()
                seg = await asyncio.to_thread(segmenter.segment, query_text)
                return seg or query_text
            except Exception as e:  # noqa: BLE001
                logger.debug(f"CKIP query segmentation failed, using raw: {e}")
                return query_text

        def _seg():
            from sqlalchemy import text as _sql
            from db.db import get_engine
            with get_engine().connect() as c:
                return c.execute(
                    _sql("SELECT array_to_string("
                         "tsvector_to_array(to_tsvector(:cfg, :q)), ' ')"),
                    {"cfg": self._text_search_config, "q": query_text}).scalar()
        try:
            seg = await asyncio.to_thread(_seg)
            return seg or query_text
        except Exception as e:  # noqa: BLE001
            logger.debug(f"sparse OR-query segmentation failed, using raw: {e}")
            return query_text

    async def _aretrieve_mode(self, query_text: str, mode: str, k: int):
        """Retrieve for a single query mode (default=dense / sparse=BM25 / hybrid), keeping only leaf nodes."""
        q = query_text
        vector_store_kwargs = {}
        if mode in ("sparse", "hybrid"):
            vector_store_kwargs["sparse_top_k"] = self._sparse_top_k
        if mode == "sparse":
            q = await self._or_query(query_text)  # fix the phrase-query bug
        if mode == "hybrid" and self._hybrid_alpha is not None:
            vector_store_kwargs["alpha"] = self._hybrid_alpha
        retriever = VectorIndexRetriever(
            index=self._index, similarity_top_k=k,
            vector_store_query_mode=mode, vector_store_kwargs=vector_store_kwargs,
        )
        nodes = await retriever.aretrieve(q)
        return [n for n in nodes if n.node.metadata.get("node_role", "leaf") == "leaf"]

    def _rrf_fuse(self, ranked_lists, k: int):
        """Reciprocal Rank Fusion: re-rank by summing 1/(rrf_k+rank) across lists.

        Only ranks matter (no score normalization), so both dense and BM25 genuinely
        influence the final order, fixing llama_index's concat fusion where dense
        dominated and hybrid was effectively vector-only.
        """
        scores: dict = {}
        node_map: dict = {}
        for lst in ranked_lists:
            for rank, n in enumerate(lst):
                nid = n.node.node_id
                scores[nid] = scores.get(nid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
                node_map.setdefault(nid, n)
        ranked = sorted(node_map.values(),
                        key=lambda n: scores[n.node.node_id], reverse=True)
        for n in ranked:
            n.score = scores[n.node.node_id]  # downstream (rerank / return) uses the RRF score
        return ranked[:k]

    async def _retrieve_candidates(self, query_text: str, k: int):
        """Return (candidate nodes, is_rrf). rrf -> retrieve dense and sparse
        separately then fuse with RRF; concat -> llama_index's native hybrid
        (legacy behavior)."""
        if self._hybrid_fusion == "rrf":
            dense = await self._aretrieve_mode(query_text, "default", k)
            sparse = await self._aretrieve_mode(query_text, "sparse", k)
            return self._rrf_fuse([dense, sparse], k), True
        return await self._aretrieve_mode(query_text, "hybrid", k), False

    async def aquery(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_cutoff: Optional[float] = None
    ) -> List[RAGSearchResult]:
        """Execute a RAG query with hybrid search (fully async).

        The entire path stays off the event loop (aretrieve -> async embedding +
        hybrid SQL; arerank -> httpx.AsyncClient).
        """
        k = top_k if top_k is not None else self._top_k
        cutoff = similarity_cutoff if similarity_cutoff is not None else self._similarity_cutoff

        # When reranker is enabled, retrieve more candidates for better precision
        retrieve_k = k * 3 if self._reranker else k

        logger.debug(f"Executing query: '{query_text[:50]}...' (top_k={k}, retrieve_k={retrieve_k}, cutoff={cutoff})")

        try:
            # Hybrid retrieval candidates: RRF (rank fusion of dense+sparse) or
            # concat (llama_index native). Leaf-only filtering already happened
            # inside _aretrieve_mode (parent nodes do not count toward the ranks).
            nodes, is_rrf = await self._retrieve_candidates(query_text, retrieve_k)
            logger.debug(
                f"Retrieved {len(nodes)} leaf candidates "
                f"(fusion={self._hybrid_fusion}) before rerank/cutoff")

            # Rerank if enabled (cross-encoder produces more accurate scores)
            if self._reranker and nodes:
                # Reranker handles top_n and score_threshold internally
                nodes = await self._reranker.arerank(query_text, nodes, top_n=k)

            results = []
            for node in nodes:
                # The similarity cutoff only applies to dense similarity scores;
                # it does not apply to RRF scores (on the order of ~0.03 for
                # 1/(k+rank)) or rerank scores, so skip it for those.
                if not self._reranker and not is_rrf and (node.score is None or node.score < cutoff):
                    continue

                metadata = RAGChunkMetadata(
                    node_id=node.node.node_id,
                    mcp_file_id=node.metadata.get("file_id"),
                    file_name=node.metadata.get("file_name"),
                    folder_name=node.metadata.get("folder_name"),
                    # Citation provenance (only present for data indexed via the docling path)
                    page=node.metadata.get("page_no"),
                    headings=node.metadata.get("headings"),
                )

                result = RAGSearchResult(
                    text=node.text,
                    score=float(node.score) if node.score else 0.0,
                    metadata=metadata
                )
                results.append(result)

            logger.info(
                f"Query completed: {len(results)} results above threshold {cutoff} "
                f"(from {len(nodes)} total)"
            )

            return results

        except sqlalchemy.exc.ProgrammingError as e:
            # vector_store table was dropped (e.g. after delete_folder_index) —
            # translate to FolderNotFoundError so middleware returns 404 instead of 500.
            err = str(e).lower()
            if "does not exist" in err or "undefined" in err:
                logger.warning(f"Vector store table missing during query: {e}")
                raise FolderNotFoundError(
                    folder_name="Vector store table missing — folder may have been deleted or its index cleared."
                ) from e
            logger.error(f"Query execution failed: {e}")
            raise

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    async def atrace(
        self,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> dict:
        """Retrieval trace (diagnostic): break out per-path ranks and scores for
        vector / BM25 / hybrid / rerank.

        Runs pure vector and pure BM25 once each in addition to the fused pass and
        aligns them by node_id. It separates signals via query mode rather than
        hybrid_alpha because pgvector's hybrid fusion ignores alpha (all alpha values
        scored identically in testing). Diagnostic path, not a hot path.
        """
        k = top_k if top_k is not None else self._top_k
        retrieve_k = k * 3 if self._reranker else k

        async def _retrieve(mode: str):
            # Reuse _aretrieve_mode: sparse goes through the OR-query fix (consistent with production)
            return await self._aretrieve_mode(query_text, mode, retrieve_k)

        def _rankmap(nodes):
            return {
                n.node.node_id: (float(n.score) if n.score is not None else None, i + 1)
                for i, n in enumerate(nodes)
            }

        vector_nodes = await _retrieve("default")
        bm25_nodes = await _retrieve("sparse")
        # Capture the raw dense/sparse scores first -- _rrf_fuse overwrites
        # node.score in place with the RRF score.
        vmap, bmap = _rankmap(vector_nodes), _rankmap(bm25_nodes)
        # Fusion mirrors production aquery: rrf -> our RRF; concat -> llama_index native hybrid
        if self._hybrid_fusion == "rrf":
            fused_nodes = self._rrf_fuse([vector_nodes, bm25_nodes], retrieve_k)
        else:
            fused_nodes = await _retrieve("hybrid")
        hmap = _rankmap(fused_nodes)

        reranked = bool(self._reranker and fused_nodes)
        if reranked:
            final = await self._reranker.arerank(query_text, fused_nodes, top_n=k)
        else:
            final = fused_nodes[:k]

        results = []
        for node in final:
            nid = node.node.node_id
            h, v, b = hmap.get(nid), vmap.get(nid), bmap.get(nid)
            results.append({
                "text": node.text,
                "final_score": float(node.score) if node.score is not None else None,
                "reranked": reranked,
                "rerank_score": (float(node.score) if node.score is not None else None) if reranked else None,
                "hybrid_score": h[0] if h else None, "hybrid_rank": h[1] if h else None,
                "vector_score": v[0] if v else None, "vector_rank": v[1] if v else None,
                "bm25_score": b[0] if b else None, "bm25_rank": b[1] if b else None,
                "metadata": {
                    "node_id": nid,
                    "file_name": node.metadata.get("file_name"),
                    "page": node.metadata.get("page_no"),
                    "headings": node.metadata.get("headings"),
                },
            })
        return {"results": results, "reranked": reranked,
                "candidates": len(fused_nodes), "fusion": self._hybrid_fusion}

    def update_config(
        self,
        top_k: Optional[int] = None,
        similarity_cutoff: Optional[float] = None,
        sparse_top_k: Optional[int] = None,
        hybrid_alpha: Optional[float] = None,
        hybrid_fusion: Optional[str] = None,
        rrf_k: Optional[int] = None,
    ):
        """Update query configuration.

        The fusion method can be hot-swapped and takes effect immediately, including
        on cached engines.
        """
        if hybrid_fusion is not None:
            self._hybrid_fusion = hybrid_fusion
        if rrf_k is not None:
            self._rrf_k = rrf_k
        if top_k is not None:
            self._top_k = top_k
            logger.debug(f"Updated top_k to {top_k}")

        if similarity_cutoff is not None:
            self._similarity_cutoff = similarity_cutoff
            logger.debug(f"Updated similarity_cutoff to {similarity_cutoff}")

        if sparse_top_k is not None:
            self._sparse_top_k = sparse_top_k
            logger.debug(f"Updated sparse_top_k to {sparse_top_k}")

        if hybrid_alpha is not None:
            self._hybrid_alpha = hybrid_alpha
            logger.debug(f"Updated hybrid_alpha to {hybrid_alpha}")

    def get_config(self) -> dict:
        """Get current query configuration."""
        return {
            'top_k': self._top_k,
            'similarity_cutoff': self._similarity_cutoff,
            'sparse_top_k': self._sparse_top_k,
            'hybrid_alpha': self._hybrid_alpha
        }
