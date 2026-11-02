
"""Auto-Merging Retriever — merge multiple leaf hits under the same parent back into that parent.

Core logic:
1. Retrieve top-k LEAF nodes through the shared hybrid search path
   (src/domain/rag/hybrid_search.py: dense + BM25, fusion, rerank) — the same
   path REST, the console and the eval harness use.
2. Apply the shared auto-merge rule (src/domain/rag/merge_plan.py): a parent
   whose hit ratio reaches merge_threshold replaces its leaves.
3. For leaves that weren't promoted, optionally expand their neighbors (expand_context).

Design trade-offs:
- Does not use LlamaIndex's built-in AutoMergingRetriever, which relies on an in-memory docstore,
  whereas this uses the PGVector + JSONB metadata approach. A small bespoke implementation is clearer.
- After merging, the original leaf score is preserved (using the highest score among the parent's
  hit children), so the final ranking stays meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llama_index.vector_stores.postgres import PGVectorStore

from src.domain.rag.hybrid_search import HybridSearcher, SearchOutcome, SearchSettings
from src.domain.rag.merge_plan import MERGE_DISABLED, MergeGroup, arrange, plan_merges
from src.log import get_api_logger

logger = get_api_logger()


@dataclass
class MergedResult:
    """A single result after auto-merge.

    May be:
    - An original leaf (no merge triggered).
    - A promoted parent (merge triggered).
    - A leaf with expanded neighbors (expand_context=True).
    """
    text: str
    score: float
    file_id: str
    file_name: str
    node_id: str
    node_role: str                    # leaf / parent / expanded
    chunk_index_range: List[int]      # parent: [start, end]; leaf: [i, i]
    merged_from_leaves: List[str] = field(default_factory=list)  # recorded when a parent merge happens
    # Citation provenance (from leaf node metadata; on a parent merge, taken from the highest-scoring hit leaf)
    page: Optional[int] = None
    headings: Optional[List[str]] = None
    content_type: Optional[str] = None


class AutoMergingRetriever:
    """Hybrid search + auto-merge + neighbour expansion (the MCP search mode).

    Usage:
        retriever = AutoMergingRetriever(vector_store=..., reranker=...,
                                         merge_threshold=0.5, expand_context_neighbors=2)
        results = await retriever.aquery("question", SearchSettings.from_config(top_k=6))
    """

    def __init__(
        self,
        vector_store: PGVectorStore,
        reranker=None,
        merge_threshold: float = 0.5,
        expand_context_neighbors: int = 2,
        searcher: Optional[HybridSearcher] = None,
    ):
        self._vector_store = vector_store
        self._merge_threshold = merge_threshold
        self._expand_context_neighbors = expand_context_neighbors
        self._searcher = searcher or HybridSearcher(vector_store, reranker=reranker)
        # The outcome of the most recent aquery on THIS instance. Instances are
        # built per request (never shared), so this is not cross-request state.
        self.last_outcome: Optional[SearchOutcome] = None

    async def aquery(
        self,
        query_text: str,
        settings: Optional[SearchSettings] = None,
        expand_context: bool = True,
        *,
        top_k: Optional[int] = None,
        similarity_cutoff: Optional[float] = None,
    ) -> List[MergedResult]:
        """Hybrid search → auto-merge → optional neighbour expansion.

        DB interaction after retrieval is batched: parent counts, parent content and
        neighbour windows are one query each.
        """
        from src.domain.rag.chunk_lookup import ChunkLookup

        settings = (settings or SearchSettings.from_config()).with_overrides(
            top_k=top_k, similarity_cutoff=similarity_cutoff)
        outcome = await self._searcher.search(query_text, settings)
        self.last_outcome = outcome
        leaf_nodes = outcome.nodes
        if not leaf_nodes:
            return []

        groups = await plan_merges(self._vector_store, leaf_nodes, self._merge_threshold)
        arranged = arrange(leaf_nodes, groups)

        # Neighbour windows for the leaves that stay leaves, in one batch.
        window_by_leaf: Dict[str, dict] = {}
        expanding = expand_context and self._expand_context_neighbors > 0
        if expanding:
            targets = []
            for item in arranged:
                if isinstance(item, MergeGroup):
                    continue
                fid = item.node.metadata.get("file_id")
                ci = item.node.metadata.get("chunk_index")
                if fid is not None and ci is not None:
                    targets.append((item.node.id_, fid, ci))
            if targets:
                windows = await ChunkLookup.afetch_neighbor_windows(
                    self._vector_store, [(fid, ci) for _, fid, ci in targets],
                    n=self._expand_context_neighbors)
                window_by_leaf = {leaf_id: w for (leaf_id, _, _), w in zip(targets, windows)}

        results: List[MergedResult] = []
        for item in arranged:
            if isinstance(item, MergeGroup):
                parent = item.parent
                results.append(MergedResult(
                    text=parent.text,
                    score=item.score,
                    file_id=parent.metadata.get("file_id", ""),
                    file_name=parent.metadata.get("file_name", ""),
                    node_id=item.parent_id,
                    node_role="parent",
                    chunk_index_range=parent.metadata.get("children_index_range", []),
                    merged_from_leaves=[h.node.id_ for h in item.hits],
                    **_provenance(item.best_hit.node),
                ))
                continue
            node = item.node
            center = node.metadata.get("chunk_index")
            if expanding:
                # A leaf without file_id/chunk_index has no window: keep its own text.
                window = window_by_leaf.get(
                    node.id_, {"text": node.text, "index_range": [center, center]})
                text, role, index_range = window["text"], "expanded", window["index_range"]
            else:
                ci = center if center is not None else -1
                text, role, index_range = node.text, "leaf", [ci, ci]
            results.append(MergedResult(
                text=text,
                score=item.score or 0.0,
                file_id=node.metadata.get("file_id", ""),
                file_name=node.metadata.get("file_name", ""),
                node_id=node.id_,
                node_role=role,
                chunk_index_range=index_range,
                **_provenance(node),
            ))

        logger.info(
            f"AutoMerge query done: {len(leaf_nodes)} leaves to {len(results)} results "
            f"({len(groups)} merged to parent)")
        return results


def _provenance(node) -> dict:
    """Leaf metadata page_no/headings/content_type → MergedResult citation fields."""
    md = node.metadata
    return {"page": md.get("page_no"), "headings": md.get("headings"),
            "content_type": md.get("content_type")}


__all__ = ["AutoMergingRetriever", "MergedResult", "MERGE_DISABLED"]
