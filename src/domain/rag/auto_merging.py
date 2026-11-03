
"""Auto-Merging Retriever — merge multiple leaf hits under the same parent back into that parent.

Core logic:
1. Retrieve top-k LEAF nodes through the shared hybrid search path
   (src/domain/rag/hybrid_search.py: dense + BM25, fusion, rerank) — the same
   path REST, the console and the eval harness use.
2. Apply the shared auto-merge rule (src/domain/rag/merge_plan.py): a parent
   whose hit ratio reaches merge_threshold replaces its leaves.
3. For leaves that weren't promoted, optionally expand their neighbors (expand_context):
   windows stay inside the hit's section and adjacent windows of a file are merged.

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

from src.domain.rag.context_windows import Window, WindowHit, plan_windows, requested_range
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
    # parent: the hit leaves it replaced; expanded: the hits one merged window covers (when > 1)
    merged_from_leaves: List[str] = field(default_factory=list)
    # Citation provenance (from leaf node metadata; on a parent merge, taken from the highest-scoring hit leaf)
    page: Optional[int] = None
    headings: Optional[List[str]] = None
    content_type: Optional[str] = None


class AutoMergingRetriever:
    """Hybrid search + auto-merge + neighbour expansion (the MCP search mode).

    Usage:
        retriever = AutoMergingRetriever(vector_store=..., reranker=...,
                                         merge_threshold=0.5, expand_context_neighbors=1)
        results = await retriever.aquery("question", SearchSettings.from_config(top_k=6))
    """

    def __init__(
        self,
        vector_store: PGVectorStore,
        reranker=None,
        merge_threshold: float = 0.5,
        expand_context_neighbors: int = 1,
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
        neighbour rows are one query each. Neighbour windows are clipped to the
        hit's section and merged per file (see context_windows.py), so adjacent
        hits come back as one passage instead of overlapping copies.
        """
        settings = (settings or SearchSettings.from_config()).with_overrides(
            top_k=top_k, similarity_cutoff=similarity_cutoff)
        outcome = await self._searcher.search(query_text, settings)
        self.last_outcome = outcome
        leaf_nodes = outcome.nodes
        if not leaf_nodes:
            return []

        groups = await plan_merges(self._vector_store, leaf_nodes, self._merge_threshold)
        arranged = arrange(leaf_nodes, groups)

        expanding = expand_context and self._expand_context_neighbors > 0
        windows_by_leaf: Dict[str, Window] = {}
        if expanding:
            windows_by_leaf = await self._plan_windows(arranged)

        results: List[MergedResult] = []
        emitted: set = set()
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
            window = windows_by_leaf.get(node.id_)
            if window is not None:
                # One result per merged window, in the slot of its best-ranked hit.
                if id(window) in emitted:
                    continue
                emitted.add(id(window))
                results.append(MergedResult(
                    text=window.text,
                    score=window.score,
                    file_id=node.metadata.get("file_id", ""),
                    file_name=node.metadata.get("file_name", ""),
                    node_id=node.id_,
                    node_role="expanded",
                    chunk_index_range=[window.start, window.end],
                    merged_from_leaves=(
                        [h.node_id for h in window.hits] if len(window.hits) > 1 else []),
                    **_provenance(node),
                ))
                continue
            center = node.metadata.get("chunk_index")
            ci = center if center is not None else -1
            results.append(MergedResult(
                text=node.text,
                score=item.score or 0.0,
                file_id=node.metadata.get("file_id", ""),
                file_name=node.metadata.get("file_name", ""),
                node_id=node.id_,
                # A leaf without file_id/chunk_index has no window: it keeps its own text.
                node_role="expanded" if expanding else "leaf",
                chunk_index_range=[ci, ci],
                **_provenance(node),
            ))

        logger.info(
            f"AutoMerge query done: {len(leaf_nodes)} leaves to {len(results)} results "
            f"({len(groups)} merged to parent, {len(set(map(id, windows_by_leaf.values())))} "
            f"window(s))")
        return results

    async def _plan_windows(self, arranged) -> Dict[str, Window]:
        """Neighbour windows for the leaves that stay leaves (one batched query),
        clipped to their section, skipping merged parents' chunks and merged per
        file. Returns {leaf node id: its (possibly shared) window}."""
        from src.domain.rag.chunk_lookup import ChunkLookup

        n = self._expand_context_neighbors
        hits: List[WindowHit] = []
        blocked: Dict[str, List[List[int]]] = {}
        for rank, item in enumerate(arranged):
            if isinstance(item, MergeGroup):
                md = item.parent.metadata
                rng = md.get("children_index_range") or []
                if md.get("file_id") is not None and len(rng) == 2:
                    blocked.setdefault(str(md["file_id"]), []).append([int(rng[0]), int(rng[1])])
                continue
            md = item.node.metadata
            fid, ci = md.get("file_id"), md.get("chunk_index")
            if fid is None or ci is None:
                continue
            hits.append(WindowHit(
                node_id=item.node.id_, file_id=str(fid), chunk_index=int(ci),
                parent_id=md.get("parent_node_id"), rank=rank,
                score=float(item.score or 0.0), text=item.node.text))
        if not hits:
            return {}
        rows = await ChunkLookup.afetch_rows_in_ranges(
            self._vector_store,
            [(h.file_id, *requested_range(h.chunk_index, n)) for h in hits])
        windows = plan_windows(hits, rows, n, blocked)
        return {h.node_id: w for w in windows for h in w.hits}


def _provenance(node) -> dict:
    """Leaf metadata page_no/headings/content_type → MergedResult citation fields."""
    md = node.metadata
    return {"page": md.get("page_no"), "headings": md.get("headings"),
            "content_type": md.get("content_type")}


__all__ = ["AutoMergingRetriever", "MergedResult", "MERGE_DISABLED"]
