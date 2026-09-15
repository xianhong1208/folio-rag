
"""Auto-Merging Retriever — merge multiple leaf hits under the same parent back into that parent.

Core logic:
1. Retrieve top-k LEAF nodes via hybrid retrieval (vector + BM25) + rerank
   (a metadata filter restricts retrieval to node_role=leaf; parents don't participate).
2. Tally the parent distribution of the hit leaves.
3. If a parent's ratio of hit children reaches merge_threshold, promote back to the parent.
4. For leaves that weren't promoted, optionally expand their neighbors (expand_context).

Design trade-offs:
- Does not use LlamaIndex's built-in AutoMergingRetriever, which relies on an in-memory docstore,
  whereas this uses the PGVector + JSONB metadata approach. A small bespoke implementation is clearer.
- After merging, the original leaf score is preserved (using the highest score among the parent's
  hit children), so the final ranking stays meaningful.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.postgres import PGVectorStore

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


class AutoMergingRetriever:
    """Retriever that intelligently merges hits after retrieval.

    Usage:
        retriever = AutoMergingRetriever(
            vector_store=...,
            reranker=...,
            merge_threshold=0.5,
            expand_context_neighbors=2,
        )
        results = await retriever.aquery(query_text="question", top_k=6, expand_context=True)
    """

    def __init__(
        self,
        vector_store: PGVectorStore,
        reranker=None,
        merge_threshold: float = 0.5,
        expand_context_neighbors: int = 2,
        sparse_top_k: int = 12,
        hybrid_alpha: float = 0.75,
    ):
        from llama_index.core import VectorStoreIndex
        from llama_index.core.retrievers import VectorIndexRetriever

        self._vector_store = vector_store
        self._reranker = reranker
        self._merge_threshold = merge_threshold
        self._expand_context_neighbors = expand_context_neighbors
        self._sparse_top_k = sparse_top_k
        self._hybrid_alpha = hybrid_alpha
        self._index = VectorStoreIndex.from_vector_store(vector_store)
        self._VectorIndexRetriever = VectorIndexRetriever

        logger.info(
            f"AutoMergingRetriever ready (merge_threshold={merge_threshold}, "
            f"expand_neighbors={expand_context_neighbors})"
        )

    async def aquery(
        self,
        query_text: str,
        top_k: int = 6,
        similarity_cutoff: float = 0.4,
        expand_context: bool = True,
    ) -> List[MergedResult]:
        """Run hybrid retrieval + auto-merge (fully async + batched DB queries).

        Flow: hybrid retrieve leaves only → rerank → tally parent hits → parents at threshold
        replace their leaves; un-merged leaves optionally expand their neighbors.

        DB interaction is fully batched (at most 3 round trips): one for parent counts, one for
        parent content, one for neighbor windows, all on the async engine.

        Args:
            query_text: Query string.
            top_k: Number of top results to return (after rerank).
            similarity_cutoff: Similarity threshold; ignored when the reranker is enabled.
            expand_context: True → expand neighbors for un-merged leaves.

        Returns:
            List[MergedResult].
        """
        from src.domain.rag.chunk_lookup import ChunkLookup

        retrieve_k = top_k * 3 if self._reranker else top_k

        retriever = self._VectorIndexRetriever(
            index=self._index,
            similarity_top_k=retrieve_k,
            vector_store_query_mode="hybrid",
            vector_store_kwargs={
                "sparse_top_k": self._sparse_top_k,
                "alpha": self._hybrid_alpha,
            },
        )

        nodes = await retriever.aretrieve(query_text)
        # Keep only leaf nodes. Parents share the table but have no embedding, so
        # vector retrieval never picks them; BM25 may occasionally surface one, so
        # filter explicitly. Legacy chunks without node_role are treated as leaves
        # for backward compatibility (they never merge, having no parent_node_id).
        leaf_nodes = [
            n for n in nodes
            if n.node.metadata.get("node_role", "leaf") == "leaf"
        ]

        logger.debug(f"Retrieved {len(nodes)} nodes, {len(leaf_nodes)} are leaves")

        if self._reranker and leaf_nodes:
            leaf_nodes = await self._reranker.arerank(query_text, leaf_nodes, top_n=top_k)

        if not self._reranker:
            leaf_nodes = [n for n in leaf_nodes if n.score and n.score >= similarity_cutoff]

        if not leaf_nodes:
            return []

        # Auto-merge decision: tally parent counts in one batch
        parent_hits: Dict[str, List[NodeWithScore]] = defaultdict(list)
        for ln in leaf_nodes:
            pid = ln.node.metadata.get("parent_node_id")
            if pid:
                parent_hits[pid].append(ln)

        children_counts = await ChunkLookup.aget_parent_children_counts(
            self._vector_store, list(parent_hits.keys())
        )

        parents_to_merge: List[str] = []
        for pid, hit_leaves in parent_hits.items():
            total_children = children_counts.get(pid, 0)
            if total_children == 0:
                continue
            hit_ratio = len(hit_leaves) / total_children
            if hit_ratio >= self._merge_threshold and len(hit_leaves) >= 2:
                parents_to_merge.append(pid)
                logger.debug(
                    f"Will merge parent {pid[:8]}: "
                    f"{len(hit_leaves)}/{total_children} children hit ({hit_ratio:.0%})"
                )

        # Batch prefetch parent content
        parent_nodes_map = await ChunkLookup.afetch_nodes(
            self._vector_store, parents_to_merge
        )
        # Neighbor windows: collect leaves taking the expand branch (parent not merged, first seen), then one batch
        window_by_leaf: Dict[str, dict] = {}
        if expand_context and self._expand_context_neighbors > 0:
            expand_targets = []
            seen_ids: set = set()
            for ln in leaf_nodes:
                pid = ln.node.metadata.get("parent_node_id")
                if pid in parents_to_merge or ln.node.id_ in seen_ids:
                    continue
                seen_ids.add(ln.node.id_)
                fid = ln.node.metadata.get("file_id")
                ci = ln.node.metadata.get("chunk_index")
                if fid is not None and ci is not None:
                    expand_targets.append((ln.node.id_, fid, ci))
            if expand_targets:
                windows = await ChunkLookup.afetch_neighbor_windows(
                    self._vector_store,
                    [(fid, ci) for _, fid, ci in expand_targets],
                    n=self._expand_context_neighbors,
                )
                window_by_leaf = {
                    leaf_id: w
                    for (leaf_id, _, _), w in zip(expand_targets, windows)
                }

        results: List[MergedResult] = []
        merged_parent_ids: set = set()
        used_leaf_ids: set = set()

        def _provenance(node) -> dict:
            """Map leaf metadata page_no/headings → MergedResult citation fields."""
            md = node.metadata
            return {"page": md.get("page_no"), "headings": md.get("headings")}

        for ln in leaf_nodes:
            pid = ln.node.metadata.get("parent_node_id")

            if pid in parents_to_merge and pid not in merged_parent_ids:
                parent_node = parent_nodes_map.get(pid)
                if parent_node:
                    merged_leaves = [hl.node.id_ for hl in parent_hits[pid]]
                    used_leaf_ids.update(merged_leaves)
                    best_hit = max(parent_hits[pid], key=lambda hl: hl.score or 0.0)
                    results.append(MergedResult(
                        text=parent_node.text,
                        score=best_hit.score or 0.0,
                        file_id=parent_node.metadata.get("file_id", ""),
                        file_name=parent_node.metadata.get("file_name", ""),
                        node_id=pid,
                        node_role="parent",
                        chunk_index_range=parent_node.metadata.get("children_index_range", []),
                        merged_from_leaves=merged_leaves,
                        **_provenance(best_hit.node),
                    ))
                    merged_parent_ids.add(pid)
                continue

            if pid in merged_parent_ids:
                continue

            if ln.node.id_ in used_leaf_ids:
                continue
            used_leaf_ids.add(ln.node.id_)

            if expand_context and self._expand_context_neighbors > 0:
                # Batch-prefetched window; a leaf missing file_id/chunk_index isn't in the map,
                # so fall back to the original leaf content.
                center = ln.node.metadata.get("chunk_index")
                expanded = window_by_leaf.get(
                    ln.node.id_,
                    {"text": ln.node.text, "index_range": [center, center]},
                )
                results.append(MergedResult(
                    text=expanded["text"],
                    score=ln.score or 0.0,
                    file_id=ln.node.metadata.get("file_id", ""),
                    file_name=ln.node.metadata.get("file_name", ""),
                    node_id=ln.node.id_,
                    node_role="expanded",
                    chunk_index_range=expanded["index_range"],
                    **_provenance(ln.node),
                ))
            else:
                results.append(MergedResult(
                    text=ln.node.text,
                    score=ln.score or 0.0,
                    file_id=ln.node.metadata.get("file_id", ""),
                    file_name=ln.node.metadata.get("file_name", ""),
                    node_id=ln.node.id_,
                    node_role="leaf",
                    chunk_index_range=[
                        ln.node.metadata.get("chunk_index", -1),
                        ln.node.metadata.get("chunk_index", -1),
                    ],
                    **_provenance(ln.node),
                ))

        results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            f"AutoMerge query done: {len(leaf_nodes)} leaves to "
            f"{len(results)} results ({len(merged_parent_ids)} merged to parent)"
        )
        return results
