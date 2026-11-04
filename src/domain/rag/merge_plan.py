"""Auto-merge rule shared by REST, MCP and the console.

A parent (a run of consecutive leaves) replaces its leaves in the results when
at least ``MIN_MERGE_HITS`` of its leaves were hit in this result set and the hit
ratio (hits / the parent's leaf count) reaches ``merge_threshold``. The merged
entry takes the place of its best-ranked leaf and the score of its best leaf.

DB access is two batched queries (children counts, parent content), or none when
the caller already holds the parent rows; either query failing degrades to "no
merge" — retrieval never fails because of merging.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.log import get_api_logger

logger = get_api_logger()

MIN_MERGE_HITS = 2

# A merge threshold above 1.0 can never be reached: callers use it to disable merging.
MERGE_DISABLED = 1.1


@dataclass
class MergeGroup:
    """A parent that replaces some hit leaves."""

    parent: Any                 # TextNode of the parent
    hits: List[Any]             # the hit leaves (NodeWithScore), in result order

    @property
    def best_hit(self):
        return max(self.hits, key=lambda h: h.score or 0.0)

    @property
    def score(self) -> float:
        return self.best_hit.score or 0.0

    @property
    def parent_id(self) -> str:
        return self.parent.node_id


def group_by_parent(leaves: List[Any]) -> Dict[str, List[Any]]:
    """Hit leaves grouped by ``parent_node_id`` (legacy leaves without one never merge)."""
    out: Dict[str, List[Any]] = defaultdict(list)
    for leaf in leaves:
        pid = leaf.node.metadata.get("parent_node_id")
        if pid:
            out[pid].append(leaf)
    return dict(out)


def select_parents(parent_hits: Dict[str, List[Any]], children_counts: Dict[str, int],
                   merge_threshold: float) -> List[str]:
    """Parent ids that meet the rule, in first-hit order."""
    chosen: List[str] = []
    for pid, hits in parent_hits.items():
        total = children_counts.get(pid, 0)
        if total <= 0 or len(hits) < MIN_MERGE_HITS:
            continue
        if len(hits) / total >= merge_threshold:
            chosen.append(pid)
    return chosen


def merge_candidates(leaves: List[Any]) -> List[str]:
    """Parents with enough hits among ``leaves`` to be worth a merge check."""
    return [pid for pid, hits in group_by_parent(leaves).items() if len(hits) >= MIN_MERGE_HITS]


def merging_enabled(merge_threshold: Optional[float]) -> bool:
    return merge_threshold is not None and merge_threshold <= 1.0


def _counts_from_nodes(parents: Dict[str, Any]) -> Dict[str, int]:
    """chunk_count of parent rows, as aget_parent_children_counts reads it."""
    out = {}
    for pid, node in parents.items():
        md = node.metadata or {}
        if md.get("node_role") == "parent" and md.get("chunk_count") is not None:
            out[pid] = int(md["chunk_count"])
    return out


async def plan_merges(vector_store, leaves: List[Any], merge_threshold: float,
                      known_parents: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, MergeGroup]:
    """Decide which parents replace their leaves and fetch their content.

    ``known_parents`` ({parent id: TextNode}, e.g. fetched while the reranker ran)
    replaces both queries when it holds every parent worth checking; otherwise
    the parents are read here.
    """
    if not merging_enabled(merge_threshold):
        return {}
    from src.domain.rag.chunk_lookup import ChunkLookup

    parent_hits = group_by_parent(leaves)
    worth_counting = merge_candidates(leaves)
    if not worth_counting:
        return {}
    if known_parents is not None and all(pid in known_parents for pid in worth_counting):
        counts = _counts_from_nodes({pid: known_parents[pid] for pid in worth_counting})
        pids = select_parents(parent_hits, counts, merge_threshold)
        parents = {pid: known_parents[pid] for pid in pids}
    else:
        counts = await ChunkLookup.aget_parent_children_counts(vector_store, worth_counting)
        pids = select_parents(parent_hits, counts, merge_threshold)
        if not pids:
            return {}
        parents = await ChunkLookup.afetch_nodes(vector_store, pids)
    groups = {pid: MergeGroup(parent=parents[pid], hits=parent_hits[pid])
              for pid in pids if pid in parents}
    if groups:
        logger.debug(f"auto-merge: {len(groups)} parent(s) replace their leaves")
    return groups


def arrange(leaves: List[Any], groups: Dict[str, MergeGroup]) -> List[Union[Any, MergeGroup]]:
    """Final order: a merged parent takes its first hit leaf's slot; its other leaves go."""
    out: List[Union[Any, MergeGroup]] = []
    placed: set = set()
    for leaf in leaves:
        pid = leaf.node.metadata.get("parent_node_id")
        if pid in groups:
            if pid not in placed:
                placed.add(pid)
                out.append(groups[pid])
            continue
        out.append(leaf)
    return out
