"""Auto-merge rule shared by REST, MCP and the console.

A parent (a run of consecutive leaves) replaces its leaves in the results when
at least ``MIN_MERGE_HITS`` of its leaves were hit in this result set and the hit
ratio (hits / the parent's leaf count) reaches ``merge_threshold``. The merged
entry takes the place of its best-ranked leaf and the score of its best leaf.

DB access is two batched queries (children counts, parent content); either one
failing degrades to "no merge" — retrieval never fails because of merging.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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


async def plan_merges(vector_store, leaves: List[Any], merge_threshold: float
                      ) -> Dict[str, MergeGroup]:
    """Decide which parents replace their leaves and fetch their content."""
    if merge_threshold is None or merge_threshold > 1.0:
        return {}
    from src.domain.rag.chunk_lookup import ChunkLookup

    parent_hits = group_by_parent(leaves)
    worth_counting = [pid for pid, hits in parent_hits.items() if len(hits) >= MIN_MERGE_HITS]
    if not worth_counting:
        return {}
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
