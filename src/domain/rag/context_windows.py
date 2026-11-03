"""Neighbour-window planning for MCP search results (pure, no I/O).

A hit leaf that was not merged into its parent is shown with up to ``n``
neighbouring leaves on each side. Three rules keep that context from repeating
itself:

1. **Clip to the section.** A window never leaves the hit's parent
   (``parent_node_id``): the parent is a run of consecutive leaves inside one
   heading section, so the neighbours shown belong to the same passage. Legacy
   leaves without a parent are not clipped.
2. **Skip what is already shown.** Chunks covered by a merged parent in the same
   result set are not repeated in a window.
3. **Merge per file.** Windows of the same file that overlap — or touch inside
   the same section — become one result (an interval union) that keeps the best
   score and the best rank of the hits it covers, so two adjacent hits no longer
   return the same text twice. Windows of neighbouring sections that merely touch
   stay separate results (nothing is repeated between them).

The caller fetches the candidate rows (one query, see
``ChunkLookup.afetch_rows_in_ranges``) and turns the returned :class:`Window`
objects into results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# (chunk_index, text, parent_node_id)
Row = Tuple[int, str, Optional[str]]


@dataclass
class WindowHit:
    """One hit leaf that gets a neighbour window."""

    node_id: str
    file_id: str
    chunk_index: int
    parent_id: Optional[str]
    rank: int            # position in the ranked result list (0 = best)
    score: float
    text: str = ""       # the hit's own text (used if its row is missing)


@dataclass
class Window:
    """A contiguous run of leaves of one file covering one or more hits."""

    file_id: str
    start: int
    end: int
    hits: List[WindowHit] = field(default_factory=list)   # best rank first
    text: str = ""

    @property
    def best(self) -> WindowHit:
        return self.hits[0]

    @property
    def rank(self) -> int:
        return self.best.rank

    @property
    def score(self) -> float:
        return max(h.score for h in self.hits)


def requested_range(center: int, n: int) -> Tuple[int, int]:
    """The ±n range to fetch around ``center`` (never below 0)."""
    n = max(0, int(n))
    return max(center - n, 0), center + n


def _blocked(idx: int, ranges: Sequence[Sequence[int]]) -> bool:
    return any(lo <= idx <= hi for lo, hi in ranges)


def clip_window(hit: WindowHit, rows: Dict[int, Tuple[str, Optional[str]]], n: int,
                blocked: Sequence[Sequence[int]] = ()) -> Tuple[int, int]:
    """The hit's window: walk outward from its chunk while the neighbour exists,
    shares the hit's parent (when the hit has one), is not already shown by a
    merged parent and stays within ``n``."""
    lo_lim, hi_lim = requested_range(hit.chunk_index, n)

    def ok(idx: int) -> bool:
        if idx not in rows or _blocked(idx, blocked):
            return False
        return hit.parent_id is None or rows[idx][1] == hit.parent_id

    lo = hi = hit.chunk_index
    while lo - 1 >= lo_lim and ok(lo - 1):
        lo -= 1
    while hi + 1 <= hi_lim and ok(hi + 1):
        hi += 1
    return lo, hi


def plan_windows(hits: Iterable[WindowHit], rows_by_file: Dict[str, Sequence[Row]], n: int,
                 blocked_by_file: Optional[Dict[str, Sequence[Sequence[int]]]] = None,
                 separator: str = "\n\n") -> List[Window]:
    """Clip, merge and render the windows of ``hits``; returned best rank first."""
    blocked_by_file = blocked_by_file or {}
    by_file: Dict[str, List[Tuple[int, int, WindowHit]]] = {}
    row_maps: Dict[str, Dict[int, Tuple[str, Optional[str]]]] = {}
    for hit in hits:
        fid = str(hit.file_id)
        rows = row_maps.get(fid)
        if rows is None:
            rows = {int(i): (t or "", p) for i, t, p in rows_by_file.get(fid, ())}
            row_maps[fid] = rows
        # The hit's own text is always available, even if its row was not fetched.
        rows.setdefault(hit.chunk_index, (hit.text, hit.parent_id))
        lo, hi = clip_window(hit, rows, n, blocked_by_file.get(fid, ()))
        by_file.setdefault(fid, []).append((lo, hi, hit))

    windows: List[Window] = []
    for fid, items in by_file.items():
        items.sort(key=lambda x: (x[0], x[1]))
        current: Optional[Window] = None
        for lo, hi, hit in items:
            if current is not None and (
                    lo <= current.end
                    or (lo == current.end + 1 and hit.parent_id is not None
                        and hit.parent_id == current.hits[-1].parent_id)):
                current.end = max(current.end, hi)
                current.hits.append(hit)
                continue
            current = Window(file_id=fid, start=lo, end=hi, hits=[hit])
            windows.append(current)
        rows = row_maps[fid]
        for w in (w for w in windows if w.file_id == fid):
            w.hits.sort(key=lambda h: h.rank)
            w.text = separator.join(rows[i][0] for i in range(w.start, w.end + 1) if i in rows)
    windows.sort(key=lambda w: w.rank)
    return windows
