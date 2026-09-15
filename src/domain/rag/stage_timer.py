
"""StageTimer — per-file indexing stage timer (loading / contextualizing / embedding / writing)

Answers "how long did each stage take". It hooks into the progress_cb chain as an
observer: progress reports already flow through here on every stage switch, so the
first time a new stage is seen the previous stage's stopwatch is stopped. It starts
no thread and does not change forwarding semantics (it passes through unchanged to the
inner cb).

Output path: the adapter calls finish() when a file completes; the result is merged
into that file's file_timings entry (stage_ms field) → persisted with IndexJobs and
directly queryable via GET /index/jobs:

    "file_timings": [{
        "file_name": "large-file.pdf", "total_ms": 183200.5,
        "stage_ms": {"loading": 92100.3, "contextualizing": 61400.8,
                     "embedding": 22300.1, "writing": 7400.2}, ...
    }]

If the same stage reappears non-contiguously (defensive; not expected in theory) the
time accumulates. Keepalive heartbeats are "repeated reports of the same stage", so
they do not trigger a switch and do not affect timing.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional


class StageTimer:
    """progress_cb observer: measures the wall-clock time of each stage.

    Usage:
        timer = StageTimer(inner=existing_cb)
        ...pass timer as the progress_cb into the indexer...
        stage_ms = timer.finish()   # {"loading": 92100.3, ...}
    """

    def __init__(self, inner: Optional[Callable] = None):
        self._inner = inner
        self._current: Optional[str] = None
        self._started_at: Optional[float] = None
        self._elapsed: Dict[str, float] = {}

    def __call__(self, stage: str, done, total) -> None:
        now = time.perf_counter()
        if stage != self._current:
            self._close(now)
            self._current = stage
            self._started_at = now
        if self._inner is not None:
            # A forwarding failure must not affect timing or propagate up
            # (same discipline as _report_chunk_progress)
            try:
                self._inner(stage, done, total)
            except Exception:
                pass

    def _close(self, now: float) -> None:
        if self._current is not None and self._started_at is not None:
            self._elapsed[self._current] = (
                self._elapsed.get(self._current, 0.0) + (now - self._started_at)
            )

    def finish(self) -> Dict[str, float]:
        """Stop the running stopwatch and return {stage: milliseconds}. Idempotent (safe to call repeatedly)."""
        self._close(time.perf_counter())
        self._current = None
        self._started_at = None
        return {k: round(v * 1000, 1) for k, v in self._elapsed.items()}
