
"""StageTimer unit tests — measuring the duration of each indexing stage.

Contract: hooked onto the progress_cb chain to observe stage transitions;
repeated reports of the same stage (keepalive heartbeats) do not affect timing;
finish() closes out the running stopwatch and returns {stage: milliseconds}.
"""

import time

from src.domain.rag.stage_timer import StageTimer


def test_measures_each_stage_duration():
    """Each stage is timed by its transition points (TC-stage-timer-01)."""
    t = StageTimer()
    t("loading", None, None)
    time.sleep(0.10)
    t("contextualizing", 0, 50)
    time.sleep(0.05)
    t("embedding", 0, 50)
    time.sleep(0.02)
    out = t.finish()

    assert set(out) == {"loading", "contextualizing", "embedding"}
    assert out["loading"] >= 90, out          # ~100ms
    assert out["contextualizing"] >= 40, out  # ~50ms
    assert out["embedding"] >= 15, out        # ~20ms
    # No stage should absorb another stage's time (loose upper bound to avoid CI jitter)
    assert out["embedding"] < 100, out


def test_same_stage_repeats_do_not_reset_clock():
    """Repeated reports of the same stage (progress ticks / keepalive heartbeats) do not reset the stopwatch (TC-stage-timer-02)."""
    t = StageTimer()
    t("embedding", 0, 100)
    time.sleep(0.03)
    t("embedding", 50, 100)   # progress tick
    t("embedding", 50, 100)   # keepalive re-send
    time.sleep(0.03)
    out = t.finish()
    assert out["embedding"] >= 50, out  # two segments total ~60ms, not reset by the tick


def test_forwards_to_inner_cb_and_swallows_inner_errors():
    """Forwarded unchanged to the inner cb; an inner failure does not affect timing (TC-stage-timer-03)."""
    seen = []

    def inner(stage, done, total):
        seen.append((stage, done, total))
        raise RuntimeError("inner 故意炸")

    t = StageTimer(inner=inner)
    t("loading", 1, 3)
    t("loading", 2, 3)
    out = t.finish()
    assert seen == [("loading", 1, 3), ("loading", 2, 3)]
    assert "loading" in out


def test_finish_is_idempotent():
    """finish() can be called repeatedly with a consistent result (TC-stage-timer-04)."""
    t = StageTimer()
    t("writing", 10, 10)
    time.sleep(0.02)
    first = t.finish()
    second = t.finish()
    assert first == second


def test_interleaved_stage_accumulates():
    """Time accumulates when a stage reappears intermittently (defensive) (TC-stage-timer-05)."""
    t = StageTimer()
    t("loading", None, None)
    time.sleep(0.02)
    t("contextualizing", 0, 10)
    time.sleep(0.02)
    t("loading", None, None)   # should not happen in theory; defensive
    time.sleep(0.02)
    out = t.finish()
    assert out["loading"] >= 35, out  # two loading segments accumulate ~40ms
