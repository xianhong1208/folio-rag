
"""Regression tests for the indexing-job keepalive heartbeat.

The IndexingJobManager watchdog flips a job to FAILED and cancels the task
after 600s without a heartbeat, but the per-file budget
(_PER_FILE_TIMEOUT_DEFAULT, currently 864000s / 10 days) is far larger than the
watchdog — the two guards' budgets differ wildly. A slow file sitting in a
heartbeat blind spot (semaphore wait / docling OCR / context-gen throttling
gaps) is killed by the watchdog before its per-file timeout expires, which
reproduces on slower machines.

_ProgressKeepalive tops up the heartbeat in place while idle so "slow" is no
longer mistaken for "dead"; a genuinely hung file is still backstopped by the
per-file wait_for.
"""

import asyncio

from src.adapter.rag_indexing import _ProgressKeepalive, _keepalive_stage_label


async def test_keepalive_emits_while_idle():
    """After idling past the interval, a heartbeat must be topped up — the only thing keeping the watchdog from a false kill."""
    seen = []
    async with _ProgressKeepalive(lambda s, d, t: seen.append((s, d, t)), interval=0.05):
        await asyncio.sleep(0.3)

    # Before any real progress is reported, the top-up is the initial "queued" (covers the semaphore-wait blind spot)
    assert len(seen) >= 3, f"expected repeated keepalive ticks, got {seen}"
    assert all(stage == "queued" for stage, _, _ in seen)


async def test_keepalive_repeats_last_stage_verbatim():
    """A heartbeat top-up must re-send the stage verbatim, with no elapsed annotation.

    The portal's IndexingStatusBadge looks up a Chinese label via
    STAGE_LABEL[stage] (falling back to the raw string). Once the stage is
    rewritten to "loading (8m20s)" the lookup misses and the badge degrades from
    the localized label to English — precisely when parsing takes longest.
    """
    seen = []
    keepalive = _ProgressKeepalive(lambda s, d, t: seen.append((s, d, t)), interval=0.05)
    async with keepalive:
        keepalive("loading", None, None)   # one real progress tick
        await asyncio.sleep(0.2)

    assert seen[0] == ("loading", None, None)
    assert len(seen) >= 2, "keepalive should have topped up after the real tick"
    assert all(stage == "loading" for stage, _, _ in seen)


async def test_real_progress_suppresses_keepalive():
    """A normal-speed file should produce no extra ticks (otherwise SSE / DB get needless extra writes)."""
    seen = []
    keepalive = _ProgressKeepalive(lambda s, d, t: seen.append((s, d, t)), interval=0.5)
    async with keepalive:
        for done in range(5):
            keepalive("embedding", done, 5)
            await asyncio.sleep(0.02)      # 0.1s total, well under the interval

    assert seen == [("embedding", i, 5) for i in range(5)]


async def test_keepalive_stops_after_exit():
    """After leaving the context, the background task must stop — otherwise the job is over but still pretending to be alive."""
    seen = []
    async with _ProgressKeepalive(lambda s, d, t: seen.append(s), interval=0.05):
        await asyncio.sleep(0.15)
    settled = len(seen)

    await asyncio.sleep(0.2)
    assert len(seen) == settled


async def test_callback_failure_does_not_break_indexing():
    """A progress-report failure must not affect indexing itself (consistent with the existing call site's handling)."""
    def boom(stage, done, total):
        raise RuntimeError("progress sink down")

    async with _ProgressKeepalive(boom, interval=0.05):
        await asyncio.sleep(0.15)     # should not raise


def test_stage_label_is_passthrough():
    """Contract test: the stage label is returned verbatim, with elapsed never encoded into it."""
    assert _keepalive_stage_label("loading", 0.0) == "loading"
    assert _keepalive_stage_label("contextualizing", 9999.0) == "contextualizing"
