
"""Regression test for LLM disconnection on mid-generation abort of context.

Contract: when a file is deleted during the contextualizing stage, beyond "no
LLM call for subsequent chunks" (an existing checkpoint), the abort watcher must
also **close the client immediately** to cut the connection for in-flight
requests -- vLLM aborts generation on detecting a disconnect, so the GPU does
not keep burning on a result destined to be discarded.
"""

import threading
import time

from src.domain.rag.context_generator import ContextGenerator


class _FakeLLMClient:
    """A fake client whose chat.completions.create blocks until close().

    Simulates an "in-flight LLM request": without an abort disconnect, each
    request takes _HANG_SECONDS to return; the moment close() is called it
    raises immediately (the connection is cut).
    """

    _HANG_SECONDS = 30.0

    def __init__(self):
        self.closed = threading.Event()
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if self.closed.wait(timeout=self._HANG_SECONDS):
            raise RuntimeError("connection closed by abort watcher")
        raise AssertionError("fake LLM 請求跑滿 30s — abort 沒有斷線")

    def close(self):
        self.closed.set()


async def test_abort_closes_client_and_cuts_inflight_calls(mocker):
    """Once the abort flag is set, in-flight LLM calls must be disconnected and wrap up quickly (TC-ctx-abort-01)"""
    fake = _FakeLLMClient()
    gen = ContextGenerator.__new__(ContextGenerator)  # skip __init__ (avoids config)
    gen._llm_config = None
    gen._model = "fake"
    gen._max_context_length = 150
    gen._max_concurrent = 4
    gen._max_doc_chars = 1000
    gen._max_tokens = 64
    gen._reasoning_effort = None
    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=fake)

    aborted = threading.Event()

    def should_abort():
        return aborted.is_set()

    import asyncio

    async def flip_abort_soon():
        await asyncio.sleep(0.3)   # let a few requests take off first
        aborted.set()

    t0 = time.monotonic()
    results, _ = await asyncio.gather(
        gen.generate_batch(
            chunks=["段落一", "段落二", "段落三", "段落四"],
            full_document="測試文件全文",
            file_name="deleted.docx",
            should_abort=should_abort,
        ),
        flip_abort_soon(),
    )
    elapsed = time.monotonic() - t0

    assert fake.closed.is_set(), "abort 後 client 必須被 close"
    # Without a disconnect it would hang at least 30s; after disconnect everything wraps up fast (fallback prefix)
    assert elapsed < 5.0, f"abort 斷線失效,耗時 {elapsed:.1f}s"
    assert len(results) == 4, "每個 chunk 都要有結果(fallback),不能缺"
