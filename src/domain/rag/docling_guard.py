"""Make docling conversion failures visible instead of silently losing pages.

docling's threaded PDF pipeline (ocr / layout / table stages) catches an
exception raised inside a stage, including a CUDA out-of-memory error, marks
the affected pages as failed and carries on: ``convert()`` then returns a
``PARTIAL_SUCCESS`` result without raising, and those pages are simply absent
from the document. This module wraps a conversion so that:

- the result status is always checked (``FAILURE`` raises, ``PARTIAL_SUCCESS``
  is retried);
- an out-of-memory error, raised or swallowed into the result, releases the
  accelerator cache and retries with a smaller stage batch (4 -> 2 -> 1);
- a non-OOM partial failure is retried once, and the attempt with the fewest
  missing pages is kept;
- pages still missing after that are reported as a warning naming the page
  numbers, which the indexer stores on the file (console + API) instead of
  pretending the file was fully indexed.

The stage batch size is overridden per conversion through a thread-local read
by a small hook on ``StandardPdfPipeline._create_run_ctx``. The pipeline
options object is never mutated: docling caches initialized pipelines by an
options hash, and changing it would load a second copy of every model.

The same hook wraps each stage's model for that run only, recording
out-of-memory errors: docling replaces the cause with "pipeline terminated
early" in the result, so the result alone cannot tell an OOM from anything else.
The recorder is captured by closure, because the stages run on worker threads.
"""

from __future__ import annotations

import contextlib
import gc
import sys
import threading
import time
from typing import Any, Callable, Iterator, List, Optional, Tuple

from src.log import get_api_logger

logger = get_api_logger()

_OOM_MARKERS = (
    "out of memory",               # torch: "CUDA out of memory. Tried to allocate ..."
    "outofmemoryerror",            # class name in wrapped messages
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "hipErrorOutOfMemory".lower(),
)

# docling's default ocr/layout/table batch size; each OOM halves it.
DEFAULT_STAGE_BATCH = 4
_BATCHED_STAGES = ("ocr", "layout", "table")

_tls = threading.local()
_hook_lock = threading.Lock()
_hook_installed = False


def is_oom(error: Any) -> bool:
    """True when an exception or error message is an accelerator out-of-memory error."""
    if error is None:
        return False
    text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
    text = text.lower()
    return any(m in text for m in _OOM_MARKERS)


def release_accelerator_cache() -> None:
    """Return cached GPU memory to the driver. A no-op when torch was never imported
    (CPU-only or cloud deployments must not pay for importing it here)."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # pragma: no cover - driver specific
        logger.debug(f"[DOCLING] cache release failed (ignored): {e}")


def _status_value(result: Any) -> str:
    status = getattr(result, "status", None)
    return str(getattr(status, "value", status) or "").lower()


def result_errors(result: Any) -> List[Any]:
    return list(getattr(result, "errors", None) or [])


def missing_pages(result: Any) -> List[int]:
    """1-based page numbers docling reported as failed (sorted, unique)."""
    pages = {getattr(e, "page_no", None) for e in result_errors(result)}
    return sorted(p for p in pages if isinstance(p, int) and p > 0)


def result_has_oom(result: Any) -> bool:
    return any(is_oom(getattr(e, "error_message", e)) for e in result_errors(result))


def _error_summary(result: Any, limit: int = 3) -> str:
    parts = []
    for e in result_errors(result)[:limit]:
        where = getattr(e, "module_name", None) or "docling"
        parts.append(f"{where}: {getattr(e, 'error_message', e)}")
    return "; ".join(parts) or "no error detail"


def partial_warning(pages: List[int], unknown: bool = False) -> Optional[str]:
    """User-facing text for pages that could not be parsed (None when nothing is missing)."""
    if pages:
        listed = ", ".join(str(p) for p in pages)
        noun = "page" if len(pages) == 1 else "pages"
        verb = "is" if len(pages) == 1 else "are"
        return (
            f"Partial parse: {noun} {listed} could not be parsed and {verb} missing "
            f"from the index. Re-index the file to retry."
        )
    if unknown:
        return (
            "Partial parse: part of the document could not be parsed and may be "
            "missing from the index. Re-index the file to retry."
        )
    return None


def oom_message(file_name: str) -> str:
    return (
        f"Out of GPU memory while parsing '{file_name}', even with a stage batch of 1. "
        f"Other services may be using the card: free memory, or point "
        f"rag.docling.device at another GPU (or cpu), then re-index the file."
    )


def _batch_sizes(start: int) -> List[int]:
    sizes, b = [], max(1, int(start))
    while True:
        sizes.append(b)
        if b == 1:
            return sizes
        b = max(1, b // 2)


@contextlib.contextmanager
def stage_batch(size: Optional[int]) -> Iterator[List[BaseException]]:
    """Run the enclosed conversion with ocr/layout/table batch ``size`` (this thread only).

    Yields a list that collects out-of-memory errors raised inside any stage.
    """
    install_batch_hook()
    prev = (getattr(_tls, "batch", None), getattr(_tls, "ooms", None))
    ooms: List[BaseException] = []
    _tls.batch, _tls.ooms = size, ooms
    try:
        yield ooms
    finally:
        _tls.batch, _tls.ooms = prev


def current_stage_batch() -> Optional[int]:
    return getattr(_tls, "batch", None)


def _recording(model: Any, ooms: List[BaseException]) -> Callable:
    def _run(conv_res, pages):
        try:
            return list(model(conv_res, pages))
        except Exception as e:
            if is_oom(e):
                ooms.append(e)
            raise

    return _run


def apply_stage_batch(run_ctx: Any) -> Any:
    """Apply this thread's override to a docling RunContext: batch size on the batched
    stages, an OOM recorder around every stage model."""
    size = current_stage_batch()
    ooms = getattr(_tls, "ooms", None)
    for stage in getattr(run_ctx, "stages", None) or []:
        if size and getattr(stage, "name", None) in _BATCHED_STAGES:
            stage.batch_size = size
        model = getattr(stage, "model", None)
        if ooms is not None and model is not None:
            stage.model = _recording(model, ooms)
    return run_ctx


def install_batch_hook() -> bool:
    """Wrap StandardPdfPipeline._create_run_ctx once. Returns False when docling has no
    such hook (an older/newer layout): conversion still works, only without the
    batch reduction on retry."""
    global _hook_installed
    with _hook_lock:
        if _hook_installed:
            return True
        try:
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        except Exception:  # pragma: no cover - docling missing
            return False
        orig = getattr(StandardPdfPipeline, "_create_run_ctx", None)
        if orig is None:
            return False

        def _create_run_ctx(self, *args, **kwargs):
            return apply_stage_batch(orig(self, *args, **kwargs))

        StandardPdfPipeline._create_run_ctx = _create_run_ctx
        _hook_installed = True
        return True


def convert_checked(
    convert: Callable[[], Any],
    file_name: str,
    *,
    start_batch: int = DEFAULT_STAGE_BATCH,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[Any, Optional[str]]:
    """Run ``convert`` (which must build a fresh source each call) until the result is
    complete or retries are exhausted. Returns ``(result, warning)``; ``warning`` names
    missing pages when the best result is still partial.

    Raises ValueError when docling reports a complete failure, RuntimeError with a
    readable message when the accelerator stays out of memory.
    """
    sizes = _batch_sizes(start_batch)
    level = 0
    partial_retried = False
    best: Any = None
    best_missing: Optional[List[int]] = None
    while True:
        size = sizes[level]
        ooms: List[BaseException] = []
        try:
            with stage_batch(size) as ooms:
                result = convert()
        except Exception as e:
            if not (is_oom(e) or ooms):
                raise
            release_accelerator_cache()
            if level + 1 >= len(sizes):
                raise RuntimeError(oom_message(file_name)) from e
            logger.warning(
                f"[DOCLING] out of GPU memory parsing {file_name} at batch {size}; "
                f"retrying at batch {sizes[level + 1]}"
            )
            level += 1
            sleep(2)
            continue

        status = _status_value(result)
        oom_inside = bool(ooms) or result_has_oom(result)
        if status in ("partial_success", "failure"):
            pages = missing_pages(result)
            if status == "partial_success" and (best is None or len(pages) < len(best_missing or [])):
                best, best_missing = result, pages
            if oom_inside and level + 1 < len(sizes):
                release_accelerator_cache()
                logger.warning(
                    f"[DOCLING] {file_name}: a stage ran out of GPU memory at batch {size} "
                    f"(pages {pages or '?'} dropped); retrying at batch {sizes[level + 1]}"
                )
                level += 1
                sleep(2)
                continue
            if not oom_inside and not partial_retried:
                partial_retried = True
                logger.warning(
                    f"[DOCLING] {file_name}: {status} ({_error_summary(result)}); retrying once"
                )
                continue
            if best is None:
                if oom_inside:
                    raise RuntimeError(oom_message(file_name))
                raise ValueError(
                    f"Docling could not parse '{file_name}': {_error_summary(result)}"
                )
            warning = partial_warning(best_missing or [], unknown=not best_missing)
            logger.warning(f"[DOCLING] {file_name}: {warning}")
            return best, warning
        return result, None
