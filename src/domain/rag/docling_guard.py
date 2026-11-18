"""Make docling conversion failures visible instead of silently losing content.

docling's ``convert()`` can return a ``PARTIAL_SUCCESS`` result without raising, with
part of the document simply absent. This module wraps a conversion so that:

- the result status is always checked (``FAILURE`` raises);
- a partial result is retried once, and the attempt with the fewest missing pages is kept;
- what is still missing after that is reported as a warning (naming pages when docling
  knows them), which the indexer stores on the file (console + API) instead of
  pretending the file was fully indexed.

Folio 1.5 runs no docling model (PDFs and images go to the vision model or the text
layer), so the out-of-memory retries with smaller stage batches that guarded the GPU
pipeline are gone with it.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from src.log import get_api_logger

logger = get_api_logger()


def _status_value(result: Any) -> str:
    status = getattr(result, "status", None)
    return str(getattr(status, "value", status) or "").lower()


def result_errors(result: Any) -> List[Any]:
    return list(getattr(result, "errors", None) or [])


def missing_pages(result: Any) -> List[int]:
    """1-based page numbers docling reported as failed (sorted, unique)."""
    pages = {getattr(e, "page_no", None) for e in result_errors(result)}
    return sorted(p for p in pages if isinstance(p, int) and p > 0)


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


def convert_checked(convert: Callable[[], Any], file_name: str) -> Tuple[Any, Optional[str]]:
    """Run ``convert`` (which must build a fresh source each call) until the result is
    complete or the one retry of a partial result is spent. Returns ``(result, warning)``;
    ``warning`` describes what is missing when the best result is still partial.

    Raises ValueError when docling reports a complete failure.
    """
    best: Any = None
    best_missing: Optional[List[int]] = None
    for attempt in range(2):
        result = convert()
        status = _status_value(result)
        if status not in ("partial_success", "failure"):
            return result, None
        pages = missing_pages(result)
        if status == "partial_success" and (best is None or len(pages) < len(best_missing or [])):
            best, best_missing = result, pages
        if attempt == 0:
            logger.warning(f"[DOCLING] {file_name}: {status} ({_error_summary(result)}); retrying once")
            continue
        if best is None:
            raise ValueError(f"Docling could not parse '{file_name}': {_error_summary(result)}")
    warning = partial_warning(best_missing or [], unknown=not best_missing)
    logger.warning(f"[DOCLING] {file_name}: {warning}")
    return best, warning
