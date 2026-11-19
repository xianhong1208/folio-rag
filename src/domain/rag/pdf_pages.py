"""PDF page counts and page rendering through PDFium (pypdfium2, Apache-2.0 / BSD-3-Clause).

Used for the indexing progress bar's page total (docling_loader) and for rendering pages
to PNG for the vision model (cloud_parser). No model, no torch: works in every profile.

PDFium is not thread-safe, so every call holds the same process-wide lock docling's own
pypdfium2 backend takes (``docling.utils.locks.pypdfium2_lock``); a page render here can
then never run inside PDFium at the same time as a docling PDF parse in another thread.
The lock is not reentrant: never call docling while holding it.
"""

from __future__ import annotations

import io
import threading
from typing import Optional

try:
    from docling.utils.locks import pypdfium2_lock as _PDFIUM_LOCK
except ImportError:  # pragma: no cover - docling-slim is a core dependency
    _PDFIUM_LOCK = threading.Lock()

# PDF user space is 72 units per inch, so a render at ``dpi`` uses scale dpi / 72.
POINTS_PER_INCH = 72
# pypdfium2 sizes the bitmap as ceil(page_points * scale). With a binary float scale an exact
# product such as 792 pt * 150/72 = 1650 comes out as 1650.0000000000002 and gains a pixel
# row. Shrinking the scale by one part in 10^9 keeps exact sizes exact (as PyMuPDF does,
# which rounds with a small tolerance) and changes nothing else visibly.
_SCALE_EPSILON = 1e-9


class PdfPasswordError(Exception):
    """The PDF needs a password to open."""


def _open(path: str):
    import pypdfium2 as pdfium

    try:
        return pdfium.PdfDocument(path)
    except pdfium.PdfiumError as e:
        if e.err_code == pdfium.raw.FPDF_ERR_PASSWORD:
            raise PdfPasswordError(str(e)) from e
        raise


def page_count(path: str) -> int:
    """Number of pages. Raises PdfPasswordError for an encrypted PDF, and pypdfium2's own
    errors (PdfiumError, FileNotFoundError) for anything that is not a readable PDF."""
    with _PDFIUM_LOCK:
        doc = _open(path)
        try:
            return len(doc)
        finally:
            doc.close()


def page_count_or_none(path: str) -> Optional[int]:
    """Page count, or None when the file cannot be opened as a PDF (never raises)."""
    try:
        return page_count(path)
    except Exception:
        return None


def render_scale(dpi: int) -> float:
    return dpi / POINTS_PER_INCH * (1 - _SCALE_EPSILON)


def render_png(path: str, page_no: int, dpi: int) -> bytes:
    """Render page ``page_no`` (1-based) at ``dpi`` to PNG bytes (RGB, white background)."""
    if page_no < 1:
        raise IndexError(f"page {page_no} out of range")
    with _PDFIUM_LOCK:
        doc = _open(path)
        try:
            page = doc[page_no - 1]
            try:
                bitmap = page.render(scale=render_scale(dpi))
                try:
                    buf = io.BytesIO()
                    bitmap.to_pil().save(buf, format="PNG")
                    return buf.getvalue()
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            doc.close()
