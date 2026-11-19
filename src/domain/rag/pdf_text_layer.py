"""PDFs read from their own text layer: no model, no network, no torch.

The parsing path when no vision model is configured (``rag.parsing.pdf: auto`` without
``rag.parsing.vision``, or ``rag.parsing.pdf: text``). PDFium (pypdfium2) extracts each
page's text in its reading order; this module rebuilds paragraphs from the lines and hands
one document, with the page number on every paragraph, to the same chunker as the vision
path (docling_loader.chunk_docling_document), so chunk records carry ``page_no`` for
citations either way.

What it cannot do, by design (Tony, 2026-11-18: a vision model is the answer for these):

- A table becomes plain text, row by row.
- A page without a usable text layer (a scanned page, a photo, a broken font encoding) is
  replaced by a visible marker naming the page and saying to configure rag.parsing.vision;
  the file's warning lists those pages. A PDF where no page has text fails indexing with
  that message: an empty file is never indexed.

Paragraphs: PDFium ends every visual line with a line break. A line that runs to the
right edge of the page's text block (and does not end a sentence) continues on the next
line, so the two are joined: without a space between CJK characters, with one otherwise,
and with a hyphen kept for a word PDFium marks as hyphenated at the break (U+FFFE).
Every other line break ends a paragraph. A line with wide gaps between its characters is a
table row (its cells), so it is never joined to a neighbour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from src.config.model import MIGRATION_GUIDE
from src.log import get_api_logger

logger = get_api_logger()

# A page with fewer non-space characters than this and an image on it is a scan (a stamped
# page number or a short caption is not a text layer worth indexing on its own).
MIN_PAGE_CHARS = 20
# Share of U+FFFD / private-use characters above which a text layer is treated as unreadable
# (a font without a usable ToUnicode map extracts as replacement or private-use glyphs).
MAX_GARBLED_RATIO = 0.3
# A line that ends within this share of the text block's width from its right edge "runs to
# the margin", i.e. wraps onto the next line.
WRAP_TOLERANCE = 0.08
# A gap between two characters of one line wider than this many character heights separates
# table cells (word spacing is well under one height).
CELL_GAP = 1.5
# An image covering at least this share of the page next to little text: the page is mostly
# a picture, so the marker is added after the text that was found.
LARGE_IMAGE_AREA = 0.5
LARGE_IMAGE_MAX_CHARS = 200

_SENTENCE_END = tuple("。！？；：.!?;:")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯＀-￯]")
_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")
_PDFIUM_HYPHEN = "￾"  # PDFium's marker for a word hyphenated at a line break

VISION_HINT = ("set rag.parsing.vision (an OpenAI-compatible vision model such as Qwen3-VL on "
               "Forgeyard) to read it")


class NoTextLayerError(ValueError):
    """No page of the PDF has a usable text layer, and no vision model reads it."""


def no_text_layer_message(file_name: str) -> str:
    return (f"'{file_name}' has no usable text layer (a scanned or image-only PDF), so there is "
            "nothing to index without a vision model. Configure rag.parsing.vision (an "
            "OpenAI-compatible vision model, e.g. Qwen3-VL on Forgeyard; rag.parsing.pdf auto or "
            f"vision) and re-index the file. See {MIGRATION_GUIDE}.")


def page_marker(page_no: int, reason: str) -> str:
    """The visible stand-in for a page whose text could not be read."""
    return f"[Page {page_no}: {reason}; {VISION_HINT}.]"


def marker_warning(pages: List[int]) -> Optional[str]:
    """The file's warning (console + API) for pages indexed only as a marker."""
    if not pages:
        return None
    listed = ", ".join(str(p) for p in pages)
    noun, verb = ("Page", "has") if len(pages) == 1 else ("Pages", "have")
    return (f"{noun} {listed} {verb} no usable text layer and {'is' if len(pages) == 1 else 'are'} "
            "indexed as a marker only. Configure rag.parsing.vision (a vision model) and re-index "
            "the file to read them.")


@dataclass
class _Line:
    text: str
    left: Optional[float] = None
    right: Optional[float] = None
    # Wide horizontal gaps between characters: a table row (cells), not running text.
    gappy: bool = False


@dataclass
class PageText:
    page_no: int
    paragraphs: List[str] = field(default_factory=list)
    marker: Optional[str] = None  # set when (part of) the page could not be read

    @property
    def has_text(self) -> bool:
        return bool(self.paragraphs)


# ---------------------------------------------------------------- pure text helpers

def _nonspace(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def garbled_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = sum(1 for c in chars if c == "�" or "" <= c <= "")
    return bad / len(chars)


def _join(prev: str, nxt: str) -> str:
    """Join a wrapped line onto the paragraph so far."""
    if prev.endswith(_PDFIUM_HYPHEN):
        return prev[:-1] + "-" + nxt
    if prev.endswith("-") and nxt[:1].islower():
        return prev + nxt
    if _CJK_RE.match(prev[-1:] or "") and _CJK_RE.match(nxt[:1] or ""):
        return prev + nxt
    return prev + " " + nxt


def lines_to_paragraphs(lines: List[_Line]) -> List[str]:
    """Rebuild paragraphs: a line that reaches the right edge of the text block (and does not
    end a sentence) continues on the next line. Without line geometry every line is its own
    paragraph."""
    lines = [ln for ln in lines if ln.text.strip()]
    rights = [ln.right for ln in lines if ln.right is not None]
    lefts = [ln.left for ln in lines if ln.left is not None]
    edge = max(rights) if rights else None
    width = (edge - min(lefts)) if (rights and lefts) else None
    paragraphs: List[str] = []
    cur: Optional[str] = None
    wraps = False
    for ln in lines:
        text = ln.text.strip()
        if cur is not None and wraps and not ln.gappy:
            cur = _join(cur, text)
        else:
            if cur is not None:
                paragraphs.append(cur)
            cur = text
        wraps = not ln.gappy and bool(
            edge is not None and width and ln.right is not None
            and ln.right >= edge - WRAP_TOLERANCE * width
            and not text.rstrip(_PDFIUM_HYPHEN).endswith(_SENTENCE_END)
        ) or text.endswith(_PDFIUM_HYPHEN)
    if cur is not None:
        paragraphs.append(cur)
    return [p.replace(_PDFIUM_HYPHEN, "-") for p in paragraphs]


# ---------------------------------------------------------------- PDFium

def _page_lines(textpage) -> List[_Line]:
    """The page's lines in PDFium's reading order, with each line's left / right edge when
    PDFium's character boxes line up with its text (they do: one box per character, including
    the generated line breaks)."""
    text = textpage.get_text_range() or ""
    geometry = len(text) == textpage.count_chars()
    out: List[_Line] = []
    pos = 0
    for m in list(_LINE_BREAK_RE.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        raw = text[pos:end]
        line = _Line(raw)
        if geometry and raw.strip():
            first = pos + (len(raw) - len(raw.lstrip()))
            last = pos + len(raw.rstrip()) - 1
            try:
                line.left = textpage.get_charbox(first)[0]
                line.right = textpage.get_charbox(last)[2]
                line.gappy = _has_cell_gap(textpage, raw, pos)
            except Exception:  # noqa: BLE001 - geometry is a nicety; the text still counts
                pass
        out.append(line)
        pos = m.end() if m else len(text)
    return out


def _has_cell_gap(textpage, raw: str, pos: int) -> bool:
    """Whether two neighbouring visible characters of the line are further apart than
    CELL_GAP times the line's character height (table cells rather than words). The tallest
    glyph box stands for the font size: a tight box such as that of 一 is far shorter."""
    boxes = [textpage.get_charbox(pos + i) for i, ch in enumerate(raw) if not ch.isspace()]
    height = max((top - bottom for _, bottom, _, top in boxes), default=0.0)
    if height <= 0:
        return False
    return any(nxt[0] - cur[2] > CELL_GAP * height for cur, nxt in zip(boxes, boxes[1:]))


def _image_coverage(page) -> Tuple[bool, float]:
    """(page has an image object, largest image's share of the page area)."""
    import pypdfium2.raw as pdfium_c

    w, h = page.get_size()
    area = max(w * h, 1.0)
    found, largest = False, 0.0
    try:
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=4):
            found = True
            try:
                left, bottom, right, top = obj.get_pos()
                largest = max(largest, abs((right - left) * (top - bottom)) / area)
            except Exception:  # noqa: BLE001
                largest = max(largest, 0.0)
    except Exception:  # noqa: BLE001 - treat an unreadable object tree as "no image"
        return found, largest
    return found, min(largest, 1.0)


def classify_page(page_no: int, text: str, paragraphs: List[str], has_image: bool,
                  image_share: float) -> PageText:
    """Decide what a page contributes: its paragraphs, a marker, or both (pure, testable)."""
    chars = _nonspace(text)
    result = PageText(page_no=page_no)
    if chars and garbled_ratio(text) > MAX_GARBLED_RATIO:
        result.marker = page_marker(page_no, "the text layer is unreadable (broken font encoding)")
        return result
    if chars == 0 or (chars < MIN_PAGE_CHARS and has_image):
        if has_image or chars:
            result.marker = page_marker(page_no, "no text layer (a scanned or image-only page)")
        return result  # a blank page contributes nothing
    result.paragraphs = paragraphs
    if image_share >= LARGE_IMAGE_AREA and chars < LARGE_IMAGE_MAX_CHARS:
        result.marker = page_marker(page_no, "most of this page is an image, which was not read")
    return result


def read_pages(file_path: str, file_name: str,
               progress_cb: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
               ) -> List[PageText]:
    """Every page's text (in order), one PDFium lock hold per page."""
    from src.domain.rag import pdf_pages

    try:
        total = pdf_pages.page_count(file_path)
    except pdf_pages.PdfPasswordError:
        raise ValueError(f"PDF is password-protected: '{file_name}'. Please decrypt it before "
                         "uploading.") from None
    except Exception as e:
        raise ValueError(f"File appears corrupted or unreadable: '{file_name}' "
                         f"({type(e).__name__}).") from e
    pages: List[PageText] = []
    for no in range(1, total + 1):
        with pdf_pages._PDFIUM_LOCK:
            doc = pdf_pages._open(file_path)
            try:
                page = doc[no - 1]
                try:
                    textpage = page.get_textpage()
                    try:
                        lines = _page_lines(textpage)
                    finally:
                        textpage.close()
                    has_image, share = _image_coverage(page)
                finally:
                    page.close()
            finally:
                doc.close()
        text = "\n".join(ln.text for ln in lines)
        pages.append(classify_page(no, text, lines_to_paragraphs(lines), has_image, share))
        if progress_cb is not None:
            try:
                progress_cb("loading", no, total)
            except Exception:  # noqa: BLE001 - progress must never stop parsing
                pass
    return pages


def build_document(pages: List[PageText], file_name: str) -> Any:
    """A DoclingDocument with one text item per paragraph (and marker), each carrying its page."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name=Path(file_name).stem or "document")
    for p in pages:
        if not (p.paragraphs or p.marker):
            continue
        doc.add_page(page_no=p.page_no, size=Size(width=1, height=1))
        for text in p.paragraphs + ([p.marker] if p.marker else []):
            doc.add_text(label=DocItemLabel.TEXT, text=text, prov=ProvenanceItem(
                page_no=p.page_no, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, len(text))))
    return doc


def parse(file_path: str, file_name: str, *, max_tokens: int = 512, tokenizer: Optional[str] = None,
          progress_cb: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
          table_max_tokens: Optional[int] = None, warnings: Optional[dict] = None,
          ) -> Tuple[str, List[dict]]:
    """(full text, chunk records) from the text layer — the contract of cloud_parser.parse.

    Raises NoTextLayerError when no page has text; pages indexed only as a marker come back
    as ``warnings["warning"]`` / ``warnings["missing_pages"]``.
    """
    pages = read_pages(file_path, file_name, progress_cb=progress_cb)
    marked = [p.page_no for p in pages if p.marker]
    if not any(p.has_text for p in pages):
        logger.warning(f"[TEXT_LAYER] {file_name}: no page has a usable text layer "
                       f"({len(pages)} page(s)); configure rag.parsing.vision to read it")
        raise NoTextLayerError(no_text_layer_message(file_name))
    if marked:
        logger.warning(f"[TEXT_LAYER] {file_name}: page(s) {marked} have no usable text layer and "
                       "are indexed as a marker; configure rag.parsing.vision to read them")
        if warnings is not None:
            warnings["warning"] = marker_warning(marked)
            warnings["missing_pages"] = marked
    from src.domain.rag.docling_loader import chunk_docling_document
    doc = build_document(pages, file_name)
    full_text, records = chunk_docling_document(doc, file_name, max_tokens=max_tokens,
                                                tokenizer=tokenizer, table_max_tokens=table_max_tokens)
    logger.info(f"[TEXT_LAYER] {file_name}: {len(pages)} page(s) → {len(records)} chunk(s)"
                + (f"; marker only: {marked}" if marked else ""))
    return full_text, records
