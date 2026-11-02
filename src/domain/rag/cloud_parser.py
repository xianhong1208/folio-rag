"""Cloud PDF / image parsing: every page is transcribed to Markdown by a vision model and
the pages are re-assembled into ONE document before chunking.

Why one document: parsing each page as its own document resets the heading breadcrumb on
every page and cuts tables that run across a page break. Here the page outputs are
stitched into a single Markdown text, parsed once by docling's Markdown backend
(torch-free), and handed to the same chunker as the local path
(docling_loader.chunk_docling_document) — so headings continue across pages and
section breadcrumbs stay intact.

Pipeline
1. Admission: page count vs ``max_pages_per_file`` (refused up front) and the day's
   ``max_pages_per_day`` budget (all pages reserved before the first call).
2. Transcription with bounded concurrency *and* context. Pages are split into K
   contiguous blocks (K = ``max_concurrency``); blocks run in parallel, pages inside a
   block run in order, and each page is sent with the previous page's trailing Markdown
   (``context_chars``) and the current heading path. The model marks a page that
   continues a paragraph / list / table with ``<<<CONTINUE>>>``. Only the boundaries between
   blocks lack model context; there a deterministic rule joins a header-less (or
   header-repeating) table onto the previous page's table. A process-wide semaphore of
   the same size bounds requests across concurrently indexed files.
3. Per-page retries: 429 / 5xx / timeouts / empty output back off exponentially with
   jitter (Retry-After honoured). 401 / 403 / 404 fail the file at once (every page
   would fail the same way). A page that still fails is marked in the text
   (``on_page_failure: mark``) or fails the file (``fail``); below
   ``min_success_ratio`` the file fails regardless.
4. Stitching: continued blocks are glued onto the previous page (a repeated table header
   is dropped), and an invisible page marker is placed after them.
5. Provenance: after the Markdown parse, marker items set the page number on every
   following item (``prov.page_no``) and are then deleted, so chunk records carry the
   page a chunk starts on — the same field the local path fills for citations.

The prompt is language-neutral: transcribe in the original language and script.
"""

from __future__ import annotations

import base64
import io
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.log import get_api_logger
from src.domain.rag.usage_budget import BUDGET
from src.utils.http_retry import RETRYABLE_STATUS, RetryPolicy, compute_delay

logger = get_api_logger()

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

CONTINUE_TOKEN = "<<<CONTINUE>>>"
# U+2063 INVISIBLE SEPARATOR around the marker: never produced by a vision model.
_MARK = "⁣PAGE {}⁣"
_MARK_RE = re.compile(r"^⁣PAGE (\d+)⁣$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.S)
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯＀-￯]")
_FATAL_STATUS = {401, 403, 404}
MAX_IMAGE_SIDE = 2048

DEFAULT_PROMPT = """\
You transcribe ONE page of a document into GitHub-flavoured Markdown.

- Transcribe all visible text in reading order, in the page's original language and
  script. Never translate, summarise, correct or add commentary.
- Headings become Markdown headings (#, ##, ###...) that follow the document's visual
  hierarchy and stay consistent with the current section path you are given.
- Tables become Markdown tables. When a table continues from the previous page, do not
  repeat its header row.
- Lists stay lists, keeping their numbering.
- A figure, chart or photo becomes one short bracketed line, e.g. [Figure: revenue by year].
- Leave out running headers, running footers and page numbers.
- If the page starts by continuing a paragraph, list or table from the previous page,
  make the first line of your output exactly <<<CONTINUE>>>.
- Output only the Markdown, without code fences."""


class CloudParseError(RuntimeError):
    """The file cannot be parsed (refused by a cap, fatal provider error, too many failed pages)."""


@dataclass
class PageResult:
    page_no: int
    markdown: str = ""
    continues: bool = False
    failed: bool = False
    error: str = ""
    attempts: int = 0


@dataclass
class ParseReport:
    pages: int
    failed_pages: List[int] = field(default_factory=list)
    seconds: float = 0.0


# ---------------------------------------------------------------- page sources

def _png(img) -> bytes:
    if max(img.size) > MAX_IMAGE_SIDE:
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _PageSource:
    """Lazily renders page N (1-based) of a PDF or an image file to PNG bytes."""

    def __init__(self, path: str, kind: str, dpi: int):
        self.path, self.kind, self.dpi = path, kind, dpi
        if kind == "pdf":
            import fitz  # PyMuPDF: page rendering only, no model
            with fitz.open(path, filetype="pdf") as d:
                if d.needs_pass:
                    raise CloudParseError("PDF is password-protected — decrypt it before uploading")
                self.count = len(d)
        else:
            from PIL import Image
            with Image.open(path) as im:
                self.count = getattr(im, "n_frames", 1) or 1

    def render(self, page_no: int) -> bytes:
        if self.kind == "pdf":
            import fitz
            with fitz.open(self.path, filetype="pdf") as d:
                return d[page_no - 1].get_pixmap(dpi=self.dpi).tobytes("png")
        from PIL import Image
        with Image.open(self.path) as im:
            if self.count > 1:
                im.seek(page_no - 1)
            return _png(im.copy())


# ---------------------------------------------------------------- stitching helpers

def _blocks(md: str) -> List[str]:
    return [b for b in re.split(r"\n\s*\n", md.strip()) if b.strip()]


def _is_table(block: str) -> bool:
    lines = [ln for ln in block.strip().splitlines() if ln.strip()]
    return bool(lines) and all(ln.lstrip().startswith("|") for ln in lines)


def _columns(line: str) -> int:
    return len([c for c in line.strip().strip("|").split("|")])


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().strip("|")).strip().lower()


def _strip_repeated_header(prev_table: str, table: str) -> str:
    """Drop ``table``'s header + separator when it repeats ``prev_table``'s header."""
    prev = prev_table.strip().splitlines()
    cur = table.strip().splitlines()
    if len(cur) >= 2 and _TABLE_SEP_RE.match(cur[1]) and _norm(cur[0]) == _norm(prev[0]):
        return "\n".join(cur[2:])
    return table.strip()


def _table_continues(prev_block: str, first_block: str) -> bool:
    """Deterministic continuation for pages sent without context: a table that starts the
    page and has the previous table's column count, and either no header row of its own
    or the very same header."""
    if not (_is_table(prev_block) and _is_table(first_block)):
        return False
    prev = prev_block.strip().splitlines()
    cur = first_block.strip().splitlines()
    if _columns(prev[0]) != _columns(cur[0]):
        return False
    headerless = not (len(cur) >= 2 and _TABLE_SEP_RE.match(cur[1]))
    return headerless or _norm(cur[0]) == _norm(prev[0])


def _glue(prev_block: str, first_block: str) -> str:
    if _is_table(prev_block) and _is_table(first_block):
        return "\n"
    a, b = prev_block.rstrip()[-1:], first_block.lstrip()[:1]
    if prev_block.rstrip().endswith("-") and b.islower():  # hyphenated word split
        return ""
    if _CJK_RE.match(a or "") and _CJK_RE.match(b or ""):
        return ""
    if re.match(r"^(\d+[.)]|[-*+])\s", first_block.lstrip()):  # list item continues a list
        return "\n"
    return " "


def stitch_pages(pages: List[PageResult]) -> str:
    """Join page outputs into one Markdown text with a page marker per page.

    A continued first block is appended to the previous page's last block and the
    page's marker goes after it, so the continued paragraph/table keeps its starting
    page (docling semantics: a chunk's page is where it starts).
    """
    parts: List[str] = []
    for p in sorted(pages, key=lambda x: x.page_no):
        mark = _MARK.format(p.page_no)
        if p.failed:
            parts += [mark, f"[Page {p.page_no}: not transcribed — {p.error}]"]
            continue
        blocks = _blocks(p.markdown)
        if not blocks:
            parts.append(mark)
            continue
        prev_content = next((x for x in reversed(parts) if not _MARK_RE.match(x)), None)
        prev_last = _blocks(prev_content)[-1] if prev_content else ""
        continues = bool(prev_content) and (
            p.continues or _table_continues(prev_last, blocks[0]))
        if continues and not prev_last.startswith("[Page "):
            first = blocks[0]
            if _is_table(prev_last) and _is_table(first):
                first = _strip_repeated_header(prev_last, first)
            idx = max(i for i, x in enumerate(parts) if x is prev_content)
            if first:
                parts[idx] = prev_content.rstrip() + _glue(prev_last, first) + first
            blocks = blocks[1:]
        parts.append(mark)
        if blocks:
            parts.append("\n\n".join(blocks))
    return "\n\n".join(parts) + "\n"


def _heading_path(md: str, path: List[str]) -> List[str]:
    """Update a heading path (one entry per level, '#'-prefixed) with ``md``'s headings."""
    path = list(path)
    for line in md.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            path = [h for h in path if len(h.split(" ", 1)[0]) < level]
            path.append(f"{m.group(1)} {m.group(2)}")
    return path


def clean_page_output(text: Optional[str]) -> Tuple[str, bool]:
    """Strip code fences / the continuation token; return (markdown, continues)."""
    md = (text or "").strip()
    m = _FENCE_RE.match(md)
    if m:
        md = m.group(1).strip()
    continues = False
    if md.startswith(CONTINUE_TOKEN):
        continues = True
        md = md[len(CONTINUE_TOKEN):].lstrip()
    md = md.replace(CONTINUE_TOKEN, "")
    return md, continues


def assign_page_provenance(doc: Any) -> int:
    """Turn page-marker items into ``prov.page_no`` on the items after them, then delete
    the markers. Returns the number of pages seen."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import ProvenanceItem

    current: Optional[int] = None
    markers, pages = [], set()
    for item, _level in doc.iterate_items():
        text = (getattr(item, "text", None) or "").strip()
        m = _MARK_RE.match(text)
        if m:
            current = int(m.group(1))
            pages.add(current)
            markers.append(item)
            continue
        if current is not None and not getattr(item, "prov", None):
            item.prov.append(ProvenanceItem(
                page_no=current, bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, len(text))))
    for no in sorted(pages):
        doc.add_page(page_no=no, size=Size(width=1, height=1))
    if markers:
        doc.delete_items(node_items=markers)
    return len(pages)


# ---------------------------------------------------------------- the parser

_slots_lock = threading.Lock()
_slots: Dict[int, threading.BoundedSemaphore] = {}


def _global_slots(n: int) -> threading.BoundedSemaphore:
    """Process-wide limit on vision requests in flight (shared by all files)."""
    with _slots_lock:
        if n not in _slots:
            _slots.clear()  # a changed setting replaces the limiter for new requests
            _slots[n] = threading.BoundedSemaphore(n)
        return _slots[n]


_md_converter = None
_md_lock = threading.Lock()


def _markdown_to_docling(markdown: str, file_name: str):
    """Parse Markdown with docling's Markdown backend only (no PDF/OCR pipelines)."""
    global _md_converter
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.document_converter import DocumentConverter
    with _md_lock:
        if _md_converter is None:
            _md_converter = DocumentConverter(allowed_formats=[InputFormat.MD])
    stream = DocumentStream(name=f"{Path(file_name).stem}.md", stream=io.BytesIO(markdown.encode("utf-8")))
    return _md_converter.convert(stream).document


class CloudDocumentParser:
    def __init__(self, parsing_cfg, *, client_factory: Optional[Callable] = None,
                 sleep: Callable[[float], Any] = time.sleep, budget=BUDGET):
        vision = parsing_cfg.vision
        if vision is None or not vision.model:
            raise CloudParseError("rag.parsing.vision.model is not configured")
        self.cfg = parsing_cfg
        self.vision = vision
        self._sleep = sleep
        self._budget = budget
        self._policy = RetryPolicy(max_retries=vision.max_retries, backoff_base=2.0, backoff_max=60.0)
        self._client_factory = client_factory or self._default_client
        self.prompt = parsing_cfg.prompt or DEFAULT_PROMPT

    @classmethod
    def from_config(cls, rag_config) -> "CloudDocumentParser":
        from src.config.model import ParsingConfig
        return cls(getattr(rag_config, "parsing", None) or ParsingConfig())

    def _default_client(self):
        from src.domain.rag.llm_client import create_openai_client
        # Retries are done per page here (with the failure policy), not inside the SDK.
        return create_openai_client(self.vision.model_copy(update={"max_retries": 0}))

    # -- one page ------------------------------------------------------------
    def _user_text(self, page_no: int, total: int, path: List[str], prev_tail: Optional[str]) -> str:
        parts = [f"Page {page_no} of {total}."]
        if path:
            parts.append("Current section path: " + " > ".join(path))
        if prev_tail:
            parts.append("End of the previous page, for context only (do not transcribe it again):\n"
                         f"<<<\n{prev_tail}\n>>>")
        elif page_no > 1:
            parts.append("The previous page is not shown; judge continuation from this page alone.")
        return "\n\n".join(parts)

    def _call(self, client, image_png: bytes, user_text: str) -> Tuple[str, Optional[str]]:
        model = self.vision.azure_deployment or self.vision.model
        uri = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": uri}},
                ]},
            ],
            max_tokens=self.vision.max_tokens,
            temperature=0,
        )
        choice = resp.choices[0]
        return choice.message.content or "", getattr(choice, "finish_reason", None)

    def transcribe_page(self, client, source: _PageSource, page_no: int, total: int,
                        path: List[str], prev_tail: Optional[str]) -> PageResult:
        result = PageResult(page_no=page_no)
        try:
            image = source.render(page_no)
        except Exception as e:  # a page that cannot be rendered cannot be retried into shape
            result.failed, result.error = True, f"render failed ({type(e).__name__})"
            return result
        user_text = self._user_text(page_no, total, path, prev_tail)
        slots = _global_slots(self.cfg.max_concurrency)
        last_error = "no attempts"
        for attempt in range(self._policy.max_retries + 1):
            result.attempts = attempt + 1
            response = None
            try:
                with slots:
                    text, finish = self._call(client, image, user_text)
                md, cont = clean_page_output(text)
                if md or cont:
                    if finish == "length":
                        logger.warning(f"[CLOUD_PARSE] page {page_no}: output hit max_tokens "
                                       f"({self.vision.max_tokens}) — the page may be truncated")
                    result.markdown, result.continues = md, cont
                    return result
                last_error = "empty output"
            except Exception as e:
                status = getattr(e, "status_code", None)
                response = getattr(e, "response", None)
                if status in _FATAL_STATUS:
                    raise CloudParseError(
                        f"vision model rejected the request (HTTP {status}) — check "
                        "rag.parsing.vision base_url / model / api_key") from e
                retryable = status in RETRYABLE_STATUS or _is_transport_error(e)
                last_error = f"HTTP {status}" if status else type(e).__name__
                if not retryable:
                    break
            if attempt < self._policy.max_retries:
                self._sleep(compute_delay(self._policy, attempt, response))
        result.failed = True
        result.error = f"{last_error} after {result.attempts} attempt(s)"
        return result

    # -- whole file ----------------------------------------------------------
    def parse(self, file_path: str, file_name: str, *, max_tokens: int = 512,
              tokenizer: Optional[str] = None,
              progress_cb: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
              ) -> Tuple[str, List[dict]]:
        """Parse a PDF or image into (full Markdown, chunk records) — the same contract as
        docling_loader.docling_convert_once."""
        markdown, report = self.parse_to_markdown(file_path, file_name, progress_cb=progress_cb)
        doc = _markdown_to_docling(markdown, file_name)
        assign_page_provenance(doc)
        from src.domain.rag.docling_loader import chunk_docling_document
        full_text, records = chunk_docling_document(doc, file_name, max_tokens=max_tokens,
                                                    tokenizer=tokenizer)
        logger.info(f"[CLOUD_PARSE] {file_name}: {report.pages} page(s) → {len(records)} chunk(s) "
                    f"in {report.seconds:.1f}s"
                    + (f"; pages not transcribed: {report.failed_pages}" if report.failed_pages else ""))
        return full_text, records

    def parse_to_markdown(self, file_path: str, file_name: str, *,
                          progress_cb: Optional[Callable] = None) -> Tuple[str, ParseReport]:
        t0 = time.time()
        kind = "pdf" if Path(file_name).suffix.lower() in PDF_EXTENSIONS else "image"
        try:
            source = _PageSource(file_path, kind, self.cfg.dpi)
        except CloudParseError:
            raise
        except Exception as e:
            raise CloudParseError(f"cannot open {file_name} as {kind}: {type(e).__name__}") from e
        total = source.count
        if total == 0:
            raise CloudParseError(f"{file_name} has no pages")
        cap = self.cfg.max_pages_per_file
        if cap and total > cap:
            raise CloudParseError(
                f"{file_name} has {total} pages; rag.parsing.max_pages_per_file is {cap}")
        self._budget.consume_or_raise("vision_pages", total, self.cfg.max_pages_per_day,
                                      "cloud-parsed pages")

        client = self._client_factory()
        done = {"n": 0}
        done_lock = threading.Lock()

        def tick():
            with done_lock:
                done["n"] += 1
                n = done["n"]
            if progress_cb is not None:
                try:
                    progress_cb("loading", n, total)
                except Exception:
                    pass

        def run_block(pages: List[int]) -> List[PageResult]:
            out: List[PageResult] = []
            path: List[str] = []
            prev_tail: Optional[str] = None
            for no in pages:
                r = self.transcribe_page(client, source, no, total, path, prev_tail)
                out.append(r)
                tick()
                if r.failed:
                    if self.cfg.on_page_failure == "fail":
                        raise CloudParseError(f"page {no} of {file_name}: {r.error}")
                    prev_tail = None
                    continue
                path = _heading_path(r.markdown, path)
                ctx = self.cfg.context_chars
                prev_tail = r.markdown[-ctx:] if ctx else None
            return out

        k = max(1, min(self.cfg.max_concurrency, total))
        size = -(-total // k)
        blocks = [list(range(s + 1, min(s + size, total) + 1)) for s in range(0, total, size)]
        try:
            with ThreadPoolExecutor(max_workers=len(blocks)) as pool:
                results = [r for block in pool.map(run_block, blocks) for r in block]
        finally:
            close = getattr(client, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass

        failed = [r.page_no for r in results if r.failed]
        ok = total - len(failed)
        if ok / total < self.cfg.min_success_ratio or ok == 0:
            raise CloudParseError(
                f"{file_name}: only {ok}/{total} pages transcribed "
                f"(rag.parsing.min_success_ratio={self.cfg.min_success_ratio}); "
                f"first error: {next(r.error for r in results if r.failed)}")
        if failed:
            logger.warning(f"[CLOUD_PARSE] {file_name}: {len(failed)}/{total} page(s) not "
                           f"transcribed and marked in the text: {failed}")
        return stitch_pages(results), ParseReport(pages=total, failed_pages=failed,
                                                   seconds=time.time() - t0)


def _is_transport_error(exc: BaseException) -> bool:
    import httpx
    import openai  # core dependency (llm_client)
    return isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError,
                            httpx.TimeoutException, httpx.TransportError))


def is_cloud_parsed(rag_config, file_name: str) -> bool:
    """Whether ``file_name`` goes to the cloud parser under the current config."""
    from src.config.roles import parsing_mode
    ext = Path(file_name).suffix.lower()
    if ext in PDF_EXTENSIONS:
        return parsing_mode(rag_config, "pdf") == "cloud"
    if ext in IMAGE_EXTENSIONS:
        return parsing_mode(rag_config, "images") == "cloud"
    return False
