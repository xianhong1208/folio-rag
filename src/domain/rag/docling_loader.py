
"""Docling-based loader for Office / OpenDocument / HTML / Markdown / e-mail / e-book files,
with structure-aware chunking.

Docling's model-free format backends only (docling-slim): PDFs and images never reach
docling. They are read by a vision model (cloud_parser) or from the PDF's text layer
(pdf_text_layer); Folio 1.5 removed docling's in-process layout, TableFormer and OCR
models together with their device settings (docs/migrate-1.5.md).

- Tables are kept as complete chunks (split at row boundaries only when oversized)
- Document hierarchy (headings, sections) is respected
- Every chunk fits the embedding model's token budget (see the helpers below); the same
  chunker serves the vision and text-layer paths (``chunk_docling_document``)
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, List, Optional

from src.log import get_api_logger
from src.utils.runtime_paths import resolve_external_dir
from src.domain.rag.docling_guard import convert_checked, missing_pages

logger = get_api_logger()

# Supported upload types. Documents map to a docling backend; PDFs and images are listed
# for the format invariant (tests/test_upload_formats.py) but go to the vision model or the
# text layer; audio goes to the rag.asr endpoint. Kept aligned to docling's official
# format list by that test (see its _EXCLUDED_FORMATS for exclusions).
DOCLING_EXTENSIONS = {
    # PDF / Office Open XML (including template/macro variants)
    ".pdf",
    ".docx", ".dotx", ".docm", ".dotm",
    ".pptx", ".ppsx", ".pptm", ".potm", ".ppsm",  # potx: docling can't open it in practice
    ".xlsx", ".xlsm", ".xltx", ".xltm",
    # Legacy Office (docling 2.119+; internally converted to modern formats via LibreOffice `soffice`
    # before parsing — the deployment image/machine must have libreoffice-writer installed, or indexing
    # fails explicitly. Quality verified: .doc and .docx of the same content give 51 chunks, 96% line overlap)
    ".doc", ".dot", ".xls", ".xlt", ".ppt", ".pot", ".pps",
    ".rtf",  # docling 2.127+: Word backend, converted through LibreOffice like .doc
    # OpenDocument (including template variants)
    ".odt", ".ods", ".odp",  # template variants ott/ots/otp: docling's odfdo backend can't open them in practice
    # Markup / scientific documents
    ".html", ".htm", ".xhtml", ".md", ".markdown", ".qmd", ".rmd",
    # MIME web archives (docling 2.127+, HTML backend; local/remote references are not fetched)
    ".mht", ".mhtml",
    ".adoc", ".asciidoc", ".asc", ".tex", ".latex",
    # Email (.msg = python-oxmsg, pure Python; .eml = MIME) / ebooks / subtitles
    ".msg", ".eml", ".epub", ".vtt",
    # Images (read by the vision model only; refused at upload without one)
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif",
    # Audio (docling's 6 official audio types; transcribed by the rag.asr endpoint, not docling)
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac",
}

# Never handed to docling: PDFs and images (vision model / text layer), audio (rag.asr).
_PAGE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}  # docling's 6 official types

# Singleton DocumentConverter (format backends only; cheap to build, but shared).
_converter = None
_converter_lock = threading.Lock()
# One conversion at a time: legacy Office formats go through LibreOffice (soffice), which
# does not tolerate concurrent runs on one profile.
DOCLING_INFER_LOCK = threading.Lock()


def resolve_assets_dir() -> Optional[Path]:
    """Resolve the assets/ resource directory (tokenizers, nltk_data shipped with a package;
    hierarchical_indexer imports this). The lookup lives in src.utils.runtime_paths."""
    return resolve_external_dir(
        "assets",
        dev_root=Path(__file__).resolve().parents[3],
    )


def get_hf_tokenizer(tokenizer_name: Optional[str]):
    """Token counter for chunk sizing (kept under its old name for callers).

    Delegates to ``chunk_tokenizer.get_chunk_tokenizer``: local assets first,
    then the bundled tokenizer, never a download. ``tokenizer_name`` is the
    embedding model name, or None for providers without a HuggingFace name
    (OpenAI / Azure), which used to crash every docling-parsed file.
    """
    from src.domain.rag.chunk_tokenizer import get_chunk_tokenizer

    return get_chunk_tokenizer(tokenizer_name)


def is_docling_supported(file_name: str) -> bool:
    """Whether docling parses this file: every listed document type except PDFs and images
    (vision model / text layer) and audio (the rag.asr endpoint)."""
    ext = Path(file_name).suffix.lower()
    return ext in DOCLING_EXTENSIONS and ext not in _PAGE_EXTENSIONS and ext not in _AUDIO_EXTENSIONS


def _document_formats() -> list:
    """docling input formats Folio converts: everything but PDF, images, audio and video,
    whose pipelines need models."""
    from docling.datamodel.base_models import InputFormat
    skip = {"PDF", "IMAGE", "AUDIO", "VIDEO"}
    return [f for f in InputFormat if f.name not in skip]


def get_converter():
    """The singleton DocumentConverter (thread-safe, built on first use). It only knows the
    model-free document formats, so it never loads a model or imports torch."""
    global _converter
    if _converter is not None:
        return _converter
    with _converter_lock:
        if _converter is None:
            from docling.document_converter import DocumentConverter
            _converter = DocumentConverter(allowed_formats=_document_formats())
            logger.info("[DOCLING] DocumentConverter ready (document formats; no PDF/image models)")
    return _converter


# Tokenizer-aware chunking helpers. We deliberately do NOT use HybridChunker: on a
# single huge chunk (e.g. an xlsx 5000-row table from HierarchicalChunker) its
# slide-window tokenization runs for tens of minutes and trips the watchdog and
# per-file timeout. Instead we re-implement its contract per-chunk — structural
# pre-split via HierarchicalChunker, then one tokenizer.encode() per chunk (bounded
# by chunk size), splitting oversized chunks by natural boundaries and merging small
# ones (mirrors merge_peers / repeat_table_header). Output is text-only, which is all
# HierarchicalIndexer needs.


# Markdown table separator (`| --- | :---: | ... |`), used to detect a table chunk
# whose first 2 lines are the header to repeat on splits.
_MD_TABLE_SEPARATOR_RE = re.compile(r'^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$')

# Sentence ends. CJK terminators (。！？；) end a sentence on their own: Chinese and
# Japanese put no space after them, so requiring whitespace (as the old pattern did)
# left a whole Chinese paragraph as one "sentence" that fell through to the character
# split and was cut mid-clause. Latin terminators need following whitespace, so
# decimals ("3.5"), version numbers and URLs stay whole. Closing quotes/brackets stay
# with the sentence they close; trailing whitespace stays with the sentence too, so
# concatenating the pieces gives back the original text.
_SENTENCE_END_RE = re.compile(
    r'[。！？；]+[」』）】》〉”’"\')]*\s*'
    r'|[.!?]+["”’\')\]]*\s+'
)
# Clause ends, for a single sentence that is still over budget: CJK commas / colons
# (no space needed) and Latin ones followed by whitespace.
_CLAUSE_END_RE = re.compile(r'[，、：]\s*|[,:;]\s+')


def _split_keep_ends(text: str, pattern: "re.Pattern[str]") -> List[str]:
    """Split ``text`` after every match of ``pattern``, keeping the separator (and
    its trailing whitespace) on the left piece. ``"".join(result) == text``."""
    out: List[str] = []
    start = 0
    for m in pattern.finditer(text):
        if m.end() > start:
            out.append(text[start:m.end()])
            start = m.end()
    if start < len(text):
        out.append(text[start:])
    return out


def _pack_balanced(counts: List[int], max_tokens: int) -> List[List[int]]:
    """Group consecutive units (token counts given) into pieces of at most
    ``max_tokens``, as evenly sized as the unit boundaries allow.

    Plain greedy packing fills every piece to the brim and leaves a runt at the end
    (242 + 45 tokens); here each piece aims at ``remaining / pieces still needed`` and
    a unit goes to the next piece when that lands closer to the target. A unit larger
    than ``max_tokens`` gets a piece of its own (callers pre-split those).
    """
    groups: List[List[int]] = []
    cur: List[int] = []
    cur_tokens = 0
    remaining = sum(counts)

    def _target() -> float:
        pieces = max(1, -(-remaining // max(max_tokens, 1)))  # ceil
        return remaining / pieces

    target = _target()
    for i, c in enumerate(counts):
        if cur:
            over_budget = cur_tokens + c > max_tokens
            overshoot = cur_tokens + c - target
            undershoot = target - cur_tokens
            if over_budget or (overshoot > 0 and overshoot > undershoot):
                groups.append(cur)
                remaining -= cur_tokens
                target = _target()
                cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += c
    if cur:
        groups.append(cur)
    return groups


def _detect_table_header(lines: List[str]) -> Optional[str]:
    """If ``lines`` is a markdown table, return the header + separator
    block to prepend on continuation chunks. ``None`` otherwise."""
    if len(lines) < 2:
        return None
    first = lines[0].strip()
    second = lines[1].strip()
    if "|" in first and _MD_TABLE_SEPARATOR_RE.match(second):
        return f"{lines[0]}\n{lines[1]}"
    return None


def _count_tokens(tokenizer, text: str) -> int:
    """Single-call token count. ``add_special_tokens=False`` matches the
    HuggingFace BPE-level count; HybridChunker uses the same flag for
    its own ≤ max_tokens checks."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_by_lines(
    text: str,
    tokenizer,
    max_tokens: int,
    header: Optional[str],
) -> List[str]:
    """Split oversized text by ``\\n`` boundaries (table rows / paragraphs).

    Each emitted sub-chunk is ≤ max_tokens (verified via tokenizer).
    When ``header`` is set, continuation sub-chunks get it prepended for
    context (the first sub-chunk already contains it as part of the
    original lines). The header's token cost is reserved upfront for
    every chunk that needs it, so the final assembled chunk stays
    under max_tokens — otherwise the header bytes overflow continuation
    chunks by header_tokens (typically 20-30 tokens for a markdown
    table header).
    """
    lines = text.split("\n")
    if header is None:
        header = _detect_table_header(lines)

    # Reserve header budget for continuation chunks. The first chunk
    # already includes the header lines as part of the natural line
    # sequence, so its effective budget is the full max_tokens;
    # continuations get max_tokens - header_tokens to leave room for
    # the prepended header at flush time.
    header_tokens = _count_tokens(tokenizer, header) if header else 0

    out: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    def _budget() -> int:
        return max_tokens - header_tokens if (header and out) else max_tokens

    def _flush():
        nonlocal buf, buf_tokens
        if not buf:
            return
        body = "\n".join(buf)
        # Continuation chunks (output index > 0) get the header back if
        # the header isn't already the first lines of this buffer.
        if header and out and not body.startswith(header):
            body = f"{header}\n{body}"
        # Tokenizer-portability safeguard: the budget check uses "per-line token sum + separator
        # reserve", which is exact under bge-m3 (sentencepiece). If a tokenizer that counts \n as
        # more than 1 token is ever used, this is a final whole-chunk verification that logs on
        # overflow (no raise — the embedding context is far larger than max_tokens, so a slight
        # overshoot isn't fatal, but it must be visible).
        final_tokens = _count_tokens(tokenizer, body)
        if final_tokens > max_tokens:
            logger.warning(
                f"[CHUNK] assembled chunk exceeds budget after join: "
                f"{final_tokens} > {max_tokens} tokens (separator drift); "
                f"check tokenizer newline behaviour"
            )
        out.append(body)
        buf = []
        buf_tokens = 0

    for line in lines:
        if not line.strip():
            continue
        line_tokens = _count_tokens(tokenizer, line)

        # Single line still too big — fall to sentence split.
        # In the table case, continuations also need the header: reserve header budget,
        # then prepend the header to each split piece.
        if line_tokens > _budget():
            _flush()
            sent_budget = max_tokens - header_tokens if header else max_tokens
            for piece in _split_by_sentence(line, tokenizer, sent_budget):
                if header and out and not piece.startswith(header):
                    piece = f"{header}\n{piece}"
                out.append(piece)
            continue

        # Would overflow if we add this line — flush first.
        # `len(buf)` = the number of \n separators the join needs after adding this line, reserving
        # budget for a tokenizer that counts \n as 1 token (slightly conservative under sentencepiece, harmless).
        if buf_tokens + line_tokens + len(buf) > _budget():
            _flush()

        buf.append(line)
        buf_tokens += line_tokens

    _flush()
    return out


def _split_by_sentence(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Mid-tier split: break a single line at sentence ends, then pack the
    sentences into balanced pieces of at most ``max_tokens``.

    Reached when a single newline-bounded line is itself > max_tokens —
    typical for long paragraphs without internal structure. A sentence that is
    still too big is cut at clause ends (，、： / ", "), and only a clause that is
    still too big falls to the character split.
    """
    units: List[str] = []
    for sent in _split_keep_ends(text, _SENTENCE_END_RE):
        if not sent.strip():
            continue
        if _count_tokens(tokenizer, sent.strip()) <= max_tokens:
            units.append(sent)
            continue
        for clause in _split_keep_ends(sent, _CLAUSE_END_RE):
            if not clause.strip():
                continue
            if _count_tokens(tokenizer, clause.strip()) <= max_tokens:
                units.append(clause)
            else:
                units.extend(_split_by_chars(clause, tokenizer, max_tokens))

    counts = [_count_tokens(tokenizer, u.strip()) for u in units]
    out: List[str] = []

    def _emit(group: List[int]) -> None:
        piece = "".join(units[i] for i in group).strip()
        # The token count of a concatenation can exceed the sum of its parts by a
        # token or two (word-start markers); halve such a group instead of overflowing.
        if len(group) > 1 and _count_tokens(tokenizer, piece) > max_tokens:
            mid = len(group) // 2
            _emit(group[:mid])
            _emit(group[mid:])
            return
        if piece:
            out.append(piece)

    for group in _pack_balanced(counts, max_tokens):
        _emit(group)
    return out


def _snap_to_space(text: str, start: int, cut: int) -> int:
    """Move a cut back to just after the last whitespace in the final fifth of the
    piece, so space-delimited text is not cut mid-word (CJK has no spaces: unchanged)."""
    floor = start + (cut - start) * 4 // 5
    ws = max(text.rfind(" ", floor, cut), text.rfind("\n", floor, cut))
    return ws + 1 if ws >= floor and ws + 1 > start else cut


def _split_by_chars(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Last-resort character split with tokenizer verification.

    Reached for pathological inputs: a clause > max_tokens with no sentence or
    clause punctuation (unpunctuated ASR output, numeric dumps). Each piece is the
    longest prefix that fits (binary search on its length, one encode per step);
    the cuts are then evened out so the last piece is no runt. A single character
    over budget is emitted as-is (progress guarantee).
    """
    n = len(text)
    # No tokenizer packs more than ~8 characters into a token in practice; bounding
    # the search keeps encode calls short on huge unpunctuated dumps.
    span = max(64, max_tokens * 8)

    def fits(a: int, b: int) -> bool:
        return _count_tokens(tokenizer, text[a:b]) <= max_tokens

    cuts: List[int] = []
    i = 0
    while i < n:
        hi = min(n, i + span)
        if fits(i, hi):
            j = hi
        elif not fits(i, i + 1):
            logger.warning(
                f"[CHUNK] a single character exceeds {max_tokens} tokens; "
                f"emitting oversized piece to guarantee progress"
            )
            j = i + 1
        else:
            lo = i + 1  # fits
            while hi - lo > 1:  # invariant: fits(i, lo), not fits(i, hi)
                mid = (lo + hi) // 2
                if fits(i, mid):
                    lo = mid
                else:
                    hi = mid
            j = _snap_to_space(text, i, lo) if lo < n else lo
        cuts.append(j)
        i = j

    if len(cuts) > 1:
        # Even out: the same number of pieces, equal character lengths, if they all fit.
        k = len(cuts)
        even: List[int] = []
        start = 0
        for idx in range(1, k):
            cut = _snap_to_space(text, start, idx * n // k)
            if cut <= start:
                break
            even.append(cut)
            start = cut
        even.append(n)
        bounds = [0] + even
        if len(even) == k and all(fits(a, b) for a, b in zip(bounds, even)):
            cuts = even

    out: List[str] = []
    start = 0
    for cut in cuts:
        if text[start:cut]:
            out.append(text[start:cut])
        start = cut
    return out


def _doc_item_kind(item: Any) -> Optional[str]:
    """Map a Docling doc_item's label to a coarse content kind, or None.

    Docling labels items (DocItemLabel: TABLE / PICTURE / TEXT / …). We only care
    about the multi-modal ones. Defensive across Docling versions — the label may be
    an enum (``.value``/``.name``) or a bare string.
    """
    label = getattr(item, "label", None)
    if label is None:
        return None
    name = str(getattr(label, "value", None) or getattr(label, "name", None) or label).lower()
    if "table" in name:
        return "table"
    if "picture" in name or "image" in name or "figure" in name:
        return "picture"
    return None


def _container_headings(item: Any, doc: Any) -> List[str]:
    """Names of the named containers a doc_item sits in, outermost first.

    Only xlsx sheets for now: docling puts each sheet's tables in a GroupItem
    labelled ``sheet`` named after the sheet, but no heading — so a table on the
    "Borealis plan" sheet never says "Borealis". Defensive: any resolution failure
    yields no extra heading.
    """
    names: List[str] = []
    ref = getattr(item, "parent", None)
    hops = 0
    while ref is not None and doc is not None and hops < 32:
        hops += 1
        try:
            node = ref.resolve(doc)
        except Exception:
            break
        label = getattr(node, "label", None)
        label = str(getattr(label, "value", label) or "").lower()
        name = getattr(node, "name", None)
        if label == "sheet" and name:
            names.append(str(name))
        ref = getattr(node, "parent", None)
    return list(reversed(names))


def _chunk_provenance(chunk: Any, doc: Any = None) -> dict:
    """Extract headings / page_no / content_type from a docling chunk object (bare strings return empty meta).

    - headings: HierarchicalChunker's meta.headings (the section path, list[str]),
      prefixed with the xlsx sheet name when ``doc`` is given (see _container_headings)
    - page_no: the page_no of the first prov of the first doc_item (present for PDF; usually absent for docx etc.)
    - content_type: "table" when the chunk carries a table doc_item (Docling keeps the
      table as Markdown in the chunk text; this flag lets the UI badge / render it).
      Defaults to None for plain text.
    Extraction is defensive throughout — any missing field returns None and doesn't affect chunking.
    """
    headings = None
    page_no = None
    content_type = None
    meta = getattr(chunk, "meta", None)
    if meta is not None:
        h = getattr(meta, "headings", None)
        if h:
            headings = list(h)
        items = list(getattr(meta, "doc_items", None) or [])
        if items and doc is not None:
            containers = _container_headings(items[0], doc)
            if containers:
                headings = containers + [x for x in (headings or []) if x not in containers]
        for item in items:
            kind = _doc_item_kind(item)
            if kind == "table":  # a table anywhere in the chunk wins
                content_type = "table"
            elif kind == "picture" and content_type is None:
                content_type = "picture"
            if page_no is None:  # keep the first doc_item's page (original semantics)
                for prov in (getattr(item, "prov", None) or []):
                    p = getattr(prov, "page_no", None)
                    if p is not None:
                        page_no = int(p)
                        break
            if page_no is not None and content_type == "table":
                break
    return {"headings": headings, "page_no": page_no, "content_type": content_type}


def _is_triplet_table(text: str) -> bool:
    """Docling's table serialization: one line of ``row, col = value`` cells joined by ". "."""
    return "\n" not in text.strip() and " = " in text and ", " in text


def _split_triplet_table(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Split an oversized triplet table at row boundaries (a row = consecutive cells
    with the same row label), so every piece is itself a valid table. A single row
    larger than the budget falls back to cell boundaries."""
    cells = [c.strip() for c in text.strip().rstrip(".").split(". ") if c.strip()]
    rows: List[List[str]] = []
    last_label = None
    for cell in cells:
        label = cell.split(", ", 1)[0] if ", " in cell else None
        if rows and label == last_label:
            rows[-1].append(cell)
        else:
            rows.append([cell])
        last_label = label
    units: List[str] = []
    for row in rows:
        joined = ". ".join(row)
        if _count_tokens(tokenizer, joined) <= max_tokens:
            units.append(joined)
        else:
            units.extend(row)
    out: List[str] = []
    buf: List[str] = []
    for unit in units:
        candidate = ". ".join(buf + [unit])
        if buf and _count_tokens(tokenizer, candidate) > max_tokens:
            out.append(". ".join(buf))
            buf = [unit]
        else:
            buf.append(unit)
    if buf:
        out.append(". ".join(buf))
    # A cell longer than the budget on its own still has to be cut.
    final: List[str] = []
    for piece in out:
        if _count_tokens(tokenizer, piece) > max_tokens:
            final.extend(_split_by_chars(piece, tokenizer, max_tokens))
        else:
            final.append(piece)
    return final


def _refine_chunks_for_token_budget(
    raw_chunks: List[Any],
    tokenizer,
    max_tokens: int,
    doc: Any = None,
    table_max_tokens: Optional[int] = None,
) -> List[dict]:
    """Refine HierarchicalChunker output into token-bounded chunk records.

    Same contract as HybridChunker (token-bounded, merge_peers, repeat_table_header)
    but with per-chunk tokenize calls, and section-aligned: small chunks merge only
    with neighbours of the same section (chunk_identity.section_key), and a merged
    chunk is labelled with the headings all its pieces share. Returns records
    ``{"text", "headings", "page_no", "content_type"}``; split pieces inherit the
    source chunk's meta, bare-string input has meta None.

    ``table_max_tokens`` (default ``max_tokens``) lets a table stay whole up to a
    larger budget than prose; an oversized docling table is split at row
    boundaries so each piece remains a readable table; its pieces carry
    ``table_part`` (0, 1, ...) so they can share one contextual prefix.
    """
    from src.domain.rag.chunk_identity import (
        common_headings,
        merge_content_type,
        section_key,
    )

    table_budget = max(max_tokens, table_max_tokens or max_tokens)
    # merge_peers heuristic — flush the buffer when the token budget reaches
    # half of max_tokens. Matches HybridChunker default behavior:
    # produce chunks at roughly 50-100% of max_tokens for retrieval.
    merge_flush_threshold = max(max_tokens // 2, 1)

    output: List[dict] = []
    pending: List[tuple] = []  # (text, meta)
    pending_tokens = 0

    def _flush_pending():
        nonlocal pending, pending_tokens
        if not pending:
            return
        metas = [m for _, m in pending]
        page = next((m.get("page_no") for m in metas if m.get("page_no") is not None), None)
        output.append({
            "text": "\n".join(t for t, _ in pending),
            "headings": common_headings(m.get("headings") for m in metas),
            "page_no": page,
            "content_type": merge_content_type(m.get("content_type") for m in metas),
        })
        pending = []
        pending_tokens = 0

    for chunk in raw_chunks:
        # Accept a HierarchicalChunker chunk object, a record dict ({"text", "headings",
        # "page_no", "content_type"}, e.g. a CSV table) or a bare string (fallback path).
        if isinstance(chunk, dict):
            text = (chunk.get("text") or "").strip()
            meta = {k: chunk.get(k) for k in ("headings", "page_no", "content_type")}
        else:
            text = (chunk.text if hasattr(chunk, "text") else chunk).strip()
            meta = _chunk_provenance(chunk, doc) if text else None
        if not text:
            continue

        token_count = _count_tokens(tokenizer, text)
        is_table = meta.get("content_type") == "table"
        budget = table_budget if is_table else max_tokens

        # Never merge across a section boundary.
        if pending and section_key(pending[0][1].get("headings")) != section_key(meta.get("headings")):
            _flush_pending()

        if token_count > budget:
            # Oversized — flush any pending small chunks first
            _flush_pending()
            if is_table and _is_triplet_table(text):
                parts = _split_triplet_table(text, tokenizer, budget)
            else:
                parts = _split_by_lines(text, tokenizer, budget, header=None)
            for k, part in enumerate(parts):
                rec = {"text": part, **meta}  # split pieces inherit the source chunk's meta
                if is_table and len(parts) > 1:
                    # Pieces of one table: the indexer writes one contextual prefix for
                    # the table and reuses it on the continuation pieces.
                    rec["table_part"] = k
                output.append(rec)
        elif token_count < merge_flush_threshold:
            # Small — accumulate (both halves stay under max_tokens), flush past the threshold
            pending.append((text, meta))
            pending_tokens += token_count
            if pending_tokens >= merge_flush_threshold:
                _flush_pending()
        else:
            # Reasonable size — flush pending then add this chunk as-is
            _flush_pending()
            output.append({"text": text, **meta})

    _flush_pending()
    return output


_HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
_UTF8_BOM = b"\xef\xbb\xbf"
_CHARSET_DECL = re.compile(
    rb"<meta[^>]+charset\s*=|<\?xml[^>]+encoding\s*=", re.IGNORECASE)


def _declare_html_utf8(data: bytes) -> bytes:
    """Mark undeclared UTF-8 HTML as UTF-8 with a byte-order mark.

    docling hands the raw bytes to BeautifulSoup, which guesses the encoding when
    the page declares none; for short CJK text it guessed cp874 (Thai) and the
    whole page came out garbled. Only bytes that are valid, non-ASCII UTF-8 with no
    BOM and no charset / XML encoding declaration are touched; everything else is
    left to the page's own declaration or to detection.
    """
    if data.startswith(_UTF8_BOM) or _CHARSET_DECL.search(data[:4096]):
        return data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    if data.isascii():
        return data
    return _UTF8_BOM + data


_MHTML_EXTENSIONS = {".mht", ".mhtml"}


def _mhtml_to_html(data: bytes) -> Optional[bytes]:
    """The root HTML of a MIME web archive, decoded with the charset its part header declares.

    docling's MHTML backend hands the HTML part to BeautifulSoup without that charset, so a
    short CJK page with no ``<meta charset>`` came out as cp874 (Thai) — the failure
    ``_declare_html_utf8`` fixes for plain HTML. With a declared charset the part is decoded
    here and returned as BOM-marked UTF-8 (BeautifulSoup trusts a BOM over any guess or
    ``<meta>``); without one it goes through ``_declare_html_utf8``. Returns None when the
    archive has no text/html part, leaving the file to docling as it is. The other parts
    (images, CSS) are dropped: docling neither fetches nor indexes them.
    """
    from email import message_from_bytes, policy

    msg = message_from_bytes(data, policy=policy.default)
    for part in msg.walk():
        if part.get_content_type() != "text/html":
            continue
        raw = part.get_payload(decode=True) or b""
        charset = part.get_content_charset()
        if not charset:
            return _declare_html_utf8(raw)
        try:
            text = raw.decode(charset)
        except (LookupError, UnicodeDecodeError):
            text = raw.decode("utf-8", errors="replace")
        return _UTF8_BOM + text.encode("utf-8")
    return None


def chunk_docling_document(
    doc: Any,
    file_name: str,
    max_tokens: int = 512,
    tokenizer: Optional[str] = None,
    table_max_tokens: Optional[int] = None,
) -> tuple[str, List[dict]]:
    """Export a DoclingDocument to Markdown and chunk it into records.

    The single chunking entry point for every parse path: docling_convert_once (document
    formats), cloud_parser (vision-model Markdown re-assembled into one document, with page
    provenance attached) and pdf_text_layer (text-layer paragraphs with their pages) all end
    here, so they produce the same record shape
    ``[{"text", "headings", "page_no", "content_type"}, ...]``.
    """
    from docling.chunking import HierarchicalChunker

    # Export full text as Markdown
    full_text = doc.export_to_markdown()
    logger.debug(f"Docling loaded {file_name}: {len(full_text)} chars")

    # Chunk via HierarchicalChunker + per-chunk tokenizer refinement
    # (see helpers above; we bypass HybridChunker for the large-table case).
    hf_tok = get_hf_tokenizer(tokenizer)

    raw_chunks = list(HierarchicalChunker().chunk(dl_doc=doc))
    texts = _refine_chunks_for_token_budget(
        raw_chunks, hf_tok, max_tokens, doc=doc, table_max_tokens=table_max_tokens)

    logger.info(
        f"Docling processed {file_name}: {len(texts)} chunks "
        f"(max_tokens={max_tokens})"
    )
    return full_text, texts


def docling_convert_once(
    file_path: str,
    file_name: str,
    max_tokens: int = 512,
    tokenizer: Optional[str] = None,
    progress_cb: Optional[Any] = None,
    report: Optional[dict] = None,
    table_max_tokens: Optional[int] = None,
) -> tuple[str, List[dict]]:
    """Parse with Docling once, returning both the full text and chunk records (with headings/page_no).

    Consolidates the previously duplicated parsing of docling_load_as_text + docling_load_and_chunk,
    reducing PDF parse time from 2 passes → 1.

    Args:
        file_path: Path to the document file
        file_name: Original file name
        max_tokens: Maximum tokens per chunk
        tokenizer: HuggingFace tokenizer name
        progress_cb: accepted for the loader's common signature; document formats have no
            pages to count, so it is not called.
        report: optional dict filled with ``warning`` (pages that could not be parsed,
            None when complete) and ``missing_pages``.
        table_max_tokens: budget a table may use before it is split (default max_tokens).

    Returns:
        (full_text, chunk_records) — full Markdown text + chunk records
            ``[{"text", "headings", "page_no"}, ...]`` (citation provenance)
    """
    try:
        # 1. Convert once (using the singleton converter)
        converter = get_converter()

        # FileStorage saves files as extensionless UUIDs. docling's _guess_format falls
        # back on extension when there are no magic bytes, so plain-text formats
        # (.md / .html / .csv / .vtt / .latex / .adoc) fail to detect. Wrapping into a
        # DocumentStream carrying the original file_name restores extension detection.
        path_obj = Path(file_path)
        suffix = Path(file_name).suffix.lower()
        is_html = suffix in _HTML_EXTENSIONS
        is_mhtml = suffix in _MHTML_EXTENSIONS
        # Only convert() holds the inference lock; keep file reading outside to shorten hold time.
        if (path_obj.suffix == "" and Path(file_name).suffix != "") or is_html or is_mhtml:
            from docling.datamodel.base_models import DocumentStream
            from io import BytesIO
            with path_obj.open("rb") as f:
                data = f.read()
            stream_name = file_name
            if is_html:
                data = _declare_html_utf8(data)
            elif is_mhtml:
                html = _mhtml_to_html(data)
                if html is not None:
                    data, stream_name = html, f"{Path(file_name).stem}.html"

            def _source():  # a stream is consumed by convert(): build a fresh one per attempt
                return DocumentStream(name=stream_name, stream=BytesIO(data))
        else:
            def _source():
                return file_path

        def _convert():
            with DOCLING_INFER_LOCK:
                return converter.convert(source=_source())

        # Status is checked: a partial result is retried once and what is still missing
        # comes back as a warning for the file record.
        result, parse_warning = convert_checked(_convert, file_name)
        if report is not None:
            report["warning"] = parse_warning
            report["missing_pages"] = missing_pages(result) if parse_warning else []
        doc = result.document

        # 2-3. Markdown export + chunking (shared with the vision and text-layer paths)
        return chunk_docling_document(doc, file_name, max_tokens=max_tokens, tokenizer=tokenizer,
                                      table_max_tokens=table_max_tokens)

    except Exception as e:
        # Detect common "the document itself is the problem" errors and give the user a clearer message.
        # These aren't server bugs; the file needs special handling.
        err_msg = str(e).lower()
        if "incorrect password" in err_msg or "encrypted" in err_msg:
            logger.error(
                f"❌ {file_name} is password-protected. "
                f"Remove password before uploading. "
                f"(original error: {e})"
            )
            raise ValueError(
                f"File is password-protected: '{file_name}'. "
                f"Please decrypt it before uploading."
            ) from e
        if "is not valid" in err_msg or "corrupt" in err_msg:
            logger.error(f"❌ {file_name} is corrupted or unreadable: {e}")
            raise ValueError(
                f"File appears corrupted or unreadable: '{file_name}'."
            ) from e
        logger.error(f"Docling processing failed for {file_name}: {e}")
        raise
