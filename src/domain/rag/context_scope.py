"""What the LLM sees when it writes a chunk's contextual prefix.

Before: the whole document, cut at ``max_doc_chars`` (60,000) characters. Every
chunk paid for the same huge prompt, and every chunk *past* the cut got a prefix
written from the start of the document, which it is not part of.

Now each chunk is described from:

- the document outline: every heading path, once, in order (cheap, and tells the
  model what the whole document is about); and
- its own section: the original text of the chunks sharing its section key
  (chunk_identity.section_key). When that section is longer than the window,
  the window is centred on the chunk, so a chunk deep inside a long, heading-less
  document still sees its own neighbourhood.

Chunks of the same section get an identical outline + section text, which keeps
vLLM's prefix cache hitting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.domain.rag.chunk_identity import BREADCRUMB_SEP, section_key

# Section text given to the LLM per chunk (characters), capped by max_doc_chars.
SECTION_WINDOW_CHARS = 12_000
OUTLINE_MAX_CHARS = 3_000
_GAP = "\n\n"


def document_outline(headings_list: Sequence[Optional[Sequence[Any]]],
                     max_chars: int = OUTLINE_MAX_CHARS) -> str:
    """Unique heading paths in document order, one per line."""
    seen, lines = set(), []
    for headings in headings_list:
        path = BREADCRUMB_SEP.join(str(h).strip() for h in (headings or []) if str(h).strip())
        if path and path not in seen:
            seen.add(path)
            lines.append(f"- {path}")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit("\n", 1)[0] + "\n- …"
    return out


def window(text: str, start: int, end: int, width: int) -> str:
    """``width`` characters of ``text`` around [start, end), marked with … when cut."""
    if len(text) <= width:
        return text
    centre = (start + end) // 2
    lo = max(0, min(centre - width // 2, len(text) - width))
    hi = lo + width
    return ("…" if lo > 0 else "") + text[lo:hi] + ("…" if hi < len(text) else "")


def section_texts(raws: Sequence[str], headings_list: Sequence[Optional[Sequence[Any]]],
                  width: int = SECTION_WINDOW_CHARS) -> List[str]:
    """Per chunk: the original text of its section, windowed around the chunk."""
    groups: Dict[Tuple[str, ...], List[int]] = {}
    for i, headings in enumerate(headings_list):
        groups.setdefault(section_key(headings), []).append(i)
    out: List[str] = [""] * len(raws)
    for members in groups.values():
        joined_parts, spans, pos = [], {}, 0
        for i in members:
            if joined_parts:
                pos += len(_GAP)
            spans[i] = (pos, pos + len(raws[i]))
            joined_parts.append(raws[i])
            pos += len(raws[i])
        joined = _GAP.join(joined_parts)
        for i in members:
            out[i] = window(joined, spans[i][0], spans[i][1], width)
    return out


def locate_window(document: str, chunk: str, width: int) -> str:
    """Fallback when no section structure is passed: the document around the chunk."""
    at = document.find(chunk[:200]) if chunk else -1
    if at < 0:
        at = 0
    return window(document, at, at + len(chunk), width)
