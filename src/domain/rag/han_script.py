"""Script hygiene for generated text: invisible characters and Simplified/Traditional drift.

LLMs asked to write in a Traditional Chinese document's language still slip
Simplified characters into their output (``文档`` for ``文件``), and sometimes
emit zero-width spaces inside words (``ROC\\u200bm``). Both break keyword search:
the Simplified form never matches a Traditional query, and a zero-width space
splits a token in two.

``fix_simplified`` is deliberately narrow. It only touches a generated text
when the document is Traditional, and within it only characters that are
Simplified-only *and* absent from the document. Converting the whole text with
OpenCC would also rewrite valid Traditional usage (``s2twp`` turns ``文件`` into
``檔案``; ``s2t`` turns Taiwan's ``台`` into ``臺``). English and other scripts are
never touched. The document's own text is never modified.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache
from typing import Optional, Tuple

# Zero-width space / non-joiner, word joiner, BOM. ZWJ (U+200D) is kept: emoji need it.
_INVISIBLE = dict.fromkeys(map(ord, "​‌⁠﻿"), None)
# No-break and fixed-width spaces -> a plain space (U+3000, the CJK full-width space, is kept).
_ODD_SPACES = dict.fromkeys(
    map(ord, "            "), " "
)
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

_cc_lock = threading.Lock()
_converters: dict = {}


def strip_invisible(text: str) -> str:
    """Remove zero-width characters and turn odd spaces into plain spaces."""
    if not text:
        return text
    return text.translate(_INVISIBLE).translate(_ODD_SPACES)


def _cc(config: str):
    with _cc_lock:
        if config not in _converters:
            import opencc  # core dependency (no torch), so it works under profile: cloud too

            _converters[config] = opencc.OpenCC(f"{config}.json")
        return _converters[config]


@lru_cache(maxsize=8192)
def _kind(ch: str) -> int:
    """+1 Traditional-only, -1 Simplified-only, 0 shared / not convertible."""
    try:
        if _cc("t2s").convert(ch) != ch:
            return 1
        if _cc("s2tw").convert(ch) != ch:
            return -1
    except Exception:  # pragma: no cover - opencc missing
        return 0
    return 0


def variant_counts(text: str) -> Tuple[int, int]:
    """(Traditional-only, Simplified-only) Han characters in ``text``."""
    trad = simp = 0
    for ch in _HAN.findall(text or ""):
        k = _kind(ch)
        if k > 0:
            trad += 1
        elif k < 0:
            simp += 1
    return trad, simp


def is_traditional(text: str, min_chars: int = 3) -> Optional[bool]:
    """True / False when ``text`` clearly uses Traditional / Simplified characters,
    None when it has too few script-specific characters to tell."""
    trad, simp = variant_counts(text)
    if trad + simp < min_chars:
        return None
    return trad > simp


def fix_simplified(generated: str, document_text: str) -> str:
    """Convert Simplified-only characters in ``generated`` to Traditional, when the
    document is Traditional and does not itself use those characters."""
    if not generated or not _HAN.search(generated):
        return generated
    if is_traditional(document_text) is not True:
        return generated
    doc_chars = set(_HAN.findall(document_text))
    targets = [i for i, ch in enumerate(generated)
               if _HAN.match(ch) and ch not in doc_chars and _kind(ch) < 0]
    if not targets:
        return generated
    converted = _cc("s2tw").convert(generated)
    if len(converted) != len(generated):  # phrase mapping changed the length: go char by char
        converted = "".join(_cc("s2tw").convert(ch) for ch in generated)
    chars = list(generated)
    for i in targets:
        chars[i] = converted[i]
    return "".join(chars)


def document_language(text: str) -> Optional[str]:
    """Name of the document's main language when it can be told from the script alone
    (CJK scripts); None for Latin and other scripts, where the prompt simply asks
    for the chunk's own language."""
    sample = (text or "")[:20000]
    kana = len(re.findall(r"[぀-ヿ]", sample))
    hangul = len(re.findall(r"[가-힯]", sample))
    han = len(_HAN.findall(sample))
    latin = len(re.findall(r"[A-Za-z]", sample))
    if kana >= 20 and kana >= han * 0.1:
        return "Japanese"
    if hangul >= 20 and hangul >= latin * 0.2:
        return "Korean"
    if han >= 20 and han >= latin * 0.2:
        trad = is_traditional(sample)
        if trad is True:
            return "Traditional Chinese"
        if trad is False:
            return "Simplified Chinese"
        return "Chinese"
    return None
