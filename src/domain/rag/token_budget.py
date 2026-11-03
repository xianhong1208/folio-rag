"""Cheap token estimates and budgets for what retrieval hands to an LLM.

Used to keep MCP search responses, read pages and CRAG answer contexts inside a
token budget without loading a tokenizer. The estimate is calibrated on common
BPE tokenizers (cl100k / o200k / Qwen): a CJK character costs roughly 0.7 tokens,
any other character roughly 0.25 (≈ 4 characters per English token). It is
deliberately a little generous, never exact — callers only use it to decide how
much to send.
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence, Tuple

CJK_TOKENS_PER_CHAR = 0.7
OTHER_TOKENS_PER_CHAR = 0.25
_EPS = 1e-6  # float sums of 0.7 drift (10 × 0.7 → 7.000000000000001)

# CJK punctuation, kana, CJK ideographs (incl. ext. A / compatibility), Hangul,
# full-width forms.
_CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿"
                  r"가-힯豈-﫿＀-￯]")


def _is_cjk(ch: str) -> bool:
    return _CJK.match(ch) is not None


def char_cost(ch: str) -> float:
    """Estimated tokens for one character."""
    return CJK_TOKENS_PER_CHAR if _is_cjk(ch) else OTHER_TOKENS_PER_CHAR


def estimate_tokens(text: str) -> int:
    """Estimated token count of ``text`` (0 for empty)."""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(math.ceil(cjk * CJK_TOKENS_PER_CHAR + other * OTHER_TOKENS_PER_CHAR - _EPS))


def cost_prefix(text: str) -> List[float]:
    """Prefix sums of per-character cost: ``p[j] - p[i]`` estimates ``text[i:j]``."""
    out = [0.0]
    acc = 0.0
    for ch in text or "":
        acc += char_cost(ch)
        out.append(acc)
    return out


def truncate_to_tokens(text: str, max_tokens: int) -> Tuple[str, bool]:
    """The longest prefix of ``text`` within ``max_tokens``; (text, truncated)."""
    if not text:
        return "", False
    if max_tokens <= 0:
        return "", True
    if estimate_tokens(text) <= max_tokens:
        return text, False
    acc = 0.0
    for i, ch in enumerate(text):
        acc += char_cost(ch)
        if acc > max_tokens + _EPS:
            return text[:i], True
    return text, False  # pragma: no cover - unreachable (estimate said it exceeds)


ELLIPSIS = " … "


def excerpt(text: str, spans: Sequence[Sequence[int]], max_tokens: int,
            step: int = 24) -> str:
    """An excerpt of ``text`` within ``max_tokens`` centred on ``spans``.

    ``spans`` are ``[start, end)`` character ranges that matter most (e.g. where
    the retrieved leaves sit inside a merged section), in priority order. The
    text is returned whole when it fits. Otherwise the spans are kept (in
    priority order while they fit) and grown evenly with surrounding context
    until the budget is spent; gaps between the kept pieces are marked with an
    ellipsis. Without usable spans the excerpt is the head of the text.
    """
    if not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    n = len(text)
    clean: List[List[int]] = []
    for sp in spans or ():
        try:
            s, e = int(sp[0]), int(sp[1])
        except (TypeError, ValueError, IndexError):
            continue
        s, e = max(0, s), min(n, e)
        if e > s:
            clean.append([s, e])
    if not clean:
        return truncate_to_tokens(text, max_tokens)[0]

    p = cost_prefix(text)

    def cost(ivs) -> float:
        return sum(p[e] - p[s] for s, e in _union(ivs))

    # Keep spans in priority order while they fit; a span that does not fit
    # entirely is cut to the remaining budget (only if nothing was kept yet).
    kept: List[List[int]] = []
    for s, e in clean:
        if cost(kept + [[s, e]]) <= max_tokens + _EPS:
            kept.append([s, e])
        elif not kept:
            piece, _ = truncate_to_tokens(text[s:e], max_tokens)
            kept.append([s, s + len(piece)])
            break
    if not kept:  # pragma: no cover - the first span always yields something
        return truncate_to_tokens(text, max_tokens)[0]

    # Grow every kept interval left and right in small steps, round robin.
    grew = True
    while grew:
        grew = False
        for idx in range(len(kept)):
            for side in (0, 1):
                trial = [list(x) for x in kept]
                if side == 0 and trial[idx][0] > 0:
                    trial[idx][0] = max(0, trial[idx][0] - step)
                elif side == 1 and trial[idx][1] < n:
                    trial[idx][1] = min(n, trial[idx][1] + step)
                else:
                    continue
                if cost(trial) <= max_tokens + _EPS:
                    kept[idx] = trial[idx]
                    grew = True
    pieces = _union(kept)
    out = []
    if pieces[0][0] > 0:
        out.append(ELLIPSIS.strip() + " ")
    for i, (s, e) in enumerate(pieces):
        if i:
            out.append(ELLIPSIS)
        out.append(text[s:e])
    if pieces[-1][1] < n:
        out.append(" " + ELLIPSIS.strip())
    return "".join(out)


def _union(intervals) -> List[List[int]]:
    """Sorted union of half-open intervals (touching ones are joined)."""
    out: List[List[int]] = []
    for s, e in sorted([list(x) for x in intervals]):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
