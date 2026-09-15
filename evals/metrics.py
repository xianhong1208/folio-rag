
"""Pure evaluation-metric functions (shared by retrieval and ASR evaluation).

No external dependencies, no I/O — behavior is pinned by
tests/test_eval_metrics.py and these are called by the scoring scripts
(scripts/run_rag_eval.py / run_asr_eval.py).
"""

from __future__ import annotations

import re
from typing import List, Sequence

_NORMALIZE_STRIP = re.compile(r"[\s,,。.、;;::!!??\"'「」『』()()\-—·…]+")


def _normalize(text: str) -> str:
    """Strip whitespace and common punctuation — ASR output often lacks
    punctuation, so we avoid penalizing punctuation differences."""
    return _NORMALIZE_STRIP.sub("", text)


def cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Character Error Rate = levenshtein(ref, hyp) / len(ref).

    Args:
        reference: Human-proofread golden transcript.
        hypothesis: ASR output.
        normalize: Compare after stripping whitespace/punctuation (default on).

    Returns:
        0.0 (exact match) up to unbounded (can exceed 1 with many insertions);
        returns 1.0 when ref is empty but hyp is not.
    """
    if normalize:
        reference, hypothesis = _normalize(reference), _normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0

    # Standard DP Levenshtein (character-level, two-row rolling)
    prev = list(range(len(hypothesis) + 1))
    for i, rc in enumerate(reference, 1):
        curr = [i]
        for j, hc in enumerate(hypothesis, 1):
            curr.append(min(
                prev[j] + 1,          # deletion
                curr[j - 1] + 1,      # insertion
                prev[j - 1] + (rc != hc),  # substitution
            ))
        prev = curr
    return prev[-1] / len(reference)


# High-frequency sample of Simplified characters that differ in form from their
# Traditional counterparts (heuristic, not exhaustive — enough to judge the
# language leaning of the output).
# WARNING: include only Simplified-exclusive glyphs — characters identical in
# both scripts must never be added, or Traditional Chinese would be misjudged.
_SIMPLIFIED_CHARS = set(
    "会议记录检统计设进军对开关买卖东车书长门问间闻风飞马鸟龙点级红绿"
    "语说读写听讲话让认识谁请谢边这远运动过还货质银钱铁钟错难题华万"
    "与专业丛严个临为丽举义乐习乡产亲亿仅从仓仪们价众优传伤体办务"
)


def simplified_char_ratio(text: str) -> float:
    """Ratio of CJK characters that belong to the Simplified-exclusive sample set.

    Heuristic: any value >0 means Simplified characters are mixed in; Traditional
    Chinese output is expected to be approximately 0. Returns 0.0 when there are
    no CJK characters.
    """
    cjk = [c for c in text if "一" <= c <= "鿿"]
    if not cjk:
        return 0.0
    return sum(1 for c in cjk if c in _SIMPLIFIED_CHARS) / len(cjk)


def recall_at_k(retrieved: Sequence[str], expected: List[str], k: int) -> float:
    """Fraction of expected items hit within the top-k (partial credit when
    multiple items are expected)."""
    if not expected:
        return 1.0
    top = set(retrieved[:k])
    return sum(1 for e in expected if e in top) / len(expected)


def mrr(retrieved: Sequence[str], expected: List[str]) -> float:
    """Mean Reciprocal Rank (single-query version): reciprocal of the first hit's
    position; 0 if there is no hit."""
    exp = set(expected)
    for i, r in enumerate(retrieved, 1):
        if r in exp:
            return 1.0 / i
    return 0.0
