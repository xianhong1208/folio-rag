"""Pure functions behind the golden-set retrieval eval (scripts/eval_retrieval.py).

Gold evidence is a file name plus quoted spans, never a chunk id, so the golden set
survives any change to chunking. Everything here is I/O-free and unit-tested
(tests/test_eval_retrieval_metrics.py):

* span matching    — does a returned text contain (most of) a gold span?
* ranking metrics  — Recall@k, MRR@k, nDCG@k over evidence items, file-level recall
* response metrics — estimated tokens, duplicate-text ratio, percentiles
* sampling         — deterministic, document-stratified passage sampling
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
from typing import Callable, Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------- text

_MARKUP = re.compile(r"[*_`#|>]+")


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x2FA1F)


def normalize(text: str) -> str:
    """Comparison form: NFKC, lower-case, and only letters / digits / CJK kept.

    Whitespace, punctuation, symbols and Markdown markup are dropped, so the same
    sentence matches whether it came from Markdown, a PDF parser (wrapped lines,
    full-width digits) or a table serialised as "row, column = value" triplets.
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(c for c in text if c.isalnum())


def text_lang(text: str) -> str:
    """'zh' when CJK characters make up at least 30% of the letters, else 'en'."""
    cjk = sum(1 for c in text if is_cjk(c))
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if cjk + latin == 0:
        return "en"
    return "zh" if cjk / (cjk + latin) >= 0.3 else "en"


# ------------------------------------------------------------------- span matching

SPAN_WINDOW = 8          # characters per window when scoring partial containment
SHORT_SPAN = 12          # normalised spans shorter than this must match exactly
DEFAULT_MIN_COVERAGE = 0.6


def span_coverage(span: str, text: str, *, window: int = SPAN_WINDOW,
                  short: int = SHORT_SPAN, _norm_text: Optional[str] = None) -> float:
    """Fraction of ``span`` that ``text`` contains, in [0, 1].

    The normalised span is cut into consecutive ``window``-character pieces (the last
    piece is kept when it has at least half a window); coverage is the share of pieces
    found verbatim in the normalised text. A span shorter than ``short`` characters
    counts only as a whole (1.0 or 0.0), since a few characters match anywhere.
    """
    s = normalize(span)
    t = _norm_text if _norm_text is not None else normalize(text)
    if not s:
        return 0.0
    if s in t:
        return 1.0
    if len(s) < short:
        return 0.0
    pieces = [s[i:i + window] for i in range(0, len(s), window)]
    if len(pieces) > 1 and len(pieces[-1]) < window // 2:
        pieces[-2] += pieces.pop()
    return sum(1 for p in pieces if p in t) / len(pieces)


def covers_spans(spans: Sequence[str], text: str,
                 min_coverage: float = DEFAULT_MIN_COVERAGE) -> bool:
    """True when ``text`` covers every span of one evidence item."""
    if not spans:
        return False
    nt = normalize(text)
    return all(span_coverage(s, text, _norm_text=nt) >= min_coverage for s in spans)


def result_hits(result: dict, evidence: Sequence[dict], gold_index: Optional[dict] = None,
                min_coverage: float = DEFAULT_MIN_COVERAGE, neighbours: int = 1) -> List[int]:
    """Indices of the evidence items one retrieved result satisfies.

    ``result``: {"text", "file_name", optional "chunk_range": [lo, hi]}.
    ``evidence``: [{"file", "spans": [...]}].
    ``gold_index``: optional {evidence_idx: (lo, hi)} — the leaf chunk range that holds
    the span in the indexed folder. A result from the same file whose chunk range lies
    within ``neighbours`` chunks of it also counts (a neighbouring chunk is how a span
    that straddles a chunk boundary is usually returned). A parent section or an
    expanded window that contains the span matches through its text.
    """
    hits = []
    for i, ev in enumerate(evidence):
        if result.get("file_name") and ev.get("file") and result["file_name"] != ev["file"]:
            continue
        if covers_spans(ev.get("spans") or [], result.get("text") or "", min_coverage):
            hits.append(i)
            continue
        rng = result.get("chunk_range")
        gold = (gold_index or {}).get(i)
        if rng and gold and None not in rng and None not in gold:
            lo, hi = rng
            if lo <= gold[1] + neighbours and hi >= gold[0] - neighbours:
                hits.append(i)
    return hits


# ------------------------------------------------------------------ ranking metrics


def ranking_metrics(hit_lists: Sequence[Sequence[int]], n_items: int,
                    ks: Iterable[int] = (5, 10, 20), mrr_k: int = 10,
                    ndcg_k: int = 10) -> Dict[str, float]:
    """Recall@k / MRR@k / nDCG@k for one query.

    ``hit_lists[r]`` lists the evidence items the result at rank r+1 satisfies. Recall
    is the share of evidence items found in the top k (multi-hop questions have two).
    For nDCG a result earns gain 1 only for evidence items not already found higher up,
    so duplicates (a parent plus its own leaf) earn nothing; the ideal ranking finds
    every item in the first positions.
    """
    out: Dict[str, float] = {}
    if n_items <= 0:
        return out
    for k in ks:
        found = set()
        for hl in hit_lists[:k]:
            found.update(hl)
        out[f"recall@{k}"] = len(found) / n_items
    rr = 0.0
    for r, hl in enumerate(hit_lists[:mrr_k], 1):
        if hl:
            rr = 1.0 / r
            break
    out[f"mrr@{mrr_k}"] = rr
    seen: set = set()
    dcg = 0.0
    for r, hl in enumerate(hit_lists[:ndcg_k], 1):
        new = set(hl) - seen
        if new:
            dcg += 1.0 / math.log2(r + 1)
            seen |= new
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_items, ndcg_k) + 1))
    out[f"ndcg@{ndcg_k}"] = dcg / idcg if idcg else 0.0
    return out


def file_recall(result_files: Sequence[str], gold_files: Sequence[str], k: int) -> float:
    """Share of gold files present among the files of the top-k results."""
    gold = set(gold_files)
    if not gold:
        return 0.0
    return len(gold & set(result_files[:k])) / len(gold)


def first_rank(hit_lists: Sequence[Sequence[int]]) -> Optional[int]:
    for r, hl in enumerate(hit_lists, 1):
        if hl:
            return r
    return None


# ---------------------------------------------------------------- response metrics

CJK_TOKENS_PER_CHAR = 0.7
OTHER_TOKENS_PER_CHAR = 0.25


def estimate_tokens(text: str) -> float:
    """Rough LLM token count: CJK ≈ 0.7 tokens/char, everything else ≈ 0.25/char."""
    cjk = sum(1 for c in text if is_cjk(c))
    return cjk * CJK_TOKENS_PER_CHAR + (len(text) - cjk) * OTHER_TOKENS_PER_CHAR


def duplicate_ratio(texts: Sequence[str], shingle: int = 10) -> float:
    """Share of the response's text that repeats text shown earlier in the same response.

    Each text is normalised and cut into overlapping ``shingle``-character pieces;
    the ratio is 1 - unique pieces / all pieces (0 = no repetition). Overlapping
    neighbour windows and a parent shown next to its own leaf push it up.
    """
    total = 0
    seen: set = set()
    for t in texts:
        n = normalize(t)
        if len(n) < shingle:
            continue
        for i in range(len(n) - shingle + 1):
            total += 1
            seen.add(n[i:i + shingle])
    return 0.0 if total == 0 else 1.0 - len(seen) / total


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Linear-interpolated percentile (p in 0..100); None for no values."""
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * p / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(vals) - 1)
    return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def group_means(rows: Sequence[dict], key: Callable[[dict], str],
                metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """{group: {metric: mean, "n": count}} over rows that carry the metric."""
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        out[g] = {"n": len(rs)}
        for m in metrics:
            v = mean(r[m] for r in rs if m in r)
            if v is not None:
                out[g][m] = round(v, 4)
    return out


# ------------------------------------------------------------------------ sampling


def stratified_sample(pool: Sequence[dict], n: int, seed: int,
                      stratum: Callable[[dict], str],
                      exclude: Optional[set] = None,
                      key: Callable[[dict], str] = lambda p: p["id"]) -> List[dict]:
    """Deterministically pick up to ``n`` items, spread round-robin over strata.

    Strata (e.g. documents) are visited in a seeded order and each contributes its next
    unused item in turn, so no single long document dominates a question type. The
    same pool, seed and exclusions always give the same list.
    """
    rng = random.Random(seed)
    exclude = exclude or set()
    by: Dict[str, List[dict]] = {}
    for item in sorted(pool, key=key):
        if key(item) not in exclude:
            by.setdefault(stratum(item), []).append(item)
    order = sorted(by)
    rng.shuffle(order)
    for s in order:
        rng.shuffle(by[s])
    out: List[dict] = []
    while len(out) < n and any(by[s] for s in order):
        for s in order:
            if by[s] and len(out) < n:
                out.append(by[s].pop(0))
    return out
