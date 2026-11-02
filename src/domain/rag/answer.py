"""Corrective-RAG answer generation for the console (both the JSON and the
streaming path share everything here).

1. **Grade** the top retrieved chunks for relevance in one LLM call. A reply that
   is not a JSON array of exactly one boolean per chunk is a grading *failure*:
   every chunk is kept (fail-open) but the answer is labelled ``unverified`` —
   never ``high`` — because nothing was actually checked.
2. **Gate**: when every chunk is graded irrelevant, decline to answer.
3. **Generate** from the kept chunks. Each chunk keeps its number from the result
   list (``[3]`` is the third result card even when chunks 1–2 were dropped), its
   source is named, and the model is asked to answer in the question's language.
4. **Clean** the answer: ``【n†…】`` / ``【n】`` become ``[n]``, Markdown bold
   markers are removed, ``[1, 3]`` becomes ``[1][3]``, and citations of chunks the
   model was not given are dropped (reported as ``invalid_citations``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

MAX_CONTEXTS = 6
GRADE_CHARS = 600
CONTEXT_CHARS = 1200

CONF_NONE = "none"
CONF_LOW = "low"
CONF_MEDIUM = "medium"
CONF_HIGH = "high"
CONF_UNVERIFIED = "unverified"

NO_BASIS_MESSAGE = ("The currently indexed content has no sufficient basis to answer this "
                    "question. Consider adding relevant documents or rephrasing the question.")
EMPTY_ANSWER_MESSAGE = ("Relevant content was found, but answer generation did not complete "
                        "(model output was truncated). Please try again or use a more "
                        "specific question.")


@dataclass
class Grading:
    grades: List[bool]
    ok: bool = True
    reason: Optional[str] = None      # llm_error | invalid_response | length_mismatch | non_boolean


@dataclass
class Context:
    """One chunk handed to the generator, with its result-list number."""
    number: int
    text: str
    source: str = ""


@dataclass
class CleanAnswer:
    text: str
    citations: List[int] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)


# ------------------------------------------------------------------ grading

_BOOL_WORDS = {"true": True, "false": False}


def parse_grades(content: Optional[str], n: int) -> Grading:
    """Strictly parse the grader's reply: a JSON array of ``n`` booleans."""
    m = re.search(r"\[.*?\]", content or "", re.S)
    if not m:
        return Grading([True] * n, ok=False, reason="invalid_response")
    try:
        arr = json.loads(m.group(0))
    except ValueError:
        return Grading([True] * n, ok=False, reason="invalid_response")
    if not isinstance(arr, list) or len(arr) != n:
        return Grading([True] * n, ok=False, reason="length_mismatch")
    out: List[bool] = []
    for x in arr:
        if isinstance(x, bool):
            out.append(x)
        elif isinstance(x, str) and x.strip().lower() in _BOOL_WORDS:
            out.append(_BOOL_WORDS[x.strip().lower()])
        else:
            return Grading([True] * n, ok=False, reason="non_boolean")
    return Grading(out)


def grade_prompt(query: str, texts: Sequence[str]) -> str:
    return ("Grade whether each numbered context can help answer the question. "
            "Reply ONLY a JSON array of booleans (one per context, true = relevant).\n\n"
            + "\n\n".join(f"[{i + 1}] {t[:GRADE_CHARS]}" for i, t in enumerate(texts))
            + f"\n\nQuestion: {query}\n\nJSON:")


def grade(client, model: str, query: str, texts: Sequence[str]) -> Grading:
    """One LLM call; any failure is reported, never silently treated as relevant."""
    if not texts:
        return Grading([])
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": grade_prompt(query, texts)}],
            max_tokens=1024, temperature=0)
        content = r.choices[0].message.content
    except Exception:  # noqa: BLE001
        return Grading([True] * len(texts), ok=False, reason="llm_error")
    return parse_grades(content, len(texts))


def confidence(grading: Grading, kept: int) -> str:
    if not grading.ok:
        return CONF_UNVERIFIED
    if kept >= 2:
        return CONF_HIGH
    return CONF_MEDIUM if kept == 1 else CONF_LOW


# --------------------------------------------------------------- generation

def source_label(result: dict) -> str:
    md = (result or {}).get("metadata") or {}
    parts = [md.get("file_name")] + list(md.get("headings") or [])
    return " › ".join(str(p) for p in parts if p)


def contexts_for(results: Sequence[dict], keep: Iterable[bool]) -> List[Context]:
    """Kept results with their original 1-based result-list numbers."""
    return [Context(number=i + 1, text=(r.get("text") or ""), source=source_label(r))
            for i, (r, k) in enumerate(zip(results, keep)) if k]


def answer_prompt(query: str, contexts: Sequence[Context]) -> str:
    ctx = "\n\n".join(
        f"[{c.number}]" + (f" ({c.source})" if c.source else "") + f" {c.text[:CONTEXT_CHARS]}"
        for c in contexts)
    return ("You are a helpful assistant. Answer the question using ONLY the context below. "
            "Cite sources inline with their bracketed numbers, e.g. [2]. If the context lacks "
            "the answer, say so. Answer in the same language as the question.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:")


# ----------------------------------------------------------------- cleanup

_FULLWIDTH_CITE = re.compile(r"【\s*(\d+)\s*(?:†[^】]*)?】")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.S)
_CITE_GROUP = re.compile(r"\[(\s*\d+(?:\s*[,，、]\s*\d+)*\s*)\]")


def normalise_citations(text: str) -> str:
    """``【n†…】`` / ``【n】`` → ``[n]`` (safe on partial streaming deltas)."""
    return _FULLWIDTH_CITE.sub(r"[\1]", text or "")


def clean_answer(text: Optional[str], allowed: Iterable[int]) -> CleanAnswer:
    """Normalise citation markers, drop bold markers, keep only citations of
    chunks the model was given."""
    allowed_set = set(allowed)
    out = _BOLD.sub(r"\2", normalise_citations(text or ""))
    used: List[int] = []
    invalid: List[int] = []

    def _fix(m):
        nums = [int(x) for x in re.split(r"[,，、]", m.group(1)) if x.strip()]
        good = [n for n in nums if n in allowed_set]
        for n in nums:
            target = used if n in allowed_set else invalid
            if n not in target:
                target.append(n)
        return "".join(f"[{n}]" for n in good)

    out = _CITE_GROUP.sub(_fix, out)
    out = re.sub(r"[ \t]+([,.;:!?，。；：！？])", r"\1", out)   # space left before punctuation
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return CleanAnswer(text=out, citations=sorted(used), invalid_citations=sorted(invalid))


# ------------------------------------------------------------------- flows

def answer(client, model: str, query: str, results: Sequence[dict]) -> dict:
    """Grade → gate → generate → clean (blocking). Returns the console payload:
    {answer, confidence, kept, dropped, note, citations, invalid_citations, grading_error?}."""
    if not results:
        return {"answer": None, "confidence": CONF_NONE, "kept": 0, "dropped": 0,
                "note": "No chunks retrieved"}
    top = list(results[:MAX_CONTEXTS])
    grading = grade(client, model, query, [r.get("text") or "" for r in top])
    contexts = contexts_for(top, grading.grades)
    dropped = len(top) - len(contexts)
    base = {"kept": len(contexts), "dropped": dropped}
    if not grading.ok:
        base["grading_error"] = grading.reason
    if not contexts:
        return {"answer": NO_BASIS_MESSAGE, "confidence": CONF_LOW, **base,
                "note": "CRAG gate: no retrieved chunk graded relevant"}

    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": answer_prompt(query, contexts)}],
        # Reasoning models spend max_tokens on reasoning first; too small a budget
        # returns empty content (finish_reason=length).
        max_tokens=2048, temperature=0.2)
    raw = r.choices[0].message.content
    finish = getattr(r.choices[0], "finish_reason", None)
    if not (raw or "").strip():
        return {"answer": EMPTY_ANSWER_MESSAGE, "confidence": CONF_LOW, **base,
                "note": f"generation returned empty (finish={finish})"}
    cleaned = clean_answer(raw, [c.number for c in contexts])
    return {"answer": cleaned.text, "confidence": confidence(grading, len(contexts)), **base,
            "note": _note(grading, len(contexts)),
            "citations": cleaned.citations, "invalid_citations": cleaned.invalid_citations}


def _note(grading: Grading, kept: int) -> str:
    if not grading.ok:
        return f"Relevance grading failed ({grading.reason}); answered from all {kept} chunk(s), unverified"
    return f"Answered from {kept} relevant chunk(s)"


def finish_stream(raw: str, contexts: Sequence[Context], grading: Grading) -> dict:
    """The streaming path's ``done`` frame: the cleaned full answer (the tokens
    already shown may contain unnormalised or invalid citations)."""
    if not (raw or "").strip():
        return {"answer": EMPTY_ANSWER_MESSAGE, "note": "generation returned empty",
                "confidence": CONF_LOW, "citations": [], "invalid_citations": []}
    cleaned = clean_answer(raw, [c.number for c in contexts])
    return {"answer": cleaned.text, "note": _note(grading, len(contexts)),
            "confidence": confidence(grading, len(contexts)),
            "citations": cleaned.citations, "invalid_citations": cleaned.invalid_citations}

