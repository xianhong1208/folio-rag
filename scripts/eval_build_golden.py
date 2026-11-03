"""Build the retrieval golden set (evals/rag/golden_qa.yaml) from the eval corpus.

    python scripts/eval_build_golden.py [--seed 42] [--offline]

Passages are cut from evals/rag/documents/ (a PDF is read from its Markdown source
in evals/rag/sources/), sampled per question type with a fixed seed and spread over
the documents, and turned into questions by the LLM. Every item then has to pass:

  * programmatic checks — every quoted span occurs verbatim in the document, the
    question is in the intended language, keyword queries have 2-4 terms, paraphrases
    do not reuse most of the source wording;
  * an LLM verification pass — the quotes answer the question, the question is
    unambiguous in this corpus, and it fits its type.

Unanswerable questions are checked against the passages a simple lexical search
finds for them (in both languages) and kept only when none answers them.

Everything dropped is logged with its reason in evals/rag/golden_qa_build_log.yaml.
LLM responses are cached in evals/rag/llm_cache.jsonl, so a re-run with the same
seed reproduces the same file without calling the LLM (--offline enforces that).
"""
from __future__ import annotations

import os
import argparse
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evals import retrieval_metrics as rm  # noqa: E402
from evals.llm_cache import CachedLLM, parse_json_block  # noqa: E402
from evals.metrics import simplified_char_ratio  # noqa: E402

DOCS = REPO / "evals" / "rag" / "documents"
SOURCES = REPO / "evals" / "rag" / "sources"
OUT = REPO / "evals" / "rag" / "golden_qa.yaml"
LOG = REPO / "evals" / "rag" / "golden_qa_build_log.yaml"
CACHE = REPO / "evals" / "rag" / "llm_cache.jsonl"

LLM_BASE_URL = "http://localhost:5052/v1"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_API_KEY = os.environ.get("EVAL_LLM_API_KEY", "not-needed")  # set EVAL_LLM_API_KEY for a keyed endpoint

# Accepted questions per type (candidates are over-sampled to survive the checks).
QUOTAS = {
    "zh_factual": 30, "en_factual": 25, "xling_zh2en": 20, "xling_en2zh": 20,
    "keyword": 25, "paraphrase": 25, "table": 20, "multihop": 20, "unanswerable": 20,
}
OVERSAMPLE = {"multihop": 2.5, "unanswerable": 2.0}
DEFAULT_OVERSAMPLE = 1.6

CORPUS_BLURB = (
    "The corpus mixes Taiwanese labour and personal-data laws (Traditional Chinese), U.S. "
    "government web pages (the Constitution, the Bill of Rights, Ready.gov and CISA advice), "
    "and documents of a fictional company, Harborlight Technologies (港光科技): a kettle "
    "user manual, HR and security policies, price and budget tables, and zh-TW meeting "
    "transcripts.")

# Multi-hop: passage A comes from the first document, the second fact from the whole
# second document (short enough to show in full). Same-document pairs need two sections.
MULTIHOP_PAIRS = [
    ("zh-meeting-2026-02-aster-k2-launch.md", "en-aster-k2-kettle-manual.md"),
    ("zh-meeting-2026-02-aster-k2-launch.md", "en-harborlight-2026-price-list.pdf"),
    ("zh-meeting-2026-05-hr-policy-review.md", "en-harborlight-remote-work-policy.md"),
    ("zh-meeting-2026-05-hr-policy-review.md", "zh-harborlight-2026-budget-and-travel.pdf"),
    ("zh-meeting-2026-08-security-incident-review.md", "en-harborlight-incident-response-runbook.md"),
    ("zh-meeting-2026-08-security-incident-review.md", "us-cisa-strong-passwords.md"),
    ("tw-labor-standards-act.md", "tw-rules-of-leave-taking-for-workers.md"),
    ("tw-gender-equality-in-employment-act.md", "tw-rules-of-leave-taking-for-workers.md"),
    ("tw-personal-data-protection-act.md", "tw-personal-data-protection-act-enforcement-rules.md"),
    ("us-constitution.md", "us-bill-of-rights.md"),
    ("en-aster-k2-kettle-manual.md", "en-harborlight-2026-price-list.pdf"),
    ("us-ready-earthquakes.md", "us-ready-build-a-kit.md"),
    ("en-harborlight-incident-response-runbook.md", "zh-harborlight-2026-budget-and-travel.pdf"),
    ("en-harborlight-remote-work-policy.md", "en-harborlight-remote-work-policy.md"),
    ("en-aster-k2-kettle-manual.md", "en-aster-k2-kettle-manual.md"),
    ("zh-harborlight-2026-budget-and-travel.pdf", "zh-harborlight-2026-budget-and-travel.pdf"),
]

TYPE_RULES = {
    "zh_factual": ("Write a natural question in Traditional Chinese (zh-TW, Taiwan usage).",
                   "the question is in Traditional Chinese"),
    "en_factual": ("Write a natural question in English.", "the question is in English"),
    "xling_zh2en": ("The passage is in English, but write the question in Traditional Chinese "
                    "(zh-TW), the way a Taiwanese user would ask it. Quotes stay in English.",
                    "the question is in Traditional Chinese while the passage is English"),
    "xling_en2zh": ("The passage is in Chinese, but write the question in English. Quotes are "
                    "still copied from the Chinese passage.",
                    "the question is in English while the passage is Chinese"),
    "keyword": ("Write a short keyword query of 2-4 terms, as typed into a search box: no "
                "question mark, not a sentence. Use the passage's language ({lang}); for "
                "Chinese separate the 2-4 terms with spaces (for example: 特別休假 未休 工資).",
                "it is a keyword query of 2-4 terms, not a full sentence"),
    "paraphrase": ("Write the question in the passage's language ({lang}) but PARAPHRASE: do "
                   "not reuse the passage's distinctive words or phrases; describe the situation "
                   "in everyday words with synonyms, as a layperson who does not know the "
                   "official terminology would.",
                   "the question avoids most of the distinctive wording of the quotes"),
    "table": ("The passage is one row of a table, shown with the table header. Ask for ONE "
              "specific cell value of this row, naming the row by its label (and the table's "
              "subject when needed). Use the document's language ({lang}). Return spans = "
              "[the row's label cell, the answer cell], each copied exactly from the row.",
              "it asks for one cell value of the row, identified by the row label"),
}

GEN_PROMPT = """You are building a retrieval evaluation set for a search engine. {blurb}

Below is a passage from the document "{title}". Write ONE search question following
these rules:
- {rule}
- It must be answerable from this passage alone, and specific enough to point to this
  passage rather than to similar passages elsewhere in the corpus (name the law,
  product, policy or meeting when that is needed to be unambiguous).
- Never mention file names, "the passage", "this document", "the text" or "the table above".
- Copy {nspans} exact quote(s) from the passage that contain the answer. Copy them
  character for character (same punctuation, no ellipsis); each 4 to 120 characters.

Return only JSON: {{"question": "...", "answer": "short answer", "spans": ["...", ...]}}

Passage:
<<<
{passage}
>>>"""

MULTIHOP_PROMPT = """You are building a retrieval evaluation set for a search engine. {blurb}

Write ONE multi-hop question that can only be answered by combining a fact from
PASSAGE A (from "{title_a}") with a fact from DOCUMENT B (from "{title_b}"){same_doc}.
For example compare two values, compute something from both, or chain a fact from A
into B. Neither source alone may be enough. Write the question in {lang_name}. Never
mention file names, "passage A" or "document B".

If no natural question combines them, return {{"question": null}}.

Otherwise return only JSON: {{"question": "...", "answer": "short answer",
"span_a": "exact quote from passage A", "span_b": "exact quote from document B"}}.
Quotes are copied character for character, 4 to 120 characters each.

PASSAGE A:
<<<
{passage_a}
>>>

DOCUMENT B:
<<<
{doc_b}
>>>"""

VERIFY_PROMPT = """You are checking one item of a retrieval evaluation set. {blurb}

Question type: {qtype}
Question: {question}
Expected answer: {answer}
Evidence quotes:
{quotes}

Source passage(s):
<<<
{passage}
>>>

Keep the item only if ALL hold:
1. The evidence quotes contain the information needed to answer the question correctly.
2. The question is clear and unambiguous for a corpus with many similar documents (other
   laws, policies, meetings, products): it identifies what it is about and does not refer
   to "the passage", "this document", "the text" or "the table above".
3. {type_check}.

Reply with only JSON: {{"keep": true or false, "reason": "short reason"}}"""

UNANSWERABLE_PROMPT = """You are building a retrieval evaluation set for a search engine. {blurb}

The documents are:
{titles}

Write {n} questions in {lang_name} that a user of this knowledge base might plausibly
ask, on the same topics, but whose answer is NOT in these documents: for example a
different product model, a provision of a different law, a decision never taken in the
meetings, a figure the documents do not give. Avoid obviously out-of-scope questions
(sports, weather, celebrities). Vary the topics across the documents.

Return only a JSON list: [{{"question": "...", "translation": "the same question in
{other_lang_name}"}}, ...]"""

UNANSWERABLE_VERIFY = """You are checking whether a search question can be answered from a
knowledge base. {blurb}

Question: {question}

The most similar passages in the knowledge base:
{passages}

Does any passage answer the question, fully or substantially? A passage that is only on
the same topic does not count. Reply with only JSON: {{"answered": true or false,
"reason": "short reason"}}"""

LANG_NAME = {"zh": "Traditional Chinese (zh-TW)", "en": "English"}


# ------------------------------------------------------------------- passages


def doc_source_text(name: str) -> str:
    """The text questions are written from: the file itself, or a PDF's Markdown source."""
    p = DOCS / name
    if p.suffix == ".pdf":
        p = SOURCES / f"{p.stem}.md"
    return p.read_text(encoding="utf-8")


def corpus_docs() -> List[str]:
    return sorted(p.name for p in DOCS.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.name != "SOURCES.md")


def _is_boilerplate(line: str) -> bool:
    s = line.strip()
    return (s.startswith(("Source:", "> 合成資料", "Written for the Folio", "本文件為 Folio"))
            or s in ("---", "------------------------------------------------------------------------"))


def split_passages(doc: str, text: str, max_chars: int = 1000, min_norm: int = 40) -> List[dict]:
    """Cut a Markdown document into prose passages (per heading section, long sections
    split on blank lines) and table-row passages (header + one row)."""
    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), doc)
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    heading, buf = title, []
    for ln in lines:
        if re.match(r"^#{1,4} ", ln):
            sections.append((heading, buf))
            heading, buf = ln.lstrip("#").strip(), []
        else:
            buf.append(ln)
    sections.append((heading, buf))

    out: List[dict] = []
    for heading, body in sections:
        prose: List[str] = []
        i = 0
        while i < len(body):
            ln = body[i]
            if ln.strip().startswith("|") and i + 1 < len(body) and re.match(r"^\s*\|[\s:|-]+\|\s*$", body[i + 1]):
                header = ln.strip()
                j = i + 2
                while j < len(body) and body[j].strip().startswith("|"):
                    row = body[j].strip()
                    out.append({"doc": doc, "title": title, "heading": heading, "kind": "table_row",
                                "text": f"{heading}\n{header}\n{body[i + 1].strip()}\n{row}",
                                "row": row, "lang": rm.text_lang(heading + row)})
                    j += 1
                i = j
                continue
            if not _is_boilerplate(ln):
                prose.append(ln)
            i += 1
        block = "\n".join(prose).strip()
        if not block:
            continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
        cur = ""
        for p in paras:
            if cur and len(cur) + len(p) > max_chars:
                out.append({"doc": doc, "title": title, "heading": heading, "kind": "prose", "text": cur})
                cur = p
            else:
                cur = f"{cur}\n\n{p}" if cur else p
        if cur:
            out.append({"doc": doc, "title": title, "heading": heading, "kind": "prose", "text": cur})
    passages = []
    for p in out:
        if p["kind"] == "prose":
            if len(rm.normalize(p["text"])) < min_norm:
                continue
            p["lang"] = rm.text_lang(p["text"])
            p["text"] = f"{p['heading']}\n{p['text']}" if p["heading"] != title else p["text"]
        passages.append(p)
    for n, p in enumerate(passages):
        p["id"] = f"{doc}#{n:03d}"
    return passages


# ------------------------------------------------------------------ span checks


def find_exact(span: str, text: str) -> Optional[str]:
    """The document's own text for ``span``: verbatim, or modulo whitespace / Markdown
    emphasis (the LLM tends to drop ** and line breaks). None when it does not occur."""
    span = (span or "").strip()
    if len(span) < 2:
        return None
    if span in text:
        return span
    core = re.sub(r"[\s*_`]+", "", span)
    if len(core) < 2:
        return None
    pattern = r"[\s*_`]*".join(re.escape(c) for c in core)
    m = re.search(pattern, text)
    return m.group(0) if m else None


def cellify(span: str) -> List[str]:
    """A quoted Markdown table row becomes its cells: parsers serialise tables differently
    (Docling writes "row, column = value" triplets), so a row quote with pipes never
    matches, while every cell of the row still lands in the same chunk."""
    if "|" not in span:
        return [span.strip()]
    return [c.strip() for c in span.split("|") if c.strip() and not re.fullmatch(r"[\s:-]+", c)]


def strip_heading_spans(spans: List[str], heading: str) -> List[str]:
    """Drop quotes that are only the section heading: headings are often kept as chunk
    metadata rather than chunk text, so they are not evidence a retriever can match."""
    h = rm.normalize(heading)
    return [s for s in spans if not (h and rm.normalize(s) and rm.normalize(s) in h)]


def bigrams(s: str) -> set:
    n = rm.normalize(s)
    return {n[i:i + 2] for i in range(len(n) - 1)}


_EN_STOP = set("""a an the of to in on for and or is are was were be by with what which who whom
whose how when where why do does did can could should would may might must at from as that this
these those it its if under about into than then there their they them i you he she we our your
much many any some""".split())


def word_overlap(question: str, source: str) -> float:
    """Share of the question's content words / character bigrams that occur in ``source``."""
    if rm.text_lang(question) == "zh":
        q = bigrams(question)
        return len(q & bigrams(source)) / len(q) if q else 0.0
    qw = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in _EN_STOP}
    sw = set(re.findall(r"[a-z0-9]+", source.lower()))
    return len(qw & sw) / len(qw) if qw else 0.0


def keyword_terms(q: str) -> int:
    if rm.text_lang(q) == "zh":
        parts = q.split()
        return len(parts) if len(parts) > 1 else (2 if len(rm.normalize(q)) <= 10 else 99)
    return len(q.split())


def programmatic_check(qtype: str, q_lang: str, question: str, spans: List[str]) -> Optional[str]:
    """Reason to drop the item, or None."""
    if not question or len(question.strip()) < 2:
        return "empty question"
    if rm.text_lang(question) != q_lang:
        return f"question language is not {q_lang}"
    if q_lang == "zh" and simplified_char_ratio(question) > 0:
        return "question contains Simplified characters"
    low = question.lower()
    if any(w in low for w in ("the passage", "this document", "the text", "this text",
                              "the table above", "本段", "本文", "上述", "這段")):
        return "question refers to the source"
    if qtype == "keyword":
        if "?" in question or "？" in question or not 2 <= keyword_terms(question) <= 4:
            return "keyword query is not 2-4 terms"
    if qtype == "paraphrase" and word_overlap(question, " ".join(spans)) > 0.5:
        return "paraphrase reuses most of the source wording"
    return None


# ------------------------------------------------------------------ generation


class Builder:
    def __init__(self, llm: CachedLLM, seed: int, workers: int = 8):
        self.llm, self.seed, self.workers = llm, seed, workers
        self.docs = corpus_docs()
        self.doc_text = {d: doc_source_text(d) for d in self.docs}
        self.passages: List[dict] = []
        for d in self.docs:
            self.passages += split_passages(d, self.doc_text[d])
        self.dropped: List[dict] = []
        self.used: set = set()

    # -- helpers
    def _map(self, fn, items):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(fn, items))

    def _json(self, prompt: str, tag: str, max_tokens: int = 2048):
        return parse_json_block(self.llm.complete(prompt, tag=tag, max_tokens=max_tokens))

    def _drop(self, qtype: str, source: str, reason: str, question=None):
        self.dropped.append({"type": qtype, "source": source, "reason": reason,
                             **({"question": question} if question else {})})

    def _sample(self, qtype: str, pool: List[dict], n: int) -> List[dict]:
        seed = self.seed * 1000 + list(QUOTAS).index(qtype)
        got = rm.stratified_sample(pool, n, seed, stratum=lambda p: p["doc"], exclude=self.used)
        if len(got) < n:  # pool exhausted: allow passages other types already used
            more = rm.stratified_sample(pool, n - len(got), seed + 1, stratum=lambda p: p["doc"],
                                        exclude={p["id"] for p in got})
            got += more
        self.used |= {p["id"] for p in got}
        return got

    def _verify(self, qtype: str, item: dict, passage: str) -> Tuple[bool, str]:
        rule = (TYPE_RULES[qtype][1] if qtype in TYPE_RULES
                else "both quotes are needed: neither quote alone answers the question")
        quotes = "\n".join(f"- ({e['file']}) {s}" for e in item["evidence"] for s in e["spans"])
        v = self._json(VERIFY_PROMPT.format(
            blurb=CORPUS_BLURB, qtype=qtype, question=item["question"],
            answer=item.get("answer", ""), quotes=quotes, passage=passage,
            type_check=rule), tag=f"verify:{qtype}", max_tokens=1024)
        if not isinstance(v, dict) or "keep" not in v:
            return False, "verifier returned no verdict"
        return bool(v["keep"]), str(v.get("reason", ""))[:200]

    # -- single-passage types
    def single(self, qtype: str) -> List[dict]:
        quota = QUOTAS[qtype]
        n_cand = math.ceil(quota * OVERSAMPLE.get(qtype, DEFAULT_OVERSAMPLE))
        prose = [p for p in self.passages if p["kind"] == "prose"]
        if qtype == "zh_factual":
            pool = [p for p in prose if p["lang"] == "zh"]
        elif qtype in ("en_factual", "xling_zh2en"):
            pool = [p for p in prose if p["lang"] == "en"]
        elif qtype == "xling_en2zh":
            pool = [p for p in prose if p["lang"] == "zh"]
        elif qtype == "table":
            pool = [p for p in self.passages if p["kind"] == "table_row"]
        else:
            pool = prose
        cands = self._sample(qtype, pool, n_cand)

        def q_lang_for(p):
            return {"zh_factual": "zh", "en_factual": "en", "xling_zh2en": "zh",
                    "xling_en2zh": "en"}.get(qtype, p["lang"])

        def work(p):
            ql = q_lang_for(p)
            rule = TYPE_RULES[qtype][0].format(lang=LANG_NAME[p["lang"]])
            g = self._json(GEN_PROMPT.format(
                blurb=CORPUS_BLURB, title=p["title"], rule=rule,
                nspans="two" if qtype == "table" else "one or two",
                passage=p["text"]), tag=f"gen:{qtype}")
            if not isinstance(g, dict) or not g.get("question"):
                return None, "generator returned no question"
            q = str(g["question"]).strip()
            spans = g.get("spans") or []
            if not isinstance(spans, list) or not spans:
                return None, "no spans"
            exact = []
            for s in spans[:2]:
                e = find_exact(str(s), p["row"] if qtype == "table" else p["text"])
                if e is None:
                    return {"question": q}, f"span not found verbatim: {str(s)[:60]}"
                if qtype != "table" and len(rm.normalize(e)) < 4:
                    return {"question": q}, "span too short"
                exact.append(e.strip(" |"))
            if qtype != "table":
                exact = strip_heading_spans(exact, p["heading"])
                if not exact:
                    return {"question": q}, "span is only the section heading"
            reason = programmatic_check(qtype, ql, q, exact)
            if reason:
                return {"question": q}, reason
            item = {"type": qtype, "lang": ql, "question": q,
                    "answer": str(g.get("answer", "")).strip(),
                    "evidence": [{"file": p["doc"], "spans": exact}],
                    "source": f"{p['doc']} :: {p['heading']}"}
            ok, why = self._verify(qtype, item, p["text"])
            return (item, None) if ok else ({"question": q}, f"verifier: {why}")

        results = self._map(work, cands)
        kept = []
        for p, (item, reason) in zip(cands, results):
            if reason is None and len(kept) < quota:
                kept.append(item)
            elif reason is not None:
                self._drop(qtype, p["id"], reason, (item or {}).get("question"))
        return kept

    # -- multi-hop
    def multihop(self) -> List[dict]:
        quota = QUOTAS["multihop"]
        n_cand = math.ceil(quota * OVERSAMPLE["multihop"])
        rng = random.Random(self.seed * 1000 + list(QUOTAS).index("multihop"))
        pairs = list(MULTIHOP_PAIRS)
        rng.shuffle(pairs)
        by_doc: Dict[str, List[dict]] = {}
        for p in self.passages:
            if p["kind"] == "prose":
                by_doc.setdefault(p["doc"], []).append(p)
        for d in by_doc:
            rng.shuffle(by_doc[d])
        cands = []
        for i in range(n_cand):
            a_doc, b_doc = pairs[i % len(pairs)]
            pa = by_doc[a_doc][(i // len(pairs)) % len(by_doc[a_doc])]
            cands.append((pa, b_doc))

        def work(c):
            pa, b_doc = c
            lang = pa["lang"]
            same = pa["doc"] == b_doc
            g = self._json(MULTIHOP_PROMPT.format(
                blurb=CORPUS_BLURB, title_a=pa["title"],
                title_b=next((ln[2:] for ln in self.doc_text[b_doc].splitlines()
                              if ln.startswith("# ")), b_doc),
                same_doc=(" (the same document: the fact from B must come from a different "
                          "section than passage A)") if same else "",
                lang_name=LANG_NAME[lang], passage_a=pa["text"],
                doc_b=self.doc_text[b_doc][:9000]), tag="gen:multihop", max_tokens=3072)
            if not isinstance(g, dict) or not g.get("question"):
                return None, "generator found no natural multi-hop question"
            q = str(g["question"]).strip()
            ea = find_exact(str(g.get("span_a", "")), pa["text"])
            eb = find_exact(str(g.get("span_b", "")), self.doc_text[b_doc])
            if not ea or not eb:
                return {"question": q}, "span not found verbatim"
            if same and (eb in pa["text"] or rm.normalize(eb) in rm.normalize(pa["text"])):
                return {"question": q}, "second span is inside passage A"
            reason = programmatic_check("multihop", lang, q, [ea, eb])
            if reason:
                return {"question": q}, reason
            sa = strip_heading_spans(cellify(ea), pa["heading"])
            sb = cellify(eb)
            if not sa or not sb:
                return {"question": q}, "span is only the section heading"
            item = {"type": "multihop", "lang": lang, "question": q,
                    "answer": str(g.get("answer", "")).strip(),
                    "evidence": [{"file": pa["doc"], "spans": sa},
                                 {"file": b_doc, "spans": sb}],
                    "source": f"{pa['doc']} :: {pa['heading']} + {b_doc}"}
            ok, why = self._verify("multihop", item, pa["text"] + "\n...\n" + self._context(b_doc, eb))
            return (item, None) if ok else ({"question": q}, f"verifier: {why}")

        results = self._map(work, cands)
        kept = []
        for (pa, b_doc), (item, reason) in zip(cands, results):
            if reason is None and len(kept) < quota:
                kept.append(item)
            elif reason is not None:
                self._drop("multihop", f"{pa['id']} + {b_doc}", reason, (item or {}).get("question"))
        return kept

    def _context(self, doc: str, span: str, width: int = 400) -> str:
        t = self.doc_text[doc]
        i = t.find(span)
        return t[max(0, i - width): i + len(span) + width] if i >= 0 else span

    # -- unanswerable
    def lexical_top(self, query: str, k: int = 6) -> List[dict]:
        qb = bigrams(query)
        scored = []
        for p in self.passages:
            pb = bigrams(p["text"])
            if qb and pb:
                scored.append((len(qb & pb) / math.sqrt(len(qb) * len(pb)), p["id"], p))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [p for _, _, p in scored[:k]]

    def unanswerable(self) -> List[dict]:
        quota = QUOTAS["unanswerable"]
        per_lang = math.ceil(quota * OVERSAMPLE["unanswerable"] / 2)
        titles = "\n".join(f"- {next((ln[2:] for ln in self.doc_text[d].splitlines() if ln.startswith('# ')), d)}"
                           for d in self.docs)
        cands = []
        for lang, other in (("zh", "en"), ("en", "zh")):
            lst = self._json(UNANSWERABLE_PROMPT.format(
                blurb=CORPUS_BLURB, titles=titles, n=per_lang, lang_name=LANG_NAME[lang],
                other_lang_name=LANG_NAME[other]), tag=f"gen:unanswerable:{lang}", max_tokens=4096)
            for c in (lst or [])[:per_lang]:
                if isinstance(c, dict) and c.get("question"):
                    cands.append((lang, str(c["question"]).strip(), str(c.get("translation", ""))))
        # interleave zh / en so the kept set is balanced
        zh = [c for c in cands if c[0] == "zh"]
        en = [c for c in cands if c[0] == "en"]
        cands = [c for pair in zip(zh, en) for c in pair] + zh[len(en):] + en[len(zh):]

        def work(c):
            lang, q, tr = c
            reason = programmatic_check("unanswerable", lang, q, [])
            if reason:
                return reason
            seen, ps = set(), []
            for p in self.lexical_top(q) + (self.lexical_top(tr) if tr else []):
                if p["id"] not in seen:
                    seen.add(p["id"])
                    ps.append(p)
            passages = "\n\n".join(f"[{i + 1}] ({p['doc']})\n{p['text'][:1200]}" for i, p in enumerate(ps))
            v = self._json(UNANSWERABLE_VERIFY.format(blurb=CORPUS_BLURB, question=q, passages=passages),
                           tag="verify:unanswerable", max_tokens=1024)
            if not isinstance(v, dict) or "answered" not in v:
                return "verifier returned no verdict"
            return f"answerable: {str(v.get('reason', ''))[:160]}" if v["answered"] else None

        results = self._map(work, cands)
        kept = []
        for (lang, q, _), reason in zip(cands, results):
            if reason is None and len(kept) < quota:
                kept.append({"type": "unanswerable", "lang": lang, "question": q, "answer": None,
                             "evidence": []})
            elif reason is not None:
                self._drop("unanswerable", "-", reason, q)
        return kept


HEADER = """\
# Folio retrieval golden set — generated by scripts/eval_build_golden.py (seed {seed}).
# Do not hand-edit entries in place: fix the builder or the corpus and regenerate, or
# add hand-written items to the end with ids starting "h-".
#
# Each item: id, type, lang (of the question), question, answer (short reference
# answer, for reviewers), evidence (file in evals/rag/documents + exact quoted spans
# from it; empty for unanswerable questions), expected_files (for
# scripts/run_rag_eval.py), source (where the question was generated from).
# Gold is never a chunk id: scripts/eval_retrieval.py matches a returned chunk when it
# contains (most of) every span of an evidence item.
#
# Types: zh_factual, en_factual, xling_zh2en (zh question, English source),
# xling_en2zh (English question, zh source), keyword, paraphrase, table, multihop,
# unanswerable. Counts: {counts}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--offline", action="store_true", help="fail on an LLM cache miss")
    args = ap.parse_args()
    llm = CachedLLM(CACHE, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, offline=args.offline)
    b = Builder(llm, args.seed, args.workers)
    print(f"{len(b.passages)} passages from {len(b.docs)} documents")
    items: List[dict] = []
    for qtype in QUOTAS:
        if qtype == "multihop":
            got = b.multihop()
        elif qtype == "unanswerable":
            got = b.unanswerable()
        else:
            got = b.single(qtype)
        print(f"  {qtype}: kept {len(got)}/{QUOTAS[qtype]}")
        items += got
    out = []
    for n, it in enumerate(items, 1):
        files = list(dict.fromkeys(e["file"] for e in it["evidence"]))
        row = {"id": f"q{n:03d}", "type": it["type"], "lang": it["lang"],
               "question": it["question"], "answer": it["answer"],
               "evidence": it["evidence"], "expected_files": files}
        if it.get("source"):
            row["source"] = it["source"]
        out.append(row)
    counts = {}
    for it in out:
        counts[it["type"]] = counts.get(it["type"], 0) + 1
    OUT.write_text(HEADER.format(seed=args.seed, counts=counts) + "\n" +
                   yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=1000),
                   encoding="utf-8")
    drop_counts: Dict[str, Dict[str, int]] = {}
    for d in b.dropped:
        key = d["reason"].split(":")[0]
        drop_counts.setdefault(d["type"], {}).setdefault(key, 0)
        drop_counts[d["type"]][key] += 1
    LOG.write_text("# Items dropped while building golden_qa.yaml (seed %d)\n" % args.seed +
                   yaml.safe_dump({"kept": counts, "dropped_by_reason": drop_counts,
                                   "dropped": b.dropped},
                                  allow_unicode=True, sort_keys=False, width=1000),
                   encoding="utf-8")
    print(f"wrote {len(out)} questions to {OUT.relative_to(REPO)}; dropped {len(b.dropped)} "
          f"(log: {LOG.relative_to(REPO)}); LLM cache hits {llm.hits}, misses {llm.misses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
