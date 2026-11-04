# Evals — RAG Retrieval and ASR Evaluation


**Purpose**: an acceptance net for every model / tuning decision. Run it before and after each change — switching the embedding model, switching ASR (FireRedASR), or tuning `ef_search` — and let the numbers decide.

## Layout

```
evals/
  metrics.py                  # Metric pure functions (CER / simplified-char rate / recall@k / MRR; unit-tested)
  rag/
    documents/                # <- Representative document set (any format in the allowlist)
    golden_qa.yaml            # <- QA set (format described in the file's comments)
  asr/
    audio/                    # <- 3-5 representative audio clips (meeting / presentation / with leading silence)
    drafts/                   # Machine drafts (produced by whisper; a proofreading starting point only, never scored)
    golden_transcripts/       # <- Same-named .txt: only **human-proofread** full Traditional-Chinese text belongs here
  baselines/                  # Scoring output (JSON; baselines are committed to the repo)
```

## Retrieval golden set (public corpus, HTTP harness)

The main retrieval benchmark. Unlike the rest of `evals/`, its material is public and
committed: `rag/documents/` holds ~20 documents that can live in an MIT repo (Taiwanese
laws, U.S. government pages, fictional company documents, two PDFs with tables and
three synthetic zh-TW meeting transcripts; sources and licenses in
`rag/documents/SOURCES.md`).

| File | What it is |
|---|---|
| `rag/golden_qa.yaml` | ~200 questions, stratified by type (zh/en factual, cross-lingual both ways, keyword, paraphrase, table, multi-hop, ~10% unanswerable). Gold = file + exact quoted spans, never chunk ids. |
| `rag/golden_qa_build_log.yaml` | every generated question that was dropped, with the reason |
| `rag/out_of_scope_qa.yaml` | 20 hand-written questions about subjects the corpus does not cover (type `out_of_scope`); they feed only the relevance-cutoff section |
| `rag/llm_cache.jsonl` | cached LLM responses — rebuilding the set or the transcripts replays them |
| `rag/sources/` | Markdown sources of the PDFs |
| `retrieval_metrics.py` | span matching, Recall/MRR/nDCG, token estimate, duplicate ratio, sampling (unit-tested) |
| `baselines/rag-<date>-<label>.{json,md}` | committed results |

```bash
# score the running server (indexes the corpus into folder folio_eval_golden_v1 first,
# reusing it when every file's sha256 matches); writes evals/baselines/rag-<date>-dev-<sha>.*
.venv/bin/python scripts/eval_retrieval.py --token-file <pat.json> --runs 2

# rebuild material (deterministic: cached LLM responses, fixed seed)
.venv/bin/python scripts/eval_fetch_laws.py            # laws from law.moj.gov.tw
.venv/bin/python scripts/eval_make_corpus.py --offline # PDFs + synthetic transcripts
.venv/bin/python scripts/eval_build_golden.py --offline
```

`<pat.json>` is `{"token": "...", "jti": "..."}` for an MCP Gate PAT issued for Folio
(with `rag:admin` so the settings snapshot is included). The eval folder belongs to
that token; use the same token for every run so the folder is reused.

What the harness measures, per question type and question language:

- **mcp** — the MCP search tool with its defaults (what an agent receives); **mcp_k20**
  — the same with `top_k=20`; **trace** — the console retrieval trace (`top_k=20`):
  fused + reranked leaves before auto-merge / expansion, with each hit's rank in the
  dense, BM25 and fused candidate lists (reported as lower bounds on stage recall).
- chunk-level Recall@5/10/20, MRR@10, nDCG@10: a result matches an evidence item when
  its text contains ≥60% of every quoted span (8-character windows after normalising
  away whitespace, punctuation and Markdown), or when it is the leaf next to the chunk
  holding the span. File-level Recall@5.
- unanswerable questions: false-accept rate = top score ≥ 0.25 / 0.5.
- relevance cutoff (`rag.retrieval.min_relevance`): for each cutoff in a sweep, the share
  of near-miss unanswerable (`golden_qa.yaml`) and out-of-scope questions it would report
  as "nothing relevant", the share of answerable questions it would wrongly reject, and
  Recall@5/10 if rejected searches returned nothing; plus the `low_relevance` flag as the
  server set it. Uses `top_relevance` from the MCP response (harness v2; `--no-out-of-scope`
  skips the extra set). False-accept and response size stay over the golden set only.
- MCP response size (estimated tokens: CJK 0.7/char, other 0.25/char), duplicate-text
  ratio, p50/p95 latency; run-to-run variance over `--runs`.
- gold reachability: evidence spans the indexed chunks no longer contain (parser loss).

## Material Source A: Common Voice zh-TW (no proofreading, recommended)

Mozilla Common Voice Scripted Speech (zh-TW, CC0-1.0) ships with **human-verified** per-sentence text, so no proofreading is needed. After logging in to the Mozilla Data Collective and downloading the tar.gz:

```bash
uv run --no-sync python scripts/import_common_voice.py <tar.gz path> --n 200
uv run --no-sync python scripts/run_asr_eval.py
```

The sampling seed is fixed (`--seed 42`) for reproducibility; re-running automatically clears the old `cv_*` batch.
Limitation: CV consists of read-aloud short sentences (phonetically clean). It measures a model's baseline CER but does not represent real-world performance on meetings / long audio, so recording your own long audio (Material Source B) is still worthwhile.

## Material Source B: Your Own Audio — Draft -> Proofread Workflow

Transcribing 20 minutes of audio from scratch is impractical, so start from a machine draft:

1. Generate drafts (whisper turbo -> `drafts/*.txt`):
   `uv run --no-sync python scripts/make_asr_drafts.py`
2. **Proofread** the contents of `drafts/` (listen to the audio and fix errors; once a FireRedASR draft is ready you can compare side by side and prioritize listening where the two disagree).
3. Once proofread, move the files into `golden_transcripts/` under the same names.

Warning: putting unproofread machine output into `golden_transcripts/` means CER is measured against whisper's own errors, distorting the scores of both ASR engines. This line is the floor of evaluation credibility.

## Running

```bash
# RAG (requires reachable PostgreSQL + embedding service; creates its own eval folder and cleans up afterward)
uv run --no-sync python scripts/run_rag_eval.py
uv run --no-sync python scripts/run_rag_eval.py --baseline evals/baselines/rag_<date>.json

# ASR (against the configured provider; --provider overrides)
uv run --no-sync python scripts/run_asr_eval.py
uv run --no-sync python scripts/run_asr_eval.py --provider openai-compatible \
    --baseline evals/baselines/asr_docling-whisper_<date>.json
```

## Metrics

| Script | Metric | Description |
|--------|--------|-------------|
| RAG | recall@5 / recall@10 | Fraction of expected files hit within the top-k retrieval results |
| RAG | MRR | Reciprocal rank of the first hit file (ranking quality) |
| ASR | CER | Character error rate (whitespace and punctuation stripped; normalization in metrics.py) |
| ASR | Simplified-char rate | For Traditional-Chinese output, acceptance is ~0 (catches cases where FireRedASR emits simplified characters) |
| ASR | Hallucination hit count | Count of audio_defense pattern hits against the provider's raw output |

## Conventions

- **Baselines go in the repo**: commit `baselines/*.json` for every meaningful run and attach the diff in the MR description.
- **Material does not go in the repo** (documents/audio may contain internal data): already excluded by `.gitignore`; keep the material in shared storage and note the path in the README. The exception is `rag/documents/`, the public golden-set corpus (its own `.gitignore` re-includes it): add only files you can publish under MIT and record each in `SOURCES.md`.
- Any change on the retrieval side (embedding / chunking / retrieval parameters / ASR) -> score before merging.
