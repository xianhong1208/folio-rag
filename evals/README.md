# Evals — RAG Retrieval, Page Reading (Vision Model) and ASR Evaluation


**Purpose**: an acceptance net for every model / tuning decision. Run it before and after each change — switching the embedding model, switching the vision model that reads PDF pages and images, switching the speech-to-text model or endpoint, or tuning `ef_search` — and let the numbers decide.

## Layout

```
evals/
  metrics.py                  # Metric pure functions (CER / char F1 / simplified-char rate / recall@k / MRR; unit-tested)
  rag/
    documents/                # <- Representative document set (any format in the allowlist)
    golden_qa.yaml            # <- QA set (format described in the file's comments)
  asr/
    audio/                    # <- 3-5 representative audio clips (meeting / presentation / with leading silence)
    drafts/                   # Machine drafts (from the rag.asr endpoint; a proofreading starting point only, never scored)
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
  dense, BM25 and fused candidate lists (reported as lower bounds on stage recall);
  **documents** (harness v3) — `POST /api/search` with `group_by: document`, top 10
  documents, scored against each question's source files: document recall@1/3/5/10 and
  MRR@10 (*Results by document* section).
- `--split N` spreads the corpus over N folders (`folio_eval_golden_v1_partIofN`,
  round-robin by file name) and asks every question once across them through
  `POST /api/search`: passages (top 20, no auto-merge — compare with the trace path of a
  one-folder run) and documents. It writes `rag-<date>-<label>-splitN.{json,md}`.
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

## Page reading: the vision model (`scripts/eval_ocr.py`)

How well the model behind `rag.parsing.vision` reads PDF pages. Every page of the two public
PDFs in `rag/documents/` (a Traditional Chinese budget and an English price list, both with
tables) is rendered at `rag.parsing.dpi` and sent with Folio's own transcription prompt; the
reference is the page's text layer. `--degrade` turns each render into a cheap office scan
first (grey, 1.5° skew, 120 dpi, blur, speckles, JPEG quality 35), which is where OCR fails.
Scores are CER and an order-free character F1 after dropping whitespace and Markdown table
syntax (`evals/metrics.py`).

```bash
VISION_BASE_URL=https://<forgeyard-host>:7860/v1 VISION_MODEL=qwen3-vl-8b \
VISION_API_KEY=<key> VISION_CA_FILE=forgeyard-ca.pem \
.venv/bin/python scripts/eval_ocr.py --degrade --out evals/baselines/ocr-<date>-degraded.json
```

Measured 2026-11-18 (the decision to drop docling's OCR, #7): Qwen3-VL-8B on Forgeyard read the
4 clean and 4 degraded pages with no misread character and kept the tables as Markdown tables;
docling's RapidOCR (PP-OCRv6) fell to a character F1 of 0.935–0.970 on the degraded Chinese pages.
With this script (2026-11-19, Qwen3-VL-8B on Forgeyard, mean char F1 / CER): clean (150 dpi) 0.9994 / 0.003;
`--degrade` at 200 dpi 0.9947 / 0.012; `--degrade` at 150 dpi (≈ 90 dpi after the scan's 0.6
shrink) 0.9639 / 0.067, where the model left out the top block of the second Chinese page
(F1 0.876). That is why `rag.parsing.dpi` defaults to 200 (an A4 page is then about 3.9 MP, inside
a 4096-image-token limit such as Forgeyard's); very low-resolution scans may still need more.
Without a vision model Folio reads these PDFs from their text layer (the reference itself), so
the eval is only meaningful for the model.

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

1. Generate drafts with the configured `rag.asr` endpoint (-> `drafts/*.txt`):
   `uv run --no-sync python scripts/make_asr_drafts.py`
   A second model's drafts are a useful second opinion:
   `uv run --no-sync python scripts/make_asr_drafts.py --model <other-model> --suffix other`
2. **Proofread** the contents of `drafts/` (listen to the audio and fix errors; where two drafts disagree is where to listen first).
3. Once proofread, move the files into `golden_transcripts/` under the same names.

Warning: putting unproofread machine output into `golden_transcripts/` means CER is measured against whisper's own errors, distorting the scores of every model compared. This line is the floor of evaluation credibility.

## Running

```bash
# RAG (requires reachable PostgreSQL + embedding service; creates its own eval folder and cleans up afterward)
uv run --no-sync python scripts/run_rag_eval.py
uv run --no-sync python scripts/run_rag_eval.py --baseline evals/baselines/rag_<date>.json

# ASR (against the rag.asr OpenAI-compatible endpoint: ASR_BASE_URL / ASR_API_KEY / ASR_MODEL;
# --model overrides the model; Folio has no in-process speech-to-text since 1.5)
uv run --no-sync python scripts/run_asr_eval.py
uv run --no-sync python scripts/run_asr_eval.py --model whisper-large-v3-turbo \
    --baseline evals/baselines/asr_<model>_<date>.json
```

## Metrics

| Script | Metric | Description |
|--------|--------|-------------|
| RAG | recall@5 / recall@10 | Fraction of expected files hit within the top-k retrieval results |
| RAG | MRR | Reciprocal rank of the first hit file (ranking quality) |
| ASR | CER | Character error rate (whitespace and punctuation stripped; normalization in metrics.py) |
| ASR | Simplified-char rate | For Traditional-Chinese output, acceptance is ~0 (whisper writes Simplified; the provider converts it per `rag.asr.chinese_conversion`) |
| ASR | Hallucination hit count | Count of audio_defense pattern hits against the provider's raw output |

## Conventions

- **Baselines go in the repo**: commit `baselines/*.json` for every meaningful run and attach the diff in the MR description.
- **Material does not go in the repo** (documents/audio may contain internal data): already excluded by `.gitignore`; keep the material in shared storage and note the path in the README. The exception is `rag/documents/`, the public golden-set corpus (its own `.gitignore` re-includes it): add only files you can publish under MIT and record each in `SOURCES.md`.
- Any change on the retrieval side (embedding / chunking / retrieval parameters / ASR) -> score before merging.
