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
- **Material does not go in the repo** (documents/audio may contain internal data): already excluded by `.gitignore`; keep the material in shared storage and note the path in the README.
- Any change on the retrieval side (embedding / chunking / retrieval parameters / ASR) -> score before merging.
