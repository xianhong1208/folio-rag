# Configuration

[繁體中文](configuration.zh-TW.md)

Settings live in `config/config.yaml`; secrets come from environment variables (`.env`). The ones you are likely to touch:

| Variable / setting | Default | Purpose |
|--------------------|---------|---------|
| `DATABASE_URL` | `postgresql://…/agentic_rag` | PostgreSQL DSN. Requires the **pgvector** extension. Use **sync** `postgresql://` (psycopg2), not `+asyncpg`. |
| `server.port` | `5039` | HTTP + MCP bind port. The MCP endpoint is `/mcp`. |
| `FOLIO_STORAGE_ROOT` | the project root | Directory that holds `storage/` (uploaded files and the console session secret). |
| `FOLIO_INDEXING_CONCURRENCY`, `FOLIO_PER_FILE_TIMEOUT_SECONDS`, `FOLIO_JOB_TIMEOUT_SECONDS` | from `rag.indexing` | Runtime overrides for the indexing limits. The pre-rename `RAG_*` names still work and log a deprecation warning. |
| `auth.enabled` / `auth.issuer` / `auth.audience` | `true` / MCP Gate URL / this server's `/mcp` URL | OAuth 2.1 verification against MCP Gate. |
| `auth.console.session_hours` | `8` | Longest a console session lasts (it also ends when the token expires). |
| `auth.console.session_secret` | *(generated)* | Key that signs the session cookie; `${ENV_VAR}` works. Unset → generated once into `storage/.console_session_secret` (gitignored), so sessions survive restarts. |
| `auth.console.secure_cookie` | `auto` | Cookie `Secure` flag: `auto` = only over https; `true` behind a TLS-terminating proxy. |
| `auth.console.admin_scope` | `rag:admin` | Scope that unlocks the console's global settings. |
| `auth.revocation.*` | disabled | MCP Gate revocation feed, so revoked tokens stop working before they expire — see [Token revocation](auth.md#token-revocation). |
| `rag.embedding.{model,dimension,base_url}` | `intfloat/multilingual-e5-large`, `1024`, `:7075/v1` | Embedding model; `dimension` **must** equal the model's real output dim. |
| `rag.llm.{model,base_url}` | `openai/gpt-oss-120b`, `:5052/v1` | LLM for Contextual Retrieval and the `answer` path. |
| `rag.rerank.{model,base_url}` | `mixedbread-ai/mxbai-rerank-base-v2`, `:8787` | Cross-encoder reranker. |
| `rag.rerank.score_threshold` / `auto_template` | `0.0` / `true` | Drop results scoring below the threshold (0.5 suits mxbai-rerank-v2); apply mxbai-v2's chat prompt automatically. |
| `rag.rerank.gap_threshold` | `0.5` | Knowledge-gap analytics: best rerank score below this counts as "no good match". |
| `rag.retrieval.rerank_pool` | `per_list` | What the cross-encoder scores: `per_list` = the dense top `rerank_dense_candidates` + the BM25 top `rerank_sparse_candidates`; `fused` = the top `max(3 × top_k, rerank_candidates)` of the fused list (the previous behaviour). Live. |
| `rag.retrieval.rerank_dense_candidates` / `rerank_sparse_candidates` | `30` / `10` | `per_list` pool depths (dense never below 3 × `top_k`). Higher = more recall margin, ~6 ms per extra pair; live-tunable (0–200). |
| `rag.retrieval.rerank_candidates` | `60` | `fused` pool only: `max(3 × top_k, this)` fused candidates go to the cross-encoder; live-tunable (1–200). |
| `rag.retrieval.mcp_response_token_budget` | `6000` | Estimated-token cap on an MCP search response; lowest-ranked results are dropped to fit and the response says so. `0` = no cap. |
| `rag.retrieval.min_relevance` | `0.9` in the shipped config (`0` when unset) | MCP search: best rerank score below this → `low_relevance: true` (flagged, not dropped). Model-specific; `0` = off. |
| `rag.asr.provider` | `fireredasr` | Audio transcription: `fireredasr` (local zh-TW) · `docling-whisper` · `openai-compatible`. `rag.asr.enabled: false` fails audio files with a clear message (nothing is transcribed). |
| `rag.retrieval.segmenter` | `ckip` (`bigram` under `profile: cloud`) | Chinese segmenter for keyword search: `ckip` · `jieba` · `bigram` (see *Chinese segmentation* in [How Folio works](how-it-works.md)). |
| `rag.docling.ocr_enabled` | `true` | OCR for scanned PDFs and images. `false` really skips it (restart to apply); a scan with no text layer then fails with a clear message. |
| `rag.contextual_retrieval.max_doc_chars` | `60000` | Upper bound on the document text the LLM reads per chunk. It reads the outline plus the chunk's own section, windowed to at most 12,000 characters. |
| `rag.retrieval.*` | — | Top-K, similarity cutoff, BM25 candidates, `hybrid_fusion` (`rrf`/`concat`), auto-merging. Tunable live from the console, or via the **High Precision / Balanced / High Recall** presets. |
| `rag.profile` | `local` | `local` · `hybrid` · `cloud` — see [Cloud profile](#cloud-profile). |
| `rag.<role>.{timeout,max_retries}` | per role | Request timeout and retries (429 / 5xx / timeouts, with backoff) for each model role. |

GPU selection is a `uv sync --extra local --group {rocm-r714|cuda|cpu}` choice — see the top of `pyproject.toml`, where each group is mapped to its PyTorch index. `cuda` installs the CUDA 13.0 build, which needs an NVIDIA driver 580 or newer.

## Cloud profile

Run with every model behind an API and nothing loaded locally — no GPU, no torch, no model
downloads. Set `rag.profile: cloud` and give each role an OpenAI-compatible endpoint:

```bash
uv sync                      # core install: no torch / transformers (the local stack is `--extra local`)
export OPENAI_API_KEY=sk-...  # keys come from the environment, never from config.yaml
uv run python main.py
```

```yaml
rag:
  profile: cloud            # local (default) | hybrid | cloud

  embedding:                # changing it means re-indexing existing folders
    provider: openai        # openai | openai-compatible | azure | vllm | ollama
    model: text-embedding-3-small
    dimension: 1536
    base_url: https://api.openai.com/v1      # any OpenAI-compatible gateway works
    api_key: "${OPENAI_API_KEY}"
    timeout: 60
    max_retries: 3

  llm:                      # default for every LLM role
    provider: openai
    model: gpt-4o-mini
    api_key: "${OPENAI_API_KEY}"
  contextual_llm:           # optional per-role overrides; unset fields inherit from llm
    model: gpt-4.1-nano
  # answer_llm: {...}       # Search Playground answers
  # eval_llm: {...}         # eval question generation / judging

  contextual_retrieval:
    enabled: true
    max_calls_per_file: 2000     # cost guards (0 = unlimited)
    max_tokens_per_day: 5000000

  rerank:
    enabled: true
    api: cohere             # score (vLLM /v1/score) | cohere | jina | generic (OpenAI-compatible /rerank)
    model: rerank-v3.5
    base_url: https://api.cohere.com
    endpoint_path: /v2/rerank
    api_key: "${COHERE_API_KEY}"

  parsing:                  # cloud is the default under profile: cloud
    vision:
      provider: openai
      model: gpt-4o-mini
      api_key: "${OPENAI_API_KEY}"
    max_concurrency: 4      # page requests in flight, process-wide
    max_pages_per_file: 500
    max_pages_per_day: 5000
    on_page_failure: mark   # mark = keep the file, mark the page | fail

  asr:
    provider: openai-compatible
    base_url: https://api.openai.com/v1
    model: whisper-1
    api_key: "${OPENAI_API_KEY}"
    segment_seconds: 600    # long audio is uploaded in silence-aligned pieces

  retrieval:
    hybrid_search: true
    # segmenter: bigram     # the cloud default (no model); jieba also works, ckip is rejected
```

What the profile does:

- **Validation.** `cloud` refuses in-process providers (`huggingface` embedding, `fireredasr` /
  `docling-whisper` ASR, local PDF/image parsing) and **rejects API keys that are not `${ENV}`
  references** — a cloud key is a billable credential and config.yaml is the file that gets
  copied around. An explicit `rag.retrieval.segmenter: ckip` is rejected too. Startup fails with the offending paths. `hybrid` only warns about literal keys;
  `local` (the default — existing configs are unchanged) checks nothing new.
- **Nothing local is imported.** The HuggingFace embedding provider, docling's layout/OCR
  models, Whisper, FireRedASR and CKIP are imported only when selected; startup skips the docling
  warm-up and logs where every role runs. `tests/test_cloud_profile_no_local_models.py` proves it
  by running the whole pipeline on an install without the local stack.
- **Cloud PDF and image parsing.** Every page is transcribed to Markdown by the vision model,
  and the pages are stitched into **one** document before chunking, so headings, breadcrumbs and
  tables continue across page breaks. Each page is sent the previous page's tail and the current
  heading path; pages run in ordered blocks for bounded concurrency; failed pages are retried with
  backoff and, if they still fail, marked in the text instead of failing the file. Chunks keep the
  page they start on for citations. Per-file and per-day page caps bound the cost. The
  unauthenticated `/v1/ocr/general` endpoint is not routed to the paid model.
- **Cloud speech-to-text.** Audio is resampled and uploaded in segments cut at silences (so no
  request exceeds an upload limit), with retries, and re-asked when a transcript is empty despite
  speech or loops.
- **Retries and timeouts per role** (`timeout`, `max_retries`): 429 / 5xx / timeouts back off
  exponentially with jitter and honour `Retry-After`.
- **Chinese BM25.** CKIP is a local torch model, so the cloud profile segments Chinese with the
  model-free `bigram` segmenter by default; `jieba` (`uv sync --extra jieba`) is the other choice.
  An explicit `rag.retrieval.segmenter: ckip` is rejected — keep CKIP with `profile: hybrid` and
  `uv sync --extra local`.

**Console.** *Settings* opens with a **Model Roles** panel: where each role runs, its provider,
model and endpoint, whether a key is set and which environment variable it comes from (keys are
never returned), and a **Test** button that sends one minimal real request with the saved settings
(an embedding — dimension checked —, a 16-token chat, an 8×8 image for the vision model, one rerank
pair, half a second of silence for speech-to-text). Key fields accept `${ENV_VAR}`; the server
stores the reference, never the key. Changes need the `rag:admin` scope.

Daily caps are kept in memory and reset at UTC midnight or on restart — they are a runaway guard,
not billing.

**Chunk token counting** never downloads anything: it uses `assets/hf_tokenizers/<org>--<model>` when present, else the bundled XLM-RoBERTa tokenizer in `resources/tokenizers/` (the tokenizer of multilingual-e5 and bge-m3), so a fresh clone with OpenAI or Azure embeddings works out of the box.

**Parsing failures are visible.** If docling cannot parse some pages (for example the GPU runs out of memory inside a stage), the parse is retried with a smaller batch; pages still missing are listed on the file (a *Partial* badge in the console, `warning` in the REST status).
