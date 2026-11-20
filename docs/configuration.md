# Configuration

[繁體中文](configuration.zh-TW.md)

Settings live in six files under `config/`, merged into one configuration at startup. Secrets
and per-machine values come from environment variables, usually from `.env`. The files only
reference them as `${VAR:-default}`, so every file under `config/` holds no secret and is safe to
commit.

| File | Holds |
|---|---|
| `config/database.yaml` | `database`: connection, pools, retention, backups |
| `config/auth.yaml` | `auth`: MCP Gate issuer and audience, scopes, console sign-in, revocation feed |
| `config/models.yaml` | `rag.profile`, and where each model runs: `rag.embedding`, `rag.llm`, `rag.contextual_llm`, `rag.rerank`, `rag.asr`, `rag.parsing` |
| `config/mcp.yaml` | `app.tool_prefix`, `auth.dynamic_tools` (per-folder tools), `modules.enabled` |
| `config/retrieval.yaml` | `rag.indexing`, `rag.chunking`, `rag.contextual_retrieval`, `rag.vector_store`, `rag.retrieval` (including `segmenter`) |
| `config/system.yaml` | `server`, `logging`, `app.name` / `title` / `description`, `metrics` ([`/metrics`](metrics.md)) |

Each file is written in the shape of the merged configuration: `models.yaml` starts with `rag:`,
`mcp.yaml` holds `app: {tool_prefix: …}`. Sections such as `rag:` open in several files, but **a
setting may be written in one file only**: the same setting in two files stops startup with an
error that names both files. All six files must exist. `--config-dir` points the server at another
directory (default `config`).

Module wiring (which router or MCP tool registration each module loads) is code, in
`src/config/modules.py`; `mcp.yaml` only lists the enabled modules (`folder`, `file`,
`rag_indexing`, `rag_query`, `rag`, `media`, `admin`).

**Environment variables.** `main.py` loads `.env` from the project root before reading the files,
and `.env` wins over a variable already exported in the shell. [`.env.example`](../.env.example)
lists every variable. An empty value means "not set".

| Variable | Setting | Default in the files |
|---|---|---|
| `DATABASE_URL` | `database.url` | `postgresql://postgres:CHANGE_ME@localhost:5432/folio` |
| `DATABASE_PORT` | `database.port` (used only when the URL has no port) | empty |
| `MCP_GATE_URL` | `auth.issuer` | `http://localhost:4568` |
| `FOLIO_AUDIENCE` | `auth.audience` | `http://127.0.0.1:5039/mcp` |
| `MCP_GATE_INTERNAL_URL` | `auth.internal_url`: where Folio fetches MCP Gate's keys and revocation feed when the issuer address does not work from here (Docker Compose: `http://mcp-gate:4568`); tokens are still checked against `auth.issuer` | empty (= the issuer) |
| `FOLIO_REVOCATION_ENABLED`, `FOLIO_REVOCATION_CLIENT_ID`, `FOLIO_REVOCATION_CLIENT_SECRET` | `auth.revocation.*` | `false`, empty, empty |
| `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY` | `rag.embedding.*` | `http://localhost:7075/v1`, `not-needed` |
| `LLM_BASE_URL`, `LLM_API_KEY` | `rag.llm.*` | `http://localhost:5052/v1`, `not-needed` |
| `CONTEXTUAL_LLM_BASE_URL`, `CONTEXTUAL_LLM_API_KEY` | `rag.contextual_llm.*` (commented out in `models.yaml`) | empty (inherit `rag.llm`) |
| `RERANK_BASE_URL`, `RERANK_API_KEY` | `rag.rerank.*` | `http://localhost:8787`, `not-needed` |
| `ASR_BASE_URL`, `ASR_API_KEY`, `ASR_MODEL` | `rag.asr.*`: the OpenAI-compatible transcription endpoint ([Upgrading to 1.5](migrate-1.5.md#speech-to-text)) | empty (audio uploads are refused) |
| `ASR_CA_FILE` | `rag.asr.ca_file`: a PEM CA certificate added to the default trust for that endpoint (Forgeyard's own CA) | empty |
| `VISION_BASE_URL`, `VISION_API_KEY`, `VISION_MODEL` | `rag.parsing.vision.*`: the OpenAI-compatible vision model that reads PDFs and images ([Upgrading to 1.5](migrate-1.5.md#pdfs-and-images)) | empty (PDFs from their text layer, images refused) |
| `VISION_CA_FILE` | `rag.parsing.vision.ca_file`: a private CA (PEM) for that endpoint, e.g. Forgeyard's | empty |
| `CKIP_DEVICE` | `rag.retrieval.ckip_device` | `cpu` |
| `FOLIO_SEGMENTER` | `rag.retrieval.segmenter` | empty (= the profile's default; the Docker image sets `jieba`) |
| `METRICS_TOKEN` | `metrics.token`: bearer token `GET /metrics` requires ([Metrics](metrics.md)) | empty (this host only) |

Read by `main.py` itself: `FOLIO_CONFIG_DIR` (the default of `--config-dir`, `config`), `FOLIO_LOG_DIR`
(where the log files go, `logs`; set but empty = the console only, as in the Docker image) and
`NO_COLOR` (no colour codes on the console). The Docker image's own `models.yaml` reads a few more
(model names, the reranker API): see [Deploying Folio](deploy.md).

The settings you are likely to touch:

| Variable / setting | Default | Purpose |
|--------------------|---------|---------|
| `DATABASE_URL` | `postgresql://…/folio` | PostgreSQL DSN. Requires the **pgvector** extension. Use **sync** `postgresql://` (psycopg2), not `+asyncpg`. |
| `server.port` | `5039` | HTTP + MCP bind port. The MCP endpoint is `/mcp`. |
| `FOLIO_STORAGE_ROOT` | the project root | Directory that holds `storage/` (uploaded files and the console session secret). |
| `database.backup.{dir,enabled,interval_hours,keep,pg_bin_dir}` | `backups`, `false`, `24`, `7`, PATH | Where `main.py backup` writes archives, and the server's own backup schedule (off by default) with how many archives it keeps — see [Backup and restore](backup.md). |
| `database.backup.folder_import_max_mb` | `2048` | Largest folder archive the console's **Import** accepts (MB); the upload is streamed and refused with 413 as soon as it passes this — see [In the console](backup.md#in-the-console). |
| `FOLIO_INDEXING_CONCURRENCY`, `FOLIO_PER_FILE_TIMEOUT_SECONDS`, `FOLIO_JOB_TIMEOUT_SECONDS` | from `rag.indexing` | Runtime overrides for the indexing limits. The pre-rename `RAG_*` names still work and log a deprecation warning. |
| `auth.enabled` / `MCP_GATE_URL` / `FOLIO_AUDIENCE` | `true` / MCP Gate URL / this server's `/mcp` URL | OAuth 2.1 verification against MCP Gate (`auth.issuer`, `auth.audience`). |
| `auth.console.session_hours` | `8` | Longest a console session lasts (it also ends when the token expires). |
| `auth.console.session_secret` | *(generated)* | Key that signs the session cookie; `${ENV_VAR}` works. Unset → generated once into `storage/.console_session_secret` (gitignored), so sessions survive restarts. |
| `auth.console.secure_cookie` | `auto` | Cookie `Secure` flag: `auto` = only over https; `true` behind a TLS-terminating proxy. |
| `auth.console.admin_scope` | `rag:admin` | Scope that unlocks the console's global settings. |
| `auth.revocation.*` | disabled | MCP Gate revocation feed, so revoked tokens stop working before they expire — see [Token revocation](auth.md#token-revocation). |
| `rag.embedding.{model,dimension}`, `EMBEDDING_BASE_URL` | `intfloat/multilingual-e5-large`, `1024`, `:7075/v1` | Embedding model; `dimension` **must** equal the model's real output dim. |
| `rag.llm.model`, `LLM_BASE_URL` | `openai/gpt-oss-120b`, `:5052/v1` | The LLM: answers (Search Playground), evaluation, and the contextual prefixes unless `rag.contextual_llm` overrides it. |
| `rag.contextual_llm` | unset (inherits `rag.llm`) | Optional model for contextual prefixes only. It writes one short sentence of context per chunk, so a small, fast model is enough, and since it runs once per chunk it saves a lot of indexing time and cost. Unset fields inherit from `rag.llm`. |
| `rag.rerank.{model,base_url}` | `mixedbread-ai/mxbai-rerank-base-v2`, `:8787` | Cross-encoder reranker. |
| `rag.rerank.score_threshold` / `auto_template` | `0.0` / `true` | Drop results scoring below the threshold (0.5 suits mxbai-rerank-v2); apply mxbai-v2's chat prompt automatically. |
| `rag.rerank.gap_threshold` | `0.5` | Knowledge-gap analytics: best rerank score below this counts as "no good match". |
| `rag.retrieval.rerank_pool` | `per_list` | What the cross-encoder scores: `per_list` = the dense top `rerank_dense_candidates` + the BM25 top `rerank_sparse_candidates`; `fused` = the top `max(3 × top_k, rerank_candidates)` of the fused list (the previous behaviour). Live. |
| `rag.retrieval.rerank_dense_candidates` / `rerank_sparse_candidates` | `30` / `10` | `per_list` pool depths (dense never below 3 × `top_k`). Higher = more recall margin, ~6 ms per extra pair; live-tunable (0–200). |
| `rag.retrieval.max_search_folders` | `20` | Search across folders (`POST /api/search`): at most this many folders per query, the most recently updated first; the rest are listed in `skipped_folders` (`folder_limit`). Live-tunable (1–200). |
| `rag.retrieval.rerank_candidates` | `60` | `fused` pool only: `max(3 × top_k, this)` fused candidates go to the cross-encoder; live-tunable (1–200). |
| `rag.retrieval.mcp_response_token_budget` | `6000` | Estimated-token cap on an MCP search response; lowest-ranked results are dropped to fit and the response says so. `0` = no cap. |
| `rag.retrieval.min_relevance` | `0.9` in the shipped config (`0` when unset) | MCP search: best rerank score below this → `low_relevance: true` (flagged, not dropped). Model-specific; `0` = off. |
| `rag.asr.*` | `openai-compatible`, no endpoint | Audio transcription through an OpenAI-compatible `POST {base_url}/audio/transcriptions` (Forgeyard, speaches, vLLM, OpenAI). Without `base_url`, or with `rag.asr.enabled: false`, audio uploads are refused with 422 and the reason. `rag.asr.provider` takes only `openai-compatible`: the in-process `fireredasr` and `docling-whisper` were **removed in 1.5** and stop the start with a pointer to [the migration guide](migrate-1.5.md#speech-to-text). |
| `rag.asr.chinese_conversion` | `s2twp` | Chinese script of the transcript: `s2twp` turns whisper's Simplified Chinese into Traditional with Taiwan phrasing (OpenCC); `none` keeps what the endpoint returns. Text without Chinese characters is never changed. Live. |
| `rag.retrieval.segmenter` | `ckip` (`bigram` under `profile: cloud`) | Chinese segmenter for keyword search: `ckip` · `jieba` · `bigram` (see *Chinese segmentation* in [How Folio works](how-it-works.md)). |
| `rag.parsing.pdf` | `auto` | How PDFs are read: `auto` = the vision model when `rag.parsing.vision` has a model and an endpoint, the PDF's text layer otherwise; `vision` = always the model (refused without one); `text` = always the text layer. Without a vision model a page with no text layer becomes a visible marker, a PDF with no text at all fails indexing, and image uploads are refused (422). Live. Folio 1.5 runs no OCR or layout model: `rag.docling`, `rag.parsing.images` and `pdf: local`/`cloud` stop the start ([Upgrading to 1.5](migrate-1.5.md#pdfs-and-images)). |
| `rag.parsing.vision.*` | none | The vision model: `provider`, `model`, `base_url`, `api_key`, `ca_file` (env `VISION_*`). Every PDF page and image is transcribed by it. |
| `rag.contextual_retrieval.max_doc_chars` | `60000` | Upper bound on the document text the LLM reads per chunk. It reads the outline plus the chunk's own section, windowed to at most 12,000 characters. |
| `rag.retrieval.*` | — | Top-K, similarity cutoff, BM25 candidates, `hybrid_fusion` (`rrf`/`concat`), auto-merging. Tunable live from the console, or via the **High Precision / Balanced / High Recall** presets. |
| `rag.profile` | `local` | `local` · `hybrid` · `cloud` — see [Cloud profile](#cloud-profile). |
| `rag.<role>.{timeout,max_retries}` | per role | Request timeout and retries (429 / 5xx / timeouts, with backoff) for each model role. |

There is no GPU choice at install time (since 1.5.1): `uv sync --extra local` installs the CPU build of torch from PyTorch's CPU index, and the in-process models (CKIP, HuggingFace embeddings) run on the CPU. `rag.retrieval.ckip_device` (`CKIP_DEVICE`) accepts only `cpu`; `cuda`, `cuda:N` and `auto` stop the start ([Upgrading to 1.5.1](migrate-1.5.md#upgrading-to-151)). Models that want a GPU run behind an OpenAI-compatible endpoint.

## Cloud profile

Run with every model behind an API and nothing loaded locally — no GPU, no torch, no model
downloads. Set `rag.profile: cloud` and give each role an OpenAI-compatible endpoint:

```bash
uv sync                      # core install: no torch / transformers (the local stack is `--extra local`)
export OPENAI_API_KEY=sk-...  # keys come from the environment (.env), never from the config files
uv run python main.py
```

The example shows the merged `rag:` section; in the files, `profile`, `embedding`, `llm`,
`contextual_llm`, `rerank`, `parsing` and `asr` go in `config/models.yaml`, and
`contextual_retrieval` and `retrieval` in `config/retrieval.yaml`.

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

  llm:                      # answers, evaluation, and contextual prefixes unless overridden
    provider: openai
    model: gpt-4o-mini
    api_key: "${OPENAI_API_KEY}"
  contextual_llm:           # optional: one short sentence per chunk, so a small, fast model
    model: gpt-4.1-nano     # is enough; unset fields inherit from llm

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

  parsing:                  # pdf: auto (default) = this model reads PDFs and images
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
    chinese_conversion: s2twp   # Simplified -> Traditional (Taiwan); none = keep
    segment_seconds: 600    # long audio is uploaded in silence-aligned pieces

  retrieval:
    hybrid_search: true
    # segmenter: bigram     # the cloud default (no model); jieba also works, ckip is rejected
```

What the profile does:

- **Validation.** `cloud` refuses in-process providers (`huggingface` embedding) and **rejects API keys that are not `${ENV}`
  references** — a cloud key is a billable credential and the config files are what gets
  copied around. An explicit `rag.retrieval.segmenter: ckip` is rejected too. Startup fails with the offending paths. `hybrid` only warns about literal keys;
  `local` (the default — existing configs are unchanged) checks nothing new.
- **Nothing local is imported.** The HuggingFace embedding provider and CKIP are imported only
  when selected (the vision model and speech-to-text are always endpoints, in every profile);
  startup logs where every role runs. `tests/test_cloud_profile_no_local_models.py` proves it
  by running the whole pipeline on an install without the local stack.
- **Vision-model PDF and image parsing** (every profile, once `rag.parsing.vision` is set; without
  it PDFs come from their text layer). Every page is transcribed to Markdown by the vision model,
  and the pages are stitched into **one** document before chunking, so headings, breadcrumbs and
  tables continue across page breaks. Each page is sent the previous page's tail and the current
  heading path; pages run in ordered blocks for bounded concurrency; failed pages are retried with
  backoff and, if they still fail, marked in the text instead of failing the file. Chunks keep the
  page they start on for citations. Per-file and per-day page caps bound the cost. A wrong key,
  an untrusted certificate (`rag.parsing.vision.ca_file`) or an unreachable endpoint fails the file
  with what to fix, never an empty index.
- **Speech-to-text** (every profile). Audio is resampled and uploaded in segments cut at silences
  (so no request exceeds an upload limit), with retries, and re-asked when a transcript is empty
  despite speech or loops.
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
pair, half a second of silence for speech-to-text). Changes need the `rag:admin` scope.

**Keys set at runtime.** A key field takes the key itself, or a `${VAR}` reference; the server
stores the reference, never the key. A reference is accepted only for a variable that
`config/models.yaml` already uses as a model `api_key`, and only while the role points at a server
(scheme, host and port of its `base_url`) that the startup configuration uses that variable with. Any other reference gets the same
"not allowed" error, whether the variable exists or not. When a role's `base_url` moves to another
server, its key is cleared unless the same change gives it a new one (the answer says so); an LLM
role that inherits `rag.llm`'s key must get its own key in that change. Resetting a setting
restores its value from the config files as they are on disk; for a key, the file's reference. At startup, stored key settings that break these rules are skipped, and a
key reference that would reach a server it is not paired with is cleared, with a warning in the
log.

Daily caps are kept in memory and reset at UTC midnight or on restart — they are a runaway guard,
not billing.

**Chunk token counting** never downloads anything: it uses `assets/hf_tokenizers/<org>--<model>` when present, else the bundled XLM-RoBERTa tokenizer in `resources/tokenizers/` (the tokenizer of multilingual-e5 and bge-m3), so a fresh clone with OpenAI or Azure embeddings works out of the box.

**Parsing failures are visible.** A docling conversion that comes back partial is retried once, and pages still missing are listed on the file (a *Partial* badge in the console, `warning` in the REST status). The same warning lists PDF pages the vision model could not transcribe, and pages indexed only as a "no text layer" marker.

## Upgrading to 1.5.1

1.5.1 installs the in-process models with `uv sync --extra local` and the CPU build of torch; the
`--group cuda | rocm-r714 | cpu` flag is gone and now fails. `CKIP_DEVICE` accepts only `cpu`. No
configuration file changes. Details: [Upgrading to 1.5.1](migrate-1.5.md#upgrading-to-151).

## Upgrading from 1.4

**Breaking change in 1.5:** the single `config/config.yaml` is no longer read. Nothing is carried
over automatically: if the file is still there, startup logs an error saying it is ignored, and
runs with the six files.

1. Before pulling, move your `config/config.yaml` out of the way (it is deleted upstream, so
   `git pull` refuses to overwrite a modified copy): `mv config/config.yaml ~/folio-config-1.4.yaml`.
2. Pull 1.5. `config/` now holds the six commented templates.
3. Convert your old file:
   ```bash
   uv run python scripts/split_config.py --old ~/folio-config-1.4.yaml --force
   ```
   It writes the six files into `config/` (`--force` replaces the templates; `git checkout --
   config/` brings them back) and the secret and per-machine values to `.env.split` (mode 600).
   It reads your `.env` (`--env-file`) to expand `${VAR}` references exactly as the server did, then
   loads the old and the new configuration and compares them field by field. It prints file
   names, variable names and notes only, never values, and exits 1 if anything differs.
4. Merge `.env.split` into `.env` (replace any line with the same name), then delete it.
5. Start the server. The log shows `Config: config/ (+ .env)`.

What else changed:

- **LLM roles 4 → 2.** `rag.answer_llm` and `rag.eval_llm` are removed: answers and evaluation use
  `rag.llm`. `rag.contextual_llm` stays as an optional override for contextual prefixes. The
  converter drops the two sections (it says so), a leftover in `models.yaml` is ignored with a
  warning, and settings stored for them on the console's Settings page are deleted at startup
  with a warning.
- **Speech-to-text only through an endpoint.** `rag.asr.provider: fireredasr` or `docling-whisper`
  stops the start. The converter rewrites it to `openai-compatible` and says so; set
  `ASR_BASE_URL`, `ASR_API_KEY` and `ASR_MODEL` in `.env` as [the migration guide](migrate-1.5.md#speech-to-text)
  shows. Until then the server runs and refuses audio uploads. A provider stored on the console's
  Settings page is deleted at startup with a warning.
- **No in-process PDF or image models.** A `rag.docling` section, `rag.parsing.images` or
  `rag.parsing.pdf: local`/`cloud` stops the start. The converter drops them and says so. Without
  `VISION_BASE_URL` / `VISION_MODEL`, PDFs are read from their text layer and image uploads are
  refused; set them as [the migration guide](migrate-1.5.md#pdfs-and-images) shows. Delete
  `DOCLING_DEVICE` from `.env`; for `hf_offline: true` set `HF_HUB_OFFLINE=1` instead. Values stored
  on the console's Settings page are deleted at startup with a warning.
- **Module wiring** (`modules.<name>.api_router` / `mcp_tools`) is no longer configurable; only
  `modules.enabled` remains. A wiring block in `mcp.yaml` is refused at startup, and an unknown
  module name too.
- **Endpoints, keys and devices** are `${VAR:-default}` references (see the table above). A
  value you wrote literally in 1.4 moves to `.env` under that variable; a reference you already
  used (for example `${MY_KEY}`) keeps its variable name.
- `main.py --config <file>` became `--config-dir <dir>`; the eval scripts take `--config-dir` too.
