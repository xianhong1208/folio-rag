# Agentic RAG

**An open-source Retrieval-Augmented Generation server for the Model Context Protocol — hierarchical chunking, auto-merging retrieval, and hybrid vector + Chinese BM25 search, exposed to MCP clients as per-folder tools.**

Agentic RAG turns a folder of documents into a retrieval tool that Claude Code, Claude Desktop, Cursor and any other MCP client can call. Each folder becomes an `Agentic_<folder>` tool with three modes — **search**, **list** and **read** — backed by a pipeline that chunks hierarchically, retrieves with dense + sparse signals, merges neighbouring hits into whole passages, and re-ranks with a cross-encoder. It is a **resource server**: it verifies OAuth tokens offline and never issues them, pairing with [MCP Gate](https://github.com/xianhong1208/MCP_Gate) as the authorization server.

License: MIT&nbsp;·&nbsp;Python 3.12&nbsp;·&nbsp;PostgreSQL + pgvector&nbsp;·&nbsp;Works with FastMCP&nbsp;·&nbsp;OAuth 2.1 (resource server)

[繁體中文](README.zh-TW.md) · [Quick start](#quick-start) · [How it works](#how-it-works) · [Configuration](#configuration) · [Console](#use-the-console) · [Architecture](#architecture)

```
┌────────────────┐   Agentic_<folder> tool call (Bearer JWT)    ┌───────────────────────────────┐
│   MCP client   │ ───────────────────────────────────────────▶ │  Agentic RAG (resource server) │
│  Claude / IDE  │ ◀─────────────────────────────────────────── │  /mcp · per-token folder tools │
└───────┬────────┘        auto-merged passages + scores         └───────────────┬───────────────┘
        │ OAuth flow                                                             │ verify JWT offline
        ▼                                                                        ▼
  ┌─────────────┐                              retrieval pipeline (per query)
  │  MCP Gate │  RS256 JWKS                  ingest ─▶ chunk ─▶ embed ─▶ pgvector ─▶ hybrid retrieve
  │ OAuth 2.1 AS│  (offline verify)            (Docling) (leaf+parent) (e5·1024d)   (vector + BM25/CKIP)
  └─────────────┘                                   └▶ auto-merge + expand ─▶ rerank ─▶ gpt-oss-120b
```

## Key features

- **Section-aligned hierarchical chunking + auto-merging retrieval.** Documents are split into small **leaves** (precise matching) and large **parents** (full context) at index time, and neither ever spans two sections. Each leaf carries its **breadcrumb** (file › heading path, xlsx sheet names included), so two documents with the same structure — two product manuals, two price sheets — stay distinguishable. When several leaves from the same passage are hit, the whole parent is returned instead of fragments; a single-leaf hit expands to its neighbours so the answer never loses narrative continuity.
- **Hybrid search, tuned for Chinese.** Dense vector retrieval (pgvector) is fused with sparse **BM25** using Reciprocal Rank Fusion, so keyword signal genuinely affects ranking. Chinese text is word-segmented before Postgres full-text indexing — with **CKIP** (Academia Sinica) by default, or jieba / a dependency-free bigram tokenizer — so space-less Chinese actually gets tokenized instead of collapsing into one term.
- **Cross-encoder reranking.** A `bge-reranker-v2-m3` / `mxbai-rerank-base-v2` model re-scores the fused candidates for fine-grained Top-K precision.
- **Contextual retrieval.** At index time an LLM writes a short context line for each chunk (Anthropic's recommended technique) from the document outline and the chunk's own section, in the document's language. The context and breadcrumb are what the chunk is *found* by (embedding, BM25, rerank); results always show the original document text.
- **Incremental re-indexing.** Re-indexing a file reuses the context line and embedding of every unchanged chunk, so editing one section costs about one LLM call and one embedding. The file's chunks are swapped in a single transaction — it never disappears from search mid-way.
- **Per-folder MCP tools, scoped per token.** Every folder is registered as an `Agentic_<folder>` tool; the tool list a caller sees is filtered by their token — you only see folders you own. Ownership is keyed on the token's stable `jti`. Tool names stay within what every MCP client accepts (`[A-Za-z0-9_-]`, 64 chars): a folder name with spaces, CJK or other symbols becomes `Agentic_<ascii part>_<8-char hash>`, the same name every time.
- **OAuth 2.1 resource server.** One `RemoteAuthProvider` + `JWTVerifier` protects `/mcp`; tokens are verified locally against MCP Gate's JWKS (RS256, offline), and are bound to this server's audience so a token for another server never works here.
- **Document + audio ingestion.** [Docling](https://github.com/DS4SD/docling) parses PDF / DOCX / PPTX / XLSX / images (with OCR); **FireRedASR** transcribes audio locally in Traditional Chinese (or any OpenAI-compatible `/v1/transcriptions` endpoint).
- **Built-in evaluation harness.** Auto-generates questions from your own chunks and scores the production pipeline (the same call as `/api/rag/query`) against `vector` / `bm25` / `rrf` / `rerank` — all through the production search path — on **Recall@k · MRR · nDCG@k**, so you can tune retrieval against your real corpus.
- **Retrieval Terminal console.** A React single-page console at `/admin`: pipeline overview, folders & files, a **Search Playground** that shows the per-signal retrieval trace, live index-job stream, query analytics, the evaluation harness, live settings, health and an audit timeline.
- **Runs on your GPU.** PyTorch is selected per platform with PEP 735 dependency groups (`rocm-r714` / `rocm-r713` / `cuda` / `cpu`) — AMD ROCm, NVIDIA CUDA or pure CPU.

## Install

**Prerequisites**

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- **PostgreSQL** with the **[pgvector](https://github.com/pgvector/pgvector)** extension.
- OpenAI-compatible endpoints for the **embedding**, **LLM** and **reranker** models (e.g. [vLLM](https://github.com/vllm-project/vllm), Ollama, or a cloud provider).
- Node.js 18+ — only if you work on the console UI.

```bash
git clone https://github.com/xianhong1208/agentic_rag.git
cd agentic_rag

# 1. Install — local models need the `local` extra; torch follows a PEP 735 dependency group
#    per platform, so ALWAYS pass --group with it. (All-cloud deployments: plain `uv sync`,
#    see "Cloud profile" below.)
uv sync --extra local --group rocm-r714    # AMD ROCm 7.1.4 (multi-arch wheel) | rocm-r713 | cuda | cpu

# 2. Configure the database and model endpoints
cp .env.example .env         # set DATABASE_URL (Postgres + pgvector)
#   edit config/config.yaml  → rag.embedding / rag.llm / rag.rerank endpoints,
#                               auth.issuer / auth.audience, server.port

# 3. Start (runs the DB bootstrap + migrations automatically)
uv run python main.py
```

The server listens on `http://0.0.0.0:5039` by default; the MCP endpoint is `/mcp` and the console is at `/admin`.

## Quick start

Agentic RAG authenticates through [MCP Gate](https://github.com/xianhong1208/MCP_Gate). The order matters: **register this server in MCP Gate first** — it only issues tokens for audiences it knows.

**1 — Register the server in MCP Gate.** In the MCP Gate console, *Services → Register Service*, enter this server's host / port / path (`/mcp`). Its audience defaults to the MCP URL, e.g. `http://127.0.0.1:5039/mcp`. Set the same value as `auth.audience` in `config/config.yaml`.

**2 — Point Agentic RAG at MCP Gate** (`config/config.yaml`):

```yaml
auth:
  enabled: true
  issuer:   "http://localhost:4568"          # MCP Gate base URL (RS256 JWKS issuer)
  audience: "http://127.0.0.1:5039/mcp"      # must equal the audience registered in the console
```

**3 — Create a folder and index documents** — from the `/admin` console (*Folders → New Folder → Upload Files*) or over REST:

```bash
# create a token-scoped folder, then upload + index files (see the API reference)
curl -X POST http://127.0.0.1:5039/api/folders/ \
  -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" \
  -d '{"name":"product-docs"}'
```

**4 — Connect a client.** With MCP Gate in front, an OAuth-capable client needs nothing up front:

```bash
claude mcp add --transport http agentic-rag http://127.0.0.1:5039/mcp
```

On first use the client registers with MCP Gate and opens the consent screen; after you allow it, the `Agentic_product-docs` tool appears. For scripts / CI, mint a **personal access token** in MCP Gate and pass it as a bearer header.

**5 — Ask.** In the client, call the folder tool:

```
Agentic_product-docs  mode="search"  query="how do I rotate the signing key?"
```

It returns the matched passages (auto-merged where relevant), each with a relevance score and its source file / heading.

## How it works

Your folder is indexed once, then every query runs through the pipeline:

| Stage | What happens |
|-------|--------------|
| **Ingest** | Docling parses PDF / DOCX / PPTX / XLSX / images (OCR); audio is transcribed by FireRedASR. |
| **Chunk** | Section-aligned splitting into small **leaves** + large **parents**; each leaf gets a breadcrumb (file › headings) and an LLM context line. Tables stay whole when they fit. |
| **Embed** | Each leaf's retrieval text (context line + breadcrumb + original text) is embedded (default `intfloat/multilingual-e5-large`, 1024-d) and written to **pgvector** together with its BM25 `text_search_tsv` (Chinese CKIP-segmented), in one transaction per file. |
| **Retrieve** | **Hybrid**: dense vector + sparse BM25, fused with Reciprocal Rank Fusion. |
| **Merge** | Leaves from the same passage **auto-merge** into their parent; single hits **expand** to neighbours. |
| **Rerank** | A cross-encoder (`bge-reranker-v2-m3` / `mxbai-rerank-base-v2`) re-scores for final Top-K precision. |
| **Answer** | For the `answer` path, `gpt-oss-120b` synthesises a response from the retrieved context, gated by a corrective-RAG confidence check. |

### Retrieval in detail

**One retrieval path.** REST `/api/rag/query`, the MCP search tool, the console Search Playground (and its trace view) and the evaluation harness all retrieve through the same code (`src/domain/rag/hybrid_search.py`): dense + BM25, fused with `rrf` (default) or `concat` (dense hits, then BM25-only hits, left for the reranker to order), reranked, then auto-merged with one shared rule. REST accepts `auto_merge` per request (`null` = follow `rag.retrieval.auto_merging.enabled`); `hybrid_alpha` is accepted and ignored. Per-request settings are passed with each call, never stored on the shared per-folder engine, so concurrent requests with different `top_k` cannot affect each other.

**Keyword search (BM25).** The query is tokenised the way the index was (segmented for Chinese), question and function words (`什麼`, `嗎`, `what`, `the`, …) are dropped, and Postgres full-text search runs over **leaf chunks only** — first requiring every term (AND), then filling up with any term (OR). Its depth is `max(top_k candidates, rag.retrieval.default_sparse_top_k)`, and it runs concurrently with the vector search. The `simple` text-search config does no English stemming (`policy` ≠ `policies`); set `text_search_config: english` for an English-only corpus.

**Chinese segmentation.** With the default `text_search_config: simple`, Chinese is segmented in Python before it reaches Postgres, by the segmenter in `rag.retrieval.segmenter` — the same function for indexing, *Rebuild FTS* and queries:

| `segmenter` | Notes |
|---|---|
| `ckip` *(default)* | Academia Sinica CKIP BERT model — most accurate on Traditional Chinese; needs torch and a model download on first use. |
| `jieba` | Dictionary-based, pure Python, no GPU. Optional extra: `uv pip install jieba` (or the `jieba` extra in `pyproject.toml`). |
| `bigram` | Overlapping CJK character pairs + Latin/digit words. No model, no dependency; bigger index. |

Each folder's table records which segmenter built its keyword index. After changing `segmenter` (live, from Settings), the console marks folders whose index was built with another one (**FTS rebuild needed**); *Rebuild FTS* re-segments them with the configured segmenter — no re-embedding.

**Reranking.** The reranker scores up to 3 × `top_k` fused candidates and keeps the best `top_k` above `rag.rerank.score_threshold`. `mxbai-rerank-*-v2` is a Qwen model trained on a chat-formatted prompt; with `rag.rerank.auto_template: true` (default) that format is applied automatically when the configured templates are the default or the old chat-less form (turn it off if your rerank server already applies a chat template). With it, scores are calibrated probabilities and **`score_threshold: 0.5`** is meaningful for that model; without it unrelated chunks score 0.8+ and no threshold filters anything. If the reranker fails, the query returns the top `top_k` candidates in retrieval order (not all 3 × `top_k`), and that degraded result is not cached.

**Knowledge gaps.** Every REST, console and MCP search is logged (masked token prefix only) with its result count and the reranker's best score. *Query Analytics* lists a question as a knowledge gap when its **latest** search in a folder returned nothing or its best score was below `rag.rerank.gap_threshold` (default `0.5`) — so a gap disappears once documents that answer it are added and the question is asked again.

**Answers (console).** With *generate answer* on, the top chunks are graded for relevance in one LLM call, irrelevant ones are dropped (nothing relevant → no answer), and the answer is generated from the rest in the **question's language**. Each chunk keeps its result-card number, so `[3]` always means card #3 (citations are links in the console); citations of chunks the model was not given are removed, `【n†…】` markers become `[n]` and Markdown bold markers are stripped — on the streaming path too, where the final `done` frame carries the cleaned answer. If the grader's reply cannot be parsed the answer is labelled **unverified** instead of `high`.

**Auth.** Agentic RAG is a *resource server*. FastMCP publishes `/.well-known/oauth-protected-resource/mcp`, which points clients at MCP Gate; the server fetches MCP Gate's JWKS once and verifies every bearer token offline (RS256 — signature, issuer, audience, expiry, scopes). MCP Gate is never on the request path.

## Configuration

Settings live in `config/config.yaml`; secrets come from environment variables (`.env`). The ones you are likely to touch:

| Variable / setting | Default | Purpose |
|--------------------|---------|---------|
| `DATABASE_URL` | `postgresql://…/agentic_rag` | PostgreSQL DSN. Requires the **pgvector** extension. Use **sync** `postgresql://` (psycopg2), not `+asyncpg`. |
| `server.port` | `5039` | HTTP + MCP bind port. The MCP endpoint is `/mcp`. |
| `auth.enabled` / `auth.issuer` / `auth.audience` | `true` / MCP Gate URL / this server's `/mcp` URL | OAuth 2.1 verification against MCP Gate. |
| `auth.console.session_hours` | `8` | Longest a console session lasts (it also ends when the token expires). |
| `auth.console.session_secret` | *(generated)* | Key that signs the session cookie; `${ENV_VAR}` works. Unset → generated once into `storage/.console_session_secret` (gitignored), so sessions survive restarts. |
| `auth.console.secure_cookie` | `auto` | Cookie `Secure` flag: `auto` = only over https; `true` behind a TLS-terminating proxy. |
| `auth.console.admin_scope` | `rag:admin` | Scope that unlocks the console's global settings. |
| `rag.embedding.{model,dimension,base_url}` | `intfloat/multilingual-e5-large`, `1024`, `:7075/v1` | Embedding model; `dimension` **must** equal the model's real output dim. |
| `rag.llm.{model,base_url}` | `openai/gpt-oss-120b`, `:5052/v1` | LLM for Contextual Retrieval and the `answer` path. |
| `rag.rerank.{model,base_url}` | `mixedbread-ai/mxbai-rerank-base-v2`, `:8787` | Cross-encoder reranker. |
| `rag.rerank.score_threshold` / `auto_template` | `0.0` / `true` | Drop results scoring below the threshold (0.5 suits mxbai-rerank-v2); apply mxbai-v2's chat prompt automatically. |
| `rag.rerank.gap_threshold` | `0.5` | Knowledge-gap analytics: best rerank score below this counts as "no good match". |
| `rag.asr.provider` | `fireredasr` | Audio transcription: `fireredasr` (local zh-TW) · `docling-whisper` · `openai-compatible`. `rag.asr.enabled: false` fails audio files with a clear message (nothing is transcribed). |
| `rag.retrieval.segmenter` | `ckip` | Chinese segmenter for keyword search: `ckip` · `jieba` · `bigram` (see *Chinese segmentation*). |
| `rag.docling.ocr_enabled` | `true` | OCR for scanned PDFs and images. `false` really skips it (restart to apply); a scan with no text layer then fails with a clear message. |
| `rag.contextual_retrieval.max_doc_chars` | `60000` | Upper bound on the document text the LLM reads per chunk. It reads the outline plus the chunk's own section, windowed to at most 12,000 characters. |
| `rag.retrieval.*` | — | Top-K, similarity cutoff, BM25 candidates, `hybrid_fusion` (`rrf`/`concat`), auto-merging. Tunable live from the console, or via the **High Precision / Balanced / High Recall** presets. |
| `rag.profile` | `local` | `local` · `hybrid` · `cloud` — see [Cloud profile](#cloud-profile). |
| `rag.<role>.{timeout,max_retries}` | per role | Request timeout and retries (429 / 5xx / timeouts, with backoff) for each model role. |

GPU selection is a `uv sync --extra local --group {rocm-r714|rocm-r713|cuda|cpu}` choice — see the top of `pyproject.toml` and `docs/testing/ENVIRONMENTS.md`.

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
    # segmenter: bigram     # model-free Chinese BM25 segmentation (CKIP is not loaded in cloud)
```

What the profile does:

- **Validation.** `cloud` refuses in-process providers (`huggingface` embedding, `fireredasr` /
  `docling-whisper` ASR, local PDF/image parsing) and **rejects API keys that are not `${ENV}`
  references** — a cloud key is a billable credential and config.yaml is the file that gets
  copied around. Startup fails with the offending paths. `hybrid` only warns about literal keys;
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
- **Chinese BM25.** CKIP is a local torch model, so the cloud profile does not load it and Chinese
  text is indexed unsegmented until a model-free segmenter is selected
  (`rag.retrieval.segmenter: bigram`).

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

## Use the console

The **Retrieval Terminal** console is a React SPA served at `/admin`.

### Console login

You sign in by pasting an **MCP Gate access token issued for this service** — usually a personal access token (PAT). The server verifies it once, offline, exactly like a REST or MCP call (signature, issuer, audience, expiry; it must carry a `jti`), then sets an HttpOnly session cookie. The token itself is never stored or sent back.

- **The session is that token's identity.** You see and manage exactly the folders owned by the token (folders are keyed on its `jti`) — the same folders it reaches over REST and MCP. Folders you create in the console belong to it too. Folders owned by another token are invisible; to manage them, sign in with that token.
- **Two tiers.** Any valid token can sign in and manage its own folders. Changing runtime **Settings**, testing arbitrary endpoints, and reading the settings **Audit Log** and server-wide storage figures need the **`rag:admin`** scope. Without it, Settings and System Health are read-only (secrets are always masked), and Overview / Index Jobs / Query Analytics cover only your own folders; with it they cover the whole server.
- **Sessions** last `auth.console.session_hours` (8 h) or until the token expires, whichever is first. Revoking the token in MCP Gate does not end a running session (verification is offline) — it ends at expiry or on *Sign out*.

**Issuing tokens in MCP Gate.** No MCP Gate change is needed:

1. *Services* → open the Agentic RAG service → *Scopes declared by this server* → add **`rag:admin`**. Do **not** mark it as a default scope (defaults are preselected on every new token). If the service restricts its *Allowed scopes*, allow `rag:admin` too.
2. *Issue Token* → *Target service*: Agentic RAG (audience = `auth.audience`). Tick `rag:admin` for operators who may change settings; leave it off for people who only manage their own folders. Give it a label and copy the token — it is shown once.

Paste the token on the login screen. Folders created earlier through the old *Owner Token* field belong to the token that was pasted there: sign in with that token to see them. Scripts can skip the cookie and call `/api/admin/*` with `Authorization: Bearer <token>` — the same rules apply, and no CSRF header is needed.

**Dev mode.** With `auth.enabled: false` there is no login: the console is open to anyone who can reach the port, with full admin rights and every folder visible (folders created there are owned by `console`). The server logs a warning at startup. Use it only on a trusted machine.

**Network exposure.** Even with login on, keep the console (`/admin`, `/api/admin/*`) reachable only from trusted networks — a VPN, an internal network, or a reverse proxy that restricts it — and serve it over HTTPS. Login protects folders and settings from people without a valid token, but the console can upload and delete documents, run queries against your model endpoints and (with `rag:admin`) change them, so it should not face the open internet. `/mcp` and `/api/rag/*` are designed to be reachable by MCP clients and need only a valid token.

| Page | What you do there |
|------|-------------------|
| **Overview** | The live retrieval pipeline, corpus readouts, index health, and model-service status. |
| **Folders** | Create folders (owned by your token); open one to upload files, watch per-file index status, view chunks (with the AI context line shown apart from the document text), reindex, or delete. Files indexed by an older chunking pipeline show **Re-index recommended**. |
| **Search Playground** | Run a live query against a folder and see the **per-signal retrieval trace** — Vector · BM25 · Hybrid · Rerank ranks side by side — plus an optional RAG answer. |
| **Index Jobs** | Live stream of indexing jobs with progress; cancel a running job. |
| **Query Analytics** | Volume, latency, zero-result rate, top queries and knowledge gaps. |
| **Evaluation** | Auto-generate a question set and score `production` (same call as `/api/rag/query`), `vector`-only and `bm25`-only on Recall@k · MRR · nDCG@k; baseline deltas only against runs on the same question set. |
| **Settings** | Live, persisted configuration for Embedding / LLM / Contextual Retrieval / Reranker / Speech-to-Text / Retrieval, with retrieval presets and connection tests. Changing anything needs `rag:admin`. |
| **System Health** | Live connectivity to every external dependency. |
| **Audit Log** | A timeline of every runtime settings change and who made it (`rag:admin`). |

### Re-indexing after an upgrade

Chunking changes are versioned per file. A file indexed by an older pipeline keeps working exactly as before, but shows **Re-index recommended** in the folder view (and `reindex_recommended: true` in `GET /api/rag/file/{id}/index/status`). Use the row's re-index button, or the header link that selects all such files, then *Reindex selected*. Per-file re-indexing is incremental from then on; *Reindex Folder* drops the folder's table and rebuilds everything.

## API reference

| Group | Endpoints |
|-------|-----------|
| **MCP** | `POST /mcp` — per-folder `Agentic_<folder>` tools (`search` / `list` / `read`); `GET /.well-known/oauth-protected-resource/mcp` |
| **Folders & files** | `/api/folders*` · `/api/folders/{id}/files*` (create, upload, list, download, delete — token-scoped) |
| **RAG** | `POST /api/rag/query` (hybrid retrieval + rerank) · `/api/rag/files/{id}/index`, `/reindex`, `/index/jobs*` (background indexing + SSE progress) |
| **Console session** | `POST /api/console/session` `{token}` (sign in) · `GET` (who am I) · `DELETE` (sign out) |
| **Console (admin)** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **Operational** | `GET /health` |

Set `ENABLE_API_DOCS=true` for the full OpenAPI reference at `/docs`.

## Architecture

- **FastMCP + FastAPI** on one ASGI app. The FastMCP app is *mounted* so its auth middleware protects `/mcp`; the FastAPI routers serve the REST API and the console.
- **PostgreSQL + pgvector** for storage; each folder gets its own `data_<folder>_<uuid>` table. Retrieval uses [LlamaIndex](https://github.com/run-llama/llama_index)'s `PGVectorStore` in hybrid mode.
- **Sync SQLAlchemy** (psycopg2) throughout — do not use `+asyncpg`.
- **Console**: Vite + React + React Router + Tailwind, built to `static/console` and served at `/admin`.

See [`ROADMAP.md`](ROADMAP.md) for what's under evaluation (Graph RAG, "LLM Wiki").

## Contributing

Issues and pull requests are welcome. Code comments are in English; commit history uses the project's own identity. See `docs/` for the testing and environment notes.

## License

[MIT](LICENSE)
