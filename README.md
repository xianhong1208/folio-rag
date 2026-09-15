# Agentic RAG

**An open-source Retrieval-Augmented Generation server for the Model Context Protocol — hierarchical chunking, auto-merging retrieval, and hybrid vector + Chinese BM25 search, exposed to MCP clients as per-folder tools.**

Agentic RAG turns a folder of documents into a retrieval tool that Claude Code, Claude Desktop, Cursor and any other MCP client can call. Each folder becomes an `Agentic_<folder>` tool with three modes — **search**, **list** and **read** — backed by a pipeline that chunks hierarchically, retrieves with dense + sparse signals, merges neighbouring hits into whole passages, and re-ranks with a cross-encoder. It is a **resource server**: it verifies OAuth tokens offline and never issues them, pairing with [MCP Center](https://github.com/xianhong1208/MCP_Center) as the authorization server.

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
  │  MCP Center │  RS256 JWKS                  ingest ─▶ chunk ─▶ embed ─▶ pgvector ─▶ hybrid retrieve
  │ OAuth 2.1 AS│  (offline verify)            (Docling) (leaf+parent) (e5·1024d)   (vector + BM25/CKIP)
  └─────────────┘                                   └▶ auto-merge + expand ─▶ rerank ─▶ gpt-oss-120b
```

## Key features

- **Hierarchical chunking + auto-merging retrieval.** Documents are split into small **leaves** (precise matching) and large **parents** (full context) at index time. When several leaves from the same passage are hit, the whole parent is returned instead of fragments; a single-leaf hit expands to its neighbours so the answer never loses narrative continuity.
- **Hybrid search, tuned for Chinese.** Dense vector retrieval (pgvector) is fused with sparse **BM25** using Reciprocal Rank Fusion, so keyword signal genuinely affects ranking. Chinese text is word-segmented with **CKIP** (Academia Sinica) before Postgres full-text indexing — space-less Chinese actually gets tokenized instead of collapsing into one term.
- **Cross-encoder reranking.** A `bge-reranker-v2-m3` / `mxbai-rerank-base-v2` model re-scores the fused candidates for fine-grained Top-K precision.
- **Contextual retrieval.** At index time an LLM prepends a short, document-aware context to each chunk (Anthropic's recommended technique), improving retrievability of otherwise ambiguous fragments.
- **Per-folder MCP tools, scoped per token.** Every folder is registered as an `Agentic_<folder>` tool; the tool list a caller sees is filtered by their token — you only see folders you own. Ownership is keyed on the token's stable `jti`.
- **OAuth 2.1 resource server.** One `RemoteAuthProvider` + `JWTVerifier` protects `/mcp`; tokens are verified locally against MCP Center's JWKS (RS256, offline), and are bound to this server's audience so a token for another server never works here.
- **Document + audio ingestion.** [Docling](https://github.com/DS4SD/docling) parses PDF / DOCX / PPTX / XLSX / images (with OCR); **FireRedASR** transcribes audio locally in Traditional Chinese (or any OpenAI-compatible `/v1/transcriptions` endpoint).
- **Built-in evaluation harness.** Auto-generates questions from your own chunks and scores `vector` / `hybrid` / `rerank` on **Recall@k · MRR · nDCG@k**, so you can tune retrieval against your real corpus.
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

# 1. Install — torch follows a PEP 735 dependency group per platform, so ALWAYS pass --group.
uv sync --group rocm-r714    # AMD ROCm 7.1.4 (multi-arch wheel)   | rocm-r713 | cuda | cpu

# 2. Configure the database and model endpoints
cp .env.example .env         # set DATABASE_URL (Postgres + pgvector)
#   edit config/config.yaml  → rag.embedding / rag.llm / rag.rerank endpoints,
#                               auth.issuer / auth.audience, server.port

# 3. Start (runs the DB bootstrap + migrations automatically)
uv run python main.py
```

The server listens on `http://0.0.0.0:5039` by default; the MCP endpoint is `/mcp` and the console is at `/admin`.

## Quick start

Agentic RAG authenticates through [MCP Center](https://github.com/xianhong1208/MCP_Center). The order matters: **register this server in MCP Center first** — it only issues tokens for audiences it knows.

**1 — Register the server in MCP Center.** In the MCP Center console, *Services → Register Service*, enter this server's host / port / path (`/mcp`). Its audience defaults to the MCP URL, e.g. `http://127.0.0.1:5039/mcp`. Set the same value as `auth.audience` in `config/config.yaml`.

**2 — Point Agentic RAG at MCP Center** (`config/config.yaml`):

```yaml
auth:
  enabled: true
  issuer:   "http://localhost:4568"          # MCP Center base URL (RS256 JWKS issuer)
  audience: "http://127.0.0.1:5039/mcp"      # must equal the audience registered in the console
```

**3 — Create a folder and index documents** — from the `/admin` console (*Folders → New Folder → Upload Files*) or over REST:

```bash
# create a token-scoped folder, then upload + index files (see the API reference)
curl -X POST http://127.0.0.1:5039/api/folders/ \
  -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" \
  -d '{"name":"product-docs"}'
```

**4 — Connect a client.** With MCP Center in front, an OAuth-capable client needs nothing up front:

```bash
claude mcp add --transport http agentic-rag http://127.0.0.1:5039/mcp
```

On first use the client registers with MCP Center and opens the consent screen; after you allow it, the `Agentic_product-docs` tool appears. For scripts / CI, mint a **personal access token** in MCP Center and pass it as a bearer header.

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
| **Chunk** | Hierarchical splitting into small **leaves** + large **parents**; an LLM adds a contextual prefix to each chunk. |
| **Embed** | Each chunk is embedded (default `intfloat/multilingual-e5-large`, 1024-d) and written to **pgvector**. Chinese text is CKIP-segmented for the BM25 `text_search_tsv`. |
| **Retrieve** | **Hybrid**: dense vector + sparse BM25, fused with Reciprocal Rank Fusion. |
| **Merge** | Leaves from the same passage **auto-merge** into their parent; single hits **expand** to neighbours. |
| **Rerank** | A cross-encoder (`bge-reranker-v2-m3` / `mxbai-rerank-base-v2`) re-scores for final Top-K precision. |
| **Answer** | For the `answer` path, `gpt-oss-120b` synthesises a response from the retrieved context, gated by a corrective-RAG confidence check. |

**Auth.** Agentic RAG is a *resource server*. FastMCP publishes `/.well-known/oauth-protected-resource/mcp`, which points clients at MCP Center; the server fetches MCP Center's JWKS once and verifies every bearer token offline (RS256 — signature, issuer, audience, expiry, scopes). MCP Center is never on the request path.

## Configuration

Settings live in `config/config.yaml`; secrets come from environment variables (`.env`). The ones you are likely to touch:

| Variable / setting | Default | Purpose |
|--------------------|---------|---------|
| `DATABASE_URL` | `postgresql://…/agentic_rag` | PostgreSQL DSN. Requires the **pgvector** extension. Use **sync** `postgresql://` (psycopg2), not `+asyncpg`. |
| `server.port` | `5039` | HTTP + MCP bind port. The MCP endpoint is `/mcp`. |
| `auth.enabled` / `auth.issuer` / `auth.audience` | `true` / MCP Center URL / this server's `/mcp` URL | OAuth 2.1 verification against MCP Center. |
| `rag.embedding.{model,dimension,base_url}` | `intfloat/multilingual-e5-large`, `1024`, `:7075/v1` | Embedding model; `dimension` **must** equal the model's real output dim. |
| `rag.llm.{model,base_url}` | `openai/gpt-oss-120b`, `:5052/v1` | LLM for Contextual Retrieval and the `answer` path. |
| `rag.rerank.{model,base_url}` | `mixedbread-ai/mxbai-rerank-base-v2`, `:8787` | Cross-encoder reranker. |
| `rag.asr.provider` | `fireredasr` | Audio transcription: `fireredasr` (local zh-TW) · `docling-whisper` · `openai-compatible`. |
| `rag.retrieval.*` | — | Top-K, similarity cutoff, BM25 candidates, `hybrid_fusion` (`rrf`/`concat`), auto-merging. Tunable live from the console, or via the **High Precision / Balanced / High Recall** presets. |

GPU selection is a `uv sync --group {rocm-r714|rocm-r713|cuda|cpu}` choice — see the top of `pyproject.toml` and `docs/testing/ENVIRONMENTS.md`.

## Use the console

The **Retrieval Terminal** console is a React SPA served at `/admin` (the previous single-file console remains at `/admin-classic`).

| Page | What you do there |
|------|-------------------|
| **Overview** | The live retrieval pipeline, corpus readouts, index health, and model-service status. |
| **Folders** | Create token-scoped folders; open one to upload files, watch per-file index status, view chunks, reindex, or delete. |
| **Search Playground** | Run a live query against a folder and see the **per-signal retrieval trace** — Vector · BM25 · Hybrid · Rerank ranks side by side — plus an optional RAG answer. |
| **Index Jobs** | Live stream of indexing jobs with progress; cancel a running job. |
| **Query Analytics** | Volume, latency, zero-result rate, top queries and knowledge gaps. |
| **Evaluation** | Auto-generate a question set and score `vector` / `hybrid` / `rerank` on Recall@k · MRR · nDCG@k. |
| **Settings** | Live, persisted configuration for Embedding / LLM / Contextual Retrieval / Reranker / Speech-to-Text / Retrieval, with retrieval presets and connection tests. |
| **System Health** | Live connectivity to every external dependency. |
| **Audit Log** | A timeline of every runtime settings change. |

## API reference

| Group | Endpoints |
|-------|-----------|
| **MCP** | `POST /mcp` — per-folder `Agentic_<folder>` tools (`search` / `list` / `read`); `GET /.well-known/oauth-protected-resource/mcp` |
| **Folders & files** | `/api/folders*` · `/api/folders/{id}/files*` (create, upload, list, download, delete — token-scoped) |
| **RAG** | `POST /api/rag/query` (hybrid retrieval + rerank) · `/api/rag/files/{id}/index`, `/reindex`, `/index/jobs*` (background indexing + SSE progress) |
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
