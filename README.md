# Agentic RAG MCP Server

[繁體中文](README.zh-TW.md)

> Released under the MIT License; see [`LICENSE`](LICENSE) for details.

A hierarchical RAG (Retrieval-Augmented Generation) MCP service. It turns arbitrary file
collections (PDF / Office / audio / images) into vector indexes that AI agents can query,
providing Hierarchical Auto-Merging Retrieval, Contextual Retrieval, cross-modal parsing
(Whisper ASR / Docling OCR), and full index lifecycle management.

---

## Design Principles

| Principle | How it is realized |
|------|----------|
| **Hierarchical retrieval** | Docling chunks → SentenceSplitter → build leaf + parent hierarchy → hit a leaf, then auto-merge back to its parent |
| **Contextual enrichment** | At index time, an LLM generates a contextual prefix for each chunk (carried over from the previous flat-RAG approach) |
| **Token-scoped isolation** | Each `user_token` sees a fully independent set of folders / files / vector tables |
| **Background-job-first** | Indexing always runs as a background job via `IndexingJobManager`, with full cancel / watchdog / restart recovery / persistence |
| **Configuration-driven** | `config.yaml` + `${VAR:-default}` environment-variable expansion; modules are loaded dynamically via `importlib` |

**Why Hierarchical + Contextual Retrieval**
- **Hierarchical (Auto-Merging)**: small leaf chunks give precise hits, but the LLM needs
  a larger context. Once leaves are hit, they auto-merge back into the parent, balancing
  precision against context richness — roughly **+20% recall** over a single fixed chunk size.
- **Contextual Retrieval (Anthropic, 2024)**: each chunk is prefixed at index time with an
  LLM-generated summary of "what this passage is about", solving the missing-context problem
  of traditional RAG — roughly **+35%** on needle-in-a-haystack tasks versus plain chunking.

---

## Quick Start

```bash
# 1. Install dependencies — torch follows a PEP 735 dependency group per deployment
#    hardware, so always pass --group
uv sync --group cuda        # NVIDIA (CUDA 12.6)
# uv sync --group rocm-r714  # AMD (ROCm 7.1.4; multi-arch wheel, covers gfx1151/90a/1201)
# uv sync --group rocm-r713  # AMD (ROCm 7.1.3)
# uv sync --group cpu        # CI / no GPU
#   ⚠️ A bare `uv sync` / `uv run` without --group swaps the three torch packages for the
#      default resolved versions, breaking an already-installed GPU stack. See the table at
#      the top of pyproject.toml and docs/testing/ENVIRONMENTS.md for details.

# 2. Start the three local model services (via vLLM or any OpenAI-compatible endpoint)
#    - Embedding (default intfloat/multilingual-e5-large, :5040; bge-m3 as fallback)
#    - LLM (default openai/gpt-oss-120b, :5052) — used by Contextual Retrieval
#    - Reranker (default mixedbread-ai/mxbai-rerank-base-v2, :8787; bge-reranker-v2-m3 fallback)
#    All URLs/models are configurable in config/config.yaml (ports follow config.yaml base_url)

# 3. Configure .env (secrets)
cp .env.example .env   # if present; otherwise edit config.yaml directly
#    At minimum set: DATABASE_URL, TOKEN_SERVER_URL

# 4. Edit config/config.yaml
#    - server.port (default 5031)
#    - rag.embedding / rag.llm / rag.rerank endpoints
#    - rag.docling.device (cuda / cuda:N / cpu)

# 5. Start (runs DB migration automatically)
uv run python main.py --config config/config.yaml
```

After startup:
- **Landing page**: `http://<host>:<port>/` — service overview + Swagger / Health links
- **Swagger UI**: `http://<host>:<port>/docs`
- **MCP endpoint**: `http://<host>:<port>/mcp`
- **Health check**: `http://<host>:<port>/health`

---

## Supported Formats (format v3 — aligned with the docling 2.124 documentation, tested per format)

> Tested = a minimal sample is generated and actually run through docling parsing, with
> content-retrieval assertions (`tests/integration/test_format_smoke.py`, 35 cases);
> coverage completeness is enforced by a programmatic invariant (`test_upload_formats.py`
> ref reads docling's official `FormatToExtensions` directly, so the test goes red
> automatically when upstream adds a format).

| Category | Extensions | Parser | Status |
|---|---|---|---|
| Plain text (direct read) | txt text json csv yaml yml xml conf log | read_text_robust (UTF-8/16/Big5 tolerant) | ✅ Unit-tested |
| PDF | pdf | docling (layout + TableFormer + OCR) | ✅ In production |
| Office OOXML | docx dotx docm dotm / pptx ppsx pptm potm ppsm / xlsx xlsm | docling | ✅ Smoke-tested |
| Legacy Office | doc dot xls xlt ppt pot pps | docling + LibreOffice (deployment needs soffice) | ✅ Smoke-tested (real files) |
| OpenDocument | odt ods odp | docling + odfdo | ✅ Smoke-tested |
| Markup / scientific | md qmd rmd html htm xhtml adoc asciidoc asc tex latex | docling | ✅ Smoke-tested |
| Mail / books / subtitles | msg eml epub vtt | docling | ✅ Smoke-tested (msg tested since v2) |
| Images | png jpg jpeg tiff tif bmp webp | docling + RapidOCR | ✅ In production (bmp uses the same engine) |
| Audio | wav mp3 m4a aac ogg flac | **AsrProvider** (local docling-whisper / local fireredasr (FireRedASR-AED-L, Traditional-Chinese output) / cloud openai-compatible, configured via `rag.asr`) | wav/mp3 ✅ in production; the 4 new types share the same pipeline (ffmpeg decode); fireredasr pending evaluation-material acceptance |
| Video | — (planned, ref) | docling video (ASR + key frames) | 🧪 Pipeline verified end to end (mp4 SUCCESS); pending speech-sample content validation before adoption |

**Deliberately excluded after testing** (listed by docling but which the backend cannot open
in practice): `ott/ots/otp` (ODF templates) and `potx` (PowerPoint template) — the rationale
is recorded in ref's `_EXCLUDED_EXTS`; they will take effect automatically once removed
after an upstream fix.

---

## Primary REST API

### File / folder management (`/api/folders` / `/api/{folder_id}/file`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/folders/` | Create a folder |
| `GET` | `/api/folders/` | List folders / look up by name or ID |
| `DELETE` | `/api/folders/{folder_id}` | Delete a folder (automatically cancels any in-progress indexing job) |
| `POST` | `/api/folders/{folder_id}/file` | Upload a file (form-data, optional auto_index) |
| `GET` | `/api/folders/{folder_id}/files/{file_id}/download` | Download the original file |

### Indexing (`/api/rag`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/files/{folder_id}/index` | Index a whole folder in the background (returns job_id) |
| `POST` | `/file/{file_id}/index` | Index a single file synchronously |
| `POST` | `/files/{folder_id}/reindex` | Delete then re-index |
| `GET` | `/files/{folder_id}/index/jobs/{job_id}` | Query job status |
| `GET` | `/files/{folder_id}/index/jobs/{job_id}/events` | **SSE realtime progress** (EventSource push) |
| `DELETE` | `/files/{folder_id}/index/jobs/{job_id}` | Cancel a running job |
| `GET` | `/index/jobs` | List jobs + counts per status |
| `GET` | `/file/{file_id}/index/status` | **Single-file status query** (`indexed`/`indexing`/`queued`/`failed`/`not_indexed`) |
| `GET` | `/indexed-files` | List indexed files in a folder |
| `DELETE` | `/files/{folder_id}/index` | Delete an entire folder's index |
| `DELETE` | `/file/{file_id}/index` | Delete a single file's index |

### Query (`/api/rag/query`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/rag/query` | Hybrid retrieval (vector + BM25) + optional rerank |

`top_k` / `similarity_cutoff` / `sparse_top_k` / `hybrid_alpha` all default from
`config.rag.retrieval.*` and can be overridden per request.

---

## Index Reliability — `IndexingJobManager`

Full lifecycle control. Key design points:

| Feature | Behavior |
|---|---|
| **Folder lock** | The same folder cannot run two indexing jobs concurrently (returns 409 Conflict) |
| **Cancel** | `DELETE /index/jobs/{id}` aborts an in-flight job immediately |
| **Watchdog** | Scans every 60s; if a job has no heartbeat for >10min it is marked failed |
| **Restart cleanup** | On server restart, any leftover PENDING/RUNNING jobs in the DB are flipped to failed (reason=server_restart) |
| **Folder-delete cleanup** | Before deleting a folder or a whole folder index, all in-flight jobs are cancelled first |
| **Per-file timeout** | Default 10 days per file (large-file OCR on slow machines; env `RAG_PER_FILE_TIMEOUT_SECONDS` / config `rag.indexing.per_file_timeout_seconds`) |
| **Job-level timeout** | Default 10-day backstop for the whole job (env `RAG_JOB_TIMEOUT_SECONDS`) |
| **File-existence gate** | If a user deletes a file mid-index, three gates abort early (no wasted GPU) |
| **`PARTIAL_SUCCESS` status** | When 9/10 succeed and 1 fails, the job status is `partial_success` rather than being misreported as succeeded |
| **Content-hash idempotency** | Re-indexing an unchanged file (same content) short-circuits — roughly **60× faster** |
| **Embedding retry** | Transient disconnects / 5xx / 429 / timeout are retried 3 times with exponential backoff (1→2→4s) |
| **Per-file timing audit** | Each file records `load_ms` (Docling) / `index_ms` (context-gen + embed + write) / `total_ms` |
| **DB persistence** | All job state is write-through to the `IndexJobs` table and remains queryable across process restarts |

---

## Directory Layout

```
├── app.py                           # FastAPI + FastMCP assembly (wire-up only)
├── main.py                          # CLI entry point (startup + DB migration)
├── config/
│   ├── config.yaml                  # ★ main configuration
│   └── instructions.md              # ★ MCP Instructions (also rendered on the landing page)
├── assets/
│   ├── docling_models/              # Docling models (gitignored, added at deploy time)
│   ├── whisper_models/              # Whisper models (same)
│   └── hf_tokenizers/               # HF tokenizers (same)
├── docs/
│   ├── integrate-existing-api.md    # Guide to integrating an existing API
│   └── testing/                     # ★ Testing process docs (TEST_PLAN / STRATEGY / specs / test-cases)
├── db/
│   ├── db.py                        # ORM Models (Folder / File / FileIndex / IndexJob)
│   ├── baseDB.py                    # Generic CRUD base class
│   ├── filedb.py / folderdb.py / fileindexdb.py / indexjobdb.py
│   ├── cached_folderdb.py           # Folder-query cache
│   └── migrate.py                   # Alembic automatic migration
├── src/
│   ├── auth/                        # Token Server remote authentication
│   ├── middleware/                  # request_id / auth / error handler
│   ├── api/
│   │   ├── dependencies/            # FastAPI Depends helpers
│   │   └── router/
│   │       ├── index.py             # landing page /
│   │       ├── health.py            # /health
│   │       ├── folder_api.py        # /api/folders/*
│   │       ├── file_api.py          # /api/folders/{id}/file*
│   │       ├── rag_indexing.py      # /api/rag/files/*, /api/rag/file/*
│   │       └── rag_query.py         # /api/rag/query
│   ├── adapter/
│   │   ├── rag.py                   # Thin facade
│   │   ├── rag_context.py           # Shared RAG state (embedding / indexer / vector_store_mgr)
│   │   ├── rag_indexing.py          # Indexing service (folder / file-list / auto-index)
│   │   ├── rag_query.py             # Query service (flat hybrid + agentic three-mode)
│   │   ├── rag_maintenance.py       # Deletion + indexed-file listing
│   │   ├── folder.py / file.py      # Folder / File adapter
│   │   └── model.py                 # Adapter-level pydantic types
│   ├── domain/
│   │   ├── exceptions.py            # DomainException family (translated to HTTP status)
│   │   └── rag/
│   │       ├── hierarchical_indexer.py    # Index orchestration (context→hierarchy→embed→write)
│   │       ├── document_loader.py         # File → Document, three load paths (text/docling/fallback)
│   │       ├── leaf_splitter.py           # Structural splitting (preserves tables) + token-budget refine
│   │       ├── hierarchy.py               # Leaf↔parent tree construction
│   │       ├── docling_loader.py          # Docling wrapper + Whisper ASR + hallucination defense
│   │       ├── agentic_handlers.py        # MCP three-mode (search/list/read) logic
│   │       ├── folder_acl.py              # token↔folder permission checks (isolated at the DB query layer)
│   │       ├── dto.py                     # Cross-layer pure data types (FileRequest, etc.)
│   │       ├── context_generator.py       # LLM-based contextual-prefix generation
│   │       ├── index_service.py           # FileIndex DB helpers
│   │       ├── index_job_manager.py       # Background job control (lifecycle / persistence / SSE)
│   │       ├── query_engine.py            # Hybrid retrieval (vector + BM25)
│   │       ├── auto_merging.py            # Merge leaves back into the parent after a hit
│   │       ├── reranker.py                # Cross-encoder rerank
│   │       ├── vector_store_manager.py    # PGVector store management
│   │       └── chunk_lookup.py            # Chunk lookup
│   ├── fastmcp_tools/                     # MCP tool registration (per-folder + global)
│   ├── infrastructure/
│   │   └── cache/                         # Query / folder cache
│   ├── storage/file_storage.py            # File storage (token/folder isolation)
│   ├── config/                            # YAML loading + Pydantic models
│   ├── utils/                             # runtime_paths / db_bootstrap
│   └── log.py                             # loguru configuration
├── scripts/                               # One-off test / utility scripts
├── storage/                               # ★ Landing zone for uploaded files (gitignored)
├── ROADMAP.md                             # Development progress: delivered / planned
└── tests/                                 # pytest unit tests (zero external dependencies, see docs/testing/)
```

---

## Development / Debugging

### Watch background job progress (SSE)

```bash
TOKEN=<your-bearer-token>
curl -sN -H "Authorization: Bearer $TOKEN" \
  "http://localhost:5031/api/rag/files/1/index/jobs/<job_id>/events"
```

Every state change (progress / file done / status transition) pushes a `data: {...}` event.
The connection closes automatically once the job reaches a terminal status.

### Check a single file's current status

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://localhost:5031/api/rag/file/<file_id>/index/status"
```

Returns one of `indexed` / `indexing` / `queued` / `failed` / `not_indexed`, including
progress and any error message.

### Re-index (skip_existing=false)

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  "http://localhost:5031/api/rag/files/<folder_id>/index?skip_existing=false"
```

If content is unchanged (same content_hash), the actual embed is skipped idempotently,
taking only tens of milliseconds.

### Inspect job persistence (backend DB)

```sql
SELECT job_id, folder_id, status, total_files, processed_files, last_updated_at
FROM "IndexJobs"
ORDER BY started_at DESC
LIMIT 10;
```

---

## Testing

The testing methodology follows the [FuSa Group "A High-Level Overview of Software Testing
Methodologies"](https://fsg.tw/software-testing-methodologies-guide-a-high-level-overview/).
Full process docs are in [`docs/testing/`](docs/testing/README.md) (test plan, strategy,
and REQ → TC → pytest traceability).

```bash
# Unit tests (zero external dependencies: no DB / vLLM / GPU / network)
uv run --no-sync pytest

# With coverage (exits non-zero if below the pyproject fail_under threshold)
uv run --no-sync pytest --cov --cov-report=term-missing
```

> On GPU machines, keep `--no-sync` (a bare `uv run`'s implicit sync swaps torch for the
> non-GPU group version); see [`docs/testing/ENVIRONMENTS.md`](docs/testing/ENVIRONMENTS.md).

| Level | Status |
|------|------|
| Unit tests | ✅ `tests/test_*.py` — pure-logic automation (chunking / caching / filename safety / config / auth cache / schema) |
| Integration / system tests | 🔶 Deployment-environment validation (needs pgvector + model services); automation tracked in the [ROADMAP](ROADMAP.md) |
| Acceptance (RAG quality) | 🔶 Manual acceptance; the regression Q&A set is tracked in the [ROADMAP](ROADMAP.md) |

Coverage is measured over the modules in unit-test scope; pipeline modules with heavy
external dependencies are in the `pyproject.toml` omit list (each with a per-file rationale)
and fall under integration-test scope. See section 5 of
[`docs/testing/TEST_PLAN.md`](docs/testing/TEST_PLAN.md).

---

## Development Progress / Roadmap

Delivered capabilities and planned work (CI, integration-test automation, RAG-quality
regression, performance benchmarks) are in [`ROADMAP.md`](ROADMAP.md).

---

## Control Center (/) — Operations Console

Open **`http://<host>:<port>/`** in a browser (the root URL; `/admin` is an alias of the
same page) to reach the all-English operations console (no authentication — a product
decision for intranet deployment; API documentation is served from `/docs`):

- **Overview**: KPIs (folders/files/chunks/jobs), index-health distribution, live progress
  of in-progress jobs, and the **live health** of the four model services (endpoint probe
  Online/Offline, ASR weights Ready)
- **Folders**: folder CRUD (creation requires an owner token — folders are token-scoped),
  file upload (auto-index) / download / delete, incremental indexing and full rebuild,
  and drill-down into file-level index status and failure reasons
- **Index Jobs**: recent job progress / messages / start-end times, cancellable while running
- **Search Playground**: run live retrieval against a folder to see hit chunks / rerank
  scores / page-number provenance (demonstrating RAG quality)
- **File-stage visualization**: after upload, the file list shows Parsing→Context→Embedding→Saving
  in real time (1-second polling) with a mini progress bar; indexed files can display their
  split chunks (including the CR prefix)
- **Overview sparkline**: 7-day indexing-volume trend; KPI cards are clickable for navigation
- **Dangerous-operation guard**: deleting a folder requires typing its name to confirm
- **Settings**: hot-edit model and retrieval parameters at runtime (see below)

The management API (`/api/admin/manage/*`) uses **act-as-owner**: it calls the existing REST
endpoint functions with the folder owner's token, so safety logic for job cancellation /
soft delete / vector cleanup is never duplicated.

| Hot-editable | How it takes effect |
|---|---|
| LLM / Rerank / ASR models and endpoints | Effective immediately (picked up by the next request) |
| Retrieval parameters (top_k / cutoff / α / auto-merging / neighbor expansion) | Effective immediately |
| Contextual Retrieval toggle and behavior | Effective on the next indexing job |
| **Embedding model/dimension/prefix** | Effective **but with a warning**: the vector space is incompatible, so existing folders must be re-indexed |
| Infrastructure such as DB / port / auth | Not hot-editable (anything outside the allowlist is rejected); requires a restart |

- Changes are persisted to the DB (`RuntimeSettings` table) and **survive restarts**; an
  "overridden" badge lets you restore the config.yaml factory value in one click
- Each model-service block has a "Test connection" (hits `{base_url}/models` to verify reachability)
- API: `GET/PUT /api/admin/settings`, `DELETE /api/admin/settings/{path}`, `POST /api/admin/probe`

## Configuration Override Precedence

1. **Environment variables** (`RAG_INDEXING_CONCURRENCY`, `RAG_PER_FILE_TIMEOUT_SECONDS`, `RAG_JOB_TIMEOUT_SECONDS`, etc.)
2. **`config.yaml` `${VAR:-default}` expansion** (deployment secret injection) — ⚠️ the
   expansion logic is not yet implemented, so placeholders enter the config literally;
   tracked as [ref-003](docs/testing/defect-reports/ref-003.md) — write literal
   values until it is implemented
3. **RuntimeSettings runtime overrides** (/admin hot-edit, DB-persisted — layered on top of the yaml)
4. **Hard-coded values in `config.yaml`**
5. **In-code fallbacks**

---

## License

[MIT](LICENSE)
