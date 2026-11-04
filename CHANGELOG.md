# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Console aggregates (Overview KPIs and trend, Index Jobs, Query Analytics) are now limited to the signed-in token's own folders for `rag:admin` tokens too. Before, an admin token saw every owner's counts, job folder names and search queries, contrary to the rule that `rag:admin` unlocks global settings only. Storage totals and the settings audit log stay admin-only.

### Retrieval quality and cost (Phase 1) — measured on the new golden set
Golden set: 205 questions over a public 19-document corpus (`evals/rag/`), baseline vs
this release on the same index host (`evals/baselines/rag-2026-11-03-*`):
MCP Recall@10 0.973 → **0.981**, Recall@5 0.957 → 0.976, nDCG@10 0.936 → 0.938,
Chinese-question/English-document Recall@10 0.80 → **0.90**; tokens per MCP search
8.2k → **2.6k** (duplicated text 48% → 7%); search p50 latency 461 → 685 ms (the
larger rerank pool). **Re-index every folder** (chunking v3, prompt v3); the console
flags them.

### Added
- **MCP search can say "nothing relevant found".** Every search response now
  carries `top_relevance` (the reranker's best score over all candidates, before
  `score_threshold`), `min_relevance` and `low_relevance` — `true` when nothing came
  back or `top_relevance` is below the new `rag.retrieval.min_relevance` (live-editable,
  console *Settings → Retrieval*; `0` = off, the shipped config sets `0.9` for
  mxbai-rerank-v2). The hint then tells the agent the folder most likely does not
  answer the question. Each reranked result carries `relevance`; results below the
  cutoff are marked `low_relevance: true`. Results are flagged, never dropped
  (`rerank.score_threshold` drops). New fields only — nothing renamed or removed. On
  the golden set (`evals/baselines/rag-2026-11-03-min-relevance.*`): out-of-scope
  questions flagged 0% → **100%**, answerable questions wrongly flagged 0%, R@5 / R@10
  unchanged (0.976 / 0.981); near-miss unanswerable questions (the right document, a
  detail it does not state) stay unflagged — the reranker scores them as relevant
  (best-score AUROC 0.79), so no cutoff separates them without losing recall. About
  +60 tokens per response.
- Eval harness v2: relevance-cutoff sweep (near-miss and out-of-scope rejection,
  answerable wrongly rejected, R@5/R@10 as if dropped) and 20 hand-written
  out-of-scope questions (`evals/rag/out_of_scope_qa.yaml`).
- Golden-set retrieval eval (`scripts/eval_retrieval.py`, `evals/rag/`), per-stage
  search timings in the trace, `docs/operations-postgres.md`.
- **MCP Gate revocation feed** (`auth.revocation`, off by default). A background
  poller mirrors MCP Gate's `POST /oauth/revoked` and rejects revoked tokens on
  `/mcp`, the REST API, console sign-in and existing console sessions, within one
  poll interval instead of at expiry (PATs can live a year). Needs a confidential
  client bound to Folio as its resource server; `fail_mode` `open` (default) or
  `closed`; status in `/health/detailed` under `checks.revocation`. See README →
  *Token revocation*.
- **Per-file Retry and Cancel in the console.** Retry (or Reindex) any idle
  file; cancel a queued or indexing file without stopping the rest of its job
  (`POST …/files/{id}/retry`, `POST …/files/{id}/cancel`, bulk
  `POST …/folders/{id}/retry`). Retries run as background jobs under the
  folder lock instead of inside the request.
- **"Interrupted" status.** Jobs that were running when the server stopped are
  marked `interrupted` (not `failed`), and so are the files they left
  unfinished, so the console offers Retry on them. Files cancelled one by one
  show as `cancelled`.
- **Rebuild FTS runs as a background job** (`kind: rebuild_fts`) with files
  done/total progress, cancel and heartbeats; the endpoint returns 202 and the
  job. A 5.6k-chunk folder used to outlive the HTTP timeout. New
  `GET …/folders/{id}/jobs/{job_id}` for polling.
- **Lookup indexes on vector tables** (`node_id`; `file_id` + `chunk_index`),
  added to existing tables by migration and to new ones after their first
  write. Neighbour expansion, read mode and per-file deletes no longer scan.
- **Backend CI** (`.github/workflows/backend.yml`): unit tests with the
  coverage gates and the integration group against pgvector. `uv.lock` is now
  committed.
- Phone-width console: a menu button opens the sidebar (it was unreachable
  below 900px).
- **Clickable answer citations** in the Search Playground; the non-streaming
  answer now shows its kept / dropped / confidence line too.
- **Knowledge gaps judged on relevance.** Query logs record the reranker's best
  score (`QueryLogs.top_score`, Alembic migration `20261102a1b2c6`, which also
  adopts the `QueryLogs` table into Alembic). Analytics lists a question as a gap
  when its latest search per folder returned nothing or scored below
  `rag.rerank.gap_threshold` (default 0.5); asking again after adding documents
  clears it. New *Knowledge Gaps* KPI on the Analytics page.
- **mxbai-rerank-v2 chat template.** `rag.rerank.auto_template` (default on)
  wraps queries and documents in the model's official Qwen chat prompt when the
  configured templates are the default or the old chat-less form. Scores become
  calibrated, so `score_threshold: 0.5` filters for that model.
- **Pluggable Chinese segmentation** — `rag.retrieval.segmenter: ckip | jieba |
  bigram` (default `ckip`, or `bigram` under `profile: cloud`; editable live). One function serves indexing,
  Rebuild FTS and queries; CKIP and jieba are imported only when selected
  (`jieba` is an optional extra) and `bigram` needs no model at all. Each
  folder's table records the segmenter that built its keyword index; the console
  flags folders that need *Rebuild FTS* after a switch
  (`GET /api/admin/manage/folders/{id}/fts`), and Rebuild FTS uses the
  configured segmenter.
- **REST auto-merge.** `POST /api/rag/query` applies the same auto-merge rule as
  the MCP search tool; `auto_merge` (true / false / null = follow
  `rag.retrieval.auto_merging.enabled`) overrides it per request. Merged results
  carry `node_role: "parent"` and `merged_from`.
- **Cloud profile** (`rag.profile: local | hybrid | cloud`). Every model role —
  embedding, contextual-prefix / answer / eval LLM (`rag.contextual_llm`,
  `rag.answer_llm`, `rag.eval_llm`, inheriting from `rag.llm`), reranker, document
  parsing (vision) and speech-to-text — takes an OpenAI-compatible `base_url`,
  `model`, `${ENV}` key, `timeout` and `max_retries`. `cloud` refuses in-process
  providers and literal API keys (startup fails; console writes get 422);
  `hybrid` warns about literal keys; `local` is the default and existing configs
  behave as before. See README → Cloud profile.
- **Cloud PDF and image parsing** (`rag.parsing`): a vision model transcribes every
  page; pages are stitched into one document (previous-page context, continued
  tables and paragraphs joined) so headings and breadcrumbs survive page breaks,
  chunks keep their start page, failed pages are retried and then marked, and
  per-file / per-day page caps bound the cost.
- **Rerank API styles** `rag.rerank.api: score | cohere | jina | generic` (+
  `endpoint_path`), behind the existing reranker.
- **Segmented cloud speech-to-text**: long recordings are uploaded in
  silence-aligned pieces with retries, and empty-despite-speech or looping
  transcripts are re-requested (`rag.asr.language`, `segment_seconds`, `timeout`,
  `max_retries`).
- **Cost guards for contextual prefixes**: `max_calls_per_file`,
  `max_tokens_per_day`. Chunks past a cap get no prefix that run; the next
  incremental re-index fills them in.
- **Per-role console settings and probes**: a Model Roles panel (where each role
  runs, key set / from which env var — never the key) and
  `POST /api/admin/probe/{role}`, which sends one minimal real request with the
  saved settings. Console key fields accept `${ENV_VAR}`; the reference is
  stored, not the key.
- `tests/test_cloud_profile_no_local_models.py` proves a cloud deployment imports
  no torch, transformers, sentence-transformers, docling models, Whisper,
  FireRedASR or CKIP, and that Chinese keyword search still segments (bigram).
- Under `profile: cloud` the Chinese segmenter defaults to the model-free
  `bigram`; an explicit `rag.retrieval.segmenter: ckip` is rejected at startup
  and in the console (422), pointing to `bigram` / `jieba` or to
  `profile: hybrid` with `uv sync --extra local`.
- **Console login.** The `/admin` console now signs in with an MCP Gate access
  token for this service (`POST /api/console/session`). The token is verified
  offline once and exchanged for an HttpOnly, SameSite=Strict session cookie
  that carries only its identity (jti / sub / scopes), never the token. Scripts
  can call `/api/admin/*` with `Authorization: Bearer <token>` instead.
  New settings `auth.console.{session_hours, session_secret, secure_cookie,
  admin_scope}`, all with defaults.
- **`rag:admin` scope.** Changing runtime settings, probing arbitrary endpoints,
  and reading the settings audit log and server-wide resources need it; without
  it Settings and System Health are read-only.
- **Section-aligned chunks with breadcrumbs.** Small chunks merge only within
  one section, and a merged chunk keeps the headings its pieces share; parents
  are grouped by section. Every leaf carries a breadcrumb (file › heading path,
  xlsx sheet names included) that is embedded and keyword-indexed with it, so
  same-structure documents (two product manuals, two price sheets) are told
  apart. Tables stay whole up to the full safe token window and are split at
  row boundaries beyond it.
- **Incremental re-indexing.** Re-indexing a file reuses the contextual prefix
  and embedding of every unchanged chunk (content hash + model keys); chunk ids
  are stable. Editing one section costs about one LLM call and one embedding.
- **Chunking pipeline version per file** (`FileIndices.pipeline_version`).
  Files indexed by an older pipeline show *Re-index recommended* in the console
  (with a per-file re-index button) and `reindex_recommended` in the REST status.
- **Parse warnings.** Pages docling could not parse are stored on the file
  (`FileIndices.index_warning`), shown as *Partial* in the console and returned
  as `warning` by the REST status endpoints.
- Bundled chunk-counting tokenizer (`resources/tokenizers/xlm-roberta`, the
  tokenizer of multilingual-e5 and bge-m3) and a `tokenizers` dependency.

### Changed
- **MCP search responses** are budgeted (`rag.retrieval.mcp_response_token_budget`,
  6000): neighbour windows stay inside the hit's section, overlapping windows merge per
  file, `expand_context_neighbors` defaults to 1, and original files are no longer
  attached by default (`return_resource_files: false`; when on, only the top hit's file,
  up to `resource_file_max_bytes`). Responses report `score_type`, `omitted_results` and
  `response_tokens_estimated`.
- **MCP tool contract**: `similarity_cutoff` is no longer advertised (accepted and
  ignored — it had no effect under rank fusion and rerank); `top_k` / `expand_context`
  defaults are read live; `top_k` is clamped to 1..20; read mode pages with
  `offset` / `next_offset`.
- **The tool payload is an explicit MCP text block.** Without an attachment, FastMCP
  used to deliver it double-encoded (`["{…}"]`).
- **Rerank pool**: the reranker sees max(3 × top_k, `rag.retrieval.rerank_candidates`
  = 60) fused candidates instead of 3 × top_k — cross-lingual hits sat at fused rank
  50+ and never reached it.
- **BM25**: one SQL statement (all-terms matches first), the folder's own stamped
  segmenter, and no bigrams across Chinese particles.
- **Chunking (pipeline v3)**: Chinese sentences split at 。！？；; no duplicated overlap
  in the plain-text/audio splitter; parents sized with the real tokenizer; CSV keeps
  its header row in every chunk; audio transcripts carry timestamps in ~5-minute
  blocks; one context prefix per split table.
- **Contextual prefixes** never use the model's reasoning text, are not cached when
  empty or failed, and failures surface as a file warning (prompt v3 regenerates
  prefixes cached by older versions).
- **Database**: all folder vector stores share the two engines (40 folders: 80 → 2
  backend connections); store cache is an LRU; indexing runs on its own thread pool
  and `max_concurrent_jobs` defaults to 4 (values above 32 are capped); QueryLogs,
  SettingsAudit and IndexJobs have retention (`database.retention.*`); duplicate
  FileIndices index dropped (migration `20261103d0b001`).
- Blocking DB / LLM calls moved off the event loop (console answers, query logging,
  folder ACL, adapters).
- **Renamed from Agentic RAG to Folio.** The console, MCP server name and
  instructions, API title, log banners and docs say Folio; the Python
  distribution is `folio-rag`. Per-folder MCP tools are now `Folio_<folder>`
  (clients pick the new names up on their next tool listing). A config.yaml
  that still has the old defaults (`app.tool_prefix: "Agentic"`,
  `app.name: "Agentic RAG"`, …) loads with the Folio values and logs a
  warning; a custom prefix is kept. Environment variables are now
  `FOLIO_CORE_DUMPS`, `FOLIO_INDEXING_CONCURRENCY`,
  `FOLIO_PER_FILE_TIMEOUT_SECONDS` and `FOLIO_JOB_TIMEOUT_SECONDS`; the old
  names still work with a deprecation warning. The console's saved theme moves
  to the `folio_theme` key on first load. Database names, the port, the
  audience URL and storage paths are unchanged.
- **The storage-root override is `FOLIO_STORAGE_ROOT`.** The previous variable
  is no longer read.
- **Evaluation scores the production search path.** Modes are `vector`,
  `bm25`, `rrf`, `rerank` (when a reranker is configured) and `production` — all
  through the shared hybrid search; `production` is the same call as
  `/api/rag/query` with the live fusion, rerank, thresholds and auto-merge (a
  merged section counts as a hit when it contains the gold chunk). The `hybrid`
  mode, which always equalled `vector`, is gone, and the old BM25 numbers skipped
  segmentation. Baseline deltas only compare runs on the same question set, k and
  harness version, so older runs are not compared with new ones. Generated
  questions follow the passage's language.
- **One retrieval path.** REST, the MCP search tool, the console Search
  Playground and the evaluation harness now all retrieve through
  `src/domain/rag/hybrid_search.py`. The MCP path used llama-index's native
  hybrid mode on the raw query (no Chinese segmentation, no RRF), so agents got
  almost no BM25 hits on Chinese questions.
- **BM25 rewritten.** Keyword search runs its own SQL over leaf chunks only,
  requires every query term first (AND) and falls back to any term (OR) to fill
  up, drops question / function words, and runs concurrently with the vector
  search. A BM25 failure now degrades to vector-only results instead of failing
  the query.
- `hybrid_fusion: concat` is now "dense hits, then BM25-only hits" (left for the
  reranker to order) instead of llama-index's native hybrid, which behaved as
  vector-only for Chinese. `hybrid_alpha` is deprecated and ignored.
- MCP tool names are always client-safe (`[A-Za-z0-9_-]`, max 64): folder names
  with spaces, CJK or symbols become `Folio_<ascii>_<hash>`. Tool
  descriptions are English.
- The settings API validates ranges and choices (e.g. `max_concurrent ≥ 1`,
  `score_threshold` 0–1, known providers — including the per-role LLM, vision
  and `openai-compatible` providers —, rerank API styles, parsing modes,
  timeouts / retries / page and token caps, http(s) URLs) and answers 422.
- Index-job progress ticks are written to the database at most every 5 s
  (status changes still immediately); SSE still sees every tick.
- `logging.max_size` / `logging.backup_count` now rotate and prune log files.
- `RuntimeSettings`, `SettingsAudit` and `QueryLogs` are created by Alembic.
- Coverage gate set to the measured 91% (it was 95% but never enforced).
- **The console is scoped to the signed-in token.** It shows and manages only
  the folders owned by the token's `jti` (the same ones REST / MCP callers with
  that token reach); folders it creates belong to it. The *Owner Token* field
  and the console's ability to act as any folder's owner are gone. Cross-folder
  views (overview, jobs, analytics) cover the whole server only for `rag:admin`.
- Settings changes are audited under the token's `sub`.
- Cookie-authenticated console writes require the `X-RAG-Console: 1` header and
  a same-origin `Origin` / `Referer` (CSRF protection).
- With `auth.enabled: false` the console stays open (dev mode) and the server
  logs a warning at startup.
- **Search results show the original document text.** The LLM contextual
  prefix is stored apart from the chunk text (`metadata.context_prefix`) and,
  with the breadcrumb, is used only to find the chunk (embedding, BM25,
  rerank). The console's chunk viewer shows it separately as *AI context*.
- **Contextual prefixes are written from the document outline plus the chunk's
  own section** (at most 12,000 characters around the chunk) instead of the
  first `max_doc_chars` of the document, with a language-neutral prompt that
  follows the document's language. Prompts are far shorter.
- A file's chunks are replaced in one transaction, with the full-text vector
  written in the same INSERT; parents are no longer keyword-searchable.
- With contextual retrieval on, a chunk's own text is capped at 330 tokens
  (was 380) to leave room for the breadcrumb in a 512-token embedding window.
- **Re-index existing folders** to get the new chunks, breadcrumbs and
  prefixes. Until then they keep working unchanged; the first re-index of a
  file after upgrading is a full one (older chunks have no content hash).
- `main.py` defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Install: the in-process model stack is now the `local` extra.** Local and hybrid
  deployments install with `uv sync --extra local --group <gpu>`; a plain
  `uv sync` is the torch-free core for the cloud profile. A `uv sync --group <gpu>`
  without `--extra local` removes the local stack from an existing venv.
  `opencc` stays in the core install (no torch): the Simplified → Traditional
  repair of contextual prefixes runs in every profile. Backend CI installs
  `--extra local --extra jieba --group cpu`, and a third job runs the unit suite
  on the core install (the cloud-profile proof included).
- Startup logs where every model role runs and warms docling only when a role uses
  its local models; CKIP is not loaded under the cloud profile.
- The unauthenticated `/v1/ocr/general` answers 503 when PDFs/images are parsed
  in the cloud instead of spending the vision budget on anonymous uploads.
- **The authorization server is now called MCP Gate** (formerly MCP Center). Docs
  and log messages follow the new name; `src/auth/mcp_center_auth.py` is now
  `src/auth/mcp_gate_auth.py` and `McpCenterTokenVerifier` is
  `McpGateTokenVerifier`. Configuration and issued tokens are unaffected.

### Fixed
- `GET /health/detailed` reported `connection_pool` as `degraded` on every call and logged
  `Connection pool stats unavailable: 'QueuePool' object has no attribute 'checked_in_connections'`:
  it called pool methods that do not exist. It now reports `size`, `checked_in`, `checked_out` and
  `overflow` (extra connections beyond `pool_size`, never negative).
- 4xx instead of 500: playground parameters of the wrong type or range,
  queries over 2000 characters or over the embedding model's context window
  (422), unsupported upload types (415), oversized uploads (413), folder names
  over 255 characters (422).
- macOS (NFD) folder names such as "café" were rejected; names are now
  NFC-normalised. Upload file names drop client paths and control characters,
  and the download `Content-Disposition` escapes quotes.
- Resetting a setting whose whole section is missing from `config.yaml`
  returned 500 after deleting the override; `${ENV}` placeholders are now
  expanded on reset.
- "Reindex selected" started one inline reindex per file outside the folder
  lock; it is now a single queued job.
- The Jobs page cancel dialog said cancellation waits for the current file; it
  is immediate.
- Thirteen stale unit tests (English messages, the `content_type` key, env
  expansion, a missing Big5 sample).
- The REST query cache ignored fusion, `rrf_k`, the text-search config, the
  segmenter, the reranker threshold / templates and the merge rule, and was not
  cleared when retrieval, rerank or embedding settings changed at runtime —
  results could be stale for up to 5 minutes. The key now covers every
  result-affecting setting and such changes clear it.
- **Console answers (CRAG), both JSON and streaming:** citations kept their
  chunk's position among the *kept* chunks, so `[n]` pointed at the wrong result
  card once a chunk was dropped — chunks now keep their result-card number and
  are labelled with their source; citations of chunks the model was not given
  are removed and `[1, 3]` becomes `[1][3]`; a grader reply that is not a list
  of booleans (or `"false"` read as true) no longer yields `high` confidence but
  `unverified`; answers follow the question's language; `【n】` markers and
  Markdown bold are cleaned; an empty streamed answer gets an explanatory
  message instead of a blank card.
- MCP searches were not logged, so analytics and knowledge gaps ignored all
  agent traffic.
- When the reranker failed, queries returned all 3 × `top_k` candidates; they
  now return `top_k` in retrieval order, and the degraded result is not cached.
- Concurrent REST queries on one folder could run with each other's `top_k` /
  cutoff: the cached per-folder engine was reconfigured per request. Settings
  are now passed with each call.
- Parent chunks crowded leaves out of BM25: they were filtered in Python after
  the SQL `LIMIT`, so a query could get almost no keyword hits.
- `rag.retrieval.default_sparse_top_k` (and the REST `sparse_top_k`) had no
  effect; it now sets the BM25 depth.
- The `openai` embedding and LLM providers ignored `base_url`, so an
  OpenAI-compatible cloud had to be configured as `vllm`; it is honoured now, and
  `openai-compatible` is a provider of its own.
- Chunk token counting never imports transformers (and with it torch) under the
  cloud profile; a tokenizer directory without `tokenizer.json` is skipped for the
  bundled tokenizer there.
- The "Whisper weights not found" startup line no longer appears when local
  Whisper is not the configured speech-to-text provider.
- Retrying a file's indexing checked folder ownership but not that the file was
  in that folder.
- The chunk viewer no longer reads a file name from outside the folder.
- Eval reports, question sets and history are keyed by folder id: two owners'
  same-named folders used to share (and could read) each other's results.
- Small chunks from different sections were merged and labelled with the first
  piece's heading, so paragraphs carried the wrong section.
- Every chunk past character ~60,000 of a long document got a contextual prefix
  written from the start of the document.
- The contextual-retrieval prompt hard-coded Chinese examples and produced
  Chinese prefixes for English documents. Simplified characters that drift into
  prefixes of Traditional documents are converted, and zero-width characters
  are stripped from all indexed text (they split keyword tokens).
- docling stage errors, CUDA out-of-memory included, silently dropped pages
  (the result was `PARTIAL_SUCCESS` and nothing raised). The result status is
  now checked; an OOM releases the GPU cache and retries with a smaller stage
  batch (4 → 2 → 1), other partial results are retried once, and pages still
  missing are reported on the file.
- `rag.docling.ocr_enabled: false` had no effect (it set an environment variable
  docling never reads); `rag.asr.enabled: false` still transcribed audio through
  docling's whisper. Both switches now work, and `/v1/ocr/general` /
  `/v1/transcriptions` return 503 when switched off.
- With OpenAI or Azure embeddings every docling-parsed file failed on a fresh
  clone (the chunk tokenizer name was `None`); token counting never downloads a
  tokenizer at runtime now.
- UTF-8 HTML files without a `<meta charset>` were decoded with a guessed
  legacy codepage (short Chinese pages came out as mojibake); undeclared UTF-8
  HTML is now marked as UTF-8 before parsing.
- Re-indexing a file left a window with no chunks, and new rows had no
  full-text vector until a separate CKIP pass ran (and none at all if it failed).

### Removed
- The inherited `docs/testing/` document set, `docs/integrate-existing-api.md`,
  `docs/ENGINEERING_BACKLOG.md` and the `web/retrieval_terminal_concept.html`
  mock-up.
- **The legacy single-file console** (`/admin-classic`, `web/admin_console.html`
  and its loader `src/api/router/admin_page_html.py`). The React console at
  `/admin` is the only console.

## [1.1.0] — 2026-09-18

### Added
- **Streaming answers (SSE)** in the Search Playground — the generated answer now
  streams token by token instead of blocking until the final token.
- **Multi-modal retrieval** — Docling-parsed tables are surfaced in retrieval
  results and rendered as real tables; the MCP `search` tool now includes
  `content_type` (`table`/`picture`) so an agent knows a hit is a table.
- **Evaluation depth** — LLM-graded answer quality (faithfulness + relevance) and
  a persisted baseline history / trend, plus stale-set auto-regeneration with
  `regenerate_reason` / `stale_dropped` notices.
- **One-click Rebuild-FTS** — re-segments existing chunks with CKIP and rebuilds
  only the keyword-search vector (no re-embedding), with an explanatory tooltip.
- **React admin console** as the front door (root redirects to `/admin`), a
  Settings **scrollspy**, an accessibility pass (keyboard-operable rows/tiles,
  labelled inputs, visible focus), themed floating tooltips, and frontend tests +
  CI.

### Fixed
- **Evaluation** — bound `n`/`k` on the run endpoint (guards a runaway/DoS run);
  never write an all-zero run to baseline history; compare the baseline delta
  against the *same* retrieval mode; keep a failed run's error visible;
  regenerate when a cached eval set is heavily stale.
- **Multi-modal** — the triplet-table parser no longer drops cells (float row
  labels, decimals, the final cell); rendered tables are no longer clamped into a
  tiny scroll box.
- **Search** — abort the SSE stream on navigation/unmount; fix SSE frame
  serialization and surface error frames.
- **Health** — the reranker check falls back to `/v1/models`, so a vLLM reranker
  no longer shows a misleading `HTTP 404`.
- **Console** — restart the file poll after upload/reindex (status refreshes
  without a manual reload); drop a stale files response after switching folders;
  retry instead of freezing on a transient load failure; format dates/numbers in
  en-US; keep the Status column stable during indexing; keep primary button labels
  on one line; fix uploads that silently did nothing; reject foreign-service tokens
  on folder create.
- **Ops** — disable the server's own core dumps by default; forward `force=`
  through the single-file indexing path.

### Notes
- The admin console is intentionally **English-only** (an i18n pass was added and
  then reverted during this cycle).

## [1.0.0]

- Initial release: Agentic RAG MCP server — hierarchical chunking + auto-merge,
  hybrid vector + BM25/CKIP retrieval, cross-encoder rerank, contextual retrieval,
  pgvector storage, per-token (`jti`) folder ownership, and the first React admin
  console.

[1.1.0]: https://github.com/xianhong1208/agentic_rag/releases/tag/v1.1.0
