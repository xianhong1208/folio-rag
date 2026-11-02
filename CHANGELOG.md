# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
  bigram` (default `ckip`, editable live). One function serves indexing,
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

### Changed
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
  with spaces, CJK or symbols become `Agentic_<ascii>_<hash>`. Tool
  descriptions are English.
- The settings API validates ranges and choices (e.g. `max_concurrent ≥ 1`,
  `score_threshold` 0–1, known providers, http(s) URLs) and answers 422.
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

### Fixed
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
- Retrying a file's indexing checked folder ownership but not that the file was
  in that folder.
- The chunk viewer no longer reads a file name from outside the folder.
- Eval reports, question sets and history are keyed by folder id: two owners'
  same-named folders used to share (and could read) each other's results.

### Removed
- **The legacy single-file console** (`/admin-classic`, `web/admin_console.html`
  and its loader `src/api/router/admin_page_html.py`). The React console at
  `/admin` is the only console.

### Changed
- **The authorization server is now called MCP Gate** (formerly MCP Center). Docs
  and log messages follow the new name; `src/auth/mcp_center_auth.py` is now
  `src/auth/mcp_gate_auth.py` and `McpCenterTokenVerifier` is
  `McpGateTokenVerifier`. Configuration and issued tokens are unaffected.

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
