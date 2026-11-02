# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
  `max_tokens_per_day`.
- **Per-role console settings and probes**: a Model Roles panel (where each role
  runs, key set / from which env var — never the key) and
  `POST /api/admin/probe/{role}`, which sends one minimal real request with the
  saved settings. Console key fields accept `${ENV_VAR}`; the reference is
  stored, not the key.
- `tests/test_cloud_profile_no_local_models.py` proves a cloud deployment imports
  no torch, transformers, sentence-transformers, docling models, Whisper,
  FireRedASR or CKIP.
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
- The `openai` embedding and LLM providers ignored `base_url`, so an
  OpenAI-compatible cloud had to be configured as `vllm`; it is honoured now, and
  `openai-compatible` is a provider of its own.
- Chunk token counting no longer crashes when no tokenizer can be loaded (it falls
  back to the Rust `tokenizers` package, then a character estimate), and no longer
  imports torch through transformers in the cloud profile.
- The "Whisper weights not found" startup line no longer appears when local
  Whisper is not the configured speech-to-text provider.
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
- **Install: the in-process model stack is now the `local` extra.** Local and hybrid
  deployments install with `uv sync --extra local --group <gpu>`; a plain
  `uv sync` is the torch-free core for the cloud profile. A `uv sync --group <gpu>`
  without `--extra local` removes the local stack from an existing venv.
- Startup logs where every model role runs and warms docling only when a role uses
  its local models; CKIP is not loaded under the cloud profile.
- The unauthenticated `/v1/ocr/general` answers 503 when PDFs/images are parsed
  in the cloud instead of spending the vision budget on anonymous uploads.
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
