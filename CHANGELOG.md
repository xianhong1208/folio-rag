# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Fixed
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
