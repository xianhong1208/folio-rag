# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Searching a folder that was never indexed returns an empty result.** A folder's vector table is created by its first indexing run; until then an MCP search answered `internal_error` (`UndefinedTable`) and `POST /api/rag/query` (and the console's Search) answered 404 "Vector store table missing". Both now return the normal empty result: MCP with `results: []`, `low_relevance: true` and a `_hint` saying the folder has nothing indexed yet; REST with 200, `results: []`, the new `not_indexed: true` and the message "Query returned 0 results: this folder has nothing indexed yet". The missing table is recognised by PostgreSQL's SQLSTATE (42P01) and only when it is that folder's own table, so any other missing relation still fails. These searches are not recorded in the query analytics (no false knowledge gap) and are never cached, so the folder answers as soon as its first file is indexed.

### Docs
- **Roadmap and README rewritten.** `ROADMAP.md` and the new `ROADMAP.zh-TW.md` list what each released version brought, the agreed plan from 1.4.2 to 1.10 (each item has an issue in its GitHub milestone), what comes later and the known limitations; the Graph RAG decision gate is kept. The READMEs are shorter, with badges, a console screenshot, a quick start and deployment options. The reference material moved, in English and Traditional Chinese, to `docs/how-it-works.md`, `docs/configuration.md`, `docs/auth.md`, `docs/console.md`, `docs/api.md` and `docs/development.md`. Corrected on the way: the API docs at `/docs` are always on (there is no `ENABLE_API_DOCS`), and folders still keyed per token are those from before 1.4.1, not 1.5.

### Security
- **Dependency advisories fixed.** The dependency audit no longer accepts any advisory (`.github/pip-audit-ignore.txt` is empty).
  - transformers 4.57.6 → 5.22.2 (`--extra local`; with huggingface-hub 0.36.2 → 1.33.0 and tokenizers 0.22.2 → 0.23.3). It drops the six accepted transformers advisories (PYSEC-2025-217, PYSEC-2026-2288, -2289, -2290, -3929, -4174). No code change was needed: CKIP word segmentation, HuggingFace embeddings, docling layout/table/OCR conversion and chunk token counts give the same output as on 4.57.6.
  - torch in the `cuda` group 2.8.0+cu126 → 2.14.1+cu130, with torchvision 0.23.0 → 0.29.1, torchaudio 2.8.0 → 2.11.0 and triton 3.4.0 → 3.8.0 (GHSA-63cw-57p8-fm3p and the `lstm_cell`, `unpack_sequence` and `jit.script` advisories). The group now installs from PyTorch's CUDA 13.0 index, which needs an NVIDIA driver ≥ 580; PyTorch publishes 2.14 for CUDA 13.0 only.
  - Mako 1.4.1 → 1.4.3 (via alembic; GHSA-5639-2j2p-m4mx) and soupsieve 2.9.2 → 4.1 (via beautifulsoup4; CVE-2026-126190, CVE-2026-126192), held there by `[tool.uv] constraint-dependencies`.

## [1.4.1] — 2026-11-09

### Changed
- **Folders belong to the person, not to one token.** A folder's owner was the `jti` of the token that created it, so every token saw its own empty set: an OAuth-connected MCP client lost its folders when its 15-minute access token was refreshed (and saw no `Folio_<folder>` tools at all), and two PATs of the same user could not see each other's folders. The owner is now the token's `sub` — the MCP Gate user — stored as `sub:<sub>`, so all of a user's tokens, PATs and OAuth sign-ins alike, see the same folders over REST, `/mcp` and the console. A `client_credentials` token's `sub` is its client id, so such a client owns its own folders. A token without a `sub` is refused for folder operations (REST 401, MCP `unauthorized`, console `missing_sub`). New folders' files live under `storage/sub_<sub>/`. `rag:admin` still never grants other people's folders; open (`auth.enabled: false`) mode is unchanged.
- **Revoking a token no longer hides its folders.** They belong to the person and stay reachable with their other tokens; delete the folders or disable the user in MCP Gate to take them away.
- **Existing folders move on first use.** Nothing is migrated in bulk. The first verified request carrying a token whose `jti` still owns folders (REST, the `/mcp` tool list or a tool call, console sign-in or session) moves all of them to that token's `sub` — rows, storage directory, stored file paths and MCP tools — in one transaction, once; concurrent requests are serialised per person (in-process lock plus a PostgreSQL advisory lock, rows locked `FOR UPDATE`). When the person already has a folder of that name, the moved one is renamed to the next free name (`docs-2`, `docs-3`, …) and the Audit Log records `RENAME folder:<id> <old> → <new>`; folders are never merged. Folders of tokens that are never used again stay under their old key; admins see how many on the Overview (`legacy_owned_folders` in `GET /api/admin/overview`).

### Fixed
- Renaming a folder left its old name in the folder lookup cache for up to five minutes.
- **A folder could be renamed to a name its owner already used** (creating one was refused). The MCP tool registry, the storage directory and name-based queries are keyed on (owner, name), so renaming either twin away removed the other's `Folio_<name>` tool, and the refused directory move still rewrote the files' stored paths into the other folder's directory. Renames to a taken name now get **409**, as creating one does (create returned 400; the API documented 409). A rename whose directory cannot be moved now changes nothing and gets 409 instead of half-applying.
- **Uploading a file whose name is already in the folder returned 200 and kept both** — files are stored under their own uuid, so every search returned the passage twice. It now gets **409** ("already in this folder; delete it first to replace it"), checked before anything is written; a batch upload reports it per file and keeps going.

- **Answers came back in the context's language instead of the question's.** An English question over Chinese documents got a Chinese answer — in Simplified characters. The answer prompt now names the language from the question's own script (English and other Latin-alphabet questions are told not to answer in Chinese, Japanese or Korean; a Chinese question too short to show its script follows the documents'), and Simplified characters that slip into a Traditional Chinese answer are converted, for the console's streamed answers too.
- **Context prefixes were thrown away for mainly English chunks with one Chinese line.** The script check counted any non-Latin character as the chunk's script, so the correct English prefix was rejected as a mismatch and the chunk was indexed without context (5 of 18 files in the QA run). The check now uses the script a chunk is mostly written in.

- **Console pages scrolled sideways.** One unbreakable string widened its column and then the page: Analytics grew to ~15,000 px with a long knowledge-gap query, Jobs' message line and the Search box ran off a phone screen, and the folder's file table squeezed file names to two letters. Columns now shrink and long text wraps; on a phone the Search bar and the Analytics filters wrap, job messages wrap, the file table drops Chunks, Size and Indexed At, and hover tooltips are not drawn (a hidden one near the right edge widened the page; phones cannot hover).
- The Overview's "Retrieval Shape" claimed a fixed `hybrid .75/.25` weighting; it now shows the fusion in use (`/api/admin/overview` gains `retrieval.fusion`). The Jobs page no longer says admins see every folder (they have not since 1.2.0).
- `/admin` without the trailing slash, which the README links to, answered 404; it now redirects to `/admin/`.

- A query the MCP tool refused (e.g. too long) was recorded as a zero-result search and listed as a knowledge gap.
- `mode=read` on a file without indexed content (failed or not yet indexed) returned empty text with the hint "Full content loaded."; the hint now says the file has no indexed content.
- Re-indexing an unchanged file answered `status: already_indexed_same_content` with the message "File indexed successfully with N chunks"; the message now says it was not re-indexed and how to force it.
- A bad `folder_id` on `GET /api/rag/indexed-files` listed the validation error twice.

- **`/v1` media endpoints:**
  - speech-to-text reported `model: whisper-turbo` and `/v1/models` listed Whisper while the configured provider (e.g. FireRedASR) did the work; both now name the provider in use;
  - audio that cannot be decoded and PDFs or images that cannot be opened got a 500 (the audio one with ffmpeg's stderr in the body); they now get **422** with a plain message, checked before the models run;
  - other failures no longer echo the exception text (it stays in the server log);
  - an unknown `response_format` was ignored (json came back); it is now **400**.

### Security
- **Signing out ends the console session on the server.** Sessions are stateless signed cookies, so signing out only removed the cookie from the browser and a copy kept working for up to `auth.console.session_hours`. Each session now carries its own id, and signing out records it (in `storage/.console_ended_sessions`, mode 0600, until the session would have expired) so a copied cookie gets 401 `session_ended`, also after a restart. Other sessions of the same token stay signed in. **Sessions issued by earlier versions cannot be ended this way and are no longer accepted: everyone signs in again once after upgrading.**

## [1.4.0] — 2026-11-09

### Added
- **Lint in CI.** `pyproject.toml` gains a `[tool.ruff]` section with the same correctness-only rules as MCP Gate, AnyDoc and Packwright (`E4`, `E7`, `E9`, `F`), and a *Lint (ruff)* job runs `ruff check` on every backend change. The 64 existing findings are fixed: statements joined with `;` or after `if …:` split onto their own lines (the AST of every such file is unchanged), an unused name and an unused mock binding removed, a duplicate `threading` import in `docling_loader.py`, an import placed after a skip marker in a test, f-strings without placeholders, an ambiguous `l`, and a malformed `# noqa` in `asr_provider.py` that ruff could not read. `main.py`, `db/migrate.py` and the Alembic `env.py` keep their late imports on purpose (they set up `sys.path`, the working directory or allocator variables first).
- **Dependency audit in CI** (`.github/workflows/audit.yml`): `pip-audit` on the locked runtime dependencies including the `local` extra, and `npm audit` (production, high and above) on the console. It runs when dependencies change and every Monday, since a new advisory against a locked version appears without any change here. Accepted advisories live in `.github/pip-audit-ignore.txt`, one per line with the reason: the transformers 4.x advisories (the package is pinned below 5) all need an attacker-controlled model repository or code paths Folio does not use,. The setuptools advisories are fixed, not accepted: setuptools is 84.2.1 since the ROCm change below.

### Changed
- **ROCm builds use torch 2.13** (`rocm-r714`: `torch[device-all]==2.13.0+rocm7.14.0`, torchvision 0.28.0; torchaudio stays 2.11.0, the newest AMD ships). AMD's 2.11 and 2.12 builds require `setuptools<82`, and because one lock serves every GPU group, that cap held CPU and CUDA installs on setuptools 81 too.

### Removed
- **The `rocm-r713` group (ROCm 7.13).** Its newest torch is 2.11, which has the same setuptools cap. Use `rocm-r714`.

### Security
- **setuptools 81 → 84.2.1** in every group (PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w), kept there by `constraint-dependencies = ["setuptools>=84.1"]` in `[tool.uv]`. setuptools 84 no longer ships `pkg_resources`; the only runtime dependency importing it, jieba, falls back to reading its dictionary directly.

## [1.3.0] — 2026-11-05

### Added
- **New upload formats from docling 2.146**, each parsed end to end first: **RTF** (docling's Word backend through LibreOffice, like `.doc`; headings and tables survive), **MHTML / MHT** web archives (the HTML part is indexed; local and remote references are never fetched), **GIF** (OCR on every frame, like a multi-page TIFF; the cloud parser counts frames against `rag.parsing.max_pages_per_file`), **`.markdown`**, and **Excel templates** `.xltx` / `.xltm`.
- MHTML is decoded with the charset its HTML part declares. docling's MHTML backend ignored it, so a short Chinese page without `<meta charset>` came out as Thai (cp874) — the failure `_declare_html_utf8` already fixed for plain HTML.

### Security
- **Dependency advisories fixed within the current major versions** (pip-audit: 168 → 89 advisories; the remaining ones need pillow 13, transformers 5, filelock 4 and setuptools 84, which follow separately). Upgraded: pyjwt 2.13.0 → 2.15.1 (the console and MCP token verifiers use it), pypdf 6.18.0 → 6.21.4, docling 2.126.0 → 2.146.0 (with docling-core 2.105, docling-parse 7.24), authlib 1.8.0 → 1.9.4, joserfc 1.7.5 → 1.8.0, aiohttp 3.14.3 → 3.14.5, urllib3 2.7.0 → 2.8.0, nltk 3.10.3 → 3.10.4, multidict 6.8.0 → 6.9.1, dnspython 2.8.0 → 2.9.2, opentelemetry-api 1.44.0 → 1.45.2 and banks 2.5.0 → 2.5.1.
- **pillow 12.2.0 → 13.0.0 and filelock 3.32.6 → 4.2.8** (major versions; 77 advisories). pillow was pinned to 12.2.0 since the first commit with no recorded reason; the pin becomes `>=13.0.0` (docling-ibm-models moves to 4.0.4 with it). Cloud-parser page rendering (multi-frame TIFF, RGBA/palette, oversized images, PDF pages) and Docling parsing of the eval corpus are unchanged on 13.0. pip-audit: 89 → 12 advisories; what remains needs transformers 5 (pinned `<5` until sentence-transformers / llama-index / CKIP are checked against it) or setuptools 84 (conflicts with the ROCm torch builds).
- **uv installs by copying** (`[tool.uv] link-mode = "copy"`). NLTK 3.10.4 refuses to open a data file with more than one hard link, and uv hard-links packages from its cache whenever the cache and the venv share a filesystem (CI, Docker), which broke the NLTK stopwords that llama-index ships. The same NLTK release also refuses data owned or writable by another account: run Folio as the user that owns `.venv` (README → Install).
- Docling 2.146 changes chunk text slightly: bare URLs and email addresses become Markdown links (`[url](url)`), and spaces between CJK characters in PDFs are dropped. Measured on the golden set with every context prefix regenerated, twice: R@10 0.981 → 0.987 in both runs, R@5 0.965 → 0.973 / 0.976, nDCG@10 0.935 → 0.940 / 0.934. Every per-question difference is within the run-to-run noise of regenerating the prefixes. Tokens per call rose 1–2%, and the 0.9 relevance cutoff behaves the same.
- Docling 2.146 lists new input formats. Each was parsed end to end before it was accepted (see *Added*); Keynote, Numbers (need the `docling[format-iwork]` extra) and AFP (mainframe print stream) stay rejected.

## [1.2.0] — 2026-11-04

### Security
- Console dependencies: React Router 6 → 7.18 (open redirect via a backslash in `<Link>` /
  `useNavigate`, the only advisory that reached the shipped bundle), Vite 6.4.3 → 6.4.5 (dev
  server path traversal / `fs.deny` bypasses), Vitest 2 → 5 (UI-server file read, Tinypool
  prototype pollution, mock path traversal), plus `source-map-js` and `postcss-selector-parser`.
  `npm audit` reports 0 vulnerabilities (was 10: 2 critical, 2 high). Console development now
  needs Node.js 22.12+ (CI moves from Node 20 to 22).
- Console aggregates (Overview KPIs and trend, Index Jobs, Query Analytics) are now limited to the signed-in token's own folders for `rag:admin` tokens too. Before, an admin token saw every owner's counts, job folder names and search queries, contrary to the rule that `rag:admin` unlocks global settings only. Storage totals and the settings audit log stay admin-only.

### Search latency — MCP search p50 ~500 → ~330 ms, p95 ~600 → ~400 ms
Same instance, interleaved runs, quality unchanged on the golden set (R@5 0.976, R@10
0.981, nDCG@10 0.938, zh→en R@5 0.90; MRR@10 0.938 → 0.939). Per-stage table and the
rejected ideas in `evals/baselines/rag-2026-11-04-search-latency.md`.

- **Rerank pool per list** (new `rag.retrieval.rerank_pool`, default `per_list`): the
  cross-encoder scores the dense top `rerank_dense_candidates` (default 30, at least
  3 × `top_k`) plus the BM25 top `rerank_sparse_candidates` (default 10) — about 35
  pairs — instead of the top 60 of the fused list. The deep pool exists for
  dense-only hits (cross-lingual, paraphrased), which RRF ranks below everything both
  lists found; cutting each list on its own reaches them with half the pairs. A
  vLLM score call costs ~6 ms per pair whatever the length, so the rerank step went
  from ~360 to ~225 ms p50. All three keys are live (console *Settings → Retrieval*);
  `rerank_pool: fused` restores the old pool (`rerank_candidates` keeps its meaning).
  With `top_k=20` the dense side is 60 deep, so that path scores slightly more pairs
  than before. A failed rerank still falls back to the fused order.
- The MCP search reads the auto-merge parents and neighbour rows for every candidate
  **while the reranker runs** (two concurrent queries, a superset of what the final
  top_k needs) instead of three sequential queries after it (~20 ms p50). If that
  read fails, the old queries run as before.
- **CKIP query segmentation runs on one dedicated thread.** On the CPU each thread
  that calls torch gets its own OpenMP worker pool; queries landing on different
  pool threads built several pools that fought over the cores (segmentation
  ~40 → ~140 ms p50, seconds on a busy host). Output is unchanged.
- The MCP search no longer waits for its QueryLog row (written in the background,
  flushed on shutdown).

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

[Unreleased]: https://github.com/xianhong1208/folio-rag/compare/v1.4.1...dev
[1.4.1]: https://github.com/xianhong1208/folio-rag/releases/tag/v1.4.1
[1.4.0]: https://github.com/xianhong1208/folio-rag/releases/tag/v1.4.0
[1.3.0]: https://github.com/xianhong1208/folio-rag/releases/tag/v1.3.0
[1.2.0]: https://github.com/xianhong1208/folio-rag/releases/tag/v1.2.0
[1.1.0]: https://github.com/xianhong1208/folio-rag/releases/tag/v1.1.0
