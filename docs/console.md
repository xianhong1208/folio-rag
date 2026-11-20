# The console

[繁體中文](console.zh-TW.md)

The **Retrieval Terminal** console is a React SPA served at `/admin`.

## Console login

You sign in by pasting an **MCP Gate access token issued for this service** — usually a personal access token (PAT). The server verifies it once, offline, exactly like a REST or MCP call (signature, issuer, audience, expiry; it must carry a `sub` and a `jti`), then sets an HttpOnly session cookie. The token itself is never stored or sent back.

- **The session is you.** You see and manage exactly the folders owned by the user the token was issued to (folders are keyed on its `sub`) — the same folders every token of that user reaches over REST and MCP. Folders you create in the console belong to you too. Other users' folders are invisible.
- **Two tiers.** Any valid token can sign in and manage its own folders. Changing runtime **Settings**, testing arbitrary endpoints, and reading the settings **Audit Log** and server-wide storage figures need the **`rag:admin`** scope. Without it, Settings and System Health are read-only (secrets are always masked), and Overview / Index Jobs / Query Analytics always cover only your own folders — `rag:admin` unlocks global settings, never another user's folders, jobs or queries. Admins also see on the Overview how many folders still belong to a pre-1.4.1 per-token key (see [Folder ownership](auth.md#folder-ownership)).
- **Sessions** last `auth.console.session_hours` (8 h) or until the token expires, whichever is first. With the [revocation feed](auth.md#token-revocation) enabled, revoking the token in MCP Gate also ends its sessions within a poll interval; without it a running session lasts until expiry or *Sign out* (verification is offline).

**Issuing tokens in MCP Gate.** No MCP Gate change is needed:

1. *Services* → open the Folio service → *Scopes declared by this server* → add **`rag:admin`**. Do **not** mark it as a default scope (defaults are preselected on every new token). If the service restricts its *Allowed scopes*, allow `rag:admin` too.
2. *Issue Token* → *Target service*: Folio (audience = `auth.audience`). Tick `rag:admin` for operators who may change settings; leave it off for people who only manage their own folders. Give it a label and copy the token — it is shown once.

Paste the token on the login screen. Folders created before 1.4.1 belong to the token that created them until that token is used again, which moves them to you (see [Folder ownership](auth.md#folder-ownership)). Scripts can skip the cookie and call `/api/admin/*` with `Authorization: Bearer <token>` — the same rules apply, and no CSRF header is needed.

**Dev mode.** With `auth.enabled: false` there is no login: the console is open to anyone who can reach the port, with full admin rights and every folder visible (folders created there are owned by `console`). The server logs a warning at startup. Use it only on a trusted machine.

**Network exposure.** Even with login on, keep the console (`/admin`, `/api/admin/*`) reachable only from trusted networks — a VPN, an internal network, or a reverse proxy that restricts it — and serve it over HTTPS. Login protects folders and settings from people without a valid token, but the console can upload and delete documents, run queries against your model endpoints and (with `rag:admin`) change them, so it should not face the open internet. `/mcp` and `/api/rag/*` are designed to be reachable by MCP clients and need only a valid token.

| Page | What you do there |
|------|-------------------|
| **Overview** | The live retrieval pipeline, corpus readouts, index health, and model-service status. |
| **Library** | Every document of all your folders in one table, paginated on the server (sort by name, upload date or size). A **filter rail** narrows it: tags with document counts (click to toggle; *any* or *all*), folders, file type, upload date range, part of the name, and metadata conditions (key, operator — `=`, *one of*, `≥`, `≤`, *is set*, *is not set* — and value). Active filters show as chips above the table (click one to remove it), and the whole view lives in the URL, so it can be bookmarked or shared. Type in **search inside documents** to rank the same filtered documents by content instead (their best passage, hits and score, the top 50); clear it to get the list back. **Select** rows (across pages) to add or remove tags or set or clear a metadata key on all of them at once, with progress and a per-document error list. **Click a row** for its panel: folder (a link to it), type, size, index status, dates, editable tags, metadata and description, a download, and a search inside just that document. On a phone the rail opens from a *Filters* button. See [below](#the-library). |
| **Folders** | Create folders (owned by you); open one to upload files (with tags), give a file tags and metadata (the tag button on its row), watch per-file index status, view chunks (with the AI context line shown apart from the document text), reindex, or delete. **Export** downloads a folder as one archive and **Import** brings one back as a new folder of yours ([details](backup.md#in-the-console)). Files indexed by an older chunking pipeline, or with another embedding model than the configured one, show **Re-index recommended**; the folder list counts them per folder (**N stale**) with a one-click re-index. |
| **Search Playground** | Run a live query against a folder and see the **per-signal retrieval trace** — Vector · BM25 · Hybrid · Rerank ranks side by side — plus an optional RAG answer. **Filters** opens the Library's tag, metadata and (with All folders) folder filters; they apply to every search on the page. Pick **All folders** to search every folder at once (each hit shows its folder; folders that were not searched are listed with the reason), and switch **Passages / Documents** to see ranked documents with their best passages instead. The answer and the trace need one folder and passages. |
| **Index Jobs** | Live stream of indexing jobs with progress; cancel a running job. |
| **Query Analytics** | Volume, latency, zero-result rate, top queries and knowledge gaps. |
| **Evaluation** | Auto-generate a question set and score `production` (same call as `/api/rag/query`), `vector`-only and `bm25`-only on Recall@k · MRR · nDCG@k; baseline deltas only against runs on the same question set. |
| **Settings** | Live, persisted configuration for Embedding / LLM / Contextual Retrieval / Reranker / Speech-to-Text / Retrieval, with retrieval presets and connection tests. Changing anything needs `rag:admin`. |
| **System Health** | Live connectivity to every external dependency. |
| **Audit Log** | A timeline of every runtime settings change and who made it (`rag:admin`). |

## The Library

The sidebar's **Library** (`/admin/library`) lists documents; **Folders** (`/admin/folders`, with the
*Index Jobs* tab) is where you create folders and upload. Old links to `/folders` and `/jobs` keep
working, and `/folders?folder=<id>` opens one folder.

- **Filters are one AND condition**, the same [document filter](api.md#document-filters) as the API:
  a document must match every chip. Several tags match *any* of them by default; switch to *all* in
  the Tags header. Rows on one metadata key combine (`year ≥ 2020` and `year ≤ 2024` is a range). A
  metadata value is typed like in the edit box: `2024` is a number, `"2024"` text, `true` a boolean,
  `2024-03-01` a date. The *to* date is inclusive.
- **The URL** holds every filter, the content query, the sort and the page, for example
  `/admin/library?tag=invoice&m=year:gte:2024&sort=name&order=asc`.
- **Bulk edits** write one `PATCH` per document that changes (a document that already has the tag or
  value is left alone), two at a time; a failure is listed by document and the rest still go through.
  Tag counts and metadata keys refresh afterwards.
- **Search inside this document** runs the folder's search with `filters.file_ids` set to that one
  document.

## Re-indexing after an upgrade

Chunking changes are versioned per file. A file indexed by an older pipeline keeps working exactly as before, but shows **Re-index recommended** in the folder view (and `reindex_recommended: true` in `GET /api/rag/file/{id}/index/status`). Use the row's re-index button, or the header link that selects all such files, then *Reindex selected*. Per-file re-indexing is incremental from then on; *Reindex Folder* drops the folder's table and rebuilds everything.

You don't have to open every folder to find them: the folder list shows an amber **N stale** badge on each folder that holds such files (`reindex_recommended_files` in `GET /api/admin/folders`), and the re-index button in that folder's row re-indexes just those files.

The current pipeline is version 3 (section-aligned chunks, with the version-3 context prompt), so every file indexed before it is marked. Until it is re-indexed, a file keeps its old chunks and is still searchable.

## Re-indexing after changing the embedding model

Each file also records the embedding model and dimension it was indexed with (`embedding_model`, `embedding_dimension`). When `rag.embedding.model` or `rag.embedding.dimension` changes — in `config/models.yaml` or on the Settings page — every file indexed with the old one is marked **Re-index recommended** the same way, at once, and counts towards the folder's **N stale** badge: its vectors came from the other model and do not match new queries. The badge's tooltip and the API's `reindex_reason` say which case it is (`embedding` or `pipeline`). Switching back to the old model clears the mark. Files indexed before 1.4.2 have their dimension filled in from the folder's vector table when the server starts; a file whose model or dimension was never recorded is not flagged for it.
