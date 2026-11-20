# API reference

[繁體中文](api.zh-TW.md)

| Group | Endpoints |
|-------|-----------|
| **MCP** | `POST /mcp` — [`folio_find`](#mcp-tools) (documents across all your folders) and per-folder `Folio_<folder>` tools (`search` / `list` / `read`, `file_ids`); `GET /.well-known/oauth-protected-resource/mcp` |
| **Folders & files** | `/api/folders*` · `/api/folders/{id}/files*` (create, upload, list, download, delete — token-scoped) |
| **Documents** | `POST /api/documents/search` · `GET /api/tags` · `GET /api/metadata/keys` — your documents across folders, with [filters](#document-filters) |
| **Search** | `POST /api/search` — one search over all your folders (or some), as passages or as documents ([below](#search-across-folders)) |
| **RAG** | `POST /api/rag/query` (hybrid retrieval + rerank, optional `filters`) · `/api/rag/files/{id}/index`, `/reindex`, `/index/jobs*` (background indexing + SSE progress) |
| **Console session** | `POST /api/console/session` `{token}` (sign in) · `GET` (who am I) · `DELETE` (sign out) |
| **Console (admin)** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **Media (removed in 1.5)** | `POST /v1/ocr/general`, `POST /v1/transcriptions/general` and `GET /v1/models` answer **410 Gone** with `Link: <…migrate-1.5.md>; rel="help"` and what to call instead ([below](#removed-in-15-the-v1-media-endpoints)). |
| **Operational** | `GET /health` (liveness) · `GET /health/detailed` (database, pgvector, connection pool, cache, revocation feed, [backups](backup.md#scheduled-backups)) · `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` · `GET /metrics` (Prometheus; [Metrics](metrics.md)) |

**Audio uploads.** An audio file (`.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`) is refused with **422** while speech-to-text cannot transcribe it — `rag.asr` has no `base_url`, or `rag.asr.enabled: false` — and the detail says what to set; the batch route lists it in `failed_uploads`. Nothing is stored, so no empty or failed entry is left behind. See [Upgrading to 1.5](migrate-1.5.md#speech-to-text).

**Image uploads.** An image (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`, `.gif`) is refused with **422** while no vision model is configured (`rag.parsing.vision` without `model` and `base_url`): only a vision model can read it. The detail says what to set; the batch route lists it in `failed_uploads`. PDFs are always accepted: without a vision model they are read from their text layer, a page with no text layer is indexed as a visible marker, and a PDF with no text at all fails indexing with the same advice. See [Upgrading to 1.5](migrate-1.5.md#pdfs-and-images).

**Upload size.** A file may be at most 100 MB. Uploads are copied to disk in 1 MiB pieces and never held in memory whole. The single-file upload routes (`POST /api/folders/{id}/file` and the console's) refuse a larger request from its `Content-Length` before reading it, or as soon as a streamed body crosses the limit, with **413** `File size exceeds limit (100.0MB)`; the batch route (`POST /api/folders/{id}/files`) reports an oversized file in `failed_uploads` and keeps the others.

## Document filters

A document (a file) carries **tags** and **metadata**, and searches and the document list can be
limited to the documents that match a **filter**.

**Tags** are short labels, one namespace per person across all their folders. Each is trimmed (inner
whitespace collapsed), at most 64 characters, at most 32 per file; tags that differ only in case are
one tag (the first spelling is kept) and match case-insensitively. **Metadata** is a flat object:
keys `^[a-z0-9_]{1,40}$`, at most 32; values are a string (at most 500 characters), a number, `true` /
`false`, or a date written `YYYY-MM-DD`. Lists and objects are refused (**422**, naming the key).

Set them when uploading (`POST /api/folders/{id}/file` and the console's upload: form fields `tags`,
comma-separated, and `metadata`, a JSON object) or later with
`PATCH /api/folders/{id}/files/{file_id}` (console: `PATCH /api/admin/manage/folders/{id}/files/{file_id}`):

```json
{"tags": ["invoice", "signed"], "metadata": {"client": "Acme", "year": 2024, "due": "2024-12-31"}}
```

A field left out stays as it is; `tags` and `metadata` replace the stored list / object (`[]` / `{}`
clear them). An edit drops the folder's cached search results at once.

**The filter** is one flat object; every condition must hold (AND):

| Field | Matches |
|---|---|
| `tags_any` / `tags_all` | at least one / all of these tags (case-insensitive) |
| `folders` | these folders, by name or id — always among your own |
| `file_type` | extensions (`pdf`, `.docx`) or MIME types (`application/pdf`, `image/*`) |
| `uploaded_after` / `uploaded_before` | upload time, inclusive / exclusive; `YYYY-MM-DD` or an ISO date-time |
| `name_contains` | part of the file name, case-insensitive |
| `file_ids` | these documents, by file id (1–100; the console's "search inside this document") |
| `metadata` | `{key: value}` (equal; strings match exactly), or `{key: {"in": [...]}}`, `{"gte": x}`, `{"lte": x}` (a number, or a `YYYY-MM-DD` date compared with dates), `{"exists": true/false}` |

```json
{"tags_any": ["invoice"], "file_type": ["pdf"],
 "metadata": {"client": {"in": ["Acme", "Beta"]}, "year": {"gte": 2024}}}
```

An unknown field or operator is a **422** that names it. Filters are turned into SQL with bound
parameters by Folio itself, and your owner scope is always added: a filter can narrow what you see,
never widen it.

Where filters apply:

- **`POST /api/rag/query`** and the console's playground API (`POST /api/admin/manage/folders/{id}/query`) take an optional `filters`: only
  that folder's matching documents are searched (dense and keyword search both; pgvector's
  iterative HNSW scan keeps a selective filter from returning too few hits). No matching document
  means no results, not an error.
- **`POST /api/documents/search`** lists your documents across all your folders, one page at a time:
  body `{"filters": {...}, "sort": "name" | "uploaded" | "size", "order": "asc" | "desc",
  "limit": 1-200 (50), "offset": 0}`. Each document has `id`, `name`, `folder {id, name}`,
  `size_bytes`, `mime`, `uploaded_at`, `updated_at`, `status` (index status), `tags`, `metadata`
  and `description`; the response adds `total` and `next_offset` (null on the last page).
- **`GET /api/tags`**: your tags with document counts (case variants grouped, shown in their most
  common spelling). **`GET /api/metadata/keys`**: your metadata keys, how many documents carry each,
  and the value types seen (`string`, `number`, `boolean`, `date`).

These three take a Bearer token (a personal access token) or the console session, like the
console's API; a cookie-authenticated `POST` needs the console's CSRF header. In MCP,
[`folio_find`](#mcp-tools) takes the same filter.

## Search across folders

`POST /api/search` searches all your folders at once (or the ones you name), optionally limited by
a [filter](#document-filters), and answers with **passages** or with **documents**:

```json
{"query": "payment terms", "folders": ["contracts", 12], "filters": {"tags_any": ["signed"]},
 "group_by": "document", "top_k": 10, "snippets": 3}
```

| Field | Default | Meaning |
|---|---|---|
| `query` | — | 2–2000 characters |
| `folders` | all yours | folder names or ids; one that is not yours is a **404**, like one that does not exist |
| `filters` | none | a [document filter](#document-filters) |
| `group_by` | `passage` | `passage` = ranked passages (`results`); `document` = ranked documents (`documents`) |
| `top_k` | `rag.retrieval.default_top_k` passages / 10 documents | 1–50 |
| `snippets` | 3 | `document`: passages shown per document (1–3) |
| `auto_merge` | as configured | `passage`: merge hit leaves into their section |

How it works: each folder is searched with its own index (its dense table and its own keyword
segmenter), in parallel (four folders at a time, inside the server's query limit), with the query
embedded once. The folders' candidates are pooled and **reranked once**: cross-encoder scores
compare passages from different folders fairly. Without a reranker (or if the rerank call fails)
the folders' lists are merged by reciprocal rank, never by raw similarity scores, which are not
comparable across tables. With several folders each one contributes a smaller pool (twice the
usual budget split over the folders, at least 10 dense + 4 keyword candidates each), and at most
150 pooled candidates are reranked. One folder with `group_by: passage` runs exactly the
single-folder search of `POST /api/rag/query`.

**Documents** are ranked by their best passage (MaxP); a tie goes to the document with more
matching passages. Each has `file_id`, `name`, `folder {id, name}`, `mime`, `tags`, `metadata`,
`score` (its best passage's), `hits` (matching passages among the ranked ones) and `snippets`
(`text`, `score`, `page`, `headings`, `content_type`, `node_id`). Names, tags and metadata are read
live, so a rename shows at once. **Passages** are the usual result objects plus `folder {id, name}`.

The response also has `folders` (searched), `skipped_folders` (`{id, name, reason, detail}`),
`score_type` (`rerank` or `rrf`), `total_results`, `retrieval_time_ms`, `timings_ms` (per stage, summed
over the folders: `embed`, `dense`, `segment`, `bm25`, `fuse`, `rerank`) and `cached`.
A folder is skipped when it has **nothing indexed** (`nothing_indexed`), when some of its files were
indexed with **another embedding model** or dimension than the configured one (`embedding_model`:
its vectors are not comparable with the query's; re-index it), when there are more searchable
folders than `rag.retrieval.max_search_folders` (default 20; `folder_limit`: the most recently
updated ones are searched), or when its search failed (`error`). A folder with no document matching
the filter is simply not searched.

Results are cached for five minutes; any index, tag or metadata change in one of the searched
folders drops the entry. Auth is the console's, as for `/api/documents/search`: a Bearer token, or
the console session with its CSRF header. Only your own folders are ever searched.

## MCP tools

Every caller sees **`folio_find`**; each folder adds its own **`Folio_<folder>`** tool (only the
caller's folders are listed).

**`folio_find(query?, filters?, folders?, limit?, offset?)`** finds documents across all of the
caller's folders, so an agent can find *which* documents first and then search *inside* them
(the split RAGFlow makes between `search_metadata` and `search_chunks`):

- with `query`: the [search across folders](#search-across-folders) with `group_by: document` —
  documents ranked by their best passage, each with up to 2 snippets (cut to about 220 tokens each),
  `hits` and `score`; `skipped_folders` lists `{folder, reason}`;
- without `query`: the documents matching `filters`, newest upload first, with `total` and
  `next_offset` (pass it as `offset`). Called with no argument at all, it also returns `vocabulary`:
  your most used tags (with counts) and metadata keys (with their value types).

| Argument | Default | Meaning |
|---|---|---|
| `query` | none | 2–2000 characters; none = a listing |
| `filters` | none | a [document filter](#document-filters); an unknown field or operator is refused with its name |
| `folders` | all yours | folder names or ids; one that is not yours is `folder_not_found`, like one that does not exist |
| `limit` | 10 | 1–50 documents |
| `offset` | 0 | listing only |

Each document has `file_id`, `name`, `folder`, **`tool`** (the folder tool to search it with),
`tags`, `metadata`, and with a query `score`, `hits` and `snippets` (`text`, `page`, `score`), or in a
listing `mime`, `uploaded_at`, `status` and a short `description`. The answer is cut to
`rag.retrieval.mcp_response_token_budget` like the folder tools': lower-ranked documents are left
out first (`omitted_results`). `_hint` suggests the next step.

The scope is always the person the token belongs to (its `sub`), never an argument: a token without
a `sub` gets `{"error": "unauthorized"}`, and a token whose `sub` owns no folder (the MCP Gate
scanner, a `client_credentials` client) gets an empty answer. With `auth.enabled: false` it covers
every folder, like the REST endpoints. Refusals come back as `{"error": code, "detail": ...}`
(`invalid_filter`, `invalid_request`, `folder_not_found`, `unauthorized`). In tools/list, the tool's
description carries the caller's own top tags and metadata keys, so the agent writes filters with
real values (refreshed at most once a minute per token).

**`Folio_<folder>`** has three modes: `search` (the default), `list` and `read`. In `search` mode two
optional arguments limit the search to some documents of that folder:

- `file_ids`: the `file_id` values `folio_find` (or `list`) returned. Ids that are not documents of
  this folder are ignored and listed in `scope.ignored_file_ids`; when none is left the call is
  refused (`error_code: invalid_request`) rather than silently searching nothing.
- `filters`: the same document filter, applied inside this folder. With `file_ids` too, a document
  must satisfy both. When nothing matches, nothing is searched and `_hint` says so.

The ids restrict both the dense and the keyword query (as `filters` does on `POST /api/rag/query`).
The response then carries `scope: {file_ids: <count>, ignored_file_ids?}`. `list` mode returns each
file's `tags`, `metadata` and `description`.

## Removed in 1.5: the /v1 media endpoints

Folio 1.5 runs no OCR or speech-to-text model, so the `/v1` media endpoints (announced in 1.4.2)
are gone. Each answers **410 Gone**, with the replacement in `detail` and
`Link: <https://github.com/xianhong1208/folio-rag/blob/main/docs/migrate-1.5.md>; rel="help"`.
The request body is not read, and the `Deprecation` headers of 1.4.2 are gone with them.

| Endpoint | Call instead |
|---|---|
| `POST /v1/ocr/general` | An OpenAI-compatible vision model, `POST {base_url}/chat/completions` with the page as an `image_url` part (e.g. Qwen3-VL on [Forgeyard](https://github.com/xianhong1208/forgeyard)) |
| `POST /v1/transcriptions/general` | An OpenAI-compatible transcription endpoint, `POST {base_url}/audio/transcriptions` |
| `GET /v1/models` | That server's own `GET {base_url}/models` |

Inside Folio the same models are configured as `rag.parsing.vision` and `rag.asr`
([Upgrading to 1.5](migrate-1.5.md)).

The full OpenAPI reference is served by the running server at `/docs` (Swagger UI) and `/openapi.json`.
