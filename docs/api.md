# API reference

[繁體中文](api.zh-TW.md)

| Group | Endpoints |
|-------|-----------|
| **MCP** | `POST /mcp` — per-folder `Folio_<folder>` tools (`search` / `list` / `read`); `GET /.well-known/oauth-protected-resource/mcp` |
| **Folders & files** | `/api/folders*` · `/api/folders/{id}/files*` (create, upload, list, download, delete — token-scoped) |
| **Documents** | `POST /api/documents/search` · `GET /api/tags` · `GET /api/metadata/keys` — your documents across folders, with [filters](#document-filters) |
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
console's API; a cookie-authenticated `POST` needs the console's CSRF header. In MCP, a folder tool's
`list` mode returns each file's `tags`, `metadata` and `description`.

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
