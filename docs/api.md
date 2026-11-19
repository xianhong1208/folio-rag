# API reference

[繁體中文](api.zh-TW.md)

| Group | Endpoints |
|-------|-----------|
| **MCP** | `POST /mcp` — per-folder `Folio_<folder>` tools (`search` / `list` / `read`); `GET /.well-known/oauth-protected-resource/mcp` |
| **Folders & files** | `/api/folders*` · `/api/folders/{id}/files*` (create, upload, list, download, delete — token-scoped) |
| **RAG** | `POST /api/rag/query` (hybrid retrieval + rerank) · `/api/rag/files/{id}/index`, `/reindex`, `/index/jobs*` (background indexing + SSE progress) |
| **Console session** | `POST /api/console/session` `{token}` (sign in) · `GET` (who am I) · `DELETE` (sign out) |
| **Console (admin)** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **Media (removed in 1.5)** | `POST /v1/ocr/general`, `POST /v1/transcriptions/general` and `GET /v1/models` answer **410 Gone** with `Link: <…migrate-1.5.md>; rel="help"` and what to call instead ([below](#removed-in-15-the-v1-media-endpoints)). |
| **Operational** | `GET /health` (liveness) · `GET /health/detailed` (database, pgvector, connection pool, cache, revocation feed, [backups](backup.md#scheduled-backups)) · `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` · `GET /metrics` (Prometheus; [Metrics](metrics.md)) |

**Audio uploads.** An audio file (`.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`) is refused with **422** while speech-to-text cannot transcribe it — `rag.asr` has no `base_url`, or `rag.asr.enabled: false` — and the detail says what to set; the batch route lists it in `failed_uploads`. Nothing is stored, so no empty or failed entry is left behind. See [Upgrading to 1.5](migrate-1.5.md#speech-to-text).

**Image uploads.** An image (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`, `.gif`) is refused with **422** while no vision model is configured (`rag.parsing.vision` without `model` and `base_url`): only a vision model can read it. The detail says what to set; the batch route lists it in `failed_uploads`. PDFs are always accepted: without a vision model they are read from their text layer, a page with no text layer is indexed as a visible marker, and a PDF with no text at all fails indexing with the same advice. See [Upgrading to 1.5](migrate-1.5.md#pdfs-and-images).

**Upload size.** A file may be at most 100 MB. Uploads are copied to disk in 1 MiB pieces and never held in memory whole. The single-file upload routes (`POST /api/folders/{id}/file` and the console's) refuse a larger request from its `Content-Length` before reading it, or as soon as a streamed body crosses the limit, with **413** `File size exceeds limit (100.0MB)`; the batch route (`POST /api/folders/{id}/files`) reports an oversized file in `failed_uploads` and keeps the others.

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
