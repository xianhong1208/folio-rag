# API reference

[繁體中文](api.zh-TW.md)

| Group | Endpoints |
|-------|-----------|
| **MCP** | `POST /mcp` — per-folder `Folio_<folder>` tools (`search` / `list` / `read`); `GET /.well-known/oauth-protected-resource/mcp` |
| **Folders & files** | `/api/folders*` · `/api/folders/{id}/files*` (create, upload, list, download, delete — token-scoped) |
| **RAG** | `POST /api/rag/query` (hybrid retrieval + rerank) · `/api/rag/files/{id}/index`, `/reindex`, `/index/jobs*` (background indexing + SSE progress) |
| **Console session** | `POST /api/console/session` `{token}` (sign in) · `GET` (who am I) · `DELETE` (sign out) |
| **Console (admin)** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **Media (no auth, deprecated)** | `POST /v1/ocr/general` (image / PDF → text, Docling OCR) · `GET /v1/models` (lists the OCR model). Unauthenticated by design; keep them on a trusted network. **Announced for removal** (see below). `POST /v1/transcriptions/general` was **removed in 1.5** and answers **410 Gone** with a pointer to [the migration guide](migrate-asr.md). |
| **Operational** | `GET /health` (liveness) · `GET /health/detailed` (database, pgvector, connection pool, cache, revocation feed, [backups](backup.md#scheduled-backups)) · `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` · `GET /metrics` (Prometheus; [Metrics](metrics.md)) |

**Audio uploads.** An audio file (`.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`) is refused with **422** while speech-to-text cannot transcribe it — `rag.asr` has no `base_url`, or `rag.asr.enabled: false` — and the detail says what to set; the batch route lists it in `failed_uploads`. Nothing is stored, so no empty or failed entry is left behind. See [Moving speech-to-text to an endpoint](migrate-asr.md).

**Upload size.** A file may be at most 100 MB. Uploads are copied to disk in 1 MiB pieces and never held in memory whole. The single-file upload routes (`POST /api/folders/{id}/file` and the console's) refuse a larger request from its `Content-Length` before reading it, or as soon as a streamed body crosses the limit, with **413** `File size exceeds limit (100.0MB)`; the batch route (`POST /api/folders/{id}/files`) reports an oversized file in `failed_uploads` and keeps the others.

## Removed in 1.5: built-in OCR, speech-to-text and the /v1 media endpoints

Folio 1.5 drops things that need heavy in-process models. 1.4.2 announced them all; the speech-to-text part is **removed in 1.5** ([migration guide](migrate-asr.md)); the OCR part is still only announced, until the vision-model path has been compared with docling's OCR:

| Goes | Use instead |
|---|---|
| **The built-in OCR** (docling's RapidOCR on scanned PDFs and images, with `rag.parsing.pdf` / `rag.parsing.images: local` and `rag.docling.ocr_enabled: true`) | An **OpenAI-compatible vision model**: `rag.parsing.pdf: cloud` and `rag.parsing.images: cloud` with `rag.parsing.vision` (`provider`, `base_url`, `model`, `api_key`), e.g. Qwen3-VL served by [Forgeyard](https://github.com/xianhong1208/forgeyard). See [Configuration](configuration.md). |
| **The built-in speech-to-text** (`rag.asr.provider: fireredasr` or `docling-whisper`) — **removed in 1.5**; an old value stops the start | An **OpenAI-compatible transcription endpoint** (`/v1/audio/transcriptions`): `rag.asr` with `base_url`, `model` and `api_key`, e.g. whisper-large-v3-turbo on Forgeyard. See [the migration guide](migrate-asr.md). |
| **`POST /v1/transcriptions/general`** — **removed in 1.5**, answers 410 | Call the transcription endpoint directly. |
| **The other `/v1` media endpoints** (`POST /v1/ocr/general`, `GET /v1/models`) | Call the vision model directly; any OpenAI-compatible server (e.g. Forgeyard) serves it. |

How Folio tells you about the parts still announced:

- **At startup**, one `[DEPRECATION]` warning in the log for each of them the configuration uses (local OCR, the `media` module).
- **On first use**, one warning per process: the first time the local OCR model is set up, and the first `/v1` call.
- **On every `/v1` response** (errors included): `Deprecation: @1794528000` ([RFC 9745](https://www.rfc-editor.org/rfc/rfc9745): deprecated since 2026-11-13) and `Link: <…this section…>; rel="deprecation"`. There is no `Sunset` header yet, because 1.5 has no release date.

The full OpenAPI reference is served by the running server at `/docs` (Swagger UI) and `/openapi.json`.
