# API reference

[繁體中文](api.zh-TW.md)

| Group | Endpoints |
|-------|-----------|
| **MCP** | `POST /mcp` — per-folder `Folio_<folder>` tools (`search` / `list` / `read`); `GET /.well-known/oauth-protected-resource/mcp` |
| **Folders & files** | `/api/folders*` · `/api/folders/{id}/files*` (create, upload, list, download, delete — token-scoped) |
| **RAG** | `POST /api/rag/query` (hybrid retrieval + rerank) · `/api/rag/files/{id}/index`, `/reindex`, `/index/jobs*` (background indexing + SSE progress) |
| **Console session** | `POST /api/console/session` `{token}` (sign in) · `GET` (who am I) · `DELETE` (sign out) |
| **Console (admin)** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **Media (no auth, deprecated)** | `POST /v1/transcriptions/general` (audio → text, follows `rag.asr.provider`) · `POST /v1/ocr/general` (image / PDF → text, Docling OCR) · `GET /v1/models`. Unauthenticated by design; keep them on a trusted network. **Removed in 1.5** (see below). |
| **Operational** | `GET /health` (liveness) · `GET /health/detailed` (database, pgvector, connection pool, cache, revocation feed) · `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` |

**Upload size.** A file may be at most 100 MB. Uploads are copied to disk in 1 MiB pieces and never held in memory whole. The single-file upload routes (`POST /api/folders/{id}/file` and the console's) refuse a larger request from its `Content-Length` before reading it, or as soon as a streamed body crosses the limit, with **413** `File size exceeds limit (100.0MB)`; the batch route (`POST /api/folders/{id}/files`) reports an oversized file in `failed_uploads` and keeps the others.

## Removed in 1.5: built-in OCR, speech-to-text and the /v1 media endpoints

Folio 1.5 drops three things that need heavy in-process models. 1.4.x keeps all of them working; it only announces the change:

| Goes in 1.5 | Use instead |
|---|---|
| **The built-in OCR** (docling's RapidOCR on scanned PDFs and images, with `rag.parsing.pdf` / `rag.parsing.images: local` and `rag.docling.ocr_enabled: true`) | An **OpenAI-compatible vision model**: `rag.parsing.pdf: cloud` and `rag.parsing.images: cloud` with `rag.parsing.vision` (`provider`, `base_url`, `model`, `api_key`), e.g. Qwen3-VL served by [Forgeyard](https://github.com/xianhong1208/forgeyard). See [Configuration](configuration.md). |
| **The built-in speech-to-text** (`rag.asr.provider: fireredasr` or `docling-whisper`) | An **OpenAI-compatible transcription endpoint** (`/v1/audio/transcriptions`): `rag.asr.provider: openai-compatible` with `base_url`, `model` and `api_key`, e.g. Forgeyard's. |
| **The `/v1` media endpoints** (`POST /v1/ocr/general`, `POST /v1/transcriptions/general`, `GET /v1/models`) | Call the vision model or the transcription endpoint directly; any OpenAI-compatible server (e.g. Forgeyard) serves both. |

How 1.4.x tells you:

- **At startup**, one `[DEPRECATION]` warning in the log for each of them the configuration uses (local OCR, an in-process ASR provider, the `media` module).
- **On first use**, one warning per process: the first time the local OCR model is set up, the first in-process transcription, and the first `/v1` call.
- **On every `/v1` response** (errors included): `Deprecation: @1794528000` ([RFC 9745](https://www.rfc-editor.org/rfc/rfc9745): deprecated since 2026-11-13) and `Link: <…this section…>; rel="deprecation"`. There is no `Sunset` header yet, because 1.5 has no release date.

The full OpenAPI reference is served by the running server at `/docs` (Swagger UI) and `/openapi.json`.
