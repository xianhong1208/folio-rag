# API reference

[繁體中文](api.zh-TW.md)

| Group | Endpoints |
|-------|-----------|
| **MCP** | `POST /mcp` — per-folder `Folio_<folder>` tools (`search` / `list` / `read`); `GET /.well-known/oauth-protected-resource/mcp` |
| **Folders & files** | `/api/folders*` · `/api/folders/{id}/files*` (create, upload, list, download, delete — token-scoped) |
| **RAG** | `POST /api/rag/query` (hybrid retrieval + rerank) · `/api/rag/files/{id}/index`, `/reindex`, `/index/jobs*` (background indexing + SSE progress) |
| **Console session** | `POST /api/console/session` `{token}` (sign in) · `GET` (who am I) · `DELETE` (sign out) |
| **Console (admin)** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **Media (no auth)** | `POST /v1/transcriptions/general` (audio → text, follows `rag.asr.provider`) · `POST /v1/ocr/general` (image / PDF → text, Docling OCR) · `GET /v1/models`. Unauthenticated by design; keep them on a trusted network. They are planned for removal in 1.5 (see the [roadmap](../ROADMAP.md)). |
| **Operational** | `GET /health` (liveness) · `GET /health/detailed` (database, pgvector, connection pool, cache, revocation feed) · `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` |

The full OpenAPI reference is served by the running server at `/docs` (Swagger UI) and `/openapi.json`.
