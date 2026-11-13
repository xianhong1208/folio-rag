# API 參考

[English](api.md)

| 群組 | 端點 |
|------|------|
| **MCP** | `POST /mcp` —— 每資料夾的 `Folio_<folder>` 工具（`search` / `list` / `read`）;`GET /.well-known/oauth-protected-resource/mcp` |
| **資料夾與檔案** | `/api/folders*` · `/api/folders/{id}/files*`（建立、上傳、列出、下載、刪除 —— token-scoped） |
| **RAG** | `POST /api/rag/query`（混合檢索 + 重排）· `/api/rag/files/{id}/index`、`/reindex`、`/index/jobs*`（背景索引 + SSE 進度） |
| **主控台 session** | `POST /api/console/session` `{token}`(登入)· `GET`(目前身分)· `DELETE`(登出) |
| **主控台（admin）** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **媒體（不需驗證）** | `POST /v1/transcriptions/general`（音訊轉文字,依 `rag.asr.provider`）· `POST /v1/ocr/general`（圖片 / PDF 轉文字,Docling OCR）· `GET /v1/models`。刻意不做驗證,請只開放在可信任的網路。預計在 1.5 移除（見[路線圖](../ROADMAP.zh-TW.md)）。 |
| **維運** | `GET /health`（存活檢查）· `GET /health/detailed`（資料庫、pgvector、連線池、快取、revocation feed）· `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` |

**上傳大小。** 單一檔案最大 100 MB。上傳以每次 1 MiB 寫入磁碟,不會整份放進記憶體。單檔上傳端點（`POST /api/folders/{id}/file` 與主控台的上傳）在 `Content-Length` 已超過上限時直接拒絕、不讀內容;沒有 `Content-Length` 的串流則在超過上限的當下中止,都回 **413** `File size exceeds limit (100.0MB)`。批次上傳（`POST /api/folders/{id}/files`）把過大的檔案列在 `failed_uploads`,其餘照常上傳。

執行中的伺服器在 `/docs`（Swagger UI）與 `/openapi.json` 提供完整的 OpenAPI 參考。
