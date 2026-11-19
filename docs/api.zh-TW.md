# API 參考

[English](api.md)

| 群組 | 端點 |
|------|------|
| **MCP** | `POST /mcp` —— 每資料夾的 `Folio_<folder>` 工具（`search` / `list` / `read`）;`GET /.well-known/oauth-protected-resource/mcp` |
| **資料夾與檔案** | `/api/folders*` · `/api/folders/{id}/files*`（建立、上傳、列出、下載、刪除 —— token-scoped） |
| **RAG** | `POST /api/rag/query`（混合檢索 + 重排）· `/api/rag/files/{id}/index`、`/reindex`、`/index/jobs*`（背景索引 + SSE 進度） |
| **主控台 session** | `POST /api/console/session` `{token}`(登入)· `GET`(目前身分)· `DELETE`(登出) |
| **主控台（admin）** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **媒體(1.5 已移除)** | `POST /v1/ocr/general`、`POST /v1/transcriptions/general` 與 `GET /v1/models` 都回 **410 Gone**,附上 `Link: <…migrate-1.5.md>; rel="help"` 並說明改呼叫什麼([見下文](#15-移除v1-媒體端點))。 |
| **維運** | `GET /health`（存活檢查）· `GET /health/detailed`（資料庫、pgvector、連線池、快取、revocation feed、[備份](backup.zh-TW.md#排程備份)）· `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` · `GET /metrics`（Prometheus;[監控指標](metrics.zh-TW.md)）|

**上傳音訊。** 當語音轉文字無法轉錄時 —— `rag.asr` 沒有 `base_url`,或 `rag.asr.enabled: false` —— 音訊檔(`.wav`、`.mp3`、`.m4a`、`.aac`、`.ogg`、`.flac`)會以 **422** 拒絕,訊息會說明要設定什麼;批次上傳把它列在 `failed_uploads`。檔案不會被存下,也就不會留下空白或失敗的索引。見[升級到 1.5](migrate-1.5.zh-TW.md#語音轉文字)。

**上傳圖片。** 沒有設定視覺模型時(`rag.parsing.vision` 缺 `model` 或 `base_url`),圖片(`.png`、`.jpg`、`.jpeg`、`.tiff`、`.tif`、`.bmp`、`.webp`、`.gif`)會以 **422** 拒絕:只有視覺模型讀得懂圖片。訊息會說明要設定什麼;批次上傳把它列在 `failed_uploads`。PDF 一律可以上傳:沒有視覺模型時從文字層讀取,沒有文字層的頁面以看得見的標記索引,完全沒有文字的 PDF 會以同樣的建議索引失敗。見[升級到 1.5](migrate-1.5.zh-TW.md#pdf-與圖片)。

**上傳大小。** 單一檔案最大 100 MB。上傳以每次 1 MiB 寫入磁碟,不會整份放進記憶體。單檔上傳端點（`POST /api/folders/{id}/file` 與主控台的上傳）在 `Content-Length` 已超過上限時直接拒絕、不讀內容;沒有 `Content-Length` 的串流則在超過上限的當下中止,都回 **413** `File size exceeds limit (100.0MB)`。批次上傳（`POST /api/folders/{id}/files`）把過大的檔案列在 `failed_uploads`,其餘照常上傳。

## 1.5 移除:/v1 媒體端點

Folio 1.5 不再執行 OCR 或語音轉文字模型,所以 `/v1` 媒體端點(1.4.2 已預告)都已移除。每一個都回 **410 Gone**,
`detail` 說明改用什麼,並帶 `Link: <https://github.com/xianhong1208/folio-rag/blob/main/docs/migrate-1.5.md>; rel="help"`。
請求內容不會被讀取,1.4.2 的 `Deprecation` 標頭也一併拿掉。

| 端點 | 改呼叫 |
|---|---|
| `POST /v1/ocr/general` | OpenAI 相容的視覺模型,`POST {base_url}/chat/completions`,把頁面放在 `image_url`(例如 [Forgeyard](https://github.com/xianhong1208/forgeyard) 上的 Qwen3-VL) |
| `POST /v1/transcriptions/general` | OpenAI 相容的轉錄端點,`POST {base_url}/audio/transcriptions` |
| `GET /v1/models` | 該伺服器自己的 `GET {base_url}/models` |

Folio 內部使用的同一批模型,設定在 `rag.parsing.vision` 與 `rag.asr`([升級到 1.5](migrate-1.5.zh-TW.md))。

執行中的伺服器在 `/docs`（Swagger UI）與 `/openapi.json` 提供完整的 OpenAPI 參考。
