# API 參考

[English](api.md)

| 群組 | 端點 |
|------|------|
| **MCP** | `POST /mcp` —— 每資料夾的 `Folio_<folder>` 工具（`search` / `list` / `read`）;`GET /.well-known/oauth-protected-resource/mcp` |
| **資料夾與檔案** | `/api/folders*` · `/api/folders/{id}/files*`（建立、上傳、列出、下載、刪除 —— token-scoped） |
| **RAG** | `POST /api/rag/query`（混合檢索 + 重排）· `/api/rag/files/{id}/index`、`/reindex`、`/index/jobs*`（背景索引 + SSE 進度） |
| **主控台 session** | `POST /api/console/session` `{token}`(登入)· `GET`(目前身分)· `DELETE`(登出) |
| **主控台（admin）** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **媒體（不需驗證,已棄用）** | `POST /v1/ocr/general`（圖片 / PDF 轉文字,Docling OCR）· `GET /v1/models`(列出 OCR 模型)。刻意不做驗證,請只開放在可信任的網路。**預告移除**（見下文）。`POST /v1/transcriptions/general` **已在 1.5 移除**,會回 **410 Gone** 並指向[遷移指南](migrate-asr.zh-TW.md)。 |
| **維運** | `GET /health`（存活檢查）· `GET /health/detailed`（資料庫、pgvector、連線池、快取、revocation feed、[備份](backup.zh-TW.md#排程備份)）· `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` · `GET /metrics`（Prometheus;[監控指標](metrics.zh-TW.md)）|

**上傳音訊。** 當語音轉文字無法轉錄時 —— `rag.asr` 沒有 `base_url`,或 `rag.asr.enabled: false` —— 音訊檔(`.wav`、`.mp3`、`.m4a`、`.aac`、`.ogg`、`.flac`)會以 **422** 拒絕,訊息會說明要設定什麼;批次上傳把它列在 `failed_uploads`。檔案不會被存下,也就不會留下空白或失敗的索引。見[把語音轉文字改到端點](migrate-asr.zh-TW.md)。

**上傳大小。** 單一檔案最大 100 MB。上傳以每次 1 MiB 寫入磁碟,不會整份放進記憶體。單檔上傳端點（`POST /api/folders/{id}/file` 與主控台的上傳）在 `Content-Length` 已超過上限時直接拒絕、不讀內容;沒有 `Content-Length` 的串流則在超過上限的當下中止,都回 **413** `File size exceeds limit (100.0MB)`。批次上傳（`POST /api/folders/{id}/files`）把過大的檔案列在 `failed_uploads`,其餘照常上傳。

## 1.5 移除:內建 OCR、語音轉文字與 /v1 媒體端點

Folio 1.5 會拿掉需要在行程內載入大型模型的功能。1.4.2 已全部預告;語音轉文字的部分**已在 1.5 移除**([遷移指南](migrate-asr.zh-TW.md));OCR 的部分仍只是預告,要等視覺模型與 docling OCR 比較過後再移除:

| 移除項目 | 改用 |
|---|---|
| **內建 OCR**(docling 的 RapidOCR,用於掃描 PDF 與圖片;`rag.parsing.pdf` / `rag.parsing.images: local` 且 `rag.docling.ocr_enabled: true`) | **OpenAI 相容的視覺模型**:`rag.parsing.pdf: cloud`、`rag.parsing.images: cloud`,並設定 `rag.parsing.vision`(`provider`、`base_url`、`model`、`api_key`),例如由 [Forgeyard](https://github.com/xianhong1208/forgeyard) 提供的 Qwen3-VL。見[設定](configuration.zh-TW.md)。 |
| **內建語音轉文字**(`rag.asr.provider: fireredasr` 或 `docling-whisper`)—— **已在 1.5 移除**,舊值會讓啟動失敗 | **OpenAI 相容的轉錄端點**(`/v1/audio/transcriptions`):在 `rag.asr` 設定 `base_url`、`model` 與 `api_key`,例如 Forgeyard 上的 whisper-large-v3-turbo。見[遷移指南](migrate-asr.zh-TW.md)。 |
| **`POST /v1/transcriptions/general`** —— **已在 1.5 移除**,回 410 | 直接呼叫轉錄端點。 |
| **其餘 `/v1` 媒體端點**(`POST /v1/ocr/general`、`GET /v1/models`) | 直接呼叫視覺模型;任何 OpenAI 相容的伺服器(例如 Forgeyard)都能提供。 |

仍在預告中的部分,Folio 會這樣提醒你:

- **啟動時**,設定中用到的每一項,都會在日誌記一行 `[DEPRECATION]` 警告(本機 OCR、`media` 模組)。
- **第一次使用時**,每個行程各警告一次:第一次建立本機 OCR 模型、第一次呼叫 `/v1`。
- **每個 `/v1` 回應**(包含錯誤)都帶 `Deprecation: @1794528000`([RFC 9745](https://www.rfc-editor.org/rfc/rfc9745):自 2026-11-13 起棄用)與 `Link: <…本節…>; rel="deprecation"`。因為 1.5 還沒有發布日期,目前不帶 `Sunset` 標頭。

執行中的伺服器在 `/docs`（Swagger UI）與 `/openapi.json` 提供完整的 OpenAPI 參考。
