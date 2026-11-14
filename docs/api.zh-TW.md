# API 參考

[English](api.md)

| 群組 | 端點 |
|------|------|
| **MCP** | `POST /mcp` —— 每資料夾的 `Folio_<folder>` 工具（`search` / `list` / `read`）;`GET /.well-known/oauth-protected-resource/mcp` |
| **資料夾與檔案** | `/api/folders*` · `/api/folders/{id}/files*`（建立、上傳、列出、下載、刪除 —— token-scoped） |
| **RAG** | `POST /api/rag/query`（混合檢索 + 重排）· `/api/rag/files/{id}/index`、`/reindex`、`/index/jobs*`（背景索引 + SSE 進度） |
| **主控台 session** | `POST /api/console/session` `{token}`(登入)· `GET`(目前身分)· `DELETE`(登出) |
| **主控台（admin）** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **媒體（不需驗證,已棄用）** | `POST /v1/transcriptions/general`（音訊轉文字,依 `rag.asr.provider`）· `POST /v1/ocr/general`（圖片 / PDF 轉文字,Docling OCR）· `GET /v1/models`。刻意不做驗證,請只開放在可信任的網路。**將在 1.5 移除**（見下文）。 |
| **維運** | `GET /health`（存活檢查）· `GET /health/detailed`（資料庫、pgvector、連線池、快取、revocation feed、[備份](backup.zh-TW.md#排程備份)）· `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` |

**上傳大小。** 單一檔案最大 100 MB。上傳以每次 1 MiB 寫入磁碟,不會整份放進記憶體。單檔上傳端點（`POST /api/folders/{id}/file` 與主控台的上傳）在 `Content-Length` 已超過上限時直接拒絕、不讀內容;沒有 `Content-Length` 的串流則在超過上限的當下中止,都回 **413** `File size exceeds limit (100.0MB)`。批次上傳（`POST /api/folders/{id}/files`）把過大的檔案列在 `failed_uploads`,其餘照常上傳。

## 1.5 移除:內建 OCR、語音轉文字與 /v1 媒體端點

Folio 1.5 會拿掉三項需要在行程內載入大型模型的功能。1.4.x 全部照常可用,只是事先公告:

| 1.5 移除 | 改用 |
|---|---|
| **內建 OCR**(docling 的 RapidOCR,用於掃描 PDF 與圖片;`rag.parsing.pdf` / `rag.parsing.images: local` 且 `rag.docling.ocr_enabled: true`) | **OpenAI 相容的視覺模型**:`rag.parsing.pdf: cloud`、`rag.parsing.images: cloud`,並設定 `rag.parsing.vision`(`provider`、`base_url`、`model`、`api_key`),例如由 [Forgeyard](https://github.com/xianhong1208/forgeyard) 提供的 Qwen3-VL。見[設定](configuration.zh-TW.md)。 |
| **內建語音轉文字**(`rag.asr.provider: fireredasr` 或 `docling-whisper`) | **OpenAI 相容的轉錄端點**(`/v1/audio/transcriptions`):`rag.asr.provider: openai-compatible`,並設定 `base_url`、`model` 與 `api_key`,例如 Forgeyard 的。 |
| **`/v1` 媒體端點**(`POST /v1/ocr/general`、`POST /v1/transcriptions/general`、`GET /v1/models`) | 直接呼叫視覺模型或轉錄端點;任何 OpenAI 相容的伺服器(例如 Forgeyard)都同時提供兩者。 |

1.4.x 會這樣提醒你:

- **啟動時**,設定中用到的每一項,都會在日誌記一行 `[DEPRECATION]` 警告(本機 OCR、行程內的 ASR provider、`media` 模組)。
- **第一次使用時**,每個行程各警告一次:第一次建立本機 OCR 模型、第一次在行程內轉錄、第一次呼叫 `/v1`。
- **每個 `/v1` 回應**(包含錯誤)都帶 `Deprecation: @1794528000`([RFC 9745](https://www.rfc-editor.org/rfc/rfc9745):自 2026-11-13 起棄用)與 `Link: <…本節…>; rel="deprecation"`。因為 1.5 還沒有發布日期,目前不帶 `Sunset` 標頭。

執行中的伺服器在 `/docs`（Swagger UI）與 `/openapi.json` 提供完整的 OpenAPI 參考。
