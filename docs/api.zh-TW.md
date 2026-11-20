# API 參考

[English](api.md)

| 群組 | 端點 |
|------|------|
| **MCP** | `POST /mcp` —— 每資料夾的 `Folio_<folder>` 工具（`search` / `list` / `read`）;`GET /.well-known/oauth-protected-resource/mcp` |
| **資料夾與檔案** | `/api/folders*` · `/api/folders/{id}/files*`（建立、上傳、列出、下載、刪除 —— token-scoped） |
| **文件** | `POST /api/documents/search` · `GET /api/tags` · `GET /api/metadata/keys`——跨資料夾列出你的文件,可用[篩選條件](#文件篩選) |
| **搜尋** | `POST /api/search`——一次搜尋你所有(或指定的)資料夾,結果可以是段落或文件([見下文](#跨資料夾搜尋)) |
| **RAG** | `POST /api/rag/query`（混合檢索 + 重排,可帶 `filters`）· `/api/rag/files/{id}/index`、`/reindex`、`/index/jobs*`（背景索引 + SSE 進度） |
| **主控台 session** | `POST /api/console/session` `{token}`(登入)· `GET`(目前身分)· `DELETE`(登出) |
| **主控台（admin）** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **媒體(1.5 已移除)** | `POST /v1/ocr/general`、`POST /v1/transcriptions/general` 與 `GET /v1/models` 都回 **410 Gone**,附上 `Link: <…migrate-1.5.md>; rel="help"` 並說明改呼叫什麼([見下文](#15-移除v1-媒體端點))。 |
| **維運** | `GET /health`（存活檢查）· `GET /health/detailed`（資料庫、pgvector、連線池、快取、revocation feed、[備份](backup.zh-TW.md#排程備份)）· `GET /health/cache/stats` · `POST /health/cache/clear` · `GET /health/database/stats` · `GET /metrics`（Prometheus;[監控指標](metrics.zh-TW.md)）|

**上傳音訊。** 當語音轉文字無法轉錄時 —— `rag.asr` 沒有 `base_url`,或 `rag.asr.enabled: false` —— 音訊檔(`.wav`、`.mp3`、`.m4a`、`.aac`、`.ogg`、`.flac`)會以 **422** 拒絕,訊息會說明要設定什麼;批次上傳把它列在 `failed_uploads`。檔案不會被存下,也就不會留下空白或失敗的索引。見[升級到 1.5](migrate-1.5.zh-TW.md#語音轉文字)。

**上傳圖片。** 沒有設定視覺模型時(`rag.parsing.vision` 缺 `model` 或 `base_url`),圖片(`.png`、`.jpg`、`.jpeg`、`.tiff`、`.tif`、`.bmp`、`.webp`、`.gif`)會以 **422** 拒絕:只有視覺模型讀得懂圖片。訊息會說明要設定什麼;批次上傳把它列在 `failed_uploads`。PDF 一律可以上傳:沒有視覺模型時從文字層讀取,沒有文字層的頁面以看得見的標記索引,完全沒有文字的 PDF 會以同樣的建議索引失敗。見[升級到 1.5](migrate-1.5.zh-TW.md#pdf-與圖片)。

**上傳大小。** 單一檔案最大 100 MB。上傳以每次 1 MiB 寫入磁碟,不會整份放進記憶體。單檔上傳端點（`POST /api/folders/{id}/file` 與主控台的上傳）在 `Content-Length` 已超過上限時直接拒絕、不讀內容;沒有 `Content-Length` 的串流則在超過上限的當下中止,都回 **413** `File size exceeds limit (100.0MB)`。批次上傳（`POST /api/folders/{id}/files`）把過大的檔案列在 `failed_uploads`,其餘照常上傳。

## 文件篩選

每份文件(檔案)都可以帶**標籤**與 **metadata**,搜尋與文件列表都可以只取符合**篩選條件**的文件。

**標籤**是簡短的標記,每個人一個命名空間,跨所有資料夾共用。每個標籤會去掉前後空白(中間連續空白合成一個),
最多 64 個字元,每個檔案最多 32 個;只差在大小寫的標籤視為同一個(保留第一次的寫法),比對時不分大小寫。
**metadata** 是扁平物件:key 須符合 `^[a-z0-9_]{1,40}$`,最多 32 個;值可以是字串(最多 500 字元)、數字、
`true` / `false`,或寫成 `YYYY-MM-DD` 的日期。清單與物件會被拒絕(**422**,並指出是哪個 key)。

上傳時設定(`POST /api/folders/{id}/file` 與主控台上傳:表單欄位 `tags` 以逗號分隔,`metadata` 為 JSON 物件),
或之後用 `PATCH /api/folders/{id}/files/{file_id}`(主控台:`PATCH /api/admin/manage/folders/{id}/files/{file_id}`):

```json
{"tags": ["invoice", "signed"], "metadata": {"client": "Acme", "year": 2024, "due": "2024-12-31"}}
```

沒給的欄位維持不變;`tags` 與 `metadata` 會整個取代原本的清單 / 物件(`[]` / `{}` 代表清空)。修改後,
該資料夾快取的搜尋結果會立刻失效。

**篩選條件**是一個扁平物件,所有條件都要成立(AND):

| 欄位 | 比對 |
|---|---|
| `tags_any` / `tags_all` | 至少有其中一個 / 全部標籤(不分大小寫) |
| `folders` | 這些資料夾,用名稱或 id——永遠只在你自己的資料夾裡 |
| `file_type` | 副檔名(`pdf`、`.docx`)或 MIME 類型(`application/pdf`、`image/*`) |
| `uploaded_after` / `uploaded_before` | 上傳時間,含 / 不含該時間;`YYYY-MM-DD` 或 ISO 日期時間 |
| `name_contains` | 檔名包含的文字,不分大小寫 |
| `metadata` | `{key: value}`(相等;字串須完全相同),或 `{key: {"in": [...]}}`、`{"gte": x}`、`{"lte": x}`(數字,或與日期比較的 `YYYY-MM-DD` 日期)、`{"exists": true/false}` |

```json
{"tags_any": ["invoice"], "file_type": ["pdf"],
 "metadata": {"client": {"in": ["Acme", "Beta"]}, "year": {"gte": 2024}}}
```

不認得的欄位或運算子會回 **422** 並指出名稱。篩選條件由 Folio 自己轉成帶參數綁定的 SQL,而且一律加上你的擁有者
範圍:篩選只能縮小你看得到的範圍,不能擴大。

可以使用篩選的地方:

- **`POST /api/rag/query`** 與主控台的 playground API(`POST /api/admin/manage/folders/{id}/query`)可帶 `filters`:只搜尋該資料夾裡符合的文件(向量與關鍵字
  搜尋都會套用;pgvector 的 iterative HNSW scan 讓高選擇性的篩選不會回傳太少結果)。沒有符合的文件時回傳空結果,
  不是錯誤。
- **`POST /api/documents/search`** 分頁列出你所有資料夾中的文件:body 為 `{"filters": {...}, "sort": "name" |
  "uploaded" | "size", "order": "asc" | "desc", "limit": 1-200(預設 50), "offset": 0}`。每份文件含 `id`、
  `name`、`folder {id, name}`、`size_bytes`、`mime`、`uploaded_at`、`updated_at`、`status`(索引狀態)、`tags`、
  `metadata` 與 `description`;回應另有 `total` 與 `next_offset`(最後一頁為 null)。
- **`GET /api/tags`**:你的標籤與文件數(大小寫不同的寫法合併,顯示最常見的寫法)。**`GET /api/metadata/keys`**:
  你的 metadata key、各有多少文件,以及出現過的值類型(`string`、`number`、`boolean`、`date`)。

這三個端點和主控台 API 一樣,接受 Bearer token(個人存取權杖)或主控台登入狀態;以 cookie 驗證的 `POST` 需要
主控台的 CSRF 標頭。在 MCP 中,資料夾工具的 `list` 模式會回傳每個檔案的 `tags`、`metadata` 與 `description`。

## 跨資料夾搜尋

`POST /api/search` 一次搜尋你所有的資料夾(或你指定的),可加[篩選條件](#文件篩選),結果可以是**段落**或**文件**:

```json
{"query": "付款條件", "folders": ["contracts", 12], "filters": {"tags_any": ["signed"]},
 "group_by": "document", "top_k": 10, "snippets": 3}
```

| 欄位 | 預設 | 說明 |
|---|---|---|
| `query` | — | 2–2000 個字元 |
| `folders` | 你全部的資料夾 | 資料夾名稱或 id;不是你的資料夾回 **404**,與不存在的資料夾相同 |
| `filters` | 無 | [文件篩選條件](#文件篩選) |
| `group_by` | `passage` | `passage` = 依排名的段落(`results`);`document` = 依排名的文件(`documents`) |
| `top_k` | 段落為 `rag.retrieval.default_top_k`、文件為 10 | 1–50 |
| `snippets` | 3 | `document`:每份文件顯示幾個段落(1–3) |
| `auto_merge` | 依設定 | `passage`:命中的 leaf 是否合併成所屬章節 |

運作方式:每個資料夾用自己的索引搜尋(自己的向量表與自己的關鍵字斷詞器),平行進行(一次四個資料夾,並受伺服器的查詢上限約束),查詢只做一次 embedding。各資料夾的候選合在一起,**只重排一次**:cross-encoder 的分數可以公平比較不同資料夾的段落。沒有 reranker(或重排失敗)時,以 reciprocal rank 合併各資料夾的清單,絕不比較原始相似度分數 —— 不同表格的分數無法比較。多個資料夾時每個資料夾提供較小的候選池(平常預算的兩倍平分給各資料夾,每個至少 10 個向量 + 4 個關鍵字候選),最多重排 150 個候選。只有一個資料夾且 `group_by: passage` 時,完全等同 `POST /api/rag/query` 的單資料夾搜尋。

**文件**依最佳段落的分數排名(MaxP);同分時,命中段落較多的文件在前。每份文件含 `file_id`、`name`、`folder {id, name}`、`mime`、`tags`、`metadata`、`score`(最佳段落的分數)、`hits`(排名段落中屬於它的數量)與 `snippets`(`text`、`score`、`page`、`headings`、`content_type`、`node_id`)。名稱、標籤與 metadata 都是即時讀取,改名後立刻生效。**段落**就是一般的結果物件,另加 `folder {id, name}`。

回應另有 `folders`(搜尋了哪些)、`skipped_folders`(`{id, name, reason, detail}`)、`score_type`(`rerank` 或 `rrf`)、`total_results`、`retrieval_time_ms`、`timings_ms`(各階段時間,多個資料夾相加:`embed`、`dense`、`segment`、`bm25`、`fuse`、`rerank`)與 `cached`。以下情況資料夾會被略過:**還沒有任何索引**(`nothing_indexed`);有檔案是以**不同的 embedding 模型**或維度索引(`embedding_model`:它的向量無法和查詢比較,請重新索引);可搜尋的資料夾超過 `rag.retrieval.max_search_folders`(預設 20;`folder_limit`:搜尋最近更新的那些);或該資料夾的搜尋失敗(`error`)。沒有任何文件符合篩選條件的資料夾則直接不搜尋。

結果快取五分鐘;被搜尋的任一資料夾有索引、標籤或 metadata 變動時,快取立即失效。驗證方式與 `/api/documents/search` 相同:Bearer token,或主控台登入狀態加上 CSRF 標頭。只會搜尋你自己的資料夾。

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
