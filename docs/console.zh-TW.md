# 主控台

[English](console.md)

**Retrieval Terminal** 主控台是 `/admin` 的 React SPA。

## 主控台登入

登入方式是貼上一把**由 MCP Gate 發給本服務的 access token**,通常是 personal access token(PAT)。伺服器會用跟 REST / MCP 呼叫完全相同的方式離線驗證一次(簽章、issuer、audience、到期時間;token 必須帶 `sub` 與 `jti`),通過後設定一個 HttpOnly 的 session cookie。Token 本身不會被儲存,也不會回傳。

- **Session 就是你本人。** 你看到、能管理的,正好是這把 token 所屬使用者擁有的資料夾(資料夾以它的 `sub` 綁定)—— 也就是該使用者每一把 token 透過 REST 與 MCP 能存取的那些。在主控台建立的資料夾同樣歸你所有。其他使用者的資料夾完全看不到。
- **兩個層級。** 任何有效的 token 都能登入並管理自己的資料夾。修改執行期 **Settings**、測試任意端點、讀取設定 **Audit Log** 與全伺服器的儲存用量,需要 **`rag:admin`** scope。沒有它時,Settings 與 System Health 為唯讀(secret 一律遮蔽),Overview / Index Jobs / Query Analytics 一律只涵蓋自己的資料夾——`rag:admin` 只開放全域設定,永遠看不到其他使用者的資料夾、工作或查詢。管理者另外會在 Overview 看到還有多少資料夾仍屬於 1.4.1 之前的「單一 token」key(見[資料夾歸屬](auth.zh-TW.md#資料夾歸屬))。
- **Session** 持續 `auth.console.session_hours`(8 小時)或到 token 到期為止,取較早者。啟用 [revocation feed](auth.zh-TW.md#token-撤銷) 後,在 MCP Gate 撤銷 token 也會在一個輪詢間隔內結束它的 session;未啟用時,進行中的 session 會持續到到期或按 *Sign out*(驗證是離線的)。

**在 MCP Gate 發 token。** MCP Gate 不需要任何程式修改:

1. *Services* → 打開 Folio 服務 → *Scopes declared by this server* → 新增 **`rag:admin`**。**不要**把它設成 default scope(default 會在每把新 token 上預先勾選)。如果該服務有限制 *Allowed scopes*,也要允許 `rag:admin`。
2. *Issue Token* → *Target service* 選 Folio(audience = `auth.audience`)。給可以改設定的維運者勾選 `rag:admin`;只管理自己資料夾的人不要勾。填上 label 後複製 token —— 它只會顯示一次。

把 token 貼到登入畫面即可。1.4.1 之前建立的資料夾,在建立它的那把 token 再次被使用之前,仍屬於那把 token;一旦使用,就會移到你名下(見[資料夾歸屬](auth.zh-TW.md#資料夾歸屬))。腳本可以不走 cookie,直接帶 `Authorization: Bearer <token>` 呼叫 `/api/admin/*` —— 規則相同,而且不需要 CSRF header。

**開發模式。** `auth.enabled: false` 時沒有登入:任何連得到這個 port 的人都能以完整管理權限使用主控台、看到所有資料夾(在這裡建立的資料夾擁有者為 `console`)。伺服器啟動時會記錄警告。只在可信任的機器上使用。

**網路暴露。** 即使已啟用登入,主控台(`/admin`、`/api/admin/*`)仍應只開放給可信任的網路 —— VPN、內部網路,或做存取限制的反向代理 —— 並以 HTTPS 提供。登入能擋下沒有有效 token 的人,但主控台可以上傳與刪除文件、對你的模型端點發出查詢,且(具 `rag:admin` 時)可修改設定,因此不應直接面對公開網際網路。`/mcp` 與 `/api/rag/*` 本來就設計給 MCP 用戶端連線,只需要有效 token。

| 頁面 | 功能 |
|------|------|
| **Overview** | 活的檢索管線、語料讀數、索引健康、模型服務狀態。 |
| **Folders** | 建立資料夾(歸你所有);點進去可上傳檔案、看逐檔索引狀態、檢視 chunk(AI 脈絡說明與文件原文分開顯示)、重新索引、刪除。以舊版切塊流程、或以不同於目前設定的 embedding 模型索引的檔案,會標示 **Re-index recommended**;資料夾列表會逐一統計(**N stale**),並可一鍵重建。 |
| **Search Playground** | 對資料夾即時查詢,顯示**逐訊號檢索軌跡** —— Vector · BM25 · Hybrid · Rerank 各自名次並列 —— 並可加 RAG 答案。 |
| **Index Jobs** | 索引任務即時串流與進度;可取消執行中的任務。 |
| **Query Analytics** | 查詢量、延遲、零結果率、熱門查詢與知識缺口。 |
| **Evaluation** | 自動生成題組,對 `production`(與 `/api/rag/query` 同一路徑)、純 `vector`、純 `bm25` 評 Recall@k · MRR · nDCG@k;基準差異只與同一題組的歷次結果比較。 |
| **Settings** | Embedding / LLM / Contextual Retrieval / Reranker / Speech-to-Text / Retrieval 的即時、持久化設定,含檢索預設與連線測試。修改需要 `rag:admin`。 |
| **System Health** | 每個外部相依的即時連線狀態。 |
| **Audit Log** | 每次執行期設定變更與變更者的時間軸(`rag:admin`)。 |

## 升級後重新索引

切塊流程的版本逐檔記錄。以舊版流程索引的檔案照常可用,但資料夾檢視會標示 **Re-index recommended**(`GET /api/rag/file/{id}/index/status` 也會回 `reindex_recommended: true`)。按該列的重新索引按鈕,或用標題列的連結一次選取所有這類檔案,再按 *Reindex selected*。之後逐檔重新索引都是增量的;*Reindex Folder* 則會刪掉資料夾的資料表、全部重建。

不用一個個點開資料夾去找:資料夾列表會在含有這類檔案的資料夾上顯示琥珀色的 **N stale** 標籤(`GET /api/admin/folders` 的 `reindex_recommended_files`),按該列的重新索引按鈕就只會重建那些檔案。

目前的切塊流程是第 3 版(依章節對齊切塊,搭配第 3 版的脈絡說明 prompt),因此在此之前索引的檔案都會被標示。重新索引之前,檔案會保留舊的 chunk,照常可以搜尋。

## 更換 embedding 模型後重新索引

每個檔案也會記錄索引時用的 embedding 模型與維度(`embedding_model`、`embedding_dimension`)。在 config.yaml 或設定頁改了 `rag.embedding.model` 或 `rag.embedding.dimension` 之後,所有用舊模型索引的檔案會立刻以同樣方式標示 **Re-index recommended**,並計入資料夾的 **N stale** 標籤:它們的向量來自另一個模型,跟新的查詢對不上。標籤的提示文字與 API 的 `reindex_reason` 會說明是哪一種情況(`embedding` 或 `pipeline`)。改回原本的模型,標示就會消失。1.4.2 之前索引的檔案,伺服器啟動時會從資料夾的向量表補上維度;從未記錄模型或維度的檔案,不會因此被標示。
