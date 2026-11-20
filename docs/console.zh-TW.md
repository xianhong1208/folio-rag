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
| **Library** | 把你所有資料夾的每一份文件放在同一張表,在伺服器端分頁(可依名稱、上傳日期或大小排序)。**篩選欄**可以縮小範圍:標籤與各自的文件數(點一下切換;*any* 或 *all*)、資料夾、檔案類型、上傳日期區間、檔名包含的文字,以及 metadata 條件(key、運算子 —— `=`、*one of*、`≥`、`≤`、*is set*、*is not set* —— 與值)。生效中的篩選會以 chip 顯示在表格上方(點一下即移除),整個檢視都存在網址裡,可以加書籤或分享。在 **search inside documents** 輸入文字,就改成依內容為同一批篩選後的文件排名(最相關段落、命中數與分數,前 50 份);清除後回到列表。**勾選**多列(可跨頁)即可一次加上或移除標籤、設定或清除某個 metadata key,有進度與逐份文件的錯誤清單。**點一列**打開文件面板:資料夾(可連過去)、類型、大小、索引狀態、日期、可編輯的標籤、metadata 與說明、下載,以及只在這份文件裡搜尋。手機上篩選欄從 *Filters* 按鈕打開。見[下方](#library)。 |
| **Folders** | 建立資料夾(歸你所有);點進去可上傳檔案(可加標籤)、為檔案設定標籤與 metadata(列上的標籤按鈕)、看逐檔索引狀態、檢視 chunk(AI 脈絡說明與文件原文分開顯示)、重新索引、刪除。**Export** 把資料夾下載成一個封存檔,**Import** 把封存檔匯入成你的新資料夾([說明](backup.zh-TW.md#在-console-裡))。以舊版切塊流程、或以不同於目前設定的 embedding 模型索引的檔案,會標示 **Re-index recommended**;資料夾列表會逐一統計(**N stale**),並可一鍵重建。 |
| **Search Playground** | 對資料夾即時查詢,顯示**逐訊號檢索軌跡** —— Vector · BM25 · Hybrid · Rerank 各自名次並列 —— 並可加 RAG 答案。**Filters** 打開與 Library 相同的標籤、metadata 與(選 All folders 時)資料夾篩選,套用在這頁的每一次搜尋。選 **All folders** 可一次搜尋所有資料夾(每筆結果標示所屬資料夾;沒被搜尋的資料夾會列出原因);切換 **Passages / Documents** 可改看依排名的文件與各自最相關的段落。RAG 答案與檢索軌跡只適用於單一資料夾的段落模式。 |
| **Index Jobs** | 索引任務即時串流與進度;可取消執行中的任務。 |
| **Query Analytics** | 查詢量、延遲、零結果率、熱門查詢與知識缺口。 |
| **Evaluation** | 自動生成題組,對 `production`(與 `/api/rag/query` 同一路徑)、純 `vector`、純 `bm25` 評 Recall@k · MRR · nDCG@k;基準差異只與同一題組的歷次結果比較。 |
| **Settings** | Embedding / LLM / Contextual Retrieval / Reranker / Speech-to-Text / Retrieval 的即時、持久化設定,含檢索預設與連線測試。修改需要 `rag:admin`。 |
| **System Health** | 每個外部相依的即時連線狀態。 |
| **Audit Log** | 每次執行期設定變更與變更者的時間軸(`rag:admin`)。 |

## Library

側欄的 **Library**(`/admin/library`)列出文件;**Folders**(`/admin/folders`,含 *Index Jobs* 分頁)
用來建立資料夾與上傳。舊的 `/folders` 與 `/jobs` 連結照常可用,`/folders?folder=<id>` 會直接打開該資料夾。

- **篩選是一個 AND 條件**,與 API 的[文件篩選](api.zh-TW.md#文件篩選)相同:文件必須符合每一個 chip。
  多個標籤預設符合*任一*即可;在 Tags 標題切換成 *all* 則須全部符合。同一個 metadata key 的多列會合併
  (`year ≥ 2020` 與 `year ≤ 2024` 就是區間)。metadata 值的型別與編輯框相同:`2024` 是數字、`"2024"`
  是文字、`true` 是布林、`2024-03-01` 是日期。*to* 日期包含當天。
- **網址**保存所有篩選、內容查詢、排序與頁碼,例如
  `/admin/library?tag=invoice&m=year:gte:2024&sort=name&order=asc`。
- **批次編輯**只對會改變的文件各送一個 `PATCH`(已有該標籤或值的文件不動),一次兩個;失敗的會逐份列出,
  其餘照常寫入。完成後標籤數量與 metadata key 會重新載入。
- **Search inside this document** 會以 `filters.file_ids` 限定這一份文件,執行該資料夾的搜尋。

## 升級後重新索引

切塊流程的版本逐檔記錄。以舊版流程索引的檔案照常可用,但資料夾檢視會標示 **Re-index recommended**(`GET /api/rag/file/{id}/index/status` 也會回 `reindex_recommended: true`)。按該列的重新索引按鈕,或用標題列的連結一次選取所有這類檔案,再按 *Reindex selected*。之後逐檔重新索引都是增量的;*Reindex Folder* 則會刪掉資料夾的資料表、全部重建。

不用一個個點開資料夾去找:資料夾列表會在含有這類檔案的資料夾上顯示琥珀色的 **N stale** 標籤(`GET /api/admin/folders` 的 `reindex_recommended_files`),按該列的重新索引按鈕就只會重建那些檔案。

目前的切塊流程是第 3 版(依章節對齊切塊,搭配第 3 版的脈絡說明 prompt),因此在此之前索引的檔案都會被標示。重新索引之前,檔案會保留舊的 chunk,照常可以搜尋。

## 更換 embedding 模型後重新索引

每個檔案也會記錄索引時用的 embedding 模型與維度(`embedding_model`、`embedding_dimension`)。在 `config/models.yaml` 或設定頁改了 `rag.embedding.model` 或 `rag.embedding.dimension` 之後,所有用舊模型索引的檔案會立刻以同樣方式標示 **Re-index recommended**,並計入資料夾的 **N stale** 標籤:它們的向量來自另一個模型,跟新的查詢對不上。標籤的提示文字與 API 的 `reindex_reason` 會說明是哪一種情況(`embedding` 或 `pipeline`)。改回原本的模型,標示就會消失。1.4.2 之前索引的檔案,伺服器啟動時會從資料夾的向量表補上維度;從未記錄模型或維度的檔案,不會因此被標示。
