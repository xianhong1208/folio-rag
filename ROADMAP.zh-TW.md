# 發展路線圖

[English](ROADMAP.md)

Folio 在 1.4.1 之後的方向。這份規劃在 2026-11-13 定案,定案前參考了 RAGFlow、Onyx、Kotaemon、R2R、
Dify、Open WebUI 與 Khoj 怎麼解決同樣的問題。每一項寫的是你會得到什麼,而不是怎麼做出來。每一項在
GitHub 上都有一張 issue,放在該版本的 milestone 裡。

## 已發布

**1.0.0（2026-09）,當時叫 Agentic RAG。** 一台把文件資料夾變成檢索工具的 MCP 伺服器:階層式切塊與
auto-merge、向量 + BM25 混合搜尋(中文以 CKIP 斷詞)、cross-encoder 重排、contextual retrieval、
pgvector 儲存、資料夾依 token 歸屬,以及第一版 React 主控台。

**1.1.0（2026-09-18）。** Search Playground 的答案逐字串流。Docling 解析出的表格以真正的表格顯示,
MCP 結果也會標明命中的是表格。評測會為答案品質打分,並保留 baseline 歷史。*Rebuild FTS* 不必重新
embedding 就能為既有資料夾重新斷詞。React 主控台成為入口。

**1.2.0（2026-11-04）,改名 Folio。** 公開的 205 題黃金測試集,以及直接評測真實 MCP 搜尋的工具。
第 3 版切塊流程、更深的重排候選池與回應 token 上限(Recall@10 0.973 → 0.981,用中文問英文內容
0.80 → 0.90,每次搜尋的 token 8.2k → 2.6k)。MCP 搜尋會說明「找不到相關內容」。搜尋更快(p50 約
500 → 330 ms)。在 MCP Gate 撤銷的 token 約 30 秒內失效。主控台的統計只涵蓋你自己的資料夾。資料夾裡
過期的檔案可以一鍵重建索引。

**1.3.0（2026-11-05）。** 相依套件安全更新(pip-audit 警告 168 → 12)。新增上傳格式:RTF、MHTML
網頁封存檔、GIF、`.markdown` 與 Excel 範本。

**1.4.0（2026-11-09）。** CI 加上 lint 與每週的相依套件稽核。AMD ROCm 版本改用 torch 2.13 與
ROCm 7.14。

**1.4.1（2026-11-09）。** 資料夾屬於「人」,而不是某一把 token:同一位 MCP Gate 使用者的每一把
personal access token 與 OAuth 登入都看到同一組資料夾,舊資料夾在第一次使用時自動移過來。登出會在
伺服器端結束主控台 session。答案使用提問的語言。重複的資料夾名稱與檔名會被拒絕。手機上的主控台不再
左右捲動。

細節見 [CHANGELOG.md](CHANGELOG.md)。

## 接下來

| 版本 | 你會得到 |
|---|---|
| **1.4.2:修正** | **修掉已知風險:** 更換 embedding 模型後,需要重建索引的資料夾會被標出來;模型金鑰設定不能再被用來讀出伺服器的環境變數;過大的上傳在占滿記憶體之前就會被拒絕;搜尋從未索引過的資料夾會回傳空結果,而不是內部錯誤。**事先公告:** 內建的 OCR 與語音轉文字,以及 `/v1` 的 OCR 與轉錄端點,將在 1.5 移除;公告會說明用什麼取代。 |
| **1.5:一行指令安裝,資料不會遺失** | **輕量的 Docker 映像檔**,以及含 PostgreSQL + pgvector 的 compose 檔,另有可選的 MCP Gate profile。內建的 OCR 與語音模型從映像檔移除:掃描 PDF 與圖片改由 **OpenAI 相容 API 上的視覺模型**讀取(例如由 [Forgeyard](https://github.com/xianhong1208/forgeyard) 提供的 Qwen3-VL),音訊交給 Forgeyard 轉錄。`/v1` 的 OCR 與語音端點一併移除;**遷移指南**說明怎麼讓 Folio 改用 Forgeyard。**部署指南。** **`folio backup` / `folio restore`** 與排程備份。**單一資料夾的匯出與匯入。** **`/metrics`** 提供 Prometheus 指標。 |
| **1.6:找到那份文件** | 文件的**中繼資料與標籤**,在 REST、主控台與 MCP 都能用來篩選。**以文件為單位的結果**,不只是段落。**跨你所有資料夾搜尋。** **Library** 頁面。回傳文件的 MCP **`find`** 模式。 |
| **1.7:更好的答案** | **查詢改寫與多重查詢**,HyDE 與翻譯為選用。每一項都只有在黃金測試集變好時才保留。**圖片**由視覺模型寫出說明,讓它們也搜得到。**引用會開啟 PDF** 並標示出該段文字。答案的**讚 / 倒讚**。 |
| **1.8:資料夾保持最新** | 資料夾的**來源**:網址或 sitemap、Git repository、伺服器上的目錄、S3 或 WebDAV。**依排程重新同步**,只處理有變動的部分,來源刪掉的檔案也會從資料夾移除。 |
| **1.9:對話** | **對話紀錄**、保留上下文的**追問**,以及對話**匯出**。 |
| **1.10:Notion 與 Google Drive** | 建立在 1.8 來源機制上的 **Notion** 與 **Google Drive** 連接器。 |

目前不規劃:把資料夾分享給其他人(見「之後」)。

## 之後

- **Confluence** 連接器。
- Kubernetes 的 **Helm chart**。
- **Python SDK。**
- 可嵌入其他網頁的**聊天元件**。
- **資料夾分享。** 有兩種做法在考慮:(A)把資料夾分享給指定帳號;(B)團隊資料夾,需要 MCP Gate 先
  有群組。在那之前,資料夾維持每人各自擁有。
- **GraphRAG**,仍然暫緩(見下方)。

### GraphRAG 與「LLM Wiki」:暫緩,先過決策門檻

Folio 已經是進階的混合式 RAG(階層切塊與 auto-merge、向量 + BM25/CKIP、cross-encoder 重排、
contextual retrieval),涵蓋了 RAG 的兩個典型弱點:chunk 破碎與精準度不足。以下兩個想法是補強,不是
取代。

- **GraphRAG**(RAG 加上知識圖譜)擅長多跳的*關係型*問題、*全域*摘要(「這些文件整體怎麼說…」)以及
  節點層級的可解釋性。建置成本高:要由 LLM 從整個語料抽出實體與關係。若採用,做法是加一層 Neo4j 或
  Microsoft GraphRAG,再加一個**查詢路由**,把關係型與「總結全部」的問題送到圖譜,查找型問題留在混合
  檢索。開工前要先回答:用哪個模型抽取、資料夾變動時圖譜怎麼同步、圖譜路徑怎麼引用回原始 chunk。
  參考:資策會 MIC〈Graph RAG〉(經濟部產業技術司,it_id=563)。
- **「LLM Wiki」**(Andrej Karpathy;是一種方法,不是產品):把來源一次「編譯」成互相連結的 Markdown
  概念頁(一個概念一頁、YAML front matter、`[[slug]]` 連結),之後查詢這些頁面而不是原始來源。它自己
  說的限制是超過約 500 頁就需要 ANN 索引,而這正是 Folio 提供的。兩種整合方式:*提供*(在別處編譯好,
  把頁面上傳到資料夾,透過混合檢索與 MCP 提供)或*合成*(在 Folio 內加一個步驟,用 contextual
  retrieval 的 LLM 從資料夾產生概念頁)。參考:blog.104.com.tw/andrej-karpathy-llm-wiki。
- **決策門檻:** 先量測實際使用情況。只有當關係型或整體摘要型的問題占了相當比例,才值得投入其中任何一個。

### 較早的想法,未排入

留在這裡以免遺失;都沒有排入計畫。

- 依 1.7 的讚 / 倒讚,為每個資料夾調整融合權重,而不是全域一個設定。
- 主控台的唯讀與管理角色,超出目前的兩個層級(任何 token / `rag:admin`)。
- 每次請求都以 RFC 7662 introspection 做即時撤銷(目前:revocation feed,一個輪詢間隔內生效)。
- 每次查詢在整條管線(向量 → BM25 → 融合 → 重排)上的追蹤,超出 Search Playground 現有的單次查詢軌跡。
- 在 chunk 中繼資料保留表格原本的 Markdown,讓表格顯示不失真。

## 尚待驗證

- **1.5 移除內建 OCR 與語音轉文字之前:** 用視覺模型流程處理黃金測試集的 PDF 與圖片,用 Forgeyard
  轉錄 `evals/asr` 的錄音,並和目前 Docling OCR 與 FireRedASR 的結果比較。
- **伺服器關閉時會 abort**(見下方):找出原因需要啟動程式的 stderr 與原生 backtrace(gdb)。

## 已知限制

- **需要 PostgreSQL 與 pgvector。** 沒有 SQLite 的退路:混合檢索與每個資料夾的資料表都依賴它。
- 設定的 DSN 走**同步 SQLAlchemy**(psycopg2);不要用 `+asyncpg`。
- **GPU 要自己選,** 用 `uv sync --extra local --group rocm-r714 | cuda | cpu`。
- **撤銷不是即時的,而且預設關閉。** Token 是離線驗證。沒開 `auth.revocation` 時,被撤銷的 token 會用到
  到期為止;開了之後,會用到下一次輪詢 MCP Gate 為止(預設 30 秒)。在 `fail_mode: open` 下,MCP Gate
  連不上期間做的撤銷,要等它恢復連線後才生效。
- **`/v1` 媒體端點沒有驗證,** 這是刻意的決定;請只開放在可信任的網路。它們會在 1.5 移除。
- **主控台只有英文,** 這是刻意的決定。(英文 + 繁體中文的完整版本曾經做出來後又移除;在 git 歷史的
  `5103cbd` / `53aa71a`,由 `be480d6` 還原。)
- **伺服器關閉時會 abort。** 每次重啟,舊的行程都會在直譯器結束時因某個原生函式庫觸發 SIGABRT 而結束,
  過去每次都留下約 16 GB 的 core 檔。Folio 現在會關掉自己的 core dump(`FOLIO_CORE_DUMPS=1` 可為除錯
  重新開啟)。
