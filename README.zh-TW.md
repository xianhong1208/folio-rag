# Folio

**一套開源的 Model Context Protocol 檢索增強生成（RAG）伺服器 —— 階層式切塊、auto-merging 檢索、向量 + 中文 BM25 混合搜尋,以「每個資料夾一個工具」的形式提供給 MCP 用戶端。**

Folio 把一個文件資料夾變成 Claude Code、Claude Desktop、Cursor 等任何 MCP 用戶端都能呼叫的檢索工具。每個資料夾成為一個 `Folio_<folder>` 工具,含三種模式 —— **search**、**list**、**read** —— 底層是一條會**階層切塊、以密集+稀疏訊號檢索、把相鄰命中合併成完整段落、再用 cross-encoder 重排**的管線。它是一台 **resource server**:只在本地離線驗證 OAuth token、從不簽發,搭配 [MCP Gate](https://github.com/xianhong1208/MCP_Gate) 當授權伺服器。

Folio 在 v1.1.0 以前名為 *Agentic RAG*。repository、預設資料庫名稱（`agentic_rag`）與儲存目錄結構沿用舊名,不需要任何遷移。

License: MIT&nbsp;·&nbsp;Python 3.12&nbsp;·&nbsp;PostgreSQL + pgvector&nbsp;·&nbsp;相容 FastMCP&nbsp;·&nbsp;OAuth 2.1（resource server）

[English](README.md) · [快速開始](#快速開始) · [運作原理](#運作原理) · [設定](#設定) · [主控台](#使用主控台) · [架構](#架構)

```
┌────────────────┐     Folio_<folder> 工具呼叫（Bearer JWT）    ┌───────────────────────────────┐
│    MCP 用戶端   │ ───────────────────────────────────────────▶ │  Folio（resource server）      │
│  Claude / IDE  │ ◀─────────────────────────────────────────── │  /mcp · 依 token 過濾的資料夾   │
└───────┬────────┘        auto-merge 後的段落 + 分數            └───────────────┬───────────────┘
        │ OAuth 流程                                                             │ 離線驗證 JWT
        ▼                                                                        ▼
  ┌─────────────┐                              檢索管線（每次查詢）
  │  MCP Gate │  RS256 JWKS                  ingest ─▶ chunk ─▶ embed ─▶ pgvector ─▶ 混合檢索
  │ OAuth 2.1 AS│  （離線驗證）                (Docling) (leaf+parent) (e5·1024d)   (向量 + BM25/CKIP)
  └─────────────┘                                   └▶ auto-merge + 擴展 ─▶ rerank ─▶ gpt-oss-120b
```

## 功能特點

- **依章節對齊的階層切塊 + auto-merging 檢索。** 索引時把文件切成小的 **leaf**（精準匹配）與大的 **parent**（完整脈絡）,兩者都不會跨越章節。每個 leaf 帶有**麵包屑**（檔名 › 標題路徑,xlsx 的工作表名稱也算在內）,結構相同的兩份文件(兩本產品手冊、兩張價目表)也能分得開。當同一段落的多個 leaf 被命中時,回傳整個 parent 而非碎片;單一 leaf 命中則向鄰居擴展,答案不會失去語意連貫性。
- **混合搜尋,為中文調校。** 密集向量檢索（pgvector）與稀疏 **BM25** 以 Reciprocal Rank Fusion 融合,讓關鍵字訊號真的影響排名。中文在進 Postgres 全文索引前先斷詞 —— 預設用 **CKIP**（中研院）,也可選 jieba 或零相依的 bigram —— 無空格的中文會被真正 tokenize,而不是塌成單一詞。
- **Cross-encoder 重排。** 以 `bge-reranker-v2-m3` / `mxbai-rerank-base-v2` 對融合後的候選重新評分,提升 Top-K 精度。
- **Contextual Retrieval。** 索引時由 LLM 依文件大綱與 chunk 所在章節,以文件本身的語言為每個 chunk 寫一句脈絡說明（Anthropic 建議的做法）。這句說明與麵包屑只用來*找到* chunk(embedding、BM25、rerank);結果一律顯示文件原文。
- **增量重建索引。** 重新索引一個檔案時,沒變的 chunk 沿用原本的脈絡說明與 embedding,只改一個章節大約只花一次 LLM 呼叫與一次 embedding。檔案的 chunk 在單一交易內替換 —— 過程中不會從搜尋結果消失。
- **每資料夾一個 MCP 工具,依 token 隔離。** 每個資料夾註冊成 `Folio_<folder>` 工具;呼叫者看到的工具清單會依其 token 過濾 —— 你只看得到自己擁有的資料夾。歸屬以 token 穩定的 `jti` 為 key。工具名稱一律符合所有 MCP 用戶端的限制(`[A-Za-z0-9_-]`、最長 64 字元):含空白、中日韓文字或其他符號的資料夾名稱會變成 `Folio_<ASCII 部分>_<8 碼雜湊>`,每次都是同一個名稱。
- **OAuth 2.1 resource server。** 一個 `RemoteAuthProvider` + `JWTVerifier` 保護 `/mcp`;token 用 MCP Gate 的 JWKS 在本地離線驗證（RS256),並綁定本伺服器的 audience,別台的 token 在這裡無效。
- **文件 + 音訊匯入。** [Docling](https://github.com/DS4SD/docling) 解析 PDF / DOCX / PPTX / XLSX / 圖片（含 OCR）;**FireRedASR** 在本地做繁體中文語音轉錄（或任何 OpenAI 相容的 `/v1/transcriptions`）。
- **內建評測 harness。** 從你自己的 chunk 自動生成問題,對正式檢索管線(與 `/api/rag/query` 同一路徑)以及 `vector` / `bm25` / `rrf` / `rerank`(全部走正式檢索路徑)評 **Recall@k · MRR · nDCG@k**,讓你針對真實語料調校檢索。
- **Retrieval Terminal 主控台。** `/admin` 的 React 單頁主控台:管線總覽、資料夾與檔案、會顯示**逐訊號檢索軌跡**的 Search Playground、即時索引任務串流、查詢分析、評測 harness、即時設定、健康狀態與稽核時間軸。
- **跑在你的 GPU 上。** PyTorch 以 PEP 735 dependency group 依平台選擇（`rocm-r714` / `rocm-r713` / `cuda` / `cpu`）—— AMD ROCm、NVIDIA CUDA 或純 CPU。

## 安裝

**前置需求**

- Python 3.12 與 [uv](https://docs.astral.sh/uv/)。
- 安裝了 **[pgvector](https://github.com/pgvector/pgvector)** 擴充的 **PostgreSQL**。
- **embedding**、**LLM**、**reranker** 三個模型的 OpenAI 相容端點（例如 [vLLM](https://github.com/vllm-project/vllm)、Ollama 或雲端供應商）。
- Node.js 22.12+ —— 只有要開發主控台 UI 時才需要(Vitest 5 需要)。

```bash
git clone https://github.com/xianhong1208/folio-rag.git
cd folio-rag

# 1. 安裝 —— 本地模型需要 `local` extra;torch 依平台走 PEP 735 dependency group,務必一起帶 --group。
#    （全雲端部署:直接 `uv sync`,見下方「雲端 profile」。）
uv sync --extra local --group rocm-r714    # AMD ROCm 7.1.4（multi-arch wheel） | rocm-r713 | cuda | cpu

# 2. 設定資料庫與模型端點
cp .env.example .env         # 設定 DATABASE_URL（Postgres + pgvector）
#   編輯 config/config.yaml  → rag.embedding / rag.llm / rag.rerank 端點、
#                               auth.issuer / auth.audience、server.port

# 3. 啟動（自動跑 DB bootstrap + migration）
uv run python main.py
```

伺服器預設監聽 `http://0.0.0.0:5039`;MCP 端點是 `/mcp`,主控台在 `/admin`。

## 快速開始

Folio 透過 [MCP Gate](https://github.com/xianhong1208/MCP_Gate) 驗證。**順序很重要:先在 MCP Gate 註冊這台伺服器** —— 它只會為它認得的 audience 簽 token。

**1 — 在 MCP Gate 註冊伺服器。** 到 MCP Gate 主控台 *Services → Register Service*,填入本伺服器的 host / port / path（`/mcp`）。它的 audience 預設為 MCP URL,例如 `http://127.0.0.1:5039/mcp`。把同一個值設進 `config/config.yaml` 的 `auth.audience`。

**2 — 讓 Folio 指向 MCP Gate**（`config/config.yaml`）:

```yaml
auth:
  enabled: true
  issuer:   "http://localhost:4568"          # MCP Gate 位址（RS256 JWKS issuer）
  audience: "http://127.0.0.1:5039/mcp"      # 必須等於主控台註冊的 audience
```

**3 — 建立資料夾並索引文件** —— 用 `/admin` 主控台（*Folders → New Folder → Upload Files*）或走 REST:

```bash
# 建立一個 token-scoped 資料夾,再上傳 + 索引檔案（見 API 參考）
curl -X POST http://127.0.0.1:5039/api/folders/ \
  -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" \
  -d '{"name":"product-docs"}'
```

**4 — 接上用戶端。** 有 MCP Gate 擋在前面,支援 OAuth 的用戶端不需預先設定:

```bash
claude mcp add --transport http folio http://127.0.0.1:5039/mcp
```

首次使用時,用戶端會向 MCP Gate 註冊並開啟同意畫面;你按 Allow 後,`Folio_product-docs` 工具就會出現。腳本 / CI 則在 MCP Gate 簽一個 **personal access token**,以 bearer header 帶入。

**5 — 開始問。** 在用戶端呼叫資料夾工具:

```
Folio_product-docs  mode="search"  query="要怎麼輪替簽章金鑰?"
```

它會回傳命中的段落（必要時 auto-merge),每筆附相關性分數與來源檔案 / 標題。

## 運作原理

資料夾索引一次,之後每次查詢都走這條管線:

| 階段 | 內容 |
|------|------|
| **Ingest** | Docling 解析 PDF / DOCX / PPTX / XLSX / 圖片（OCR）;音訊由 FireRedASR 轉錄。 |
| **Chunk** | 依章節切成小 **leaf** + 大 **parent**;每個 leaf 帶麵包屑（檔名 › 標題）與 LLM 寫的脈絡說明。表格放得下時保持完整。 |
| **Embed** | 每個 leaf 的檢索文字（脈絡說明 + 麵包屑 + 原文）做 embedding（預設 `intfloat/multilingual-e5-large`,1024 維）,與 BM25 的 `text_search_tsv`（中文以 CKIP 斷詞）在同一筆交易內寫入 **pgvector**,每個檔案一筆交易。 |
| **Retrieve** | **混合**:密集向量 + 稀疏 BM25,以 Reciprocal Rank Fusion 融合。 |
| **Merge** | 同段落的 leaf **auto-merge** 成 parent;單一命中**擴展**到鄰居。 |
| **Rerank** | cross-encoder（`bge-reranker-v2-m3` / `mxbai-rerank-base-v2`）重排,決定最終 Top-K 精度。 |
| **Answer** | 走 `answer` 路徑時,`gpt-oss-120b` 依檢索到的脈絡合成回答,並由 corrective-RAG 信心檢查把關。 |

### 檢索細節

**單一檢索路徑。** REST `/api/rag/query`、MCP search 工具、主控台 Search Playground（含檢索軌跡）與評測 harness 都走同一份程式碼（`src/domain/rag/hybrid_search.py`）:密集向量 + BM25,以 `rrf`（預設）或 `concat`（先放向量命中,再接只有 BM25 命中的段落,交給 reranker 排序）融合,接著 rerank,再用同一條規則做 auto-merge。REST 每次請求可帶 `auto_merge`（`null` = 依 `rag.retrieval.auto_merging.enabled`）;`hybrid_alpha` 仍接受但不再生效。每次請求的參數隨呼叫傳入,不寫進各資料夾共用的 engine,所以不同 `top_k` 的並行請求不會互相干擾。

**關鍵字檢索（BM25）。** 查詢用與索引相同的方式切詞（中文先斷詞）,去掉疑問詞與虛詞（`什麼`、`嗎`、`what`、`the`…）,再以 Postgres 全文檢索**只搜 leaf chunk** —— 先要求所有詞都出現（AND）,不足再用任一詞（OR）補滿。檢索深度為 `max(top_k 候選數, rag.retrieval.default_sparse_top_k)`,並與向量檢索同時執行。`simple` 全文設定不做英文詞幹還原（`policy` ≠ `policies`）;純英文語料可設 `text_search_config: english`。

**中文斷詞。** 預設 `text_search_config: simple` 時,中文在進 Postgres 前先在 Python 端斷詞,斷詞器由 `rag.retrieval.segmenter` 決定 —— 索引、*Rebuild FTS* 與查詢都走同一個函式:

| `segmenter` | 說明 |
|---|---|
| `ckip` *(預設)* | 中研院 CKIP BERT 模型 —— 繁體中文最準;需要 torch(`uv sync --extra local`),第一次使用時下載模型。`profile: cloud` 不允許使用。 |
| `jieba` | 詞典式、純 Python、不需 GPU。選用套件:`uv sync --extra jieba`。 |
| `bigram` | 中日韓字元兩兩重疊切分 + 英數詞。不需模型、零相依;索引較大。`profile: cloud` 的預設值。 |

`segmenter` 未設定時依 profile 決定:`ckip`,`profile: cloud` 則為 `bigram`。

每個資料夾的資料表會記錄它的關鍵字索引是哪個斷詞器建的。改了 `segmenter`（可在 Settings 即時改）之後,主控台會把用其他斷詞器建索引的資料夾標成 **FTS rebuild needed**;*Rebuild FTS* 會用目前設定的斷詞器重新斷詞 —— 不需重新 embedding。

**重排。** reranker 對候選池(見下)評分,保留分數不低於 `rag.rerank.score_threshold` 的前 `top_k` 個。`mxbai-rerank-*-v2` 是以對話格式 prompt 訓練的 Qwen 模型;`rag.rerank.auto_template: true`（預設）時,若設定的模板是預設值或舊的無對話格式版本,會自動套上官方格式（若 rerank 服務已自行套用 chat template,請關掉)。套上後分數是校準過的機率,對這個模型而言 **`score_threshold: 0.5`** 才有意義;沒套的話無關段落也有 0.8 以上,任何門檻都濾不掉東西。reranker 失敗時,查詢會依檢索順序回傳前 `top_k` 個候選（而不是整個候選池）,且這個降級結果不會被快取。

**候選池。** 候選池夠深對跨語言問題最重要:用中文問英文段落時,那段文字在向量與 BM25 檢索裡常常排得很後面,只有 cross-encoder 能把它拉上來。這類段落只有向量檢索找得到(大約在向量排名 15–30),而 RRF 會把兩路都找到的段落全排在它們前面,所以若從*融合後*的清單截取,要截到 60 才碰得到。`rag.retrieval.rerank_pool: per_list`(預設)改成讓 reranker 評分向量檢索前 `rerank_dense_candidates` 名(預設 30,至少 3 × `top_k`)加上 BM25 前 `rerank_sparse_candidates` 名(預設 10)—— 約 35 對。在 golden set 上,Recall@5 0.976 與中→英 Recall@5 0.90 都和 60 筆融合候選池完全相同(候選池只有 3 × `top_k` 時是 0.957 / 0.80);而 cross-encoder 每一對大約花 6 ms、與文字長短幾乎無關,所以重排步驟從約 360 ms 降到約 225 ms。`rerank_pool: fused` 可恢復先前的行為(`max(3 × top_k, rerank_candidates)` 個融合候選)。這四個設定都能在 Settings 即時調整;候選池越深,召回的餘裕越大、延遲也越高。沒有 reranker 時候選池就是 `top_k`。

**延遲。** 在 golden set 上,一次預設的 MCP 搜尋在用戶端量到約 330 ms p50 / 約 400 ms p95(原本約 500 / 600 ms;各階段的耗時表見 `evals/baselines/rag-2026-11-04-search-latency.md`)。大部分時間花在重排呼叫(約 35 對,約 225 ms)。在那之前,查詢 embedding(約 55 ms,接著是向量查詢)與斷詞、BM25(約 40 + 20 ms)同時進行,這一段約 80 ms。所有候選的 auto-merge 與鄰近視窗查詢會在 reranker 運算時先讀好,重排完不再多花時間;查詢紀錄則在回應送出後才寫入。CKIP 在一條專用執行緒上斷詞:在 CPU 上,每個呼叫 torch 的執行緒都會有自己的一組工作執行緒,好幾組搶同一批核心時斷詞會慢 3 倍(主機忙碌時甚至要好幾秒)。

**回應大小。** MCP 搜尋的回應會控制在 `rag.retrieval.mcp_response_token_budget`(估算 token 數,預設 6000)以內:排名最低的結果會被省略,回應中以 `omitted_results` 與 `response_tokens_estimated` 註明,`_hint` 也會提示 agent 改問更精確的問題。設為 `0` 即不限制。

**說出「找不到相關內容」。** 每次 MCP 搜尋都會回報資料夾的匹配程度:`top_relevance`（reranker 在所有候選上的最高分,在 `score_threshold` 之前)、`min_relevance`(設定的門檻)與 `low_relevance` —— 沒有結果、或 `top_relevance` 低於 `rag.retrieval.min_relevance` 時為 `true`,此時 `_hint` 會告訴 agent 這個資料夾多半無法回答這個問題。每筆經過重排的結果都帶 `relevance`,低於門檻的結果標上 `low_relevance: true`。結果**只標記、不刪除**(刪除是 `rag.rerank.score_threshold` 的工作):在 golden set 上,這個門檻能擋下所有跟語料無關的問題,但「差一點」的問題(問對了法條、問的細節卻不在文中)分數跟可回答的問題一樣高,所以最後的判斷留給 agent。預設的 `0.9` 是針對 `mxbai-rerank-v2` 校準的(掃描結果見 `evals/baselines/rag-2026-11-03-min-relevance.md`);設 `0` 關閉,沒有 reranker 時只有空結果會被標記。

**知識缺口。** 每次 REST、主控台與 MCP 搜尋都會記錄（token 只存遮罩後的前綴）結果數與 reranker 的最高分。*Query Analytics* 把某資料夾中**最近一次**搜尋沒有結果、或最高分低於 `rag.rerank.gap_threshold`（預設 `0.5`）的問題列為知識缺口 —— 補上能回答它的文件、再問一次後,缺口就會消失。

**回答（主控台）。** 勾選 *generate answer* 時,先用一次 LLM 呼叫評估各 chunk 是否相關,丟掉不相關的（都不相關就不回答）,再用剩下的以**問題所用的語言**生成回答。每個 chunk 保留它在結果卡片中的編號,`[3]` 一定是第 3 張卡（主控台中引用可點擊）;引用了沒提供給模型的 chunk 會被移除,`【n†…】` 會轉成 `[n]`,Markdown 粗體標記會去掉 —— 串流路徑也一樣,最後的 `done` frame 帶有整理過的完整回答。評估結果無法解析時,回答會標成 **unverified**,而不是 `high`。

**驗證。** Folio 是 *resource server*。FastMCP 會發佈 `/.well-known/oauth-protected-resource/mcp`,把用戶端導向 MCP Gate;伺服器抓一次 MCP Gate 的 JWKS,之後每個 bearer token 都離線驗證（RS256 —— 簽章、issuer、audience、效期、scope）。MCP Gate 從不在請求路徑上。

## 設定

設定在 `config/config.yaml`,機密從環境變數（`.env`）讀。你最可能會動到的:

| 變數 / 設定 | 預設 | 用途 |
|------------|------|------|
| `DATABASE_URL` | `postgresql://…/agentic_rag` | PostgreSQL DSN。需 **pgvector** 擴充。用**同步** `postgresql://`（psycopg2),不要 `+asyncpg`。 |
| `server.port` | `5039` | HTTP + MCP 綁定 port。MCP 端點是 `/mcp`。 |
| `FOLIO_STORAGE_ROOT` | 專案根目錄 | 放 `storage/`（上傳檔案與主控台 session 金鑰）的目錄。 |
| `FOLIO_INDEXING_CONCURRENCY`、`FOLIO_PER_FILE_TIMEOUT_SECONDS`、`FOLIO_JOB_TIMEOUT_SECONDS` | 取自 `rag.indexing` | 索引上限的執行期覆寫。改名前的 `RAG_*` 舊名仍可用,但會記一筆 deprecation 警告。 |
| `auth.enabled` / `auth.issuer` / `auth.audience` | `true` / MCP Gate 位址 / 本伺服器 `/mcp` URL | 對 MCP Gate 做 OAuth 2.1 驗證。 |
| `auth.console.session_hours` | `8` | 主控台 session 最長時數(token 到期時也會一併結束)。 |
| `auth.console.session_secret` | *(自動產生)* | 簽 session cookie 的金鑰,可用 `${ENV_VAR}`。未設定時會產生一次並存到 `storage/.console_session_secret`(已 gitignore),重啟後 session 仍有效。 |
| `auth.console.secure_cookie` | `auto` | Cookie 的 `Secure` 旗標:`auto` = 只在 https 時加上;放在 TLS 終結的 proxy 後面時設 `true`。 |
| `auth.console.admin_scope` | `rag:admin` | 解鎖主控台全域設定的 scope。 |
| `auth.revocation.*` | 停用 | MCP Gate revocation feed,讓被撤銷的 token 在到期前就失效——見 [Token 撤銷](#token-撤銷)。 |
| `rag.embedding.{model,dimension,base_url}` | `intfloat/multilingual-e5-large`、`1024`、`:7075/v1` | embedding 模型;`dimension` **必須**等於模型實際輸出維度。 |
| `rag.llm.{model,base_url}` | `openai/gpt-oss-120b`、`:5052/v1` | Contextual Retrieval 與 `answer` 路徑用的 LLM。 |
| `rag.rerank.{model,base_url}` | `mixedbread-ai/mxbai-rerank-base-v2`、`:8787` | cross-encoder 重排器。 |
| `rag.rerank.score_threshold` / `auto_template` | `0.0` / `true` | 濾掉分數低於門檻的結果（mxbai-rerank-v2 適合 0.5）;自動套用 mxbai-v2 的對話格式 prompt。 |
| `rag.rerank.gap_threshold` | `0.5` | 知識缺口分析:最高 rerank 分數低於此值即視為「沒有好的匹配」。 |
| `rag.retrieval.rerank_pool` | `per_list` | cross-encoder 評分的對象:`per_list` = 向量檢索前 `rerank_dense_candidates` 名 + BM25 前 `rerank_sparse_candidates` 名;`fused` = 融合清單的前 `max(3 × top_k, rerank_candidates)` 名(先前的行為)。即時生效。 |
| `rag.retrieval.rerank_dense_candidates` / `rerank_sparse_candidates` | `30` / `10` | `per_list` 候選池的深度(向量端不低於 3 × `top_k`)。越大召回餘裕越多,每多一對約多 6 ms;可即時調整(0–200)。 |
| `rag.retrieval.rerank_candidates` | `60` | 僅用於 `fused` 候選池:`max(3 × top_k, 此值)` 個融合候選送進 cross-encoder;可即時調整(1–200)。 |
| `rag.retrieval.mcp_response_token_budget` | `6000` | MCP 搜尋回應的估算 token 上限;超過時省略排名最低的結果並在回應中註明。`0` = 不限制。 |
| `rag.retrieval.min_relevance` | 隨附設定檔為 `0.9`(未設定時 `0`) | MCP 搜尋:最高 rerank 分數低於此值 → `low_relevance: true`(只標記、不刪除)。依模型而定;`0` = 關閉。 |
| `rag.asr.provider` | `fireredasr` | 音訊轉錄:`fireredasr`（本地繁中）· `docling-whisper` · `openai-compatible`。`rag.asr.enabled: false` 時音檔會以明確訊息標為失敗(不做任何轉錄)。 |
| `rag.retrieval.segmenter` | `ckip`(`profile: cloud` 為 `bigram`) | 關鍵字檢索的中文斷詞器:`ckip` · `jieba` · `bigram`（見「中文斷詞」）。 |
| `rag.docling.ocr_enabled` | `true` | 掃描 PDF 與圖片的 OCR。設 `false` 會真的關掉(重啟後生效);沒有文字層的掃描檔會以明確訊息失敗。 |
| `rag.contextual_retrieval.max_doc_chars` | `60000` | 每個 chunk 交給 LLM 的文件文字上限。LLM 讀的是大綱加上 chunk 所在章節,以 chunk 為中心最多 12,000 字元。 |
| `rag.retrieval.*` | — | Top-K、相似度門檻、BM25 候選數、`hybrid_fusion`（`rrf`/`concat`）、auto-merging。可從主控台即時調,或用 **High Precision / Balanced / High Recall** 預設。 |
| `rag.profile` | `local` | `local` · `hybrid` · `cloud` —— 見[雲端 profile](#雲端-profile)。 |
| `rag.<role>.{timeout,max_retries}` | 各角色不同 | 每個模型角色的請求逾時與重試（429 / 5xx / 逾時,含退避）。 |

GPU 選擇是 `uv sync --extra local --group {rocm-r714|rocm-r713|cuda|cpu}` 的取捨 —— 見 `pyproject.toml` 開頭,每個 group 對應到各自的 PyTorch index。

### Token 撤銷

Folio 以離線方式驗證 MCP Gate 的 token,因此預設情況下,在 MCP Gate 撤銷的 token 在這裡會一直有效到過期為止——而 PAT 最長可活一年。啟用 **revocation feed** 後,Folio 會在背景輪詢 MCP Gate 的 `POST /oauth/revoked`(每 `poll_interval_sec` 秒,帶 jitter 與錯誤退避),在記憶體保存「已撤銷但尚未過期」的 `jti`,並在所有路徑拒絕它們:`/mcp`(401 `invalid_token`)、REST API(401)、主控台登入,以及既有主控台 session 的每一個請求(session 以 401 `token_revoked` 結束)。撤銷約在一個輪詢間隔內生效。

| 設定 | 預設 | 用途 |
|------|------|------|
| `auth.revocation.enabled` | `false`(`FOLIO_REVOCATION_ENABLED`) | 開啟 feed。關閉 = 舊行為,不輪詢。 |
| `auth.revocation.client_id` | `${FOLIO_REVOCATION_CLIENT_ID}` | 在 MCP Gate 註冊的 resource-server client(見下方)。 |
| `auth.revocation.client_secret` | `${FOLIO_REVOCATION_CLIENT_SECRET}` | 它的 secret。**必須**是 `${ENV_VAR}` 參照——寫成字面值會拒絕啟動。 |
| `auth.revocation.poll_interval_sec` | `30` | 輪詢頻率(5–3600 秒)。 |
| `auth.revocation.fail_mode` | `open` | MCP Gate 連不上時:`open` 繼續接受 token 並大聲記錄;`closed` 在連續 `stale_after_intervals` × `poll_interval_sec` 沒有成功輪詢後拒絕所有 token(`/mcp` 與 REST 回 401,主控台回 503),啟動後第一次輪詢成功前也一樣。 |
| `auth.revocation.stale_after_intervals` | `4` | 清單何時算過期(`/health/detailed` 顯示 `degraded`;`closed` 開始拒絕)。 |

第一次輪詢抓完整清單;之後只問上次以來的撤銷(重疊 30 秒),每小時再完整抓一次。token 本來就會過期之後,項目即被移除。啟動時會記錄 feed 是否啟用,`GET /health/detailed` 的 `checks.revocation` 會顯示 `last_success`、`revoked_count`、`consecutive_failures`、`last_error`、`stale`。

**設定步驟**(每個部署做一次):

1. **在 MCP Gate 註冊 confidential client。** 主控台 → *OAuth Clients* → *Register trusted client*。名稱例如 `Folio revocation feed`;*Redirect URIs* 留空;*Grant types* 只選 `client_credentials`;*Token endpoint authentication*:**Client secret (POST body)**(HTTP Basic 也可以,MCP Gate 兩者都接受);*Resource server for*:**Folio**。註冊後複製 `client_id`(`mcpc_…`)與 client secret——secret 只顯示一次。
2. **綁定到 Folio**——若步驟 1 已選 *Resource server for* 就已完成。否則在該 client 那一列點伺服器圖示(*Resource server…*),選 Folio 服務後 *Save*;或以主控台管理員呼叫 `PUT /api/oauth/clients/<client_id>/resource-server`,body `{"service_id": "<Folio service id>"}`。該列會顯示 *Resource server for Folio*。feed 只列出 audience 等於該服務 audience 的 token,所以它必須等於 Folio 的 `auth.audience`。
3. **把憑證交給 Folio**(環境變數或 `.env`),然後重啟:
   ```bash
   FOLIO_REVOCATION_ENABLED=true
   FOLIO_REVOCATION_CLIENT_ID=mcpc_...
   FOLIO_REVOCATION_CLIENT_SECRET=...
   ```
   (若你的 `config/config.yaml` 早於此功能,先從範本複製 `auth.revocation` 區塊。)
4. **確認。** 啟動 log 會出現 `MCP Gate revocation feed: ENABLED …`;`curl -s localhost:5039/health/detailed | jq .checks.revocation` 會看到最近的 `last_success`。在 MCP Gate 撤銷一個測試 PAT,`poll_interval_sec` 內用它發請求會得到 401。`last_error` 出現 `HTTP 401: invalid_client` 表示憑證錯誤;`HTTP 403: unauthorized_client` 表示 client 尚未綁定伺服器。

## 雲端 profile

所有模型都走 API、本機完全不載入任何模型 —— 不需要 GPU、不 import torch、不下載模型。設定 `rag.profile: cloud`,每個角色各給一個 OpenAI 相容端點:

```bash
uv sync                      # 核心安裝:沒有 torch / transformers（本地模型堆疊在 `--extra local`）
export OPENAI_API_KEY=sk-...  # 金鑰從環境變數來,絕不寫進 config.yaml
uv run python main.py
```

```yaml
rag:
  profile: cloud            # local (default) | hybrid | cloud

  embedding:                # changing it means re-indexing existing folders
    provider: openai        # openai | openai-compatible | azure | vllm | ollama
    model: text-embedding-3-small
    dimension: 1536
    base_url: https://api.openai.com/v1      # any OpenAI-compatible gateway works
    api_key: "${OPENAI_API_KEY}"
    timeout: 60
    max_retries: 3

  llm:                      # default for every LLM role
    provider: openai
    model: gpt-4o-mini
    api_key: "${OPENAI_API_KEY}"
  contextual_llm:           # optional per-role overrides; unset fields inherit from llm
    model: gpt-4.1-nano
  # answer_llm: {...}       # Search Playground answers
  # eval_llm: {...}         # eval question generation / judging

  contextual_retrieval:
    enabled: true
    max_calls_per_file: 2000     # cost guards (0 = unlimited)
    max_tokens_per_day: 5000000

  rerank:
    enabled: true
    api: cohere             # score (vLLM /v1/score) | cohere | jina | generic (OpenAI-compatible /rerank)
    model: rerank-v3.5
    base_url: https://api.cohere.com
    endpoint_path: /v2/rerank
    api_key: "${COHERE_API_KEY}"

  parsing:                  # cloud is the default under profile: cloud
    vision:
      provider: openai
      model: gpt-4o-mini
      api_key: "${OPENAI_API_KEY}"
    max_concurrency: 4      # page requests in flight, process-wide
    max_pages_per_file: 500
    max_pages_per_day: 5000
    on_page_failure: mark   # mark = keep the file, mark the page | fail

  asr:
    provider: openai-compatible
    base_url: https://api.openai.com/v1
    model: whisper-1
    api_key: "${OPENAI_API_KEY}"
    segment_seconds: 600    # long audio is uploaded in silence-aligned pieces

  retrieval:
    hybrid_search: true
    # segmenter: bigram     # the cloud default (no model); jieba also works, ckip is rejected
```

profile 做了什麼:

- **驗證。** `cloud` 會拒絕在本機程序內載入的 provider（`huggingface` embedding、`fireredasr` /
  `docling-whisper` ASR、本地 PDF/圖片解析）,而且**拒絕不是 `${ENV}` 參照的 API 金鑰** —— 雲端金鑰會計費,
  而 config.yaml 是最常被複製流傳的檔案。明確設定 `rag.retrieval.segmenter: ckip` 也會被拒絕。啟動會失敗並列出有問題的設定路徑。`hybrid` 對寫死的金鑰只發警告;
  `local`（預設 —— 既有設定不受影響）不做任何新檢查。
- **不 import 任何本地模型。** HuggingFace embedding、docling 的版面/OCR 模型、Whisper、FireRedASR、CKIP
  只在被選用時才 import;啟動時略過 docling 預熱,並在 log 中列出每個角色跑在哪裡。
  `tests/test_cloud_profile_no_local_models.py` 在沒有本地堆疊的安裝上跑完整流程來證明這點。
- **雲端 PDF 與圖片解析。** 每一頁由視覺模型轉成 Markdown,所有頁面先接回**同一份**文件再切 chunk,
  標題、麵包屑與表格都能跨頁延續。每頁都會附上前一頁的結尾與目前的標題路徑;頁面以有序區塊並行
  （並行數有上限）;失敗的頁面會退避重試,仍失敗就在文字中標記,而不是讓整個檔案失敗。chunk 保留起始頁碼
  供引用。單檔與每日頁數上限控制成本。未驗證的 `/v1/ocr/general` 端點不會轉給付費模型。
- **雲端語音轉文字。** 音訊重新取樣後,在靜音處切段上傳（每個請求都不會超過上傳上限）,有重試;
  轉錄結果在有語音卻為空、或疑似生成迴圈時會重新請求。
- **每個角色各自的重試與逾時**（`timeout`、`max_retries`）:429 / 5xx / 逾時以指數退避加抖動重試,並遵守 `Retry-After`。
- **中文 BM25。** CKIP 是本地 torch 模型,cloud profile 不載入它,預設改用不需模型的 `bigram` 斷詞;
  也可選 `jieba`(`uv sync --extra jieba`)。明確設定 `rag.retrieval.segmenter: ckip` 會被拒絕 ——
  要用 CKIP 請改用 `profile: hybrid` 並 `uv sync --extra local`。

**主控台。** *Settings* 最上方是 **Model Roles** 面板:每個角色跑在哪、provider、模型與端點、
是否已設定金鑰以及來自哪個環境變數（金鑰值永不回傳）,還有 **Test** 按鈕,用已儲存的設定送出一次最小的真實請求
（一次 embedding —— 會檢查維度 ——、16 token 的對話、給視覺模型的 8×8 圖片、一組 rerank、給語音轉文字的半秒靜音）。
金鑰欄位可填 `${ENV_VAR}`;伺服器存的是參照,不是金鑰。修改需要 `rag:admin` scope。

每日上限存在記憶體中,於 UTC 午夜或重啟時歸零 —— 它是防失控的保護,不是計費。

**計算 chunk token 數**從不下載任何東西:有 `assets/hf_tokenizers/<org>--<model>` 就用它,否則用 `resources/tokenizers/` 內附的 XLM-RoBERTa tokenizer（multilingual-e5 與 bge-m3 用的就是它）,所以全新 clone 搭配 OpenAI 或 Azure embedding 也能直接用。

**解析失敗看得到。** docling 若有頁面解析失敗(例如某個階段 GPU 記憶體不足),會以較小的 batch 重試;仍缺的頁碼會記在檔案上(主控台顯示 *Partial*,REST 狀態回 `warning`)。

## 使用主控台

**Retrieval Terminal** 主控台是 `/admin` 的 React SPA。

### 主控台登入

登入方式是貼上一把**由 MCP Gate 發給本服務的 access token**,通常是 personal access token(PAT)。伺服器會用跟 REST / MCP 呼叫完全相同的方式離線驗證一次(簽章、issuer、audience、到期時間;token 必須帶 `jti`),通過後設定一個 HttpOnly 的 session cookie。Token 本身不會被儲存,也不會回傳。

- **Session 就是那把 token 的身分。** 你看到、能管理的,正好是這把 token 擁有的資料夾(資料夾以它的 `jti` 綁定)—— 也就是它透過 REST 與 MCP 能存取的那些。在主控台建立的資料夾同樣歸它所有。其他 token 的資料夾完全看不到;要管理它們,請用那把 token 登入。
- **兩個層級。** 任何有效的 token 都能登入並管理自己的資料夾。修改執行期 **Settings**、測試任意端點、讀取設定 **Audit Log** 與全伺服器的儲存用量,需要 **`rag:admin`** scope。沒有它時,Settings 與 System Health 為唯讀(secret 一律遮蔽),Overview / Index Jobs / Query Analytics 一律只涵蓋自己的資料夾——`rag:admin` 只開放全域設定,永遠看不到其他 token 的資料夾、工作或查詢。
- **Session** 持續 `auth.console.session_hours`(8 小時)或到 token 到期為止,取較早者。啟用 [revocation feed](#token-撤銷) 後,在 MCP Gate 撤銷 token 也會在一個輪詢間隔內結束它的 session;未啟用時,進行中的 session 會持續到到期或按 *Sign out*(驗證是離線的)。

**在 MCP Gate 發 token。** MCP Gate 不需要任何程式修改:

1. *Services* → 打開 Folio 服務 → *Scopes declared by this server* → 新增 **`rag:admin`**。**不要**把它設成 default scope(default 會在每把新 token 上預先勾選)。如果該服務有限制 *Allowed scopes*,也要允許 `rag:admin`。
2. *Issue Token* → *Target service* 選 Folio(audience = `auth.audience`)。給可以改設定的維運者勾選 `rag:admin`;只管理自己資料夾的人不要勾。填上 label 後複製 token —— 它只會顯示一次。

把 token 貼到登入畫面即可。以前透過舊的 *Owner Token* 欄位建立的資料夾,屬於當時貼上的那把 token:用那把 token 登入就看得到。腳本可以不走 cookie,直接帶 `Authorization: Bearer <token>` 呼叫 `/api/admin/*` —— 規則相同,而且不需要 CSRF header。

**開發模式。** `auth.enabled: false` 時沒有登入:任何連得到這個 port 的人都能以完整管理權限使用主控台、看到所有資料夾(在這裡建立的資料夾擁有者為 `console`)。伺服器啟動時會記錄警告。只在可信任的機器上使用。

**網路暴露。** 即使已啟用登入,主控台(`/admin`、`/api/admin/*`)仍應只開放給可信任的網路 —— VPN、內部網路,或做存取限制的反向代理 —— 並以 HTTPS 提供。登入能擋下沒有有效 token 的人,但主控台可以上傳與刪除文件、對你的模型端點發出查詢,且(具 `rag:admin` 時)可修改設定,因此不應直接面對公開網際網路。`/mcp` 與 `/api/rag/*` 本來就設計給 MCP 用戶端連線,只需要有效 token。

| 頁面 | 功能 |
|------|------|
| **Overview** | 活的檢索管線、語料讀數、索引健康、模型服務狀態。 |
| **Folders** | 建立資料夾(歸你的 token 所有);點進去可上傳檔案、看逐檔索引狀態、檢視 chunk(AI 脈絡說明與文件原文分開顯示)、重新索引、刪除。以舊版切塊流程索引的檔案會標示 **Re-index recommended**;資料夾列表會逐一統計(**N stale**),並可一鍵重建。 |
| **Search Playground** | 對資料夾即時查詢,顯示**逐訊號檢索軌跡** —— Vector · BM25 · Hybrid · Rerank 各自名次並列 —— 並可加 RAG 答案。 |
| **Index Jobs** | 索引任務即時串流與進度;可取消執行中的任務。 |
| **Query Analytics** | 查詢量、延遲、零結果率、熱門查詢與知識缺口。 |
| **Evaluation** | 自動生成題組,對 `production`(與 `/api/rag/query` 同一路徑)、純 `vector`、純 `bm25` 評 Recall@k · MRR · nDCG@k;基準差異只與同一題組的歷次結果比較。 |
| **Settings** | Embedding / LLM / Contextual Retrieval / Reranker / Speech-to-Text / Retrieval 的即時、持久化設定,含檢索預設與連線測試。修改需要 `rag:admin`。 |
| **System Health** | 每個外部相依的即時連線狀態。 |
| **Audit Log** | 每次執行期設定變更與變更者的時間軸(`rag:admin`)。 |

### 升級後重新索引

切塊流程的版本逐檔記錄。以舊版流程索引的檔案照常可用,但資料夾檢視會標示 **Re-index recommended**(`GET /api/rag/file/{id}/index/status` 也會回 `reindex_recommended: true`)。按該列的重新索引按鈕,或用標題列的連結一次選取所有這類檔案,再按 *Reindex selected*。之後逐檔重新索引都是增量的;*Reindex Folder* 則會刪掉資料夾的資料表、全部重建。

不用一個個點開資料夾去找:資料夾列表會在含有這類檔案的資料夾上顯示琥珀色的 **N stale** 標籤(`GET /api/admin/folders` 的 `reindex_recommended_files`),按該列的重新索引按鈕就只會重建那些檔案。

目前的切塊流程是第 3 版(依章節對齊切塊,搭配第 3 版的脈絡說明 prompt),因此在此之前索引的檔案都會被標示。重新索引之前,檔案會保留舊的 chunk,照常可以搜尋。

## API 參考

| 群組 | 端點 |
|------|------|
| **MCP** | `POST /mcp` —— 每資料夾的 `Folio_<folder>` 工具（`search` / `list` / `read`）;`GET /.well-known/oauth-protected-resource/mcp` |
| **資料夾與檔案** | `/api/folders*` · `/api/folders/{id}/files*`（建立、上傳、列出、下載、刪除 —— token-scoped） |
| **RAG** | `POST /api/rag/query`（混合檢索 + 重排）· `/api/rag/files/{id}/index`、`/reindex`、`/index/jobs*`（背景索引 + SSE 進度） |
| **主控台 session** | `POST /api/console/session` `{token}`(登入)· `GET`(目前身分)· `DELETE`(登出) |
| **主控台（admin）** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **維運** | `GET /health` |

設 `ENABLE_API_DOCS=true` 可在 `/docs` 取得完整 OpenAPI 參考。

## 量測檢索品質

檢索與切塊的調整以數字判斷,不靠肉眼。`evals/` 有一套公開的 golden set:19 份文件(台灣法規、美國政府網頁、虛構公司文件、含表格的 PDF、合成的繁中會議逐字稿)上的 205 題,依題型與語言分層,約一成是無解題;另有 20 題超出範圍的問題,用來校準相關度門檻。評估工具透過 agent 實際使用的同一個 MCP 搜尋,對執行中的服務評分:

```bash
.venv/bin/python scripts/eval_retrieval.py --token-file <pat.json> --runs 2
```

它依題型回報 Recall@5/10/20、MRR、nDCG、回應 token 數、重複率與延遲,以及無解題的誤收率和 `min_relevance` 門檻掃描,結果寫到 `evals/baselines/rag-<日期>-<標籤>.{json,md}`。已提交的 baseline 記錄了每次改動前後的數字。token 檔格式、語料怎麼建、各指標的意義見 [`evals/README.md`](evals/README.md)。

## 架構

- **FastMCP + FastAPI** 同一個 ASGI app。FastMCP app 以 *mount* 掛載,讓它的 auth middleware 保護 `/mcp`;FastAPI router 提供 REST API 與主控台。
- **PostgreSQL + pgvector** 儲存;每個資料夾一張 `data_<folder>_<uuid>` 表。檢索用 [LlamaIndex](https://github.com/run-llama/llama_index) 的 `PGVectorStore` 混合模式。
- 全程 **同步 SQLAlchemy**（psycopg2）—— 不要用 `+asyncpg`。
- **主控台**:Vite + React + React Router + Tailwind,build 到 `static/console`,由 `/admin` 提供。

正在評估中的方向（Graph RAG、「LLM Wiki」）見 [`ROADMAP.md`](ROADMAP.md)。

## 參與貢獻

歡迎 issue 與 PR。程式註解一律英文。單元測試用 `uv run pytest`;`tests/integration` 下的 PostgreSQL 整合測試需明確開啟（`RAG_RUN_DB_ITESTS=1`),會自行建立並刪除測試資料庫。

## 授權

[MIT](LICENSE)
