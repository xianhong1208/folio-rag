# Agentic RAG

**一套開源的 Model Context Protocol 檢索增強生成（RAG）伺服器 —— 階層式切塊、auto-merging 檢索、向量 + 中文 BM25 混合搜尋,以「每個資料夾一個工具」的形式提供給 MCP 用戶端。**

Agentic RAG 把一個文件資料夾變成 Claude Code、Claude Desktop、Cursor 等任何 MCP 用戶端都能呼叫的檢索工具。每個資料夾成為一個 `Agentic_<folder>` 工具,含三種模式 —— **search**、**list**、**read** —— 底層是一條會**階層切塊、以密集+稀疏訊號檢索、把相鄰命中合併成完整段落、再用 cross-encoder 重排**的管線。它是一台 **resource server**:只在本地離線驗證 OAuth token、從不簽發,搭配 [MCP Center](https://github.com/xianhong1208/MCP_Center) 當授權伺服器。

License: MIT&nbsp;·&nbsp;Python 3.12&nbsp;·&nbsp;PostgreSQL + pgvector&nbsp;·&nbsp;相容 FastMCP&nbsp;·&nbsp;OAuth 2.1（resource server）

[English](README.md) · [快速開始](#快速開始) · [運作原理](#運作原理) · [設定](#設定) · [主控台](#使用主控台) · [架構](#架構)

```
┌────────────────┐   Agentic_<folder> 工具呼叫（Bearer JWT）    ┌───────────────────────────────┐
│    MCP 用戶端   │ ───────────────────────────────────────────▶ │  Agentic RAG（resource server）│
│  Claude / IDE  │ ◀─────────────────────────────────────────── │  /mcp · 依 token 過濾的資料夾   │
└───────┬────────┘        auto-merge 後的段落 + 分數            └───────────────┬───────────────┘
        │ OAuth 流程                                                             │ 離線驗證 JWT
        ▼                                                                        ▼
  ┌─────────────┐                              檢索管線（每次查詢）
  │  MCP Center │  RS256 JWKS                  ingest ─▶ chunk ─▶ embed ─▶ pgvector ─▶ 混合檢索
  │ OAuth 2.1 AS│  （離線驗證）                (Docling) (leaf+parent) (e5·1024d)   (向量 + BM25/CKIP)
  └─────────────┘                                   └▶ auto-merge + 擴展 ─▶ rerank ─▶ gpt-oss-120b
```

## 功能特點

- **階層切塊 + auto-merging 檢索。** 索引時把文件切成小的 **leaf**（精準匹配）與大的 **parent**（完整脈絡）。當同一段落的多個 leaf 被命中時,回傳整個 parent 而非碎片;單一 leaf 命中則向鄰居擴展,答案不會失去語意連貫性。
- **混合搜尋,為中文調校。** 密集向量檢索（pgvector）與稀疏 **BM25** 以 Reciprocal Rank Fusion 融合,讓關鍵字訊號真的影響排名。中文在進 Postgres 全文索引前先用 **CKIP**（中研院）斷詞 —— 無空格的中文會被真正 tokenize,而不是塌成單一詞。
- **Cross-encoder 重排。** 以 `bge-reranker-v2-m3` / `mxbai-rerank-base-v2` 對融合後的候選重新評分,提升 Top-K 精度。
- **Contextual Retrieval。** 索引時由 LLM 為每個 chunk 加上一段依文件脈絡的前綴（Anthropic 建議的做法),提升原本模稜兩可片段的可檢索性。
- **每資料夾一個 MCP 工具,依 token 隔離。** 每個資料夾註冊成 `Agentic_<folder>` 工具;呼叫者看到的工具清單會依其 token 過濾 —— 你只看得到自己擁有的資料夾。歸屬以 token 穩定的 `jti` 為 key。
- **OAuth 2.1 resource server。** 一個 `RemoteAuthProvider` + `JWTVerifier` 保護 `/mcp`;token 用 MCP Center 的 JWKS 在本地離線驗證（RS256),並綁定本伺服器的 audience,別台的 token 在這裡無效。
- **文件 + 音訊匯入。** [Docling](https://github.com/DS4SD/docling) 解析 PDF / DOCX / PPTX / XLSX / 圖片（含 OCR）;**FireRedASR** 在本地做繁體中文語音轉錄（或任何 OpenAI 相容的 `/v1/transcriptions`）。
- **內建評測 harness。** 從你自己的 chunk 自動生成問題,對 `vector` / `hybrid` / `rerank` 評 **Recall@k · MRR · nDCG@k**,讓你針對真實語料調校檢索。
- **Retrieval Terminal 主控台。** `/admin` 的 React 單頁主控台:管線總覽、資料夾與檔案、會顯示**逐訊號檢索軌跡**的 Search Playground、即時索引任務串流、查詢分析、評測 harness、即時設定、健康狀態與稽核時間軸。
- **跑在你的 GPU 上。** PyTorch 以 PEP 735 dependency group 依平台選擇（`rocm-r714` / `rocm-r713` / `cuda` / `cpu`）—— AMD ROCm、NVIDIA CUDA 或純 CPU。

## 安裝

**前置需求**

- Python 3.12 與 [uv](https://docs.astral.sh/uv/)。
- 安裝了 **[pgvector](https://github.com/pgvector/pgvector)** 擴充的 **PostgreSQL**。
- **embedding**、**LLM**、**reranker** 三個模型的 OpenAI 相容端點（例如 [vLLM](https://github.com/vllm-project/vllm)、Ollama 或雲端供應商）。
- Node.js 18+ —— 只有要開發主控台 UI 時才需要。

```bash
git clone https://github.com/xianhong1208/agentic_rag.git
cd agentic_rag

# 1. 安裝 —— torch 依平台走 PEP 735 dependency group,務必帶 --group。
uv sync --group rocm-r714    # AMD ROCm 7.1.4（multi-arch wheel） | rocm-r713 | cuda | cpu

# 2. 設定資料庫與模型端點
cp .env.example .env         # 設定 DATABASE_URL（Postgres + pgvector）
#   編輯 config/config.yaml  → rag.embedding / rag.llm / rag.rerank 端點、
#                               auth.issuer / auth.audience、server.port

# 3. 啟動（自動跑 DB bootstrap + migration）
uv run python main.py
```

伺服器預設監聽 `http://0.0.0.0:5039`;MCP 端點是 `/mcp`,主控台在 `/admin`。

## 快速開始

Agentic RAG 透過 [MCP Center](https://github.com/xianhong1208/MCP_Center) 驗證。**順序很重要:先在 MCP Center 註冊這台伺服器** —— 它只會為它認得的 audience 簽 token。

**1 — 在 MCP Center 註冊伺服器。** 到 MCP Center 主控台 *Services → Register Service*,填入本伺服器的 host / port / path（`/mcp`）。它的 audience 預設為 MCP URL,例如 `http://127.0.0.1:5039/mcp`。把同一個值設進 `config/config.yaml` 的 `auth.audience`。

**2 — 讓 Agentic RAG 指向 MCP Center**（`config/config.yaml`）:

```yaml
auth:
  enabled: true
  issuer:   "http://localhost:4568"          # MCP Center 位址（RS256 JWKS issuer）
  audience: "http://127.0.0.1:5039/mcp"      # 必須等於主控台註冊的 audience
```

**3 — 建立資料夾並索引文件** —— 用 `/admin` 主控台（*Folders → New Folder → Upload Files*）或走 REST:

```bash
# 建立一個 token-scoped 資料夾,再上傳 + 索引檔案（見 API 參考）
curl -X POST http://127.0.0.1:5039/api/folders/ \
  -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" \
  -d '{"name":"product-docs"}'
```

**4 — 接上用戶端。** 有 MCP Center 擋在前面,支援 OAuth 的用戶端不需預先設定:

```bash
claude mcp add --transport http agentic-rag http://127.0.0.1:5039/mcp
```

首次使用時,用戶端會向 MCP Center 註冊並開啟同意畫面;你按 Allow 後,`Agentic_product-docs` 工具就會出現。腳本 / CI 則在 MCP Center 簽一個 **personal access token**,以 bearer header 帶入。

**5 — 開始問。** 在用戶端呼叫資料夾工具:

```
Agentic_product-docs  mode="search"  query="要怎麼輪替簽章金鑰?"
```

它會回傳命中的段落（必要時 auto-merge),每筆附相關性分數與來源檔案 / 標題。

## 運作原理

資料夾索引一次,之後每次查詢都走這條管線:

| 階段 | 內容 |
|------|------|
| **Ingest** | Docling 解析 PDF / DOCX / PPTX / XLSX / 圖片（OCR）;音訊由 FireRedASR 轉錄。 |
| **Chunk** | 階層切成小 **leaf** + 大 **parent**;LLM 為每個 chunk 加上脈絡前綴。 |
| **Embed** | 每個 chunk 做 embedding（預設 `intfloat/multilingual-e5-large`,1024 維）寫入 **pgvector**。中文以 CKIP 斷詞供 BM25 的 `text_search_tsv`。 |
| **Retrieve** | **混合**:密集向量 + 稀疏 BM25,以 Reciprocal Rank Fusion 融合。 |
| **Merge** | 同段落的 leaf **auto-merge** 成 parent;單一命中**擴展**到鄰居。 |
| **Rerank** | cross-encoder（`bge-reranker-v2-m3` / `mxbai-rerank-base-v2`）重排,決定最終 Top-K 精度。 |
| **Answer** | 走 `answer` 路徑時,`gpt-oss-120b` 依檢索到的脈絡合成回答,並由 corrective-RAG 信心檢查把關。 |

**驗證。** Agentic RAG 是 *resource server*。FastMCP 會發佈 `/.well-known/oauth-protected-resource/mcp`,把用戶端導向 MCP Center;伺服器抓一次 MCP Center 的 JWKS,之後每個 bearer token 都離線驗證（RS256 —— 簽章、issuer、audience、效期、scope）。MCP Center 從不在請求路徑上。

## 設定

設定在 `config/config.yaml`,機密從環境變數（`.env`）讀。你最可能會動到的:

| 變數 / 設定 | 預設 | 用途 |
|------------|------|------|
| `DATABASE_URL` | `postgresql://…/agentic_rag` | PostgreSQL DSN。需 **pgvector** 擴充。用**同步** `postgresql://`（psycopg2),不要 `+asyncpg`。 |
| `server.port` | `5039` | HTTP + MCP 綁定 port。MCP 端點是 `/mcp`。 |
| `auth.enabled` / `auth.issuer` / `auth.audience` | `true` / MCP Center 位址 / 本伺服器 `/mcp` URL | 對 MCP Center 做 OAuth 2.1 驗證。 |
| `rag.embedding.{model,dimension,base_url}` | `intfloat/multilingual-e5-large`、`1024`、`:7075/v1` | embedding 模型;`dimension` **必須**等於模型實際輸出維度。 |
| `rag.llm.{model,base_url}` | `openai/gpt-oss-120b`、`:5052/v1` | Contextual Retrieval 與 `answer` 路徑用的 LLM。 |
| `rag.rerank.{model,base_url}` | `mixedbread-ai/mxbai-rerank-base-v2`、`:8787` | cross-encoder 重排器。 |
| `rag.asr.provider` | `fireredasr` | 音訊轉錄:`fireredasr`（本地繁中）· `docling-whisper` · `openai-compatible`。 |
| `rag.retrieval.*` | — | Top-K、相似度門檻、BM25 候選數、`hybrid_fusion`（`rrf`/`concat`）、auto-merging。可從主控台即時調,或用 **High Precision / Balanced / High Recall** 預設。 |

GPU 選擇是 `uv sync --group {rocm-r714|rocm-r713|cuda|cpu}` 的取捨 —— 見 `pyproject.toml` 開頭與 `docs/testing/ENVIRONMENTS.md`。

## 使用主控台

**Retrieval Terminal** 主控台是 `/admin` 的 React SPA(舊的單檔主控台仍保留在 `/admin-classic`)。

| 頁面 | 功能 |
|------|------|
| **Overview** | 活的檢索管線、語料讀數、索引健康、模型服務狀態。 |
| **Folders** | 建立 token-scoped 資料夾;點進去可上傳檔案、看逐檔索引狀態、檢視 chunk、重新索引、刪除。 |
| **Search Playground** | 對資料夾即時查詢,顯示**逐訊號檢索軌跡** —— Vector · BM25 · Hybrid · Rerank 各自名次並列 —— 並可加 RAG 答案。 |
| **Index Jobs** | 索引任務即時串流與進度;可取消執行中的任務。 |
| **Query Analytics** | 查詢量、延遲、零結果率、熱門查詢與知識缺口。 |
| **Evaluation** | 自動生成題組,對 `vector` / `hybrid` / `rerank` 評 Recall@k · MRR · nDCG@k。 |
| **Settings** | Embedding / LLM / Contextual Retrieval / Reranker / Speech-to-Text / Retrieval 的即時、持久化設定,含檢索預設與連線測試。 |
| **System Health** | 每個外部相依的即時連線狀態。 |
| **Audit Log** | 每次執行期設定變更的時間軸。 |

## API 參考

| 群組 | 端點 |
|------|------|
| **MCP** | `POST /mcp` —— 每資料夾的 `Agentic_<folder>` 工具（`search` / `list` / `read`）;`GET /.well-known/oauth-protected-resource/mcp` |
| **資料夾與檔案** | `/api/folders*` · `/api/folders/{id}/files*`（建立、上傳、列出、下載、刪除 —— token-scoped） |
| **RAG** | `POST /api/rag/query`（混合檢索 + 重排）· `/api/rag/files/{id}/index`、`/reindex`、`/index/jobs*`（背景索引 + SSE 進度） |
| **主控台（admin）** | `/api/admin/overview` · `/api/admin/folders*` · `/api/admin/jobs` · `/api/admin/analytics` · `/api/admin/eval/{run,status,report}` · `/api/admin/settings*` · `/api/admin/health` · `/api/admin/audit` · `/api/admin/probe` |
| **維運** | `GET /health` |

設 `ENABLE_API_DOCS=true` 可在 `/docs` 取得完整 OpenAPI 參考。

## 架構

- **FastMCP + FastAPI** 同一個 ASGI app。FastMCP app 以 *mount* 掛載,讓它的 auth middleware 保護 `/mcp`;FastAPI router 提供 REST API 與主控台。
- **PostgreSQL + pgvector** 儲存;每個資料夾一張 `data_<folder>_<uuid>` 表。檢索用 [LlamaIndex](https://github.com/run-llama/llama_index) 的 `PGVectorStore` 混合模式。
- 全程 **同步 SQLAlchemy**（psycopg2）—— 不要用 `+asyncpg`。
- **主控台**:Vite + React + React Router + Tailwind,build 到 `static/console`,由 `/admin` 提供。

正在評估中的方向（Graph RAG、「LLM Wiki」）見 [`ROADMAP.md`](ROADMAP.md)。

## 參與貢獻

歡迎 issue 與 PR。程式註解一律英文。詳見 `docs/`。

## 授權

[MIT](LICENSE)
