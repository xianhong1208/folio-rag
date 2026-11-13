# Folio

[![License: MIT](https://img.shields.io/github/license/xianhong1208/folio-rag)](LICENSE)
[![Backend CI](https://github.com/xianhong1208/folio-rag/actions/workflows/backend.yml/badge.svg?branch=dev)](https://github.com/xianhong1208/folio-rag/actions/workflows/backend.yml)
[![Console CI](https://github.com/xianhong1208/folio-rag/actions/workflows/frontend.yml/badge.svg?branch=dev)](https://github.com/xianhong1208/folio-rag/actions/workflows/frontend.yml)
[![Release](https://img.shields.io/github/v/release/xianhong1208/folio-rag)](https://github.com/xianhong1208/folio-rag/releases)

**把一個文件資料夾變成 Claude、Cursor 等任何 MCP 用戶端都能用的搜尋工具:向量 + 中文 BM25 混合檢索、重排,以及自動合併的完整段落,每個資料夾一個 `Folio_<folder>` 工具。**

[文件](docs/how-it-works.zh-TW.md) · [English](README.md) · [路線圖](ROADMAP.zh-TW.md) · [更新紀錄](CHANGELOG.md)

![Search Playground:附引用的答案,以及每筆結果的向量、BM25、融合與重排分數](docs/images/console-search.png)

Folio 是一套開源、給 Model Context Protocol 用的檢索增強生成(RAG)伺服器。每個資料夾會成為一個
`Folio_<folder>` 工具,有 **search**、**list**、**read** 三種模式。它是 OAuth 2.1 的
**resource server**:只在本地離線驗證 token,從不簽發,搭配 [MCP Gate](https://github.com/xianhong1208/MCP_Gate)
當授權伺服器。Folio 在 v1.1.0 以前叫 *Agentic RAG*;資料庫名稱(`agentic_rag`)與儲存目錄結構沿用舊名,
不需要任何遷移。

## 最新消息

**1.4.1:** 資料夾屬於「人」而不是某一把 token,同一位 MCP Gate 使用者的每一把 personal access token
與 OAuth 登入都看到同一組資料夾;登出會在伺服器端結束主控台 session;答案使用提問的語言。詳見
[CHANGELOG](CHANGELOG.md);接下來的規劃見[路線圖](ROADMAP.zh-TW.md)(1.5:Docker 映像檔、備份,以及
透過 OpenAI 相容 API 做 OCR 與語音轉文字)。

## 主要功能

- **會合併回來的階層式切塊。** 小的 leaf 用來精準比對,大的 parent 提供完整脈絡,兩者都不跨章節;同一段落
  多處命中時,回傳的是整段。
- **為中文調校的混合搜尋。** 向量(pgvector)與 BM25 以 Reciprocal Rank Fusion 融合;中文先斷詞
  (CKIP、jieba 或 bigram)。
- **Cross-encoder 重排**(`mxbai-rerank-base-v2`、`bge-reranker-v2-m3`),MCP 搜尋也會說明「找不到相關
  內容」。
- **Contextual retrieval。** 索引時由 LLM 為每個 chunk 寫一句脈絡說明;重新索引改過的檔案時,只重做有變動的
  chunk。
- **每人各自的資料夾。** 你只看得到自己 MCP Gate 帳號的資料夾,MCP、REST 與主控台都一樣。
- **多種格式。** PDF、Office(含舊版與 RTF)、OpenDocument、HTML 與 MHTML、Markdown、EPUB、電子郵件、
  圖片(OCR)與音訊。
- **用數字說話。** 公開的 205 題黃金測試集與評測工具;主控台可以用 Recall@k、MRR、nDCG@k 為你自己的
  資料夾評分。
- **主控台**在 `/admin`:資料夾與檔案、顯示逐訊號軌跡的 Search Playground、索引任務、查詢分析與知識缺口、
  評測、即時設定、健康狀態與稽核紀錄。
- **本機 GPU、CPU 或雲端。** 行程內模型可跑在 AMD ROCm、NVIDIA CUDA 或 CPU 上,也可以讓所有模型都走
  OpenAI 相容 API,完全不需要 torch。

## 安裝

需要:

- Python 3.12 與 [uv](https://docs.astral.sh/uv/);
- 裝了 **[pgvector](https://github.com/pgvector/pgvector)** 擴充的 **PostgreSQL**;
- **embedding**、**LLM** 與 **reranker** 模型的 OpenAI 相容端點([vLLM](https://github.com/vllm-project/vllm)、
  Ollama、[Forgeyard](https://github.com/xianhong1208/forgeyard) 或雲端服務);
- 處理舊版 Office 檔與 RTF 需要 LibreOffice(`soffice`)。

```bash
git clone https://github.com/xianhong1208/folio-rag.git
cd folio-rag

# 本地模型需要 `local` extra,torch 則來自 dependency group:
# 一定要一起帶 rocm-r714 | cuda | cpu 其中一個。全雲端:直接 `uv sync`。
uv sync --extra local --group cuda

cp .env.example .env         # 設定 DATABASE_URL(PostgreSQL + pgvector)
# 編輯 config/config.yaml:rag.embedding / rag.llm / rag.rerank 端點、
#                         auth.issuer / auth.audience、server.port

uv run python main.py        # 啟動時會建立 schema 並執行 migration
```

Folio 監聽 `http://0.0.0.0:5039`:MCP 端點是 `/mcp`,主控台是 `/admin`。請用擁有 `.venv` 的使用者執行:
NLTK 3.10.4 起會拒絕其他帳號擁有或可寫入的資料檔,較長的純文字段落就會無法切分
(`chown -R <執行服務的使用者> .venv`)。

## 快速開始

Folio 透過 [MCP Gate](https://github.com/xianhong1208/MCP_Gate) 驗證身分。請先在 MCP Gate 註冊 Folio:
MCP Gate 只會為它認得的 audience 簽發 token。

**1. 在 MCP Gate 註冊伺服器。** *Services → Register Service*,填入這台伺服器的 host、port 與路徑
(`/mcp`)。audience 預設就是 MCP 網址,例如 `http://127.0.0.1:5039/mcp`。

**2. 讓 Folio 指向 MCP Gate**,在 `config/config.yaml`:

```yaml
auth:
  enabled: true
  issuer:   "http://localhost:4568"          # MCP Gate 網址(RS256 JWKS issuer)
  audience: "http://127.0.0.1:5039/mcp"      # 必須和在 MCP Gate 註冊的 audience 一字不差
```

**3. 建立資料夾並加入文件**,可以在主控台(*Library → New Folder → Upload Files*,用 MCP Gate 發的
personal access token 登入),或透過 REST:

```bash
curl -X POST http://127.0.0.1:5039/api/folders/ \
  -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" \
  -d '{"name":"product-docs"}'
```

**4. 連接用戶端。** 支援 OAuth 的用戶端不需要事先準備任何東西:

```bash
claude mcp add --transport http folio http://127.0.0.1:5039/mcp
```

第一次使用時,用戶端會向 MCP Gate 註冊並顯示同意畫面;之後就會出現 `Folio_product-docs` 工具。腳本則把
personal access token 放在 bearer header 傳入。

**5. 提問。**

```
Folio_product-docs  mode="search"  query="how do I rotate the signing key?"
```

你會拿到相符的段落,屬於同一段的會合併在一起,每筆都附相關度分數、來源檔案與標題。

## 部署方式

| Profile | 模型 | 安裝 |
|---|---|---|
| `local`(預設) | 什麼都可以。內附的設定呼叫自架的 embedding、LLM 與 reranker 端點,並在行程內執行 CKIP 斷詞、含 OCR 的 Docling 解析與 FireRedASR 語音轉文字 | `uv sync --extra local --group rocm-r714 \| cuda \| cpu` |
| `hybrid` | 同上,用於自架與雲端端點混用;直接寫在 `config.yaml` 裡的 API 金鑰會被警告 | 同 `local` |
| `cloud` | 每個角色都是 HTTP 端點,金鑰必須是 `${ENV}` 參照;不需要 torch,也不下載模型 | `uv sync` |

用 `rag.profile` 設定,詳見[雲端 profile](docs/configuration.zh-TW.md#雲端-profile)。主控台與不需驗證的
`/v1` 媒體端點請只開放在可信任的網路,或放在有 HTTPS 的反向代理後面;`/mcp` 與 `/api/rag/*` 需要有效的
token。Docker 映像檔與 compose 檔預計在 1.5 推出。

## 文件

| | |
|---|---|
| [運作原理](docs/how-it-works.zh-TW.md) | 檢索管線、檢索細節(BM25、斷詞、重排、延遲、相關度門檻)、量測檢索品質、每項功能的細節 |
| [設定](docs/configuration.zh-TW.md) | 常用的每個設定,以及雲端 profile |
| [驗證](docs/auth.zh-TW.md) | 依 MCP Gate 使用者的資料夾歸屬,以及 token 撤銷 feed |
| [主控台](docs/console.zh-TW.md) | 登入、兩種權限層級、每個頁面、升級後重新索引 |
| [API 參考](docs/api.zh-TW.md) | MCP、REST、主控台與健康檢查端點 |
| [PostgreSQL 維運](docs/operations-postgres.md) | 連線數預算與資料庫維護(英文) |
| [開發](docs/development.zh-TW.md) | 架構、檢查、CI 與發布 |
| [檢索評測](evals/README.md) | 黃金測試集、評測工具與 baseline(英文) |

## 參與貢獻

歡迎 issue 與 pull request。從 `dev` 開分支,pull request 開回 `dev`。程式碼與註解使用英文;
主控台依決定只有英文。送出前先跑 [docs/development.zh-TW.md](docs/development.zh-TW.md) 裡的檢查:
`uv run pytest`、`uvx ruff check .`,改了主控台的話再到 `frontend/` 跑 `npm test` 與 `npm run build`。

## 授權

[MIT](LICENSE)
