# 開發

[English](development.md)

## 架構

- **FastMCP + FastAPI** 同一個 ASGI app。FastMCP app 以 *mount* 掛載,讓它的 auth middleware 保護 `/mcp`;FastAPI router 提供 REST API 與主控台。
- **PostgreSQL + pgvector** 儲存;每個資料夾一張 `data_<folder>_<uuid>` 表。檢索用 [LlamaIndex](https://github.com/run-llama/llama_index) 的 `PGVectorStore` 混合模式。
- 全程 **同步 SQLAlchemy**（psycopg2）—— 不要用 `+asyncpg`。
- **主控台**:Vite + React + React Router + Tailwind,build 到 `static/console`,由 `/admin` 提供。

接下來的規劃,以及 Graph RAG 與「LLM Wiki」為何暫緩:見 [`ROADMAP.zh-TW.md`](../ROADMAP.zh-TW.md)。

## 檢查

開 pull request 前先跑過這些;CI 跑的是同一組。

```bash
uv sync --extra local --extra jieba --group cpu     # CI 安裝的組合
uv run --no-sync pytest --cov                       # 單元測試;整體覆蓋率低於 91% 就失敗
uv run --no-sync python scripts/check_coverage_floor.py reports/coverage.json   # 每個檔案至少 50%
uvx ruff check .                                    # 正確性規則(E4、E7、E9、F)
RAG_RUN_DB_ITESTS=1 uv run --no-sync pytest tests/integration   # 需要 PostgreSQL + pgvector;會自行建立並刪除測試資料庫
cd frontend && npm ci && npm test && npm run build  # 主控台:Vitest,再做正式 build
```

逐檔門檻讀的是 JSON 報告:先在 pytest 那一行加上 `--cov-report=json:reports/coverage.json`。主控台需要
Node.js 22.12 以上。它的 build 產物 `static/console` **有提交進版本庫**,伺服器直接從工作目錄提供它:
每次改主控台都要重新 build。

CI:*Backend CI*(`.github/workflows/backend.yml`:ruff、含覆蓋率的單元測試、PostgreSQL 整合測試、
cloud profile 安裝)、*Console CI*(`frontend.yml`)與 *Dependency audit*(`audit.yml`:pip-audit 與
npm audit,每週一也會跑)。每個 workflow 只在相關路徑有變動時執行。

## 發布

1. 更新版本號:`pyproject.toml`(接著 `uv lock`)、`frontend/package.json`(接著
   `npm install --package-lock-only`),以及 `frontend/src/components/Layout.jsx` 側欄上的字串。
2. 為 `CHANGELOG.md` 的段落填上日期,更新 `ROADMAP.md` 與 `ROADMAP.zh-TW.md` 的「已發布」,並重新
   build 主控台。
3. 發布用的 pull request 合併進 `dev`;把 `main` fast-forward 到 `dev`;打 tag
   `git tag -a vX.Y.Z -m "Folio X.Y.Z"`;以 CHANGELOG 段落建立標題為 `vX.Y.Z` 的 GitHub Release。

`src/version.py` 先讀已安裝套件的 metadata,所以執行中的伺服器要等 `uv sync` 之後才會回報新版本。
