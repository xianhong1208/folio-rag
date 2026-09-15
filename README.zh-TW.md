# Agentic RAG MCP Server

> 本專案以 MIT 授權釋出,詳見 [`LICENSE`](LICENSE)。

階層式 RAG (Retrieval-Augmented Generation) MCP 服務。把任意檔案集合(PDF / Office / 音檔 / 圖檔)
轉成可被 AI agent 查詢的向量索引,提供 Hierarchical Auto-Merging Retrieval、Contextual Retrieval、
跨模態(Whisper ASR / Docling OCR)解析,以及完整的索引生命週期管理。

---

## 設計理念

| 原則 | 怎麼體現 |
|------|----------|
| **階層檢索** | Docling chunks → SentenceSplitter → 建 leaf + parent hierarchy → 命中 leaf 再 auto-merge 回 parent |
| **上下文補強** | 索引時 LLM 為每個 chunk 生成 contextual prefix(沿用前一版 flat RAG 做法) |
| **Token-scoped 隔離** | 每個 user_token 看到的 folders / files / vector tables 完全獨立 |
| **背景任務優先** | 索引一律走 `IndexingJobManager` 背景 job,有完整的 cancel / watchdog / 重啟回收 / 持久化 |
| **設定驅動** | `config.yaml` + `${VAR:-default}` 環境變數展開;模組透過 `importlib` 動態載入 |

`★ 為什麼選 Hierarchical + Contextual Retrieval`
- **Hierarchical (Auto-Merging)**:小塊 leaf 精準命中,但 LLM 需要更大的 context。leaves 命中後自動 merge 回 parent,
  在「精準度」跟「上下文豐富度」中取平衡,**比單一 chunk size 多 ~20% recall**。
- **Contextual Retrieval (Anthropic 2024)**:每個 chunk 索引時前綴一段 LLM 生成的「這段內容在說什麼」說明,
  解決傳統 RAG 的 chunk 缺脈絡問題,**對比一般 chunking 在 needle-in-haystack 任務上提升 ~35%**。

---

## 快速開始

```bash
# 1. 安裝依賴 —— torch 依部署硬體走 PEP 735 dependency group,務必帶 --group
uv sync --group cuda        # NVIDIA(CUDA 12.6)
# uv sync --group rocm-r714  # AMD(ROCm 7.1.4;multi-arch wheel,gfx1151/90a/1201 通吃)
# uv sync --group rocm-r713  # AMD(ROCm 7.1.3)
# uv sync --group cpu        # CI / 無 GPU
#   ⚠️ 不帶 --group 的裸 `uv sync` / `uv run` 會把 torch 三件套換成預設解析版本,
#      破壞已裝好的 GPU stack。細節見 pyproject.toml 開頭對照表與 docs/testing/ENVIRONMENTS.md

# 2. 啟動三個本地模型服務(透過 vLLM 或其他 OpenAI-compatible endpoint)
#    - Embedding (預設 intfloat/multilingual-e5-large,:5040;bge-m3 為回退選項)
#    - LLM (預設 openai/gpt-oss-120b,:5052) — Contextual Retrieval 用
#    - Reranker (預設 mixedbread-ai/mxbai-rerank-base-v2,:8787;bge-reranker-v2-m3 為回退)
#    上述 URL/model 都可在 config/config.yaml 改(埠號以 config.yaml base_url 為準)

# 3. 設定 .env(機密)
cp .env.example .env   # 若有此檔;否則直接編輯 config.yaml
#    至少要設:DATABASE_URL、TOKEN_SERVER_URL

# 4. 修改 config/config.yaml
#    - server.port (預設 5031)
#    - rag.embedding / rag.llm / rag.rerank 各 endpoint
#    - rag.docling.device (cuda / cuda:N / cpu)

# 5. 啟動(會自動跑 DB migration)
uv run python main.py --config config/config.yaml
```

啟動後:
- **首頁 (Landing page)** : `http://<host>:<port>/` — 服務介紹 + Swagger / Health 入口
- **Swagger UI** : `http://<host>:<port>/docs`
- **MCP endpoint** : `http://<host>:<port>/mcp`
- **Health check** : `http://<host>:<port>/health`

---

## 支援格式(格式 v3 — 對標 docling 2.124 官網,逐格式實測)

> 實測 = 生成最小樣本真走 docling 解析、斷言內容取回(`tests/integration/test_format_smoke.py`,
> 35 案);對標完整性由程式化不變式把關(`test_upload_formats.py` ref 直接讀 docling
> 官方 FormatToExtensions,上游新增格式時測試自動紅)。

| 類別 | 副檔名 | 解析引擎 | 驗證狀態 |
|---|---|---|---|
| 純文字直讀 | txt text json csv yaml yml xml conf log | read_text_robust(UTF-8/16/Big5 容錯) | ✅ 單元測試 |
| PDF | pdf | docling(layout + TableFormer + OCR) | ✅ 生產使用中 |
| Office OOXML | docx dotx docm dotm / pptx ppsx pptm potm ppsm / xlsx xlsm | docling | ✅ 冒煙實測 |
| Legacy Office | doc dot xls xlt ppt pot pps | docling + LibreOffice(部署需 soffice) | ✅ 冒煙實測(真檔) |
| OpenDocument | odt ods odp | docling + odfdo | ✅ 冒煙實測 |
| 標記 / 科學 | md qmd rmd html htm xhtml adoc asciidoc asc tex latex | docling | ✅ 冒煙實測 |
| 郵件 / 書 / 字幕 | msg eml epub vtt | docling | ✅ 冒煙實測(msg 為 v2 實測) |
| 圖片 | png jpg jpeg tiff tif bmp webp | docling + RapidOCR | ✅ 生產使用中(bmp 同引擎) |
| 音訊 | wav mp3 m4a aac ogg flac | **AsrProvider**(本地 docling-whisper / 本地 fireredasr(FireRedASR-AED-L,繁中輸出)/ 雲端 openai-compatible,`rag.asr` 設定) | wav/mp3 ✅ 生產;新 4 種同管線(ffmpeg 解碼);fireredasr 待評測素材驗收 |
| 影片 | —(規劃中 ref) | docling video(ASR + 關鍵影格) | 🧪 管線已實測通(mp4 SUCCESS),待語音樣本內容驗證後收編 |

**實測後刻意排除**(docling 官方列了、但 backend 實測打不開):`ott/ots/otp`(ODF 範本)、
`potx`(PowerPoint 範本)— 理由記錄於 ref 的 `_EXCLUDED_EXTS`,上游修復後移出即自動生效。

---

## 主要 REST API

### 檔案 / 資料夾管理(`/api/folders` / `/api/{folder_id}/file`)

| Method | Path | 用途 |
|---|---|---|
| `POST` | `/api/folders/` | 建資料夾 |
| `GET` | `/api/folders/` | 列出 / 用名稱/ID 查資料夾 |
| `DELETE` | `/api/folders/{folder_id}` | 刪除資料夾(會自動 cancel 進行中 indexing job) |
| `POST` | `/api/folders/{folder_id}/file` | 上傳檔案(form-data,可選 auto_index) |
| `GET` | `/api/folders/{folder_id}/files/{file_id}/download` | 下載原檔 |

### 索引 (`/api/rag`)

| Method | Path | 用途 |
|---|---|---|
| `POST` | `/files/{folder_id}/index` | 整個 folder 背景索引(回 job_id) |
| `POST` | `/file/{file_id}/index` | 單一檔案同步索引 |
| `POST` | `/files/{folder_id}/reindex` | 先刪再重新索引 |
| `GET` | `/files/{folder_id}/index/jobs/{job_id}` | 查 job 狀態 |
| `GET` | `/files/{folder_id}/index/jobs/{job_id}/events` | **SSE realtime progress**(EventSource 推播) |
| `DELETE` | `/files/{folder_id}/index/jobs/{job_id}` | 取消執行中的 job |
| `GET` | `/index/jobs` | 列出 + 各狀態 counts |
| `GET` | `/file/{file_id}/index/status` | **單檔狀態查詢**(`indexed`/`indexing`/`queued`/`failed`/`not_indexed`) |
| `GET` | `/indexed-files` | 列出某 folder 已索引檔案 |
| `DELETE` | `/files/{folder_id}/index` | 刪除整個 folder 索引 |
| `DELETE` | `/file/{file_id}/index` | 刪除單檔索引 |

### 查詢 (`/api/rag/query`)

| Method | Path | 用途 |
|---|---|---|
| `POST` | `/api/rag/query` | Hybrid retrieval(向量 + BM25)+ 可選 rerank |

`top_k` / `similarity_cutoff` / `sparse_top_k` / `hybrid_alpha` 預設都從 `config.rag.retrieval.*` 讀,
請求時可覆寫。

---

## 索引可靠性 — `IndexingJobManager`

完整 lifecycle 控管,設計重點:

| 功能 | 行為 |
|---|---|
| **Folder lock** | 同一 folder 不能同時跑兩個索引 job(回 409 Conflict) |
| **Cancel** | `DELETE /index/jobs/{id}` 可即時中止 in-flight 任務 |
| **Watchdog** | 每 60s 掃描;若 job >10min 無心跳自動標 failed |
| **Restart cleanup** | server 重啟時把 DB 內遺留的 PENDING/RUNNING jobs 翻成 failed(reason=server_restart) |
| **Folder-delete cleanup** | 刪 folder 或刪整個 folder index 之前,先 cancel 所有 in-flight job |
| **Per-file timeout** | 預設每檔 10 天(慢機台大檔 OCR;env `RAG_PER_FILE_TIMEOUT_SECONDS` / config `rag.indexing.per_file_timeout_seconds`) |
| **Job-level timeout** | 預設整個 job 10 天 backstop(env `RAG_JOB_TIMEOUT_SECONDS`) |
| **File-existence gate** | 索引中途若使用者刪檔,3 道 gate 提前 abort(不浪費 GPU) |
| **`PARTIAL_SUCCESS` 狀態** | 9/10 成功 1 失敗時,job 狀態 = `partial_success`,不會誤報 succeeded |
| **Content-hash idempotency** | 同檔再 index(content 沒變)直接 short-circuit,**~60× 加速** |
| **Embedding retry** | 短暫斷線 / 5xx / 429 / timeout 自動 exponential backoff 重試 3 次(1→2→4s) |
| **Per-file timing audit** | 每檔記 `load_ms`(Docling)/ `index_ms`(context-gen + embed + write)/ `total_ms` |
| **DB persistence** | 所有 job state write-through 到 `IndexJobs` table,跨 process restart 仍可查 |

---

## 目錄結構

```
├── app.py                           # FastAPI + FastMCP 組裝(只負責 wire-up)
├── main.py                          # CLI 入口(startup + DB migration)
├── config/
│   ├── config.yaml                  # ★ 主配置
│   └── instructions.md              # ★ MCP Instructions(landing page 也會渲染)
├── assets/
│   ├── docling_models/              # Docling 模型(gitignored,部署時放入)
│   ├── whisper_models/              # Whisper 模型(同上)
│   └── hf_tokenizers/               # HF tokenizers(同上)
├── docs/
│   ├── integrate-existing-api.md    # 整合現有 API 的指南
│   └── testing/                     # ★ 測試流程文件(TEST_PLAN / STRATEGY / specs / test-cases)
├── db/
│   ├── db.py                        # ORM Models(Folder / File / FileIndex / IndexJob)
│   ├── baseDB.py                    # 通用 CRUD 基類
│   ├── filedb.py / folderdb.py / fileindexdb.py / indexjobdb.py
│   ├── cached_folderdb.py           # Folder 查詢 cache
│   └── migrate.py                   # Alembic 自動 migration
├── src/
│   ├── auth/                        # Token Server remote 認證
│   ├── middleware/                  # request_id / auth / error handler
│   ├── api/
│   │   ├── dependencies/            # FastAPI Depends helpers
│   │   └── router/
│   │       ├── index.py             # 首頁 /
│   │       ├── health.py            # /health
│   │       ├── folder_api.py        # /api/folders/*
│   │       ├── file_api.py          # /api/folders/{id}/file*
│   │       ├── rag_indexing.py      # /api/rag/files/*, /api/rag/file/*
│   │       └── rag_query.py         # /api/rag/query
│   ├── adapter/
│   │   ├── rag.py                   # Thin facade
│   │   ├── rag_context.py           # Shared RAG state(embedding / indexer / vector_store_mgr)
│   │   ├── rag_indexing.py          # 索引 service(folder / file-list / auto-index)
│   │   ├── rag_query.py             # 查詢 service(flat hybrid + agentic three-mode)
│   │   ├── rag_maintenance.py       # 刪除 + 已索引列表
│   │   ├── folder.py / file.py      # Folder / File adapter
│   │   └── model.py                 # Adapter-level pydantic types
│   ├── domain/
│   │   ├── exceptions.py            # DomainException 家族(translate 成 HTTP status)
│   │   └── rag/
│   │       ├── hierarchical_indexer.py    # 索引編排(context→hierarchy→embed→寫入)
│   │       ├── document_loader.py         # 檔案 → Document 三路載入(文字/docling/fallback)
│   │       ├── leaf_splitter.py           # 結構切分(表格保留)+ token budget refine
│   │       ├── hierarchy.py               # Leaf↔parent 樹狀結構建構
│   │       ├── docling_loader.py          # Docling 包裝 + Whisper ASR + hallucination 防禦
│   │       ├── agentic_handlers.py        # MCP 三模式(search/list/read)實際邏輯
│   │       ├── folder_acl.py              # token↔folder 權限檢查(隔離在 DB 查詢層)
│   │       ├── dto.py                     # 跨層共用純資料型別(FileRequest 等)
│   │       ├── context_generator.py       # LLM-based contextual prefix 生成
│   │       ├── index_service.py           # FileIndex DB helpers
│   │       ├── index_job_manager.py       # 背景 job 控管(lifecycle / persistence / SSE)
│   │       ├── query_engine.py            # Hybrid retrieval(vector + BM25)
│   │       ├── auto_merging.py            # Leaf 命中後 merge 回 parent
│   │       ├── reranker.py                # Cross-encoder rerank
│   │       ├── vector_store_manager.py    # PGVector store 管理
│   │       └── chunk_lookup.py            # Chunk 查表
│   ├── fastmcp_tools/                     # MCP 工具註冊(per-folder + global)
│   ├── infrastructure/
│   │   └── cache/                         # Query / folder cache
│   ├── storage/file_storage.py            # 檔案存放(token/folder 隔離)
│   ├── config/                            # YAML 載入 + Pydantic models
│   ├── utils/                             # runtime_paths / db_bootstrap
│   └── log.py                             # loguru 設定
├── scripts/                               # 一次性測試 / 工具腳本
├── storage/                               # ★ 上傳檔案落地(gitignored)
├── ROADMAP.md                             # 開發進度:已交付 / 規劃中
└── tests/                                 # pytest 單元測試(零外部依賴,見 docs/testing/)
```

---

## 開發 / 除錯

### 看背景 job 進度(SSE)

```bash
TOKEN=<your-bearer-token>
curl -sN -H "Authorization: Bearer $TOKEN" \
  "http://localhost:5031/api/rag/files/1/index/jobs/<job_id>/events"
```

每次 state 改變(progress / file done / status transition)會推一個 `data: {...}` 事件。
連線會在 job 達到 terminal status 後自動關閉。

### 看單檔目前狀態

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://localhost:5031/api/rag/file/<file_id>/index/status"
```

回傳 `indexed` / `indexing` / `queued` / `failed` / `not_indexed` 其一,含 progress 跟錯誤訊息。

### 重新索引(skip_existing=false)

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  "http://localhost:5031/api/rag/files/<folder_id>/index?skip_existing=false"
```

若內容沒變(content_hash 相同)會 idempotent 跳過實際 embed,只需幾十 ms。

### 看 Job 持久化(後端 DB)

```sql
SELECT job_id, folder_id, status, total_files, processed_files, last_updated_at
FROM "IndexJobs"
ORDER BY started_at DESC
LIMIT 10;
```

---

## 測試 (Testing)

測試方法論參照 [FuSa Group《軟體測試方法論完整指南》](https://fsg.tw/software-testing-methodologies-guide-a-high-level-overview/),
完整流程文件見 [`docs/testing/`](docs/testing/README.md)(測試計畫、策略、REQ → TC → pytest 可追溯性)。

```bash
# 單元測試(零外部依賴:不需 DB / vLLM / GPU / 網路)
uv run --no-sync pytest

# 附覆蓋率(低於 pyproject fail_under 門檻會以非零退出)
uv run --no-sync pytest --cov --cov-report=term-missing
```

> GPU 機器請保留 `--no-sync`(裸 `uv run` 的隱式 sync 會把 torch 換成非 GPU group 版本);
> 詳見 [`docs/testing/ENVIRONMENTS.md`](docs/testing/ENVIRONMENTS.md)。

| 層級 | 現況 |
|------|------|
| 單元測試 | ✅ `tests/test_*.py` — 純邏輯自動化(分塊 / 快取 / 檔名安全 / 設定 / 認證快取 / schema) |
| 整合 / 系統測試 | 🔶 部署環境驗證(需 pgvector + 模型服務);自動化列於 [ROADMAP](ROADMAP.md) |
| 驗收(RAG 品質) | 🔶 人工驗收;回歸問答集列於 [ROADMAP](ROADMAP.md) |

覆蓋率統計範圍 = 單元測試範疇模組;重外部依賴的 pipeline 模組列於
`pyproject.toml` omit 清單(逐檔附理由),屬整合測試範疇。詳見
[`docs/testing/TEST_PLAN.md`](docs/testing/TEST_PLAN.md) 第 5 節。

---

## 開發進度 / Roadmap

已交付能力與規劃中工作(CI、整合測試自動化、RAG 品質回歸、效能基準)見 [`ROADMAP.md`](ROADMAP.md)。

---

## Control Center(/)— 營運主控台

瀏覽器開 **`http://<host>:<port>/`**(根網址;`/admin` 同頁別名)即進入
全英文營運主控台(免認證 — 內網部署產品決策;API 說明頁由 `/docs` 承擔):

- **Overview**:KPI(folders/files/chunks/jobs)、索引健康分佈、進行中任務
  即時進度、四個模型服務的**即時健康**(端點探測 Online/Offline、ASR 權重 Ready)
- **Folders**:資料夾 CRUD(建立需 owner token — folder 為 token-scoped)、
  檔案上傳(自動索引)/下載/刪除、增量索引與全量重建、鑽取檔案級索引狀態與失敗原因
- **Index Jobs**:近期 job 進度/訊息/起迄,執行中可取消
- **Search Playground**:對 folder 跑即時檢索,看命中 chunk / rerank 分數 /
  頁碼溯源(demo RAG 品質)
- **檔案階段可視**:上傳後檔案清單即時顯示 Parsing→Context→Embedding→Saving
  (1 秒輪詢)+ 迷你進度條;indexed 檔可檢視切好的 chunks(含 CR 前綴)
- **總覽 sparkline**:近 7 天索引量趨勢;KPI 卡可點擊導航
- **危險操作防護**:刪 folder 需輸入名稱確認
- **Settings**:模型與檢索參數執行期熱改(見下)

管理面 API(`/api/admin/manage/*`)採 **act-as-owner**:以 folder 擁有者
token 呼叫既有 REST 端點函式,job 取消/軟刪/向量清理等安全邏輯零重複。

| 可熱改 | 生效方式 |
|---|---|
| LLM / Rerank / ASR 模型與端點 | 即改即生效(接手下一個請求) |
| 檢索參數(top_k / cutoff / α / auto-merging / 展開鄰居) | 即改即生效 |
| Contextual Retrieval 開關與行為 | 下一個索引 job 生效 |
| **Embedding 模型/維度/前綴** | 生效但**附警告**:向量空間不相容,既有 folder 需重建索引 |
| DB / port / auth 等基礎設施 | 不開放熱改(白名單外一律拒絕),重啟才能改 |

- 改動持久化到 DB(`RuntimeSettings` 表),**重啟不丟**;「已覆寫」徽章一鍵還原 config.yaml 出廠值
- 每個模型服務區塊有「測試連線」(打 `{base_url}/models` 驗證可達)
- API:`GET/PUT /api/admin/settings`、`DELETE /api/admin/settings/{path}`、`POST /api/admin/probe`

## 配置覆寫優先順序

1. **環境變數**(`RAG_INDEXING_CONCURRENCY`、`RAG_PER_FILE_TIMEOUT_SECONDS`、`RAG_JOB_TIMEOUT_SECONDS` 等)
2. **`config.yaml` `${VAR:-default}` 展開**(部署 secret 注入)——⚠️ 展開邏輯尚未實作,
   佔位符會以字面進 config;已登記 [ref-003](docs/testing/defect-reports/ref-003.md),實作前請直接寫值
3. **RuntimeSettings 執行期覆寫**(/admin 熱改,DB 持久化 — 疊在 yaml 之上)
4. **`config.yaml` 寫死值**
5. **程式碼內 fallback**

---

## 授權 / License

[MIT](LICENSE)
