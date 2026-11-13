# 設定

[English](configuration.md)

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
| `auth.revocation.*` | 停用 | MCP Gate revocation feed,讓被撤銷的 token 在到期前就失效——見 [Token 撤銷](auth.zh-TW.md#token-撤銷)。 |
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
| `rag.asr.provider` | `fireredasr` | 音訊轉錄:`fireredasr`（本地繁中）· `docling-whisper` · `openai-compatible`。`rag.asr.enabled: false` 時音檔會以明確訊息標為失敗(不做任何轉錄)。`fireredasr` 與 `docling-whisper` **將在 1.5 移除**,請改用 `openai-compatible`([詳見](api.zh-TW.md#15-移除內建-ocr語音轉文字與-v1-媒體端點))。 |
| `rag.retrieval.segmenter` | `ckip`(`profile: cloud` 為 `bigram`) | 關鍵字檢索的中文斷詞器:`ckip` · `jieba` · `bigram`（見[運作原理](how-it-works.zh-TW.md)的「中文斷詞」）。 |
| `rag.docling.ocr_enabled` | `true` | 掃描 PDF 與圖片的 OCR。設 `false` 會真的關掉(重啟後生效);沒有文字層的掃描檔會以明確訊息失敗。內建 OCR **將在 1.5 移除**,請改用搭配 `rag.parsing.vision` 的雲端解析([詳見](api.zh-TW.md#15-移除內建-ocr語音轉文字與-v1-媒體端點))。 |
| `rag.contextual_retrieval.max_doc_chars` | `60000` | 每個 chunk 交給 LLM 的文件文字上限。LLM 讀的是大綱加上 chunk 所在章節,以 chunk 為中心最多 12,000 字元。 |
| `rag.retrieval.*` | — | Top-K、相似度門檻、BM25 候選數、`hybrid_fusion`（`rrf`/`concat`）、auto-merging。可從主控台即時調,或用 **High Precision / Balanced / High Recall** 預設。 |
| `rag.profile` | `local` | `local` · `hybrid` · `cloud` —— 見[雲端 profile](#雲端-profile)。 |
| `rag.<role>.{timeout,max_retries}` | 各角色不同 | 每個模型角色的請求逾時與重試（429 / 5xx / 逾時,含退避）。 |

GPU 選擇是 `uv sync --extra local --group {rocm-r714|cuda|cpu}` 的取捨 —— 見 `pyproject.toml` 開頭,每個 group 對應到各自的 PyTorch index。`cuda` 安裝 CUDA 13.0 版本,需要 NVIDIA driver 580 以上。

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
