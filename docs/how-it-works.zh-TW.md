# Folio 的運作原理

[English](how-it-works.md)

```
┌────────────────┐     Folio_<folder> 工具呼叫（Bearer JWT）    ┌───────────────────────────────┐
│    MCP 用戶端   │ ───────────────────────────────────────────▶ │  Folio（resource server）      │
│  Claude / IDE  │ ◀─────────────────────────────────────────── │  /mcp · 依使用者過濾的資料夾  │
└───────┬────────┘        auto-merge 後的段落 + 分數            └───────────────┬───────────────┘
        │ OAuth 流程                                                             │ 離線驗證 JWT
        ▼                                                                        ▼
  ┌─────────────┐                              檢索管線（每次查詢）
  │  MCP Gate │  RS256 JWKS                  ingest ─▶ chunk ─▶ embed ─▶ pgvector ─▶ 混合檢索
  │ OAuth 2.1 AS│  （離線驗證）                (Docling) (leaf+parent) (e5·1024d)   (向量 + BM25/CKIP)
  └─────────────┘                                   └▶ auto-merge + 擴展 ─▶ rerank ─▶ gpt-oss-120b
```

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

## 檢索細節

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

## 量測檢索品質

檢索與切塊的調整以數字判斷,不靠肉眼。`evals/` 有一套公開的 golden set:19 份文件(台灣法規、美國政府網頁、虛構公司文件、含表格的 PDF、合成的繁中會議逐字稿)上的 205 題,依題型與語言分層,約一成是無解題;另有 20 題超出範圍的問題,用來校準相關度門檻。評估工具透過 agent 實際使用的同一個 MCP 搜尋,對執行中的服務評分:

```bash
.venv/bin/python scripts/eval_retrieval.py --token-file <pat.json> --runs 2
```

它依題型回報 Recall@5/10/20、MRR、nDCG、回應 token 數、重複率與延遲,以及無解題的誤收率和 `min_relevance` 門檻掃描,結果寫到 `evals/baselines/rag-<日期>-<標籤>.{json,md}`。已提交的 baseline 記錄了每次改動前後的數字。token 檔格式、語料怎麼建、各指標的意義見 [`evals/README.md`](../evals/README.md)。

## 功能細節

- **依章節對齊的階層切塊 + auto-merging 檢索。** 索引時把文件切成小的 **leaf**（精準匹配）與大的 **parent**（完整脈絡）,兩者都不會跨越章節。每個 leaf 帶有**麵包屑**（檔名 › 標題路徑,xlsx 的工作表名稱也算在內）,結構相同的兩份文件(兩本產品手冊、兩張價目表)也能分得開。當同一段落的多個 leaf 被命中時,回傳整個 parent 而非碎片;單一 leaf 命中則向鄰居擴展,答案不會失去語意連貫性。
- **混合搜尋,為中文調校。** 密集向量檢索（pgvector）與稀疏 **BM25** 以 Reciprocal Rank Fusion 融合,讓關鍵字訊號真的影響排名。中文在進 Postgres 全文索引前先斷詞 —— 預設用 **CKIP**（中研院）,也可選 jieba 或零相依的 bigram —— 無空格的中文會被真正 tokenize,而不是塌成單一詞。
- **Cross-encoder 重排。** 以 `bge-reranker-v2-m3` / `mxbai-rerank-base-v2` 對融合後的候選重新評分,提升 Top-K 精度。
- **Contextual Retrieval。** 索引時由 LLM 依文件大綱與 chunk 所在章節,以文件本身的語言為每個 chunk 寫一句脈絡說明（Anthropic 建議的做法）。這句說明與麵包屑只用來*找到* chunk(embedding、BM25、rerank);結果一律顯示文件原文。
- **增量重建索引。** 重新索引一個檔案時,沒變的 chunk 沿用原本的脈絡說明與 embedding,只改一個章節大約只花一次 LLM 呼叫與一次 embedding。檔案的 chunk 在單一交易內替換 —— 過程中不會從搜尋結果消失。
- **每資料夾一個 MCP 工具,依使用者隔離。** 每個資料夾註冊成 `Folio_<folder>` 工具;呼叫者看到的工具清單會依其 token 所屬的使用者過濾 —— 你只看得到自己擁有的資料夾。歸屬以 token 的 `sub`(MCP Gate 使用者)為 key,所以同一位使用者的每一把 token —— personal access token 與 OAuth 登入皆然 —— 看到的是同一組資料夾(見[資料夾歸屬](auth.zh-TW.md#資料夾歸屬))。工具名稱一律符合所有 MCP 用戶端的限制(`[A-Za-z0-9_-]`、最長 64 字元):含空白、中日韓文字或其他符號的資料夾名稱會變成 `Folio_<ASCII 部分>_<8 碼雜湊>`,每次都是同一個名稱。
- **OAuth 2.1 resource server。** 一個 `RemoteAuthProvider` + `JWTVerifier` 保護 `/mcp`;token 用 MCP Gate 的 JWKS 在本地離線驗證（RS256),並綁定本伺服器的 audience,別台的 token 在這裡無效。
- **文件 + 音訊匯入。** [Docling](https://github.com/DS4SD/docling) 解析 PDF、Office（含舊版 `.doc` / `.xls` / `.ppt`、RTF 與 Excel 範本）、OpenDocument、HTML 與 MHTML 網頁封存檔、Markdown、EPUB、電子郵件與圖片（含 OCR）;舊版 Office 與 RTF 經 LibreOffice 轉換,需安裝 `soffice`。**FireRedASR** 在本地做繁體中文語音轉錄（或任何 OpenAI 相容的 `/v1/transcriptions`）。
- **內建評測 harness。** 從你自己的 chunk 自動生成問題,對正式檢索管線(與 `/api/rag/query` 同一路徑)以及 `vector` / `bm25` / `rrf` / `rerank`(全部走正式檢索路徑)評 **Recall@k · MRR · nDCG@k**,讓你針對真實語料調校檢索。
- **Retrieval Terminal 主控台。** `/admin` 的 React 單頁主控台:管線總覽、資料夾與檔案、會顯示**逐訊號檢索軌跡**的 Search Playground、即時索引任務串流、查詢分析、評測 harness、即時設定、健康狀態與稽核時間軸。
- **跑在你的 GPU 上。** PyTorch 以 PEP 735 dependency group 依平台選擇（`rocm-r714` / `cuda` / `cpu`）—— AMD ROCm、NVIDIA CUDA 或純 CPU。
