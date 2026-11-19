# 監控指標

[English](metrics.md)

Folio 以 Prometheus 文字格式提供 **`GET /metrics`**:HTTP 請求、檢索各階段延遲、MCP 工具呼叫、
索引工作、模型端點呼叫、撤銷清單同步、資料庫連線池與版本。讓 Prometheus(或任何讀得懂這個格式的
工具,例如 Grafana Agent、VictoriaMetrics、OpenTelemetry Collector)來抓取即可。

## 開啟與保護

預設開啟(`config/system.yaml`):

```yaml
metrics:
  enabled: true                 # false:不提供 /metrics(404)
  token: "${METRICS_TOKEN:-}"   # 在 .env 設定 METRICS_TOKEN 即需要 bearer token
```

- **設定了 `METRICS_TOKEN`** 時,抓取必須帶 `Authorization: Bearer <token>`,不論從哪個位址來,
  本機也一樣。沒帶或錯誤一律回 `401`。token 以固定時間比對,也不會寫進 log。
- **沒有設定**時,只有同一台機器上的程式可以抓取:直接從 loopback 位址(`127.0.0.0/8`、`::1`)
  連進來的請求。其他來源一律回 `403`。帶有 `X-Forwarded-For`、`Forwarded` 或 `X-Real-IP` 的請求
  是經過代理轉送的,即使代理本身從 `127.0.0.1` 連進來也會被拒絕。所以同一台機器上的 Prometheus
  不用任何設定就能抓,跨網路(另一台主機、另一個容器)則需要 token。

指標內容不含任何資料夾、檔案、使用者或查詢名稱(見 [標籤](#標籤)),但看得到流量、錯誤率與版本。
即使設了 token,也請在反向代理的公開端擋掉 `/metrics`。nginx 範例:

```nginx
location = /metrics {
    allow 10.0.0.0/8;     # Prometheus 所在的網段
    deny  all;
    proxy_set_header X-Forwarded-For $remote_addr;   # 少了這行,每次抓取看起來都像本機
    proxy_pass http://127.0.0.1:5039;
}
```

同一台機器上的代理若轉送 `/metrics` 時**沒有**帶任何轉送標頭,每個請求看起來都像來自本機,
所以只要有代理轉送 `/metrics`,就請設定 `METRICS_TOKEN`。Caddy 則在公開站台用 `@metrics path /metrics`
加 `respond @metrics 403`,再從內網直接抓 Folio 自己的埠。

## 抓取設定

```yaml
scrape_configs:
  - job_name: folio
    scrape_interval: 30s
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/folio-metrics-token   # METRICS_TOKEN 的值
    static_configs:
      - targets: ["folio.internal:5039"]
```

Folio 是單一程序(一個 uvicorn worker),一個 target 就是整台伺服器。伺服器重啟後計數器從零開始,
`rate()` 與 `increase()` 會自動處理。

## 量測項目

時間單位為秒。直方圖(histogram)會有 `_bucket`、`_sum`、`_count` 三組序列。

| 指標 | 類型 | 標籤 | 內容 |
|---|---|---|---|
| `folio_http_requests_total` | counter | `route`、`method`、`status` | 每個 HTTP 請求,依路由樣板分類 |
| `folio_http_request_duration_seconds` | histogram | `route`、`method` | 到回應開始(送出標頭)為止的時間。串流(工作進度、MCP streamable HTTP)只算到建立為止,不算持續開著的時間 |
| `folio_retrieval_stage_duration_seconds` | histogram | `stage` | 一次搜尋的各檢索階段:`embed`、`dense`、`segment`、`bm25`、`fuse`、`rerank`、`prefetch`、`merge`、`expand`(與查詢追蹤的時間相同)。dense 與 BM25 同時進行,所以各階段加總不等於總時間 |
| `folio_retrieval_search_duration_seconds` | histogram | | 一次混合搜尋:候選、融合與重排序 |
| `folio_mcp_tool_calls_total` | counter | `mode`、`outcome` | 各資料夾 MCP 工具的呼叫。`mode`:`search` / `list` / `read`。`outcome`:`ok`、`invalid_request`(例如缺少 `file_id`)、`unauthorized`、`forbidden`、`error` |
| `folio_mcp_tool_call_duration_seconds` | histogram | `mode` | MCP 工具呼叫時間 |
| `folio_index_jobs_submitted_total` | counter | `kind` | 建立的索引工作(立即開始或排隊)。`kind`:`index` 或 `rebuild_fts` |
| `folio_index_jobs_started_total` | counter | `kind` | 開始執行的工作 |
| `folio_index_jobs_finished_total` | counter | `kind`、`status` | 進入最終狀態的工作:`succeeded`、`partial_success`、`failed`、`cancelled`、`interrupted`(被伺服器重啟中斷,於下次啟動時計入) |
| `folio_index_job_duration_seconds` | histogram | `kind`、`status` | 從建立到最終狀態的時間,含排隊時間 |
| `folio_index_jobs_active` | gauge | `kind`、`status` | 目前 `pending` 或 `running` 的工作數 |
| `folio_index_files_total` | counter | `status` | 索引工作處理的檔案:`success`、`failed`、`skipped` |
| `folio_model_requests_total` | counter | `role`、`outcome` | 對模型端點的 HTTP 請求。`role`:`embedding`、`llm`(回答)、`contextual_llm`(上下文前綴)、`rerank`、`asr`、`vision`。`outcome`:`ok`、`client_error`(4xx)、`server_error`(5xx)、`error`(逾時、連線被拒或中斷)。每次重試各算一次 |
| `folio_model_request_duration_seconds` | histogram | `role` | 模型端點請求時間 |
| `folio_revocation_feed_enabled` | gauge | | 有設定 MCP Gate 撤銷清單同步時為 1 |
| `folio_revocation_feed_last_success_timestamp_seconds` | gauge | | 上次成功輪詢的 Unix 時間(0 表示尚未成功) |
| `folio_revocation_feed_consecutive_failures` | gauge | | 上次成功後連續失敗的輪詢次數 |
| `folio_revocation_feed_stale` | gauge | | 撤銷清單超過過期門檻時為 1 |
| `folio_revocation_revoked_tokens` | gauge | | 記憶體中已撤銷且未到期的 token 數 |
| `folio_db_pool_size`、`folio_db_pool_checked_out`、`folio_db_pool_checked_in`、`folio_db_pool_overflow` | gauge | `pool` | 共用連線池:`sync`(ORM)與 `async`(查詢路徑) |
| `folio_build_info` | gauge | `version`、`profile` | 固定為 1;標籤帶版本與 `rag.profile` |
| `process_*`、`python_info` | | | 程序的 CPU、常駐記憶體、開啟檔案數與啟動時間;Python 版本 |

不計入:在程序內執行的模型(`rag.profile: local` / `hybrid`,例如 HuggingFace 嵌入模型或 CKIP
斷詞模型)不發 HTTP 請求,所以 `folio_model_*` 只涵蓋端點;主控台的模型探測也不計入。
設定 `database.vector_store_engines: per_folder` 時,每個資料夾各有一個小連線池,這些不會回報。

### 標籤

沒有任何標籤帶資料夾名稱或 id、檔名、使用者、token 或查詢內容:這些會讓序列數無限增長,也會洩漏
大家存了什麼、搜尋了什麼。路由一律用樣板(`/api/admin/folders/{folder_id}`,不是實際 id)。沒有
對應路由的請求記為 `unmatched`,主控台的靜態檔案記為 `/admin/*`,MCP 流量記為 `/mcp`。其餘標籤只會
是固定的幾個值,其他值一律記為 `other`。

## 查詢範例

```promql
# 各路由每秒請求數,以及失敗(5xx)比例
sum by (route) (rate(folio_http_requests_total[5m]))
sum(rate(folio_http_requests_total{status=~"5.."}[5m])) / sum(rate(folio_http_requests_total[5m]))

# 搜尋延遲的第 95 百分位,以及各階段
histogram_quantile(0.95, sum by (le) (rate(folio_retrieval_search_duration_seconds_bucket[5m])))
histogram_quantile(0.95, sum by (stage, le) (rate(folio_retrieval_stage_duration_seconds_bucket[5m])))

# 各角色模型端點的錯誤比例
sum by (role) (rate(folio_model_requests_total{outcome!="ok"}[5m]))
  / sum by (role) (rate(folio_model_requests_total[5m]))

# 過去一天失敗的索引工作
sum by (kind) (increase(folio_index_jobs_finished_total{status="failed"}[1d]))
```

值得設定的告警:

```yaml
- alert: FolioRevocationFeedStale
  expr: folio_revocation_feed_enabled == 1 and folio_revocation_feed_stale == 1
  for: 5m
- alert: FolioModelEndpointFailing
  expr: sum by (role) (rate(folio_model_requests_total{outcome=~"server_error|error"}[5m])) > 0.1
  for: 10m
- alert: FolioDbPoolExhausted
  expr: folio_db_pool_checked_out >= folio_db_pool_size and folio_db_pool_overflow > 0
  for: 5m
```

**Grafana。** 把 Prometheus 加為資料來源,用上面的查詢建立面板:依 `route` 的請求率時間序列、
`folio_retrieval_search_duration_seconds_bucket` 的熱圖(格式選 "Heatmap",`sum by (le)`)、依 `stage`
的 `folio_retrieval_stage_duration_seconds` p95 長條圖、`folio_index_jobs_active` 的數值面板,以及
`folio_build_info` 的表格,看目前跑的是哪個版本。
