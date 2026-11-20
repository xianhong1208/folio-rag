# Metrics

[繁體中文](metrics.zh-TW.md)

Folio serves **`GET /metrics`** in the Prometheus text format: HTTP requests, retrieval stage
latency, MCP tool calls, index jobs, model endpoint calls, the revocation feed, database pools
and the version. Point Prometheus (or anything that reads the format, such as Grafana Agent,
VictoriaMetrics or the OpenTelemetry Collector) at it.

## Turning it on and protecting it

It is on by default (`config/system.yaml`):

```yaml
metrics:
  enabled: true                 # false: /metrics is not served (404)
  token: "${METRICS_TOKEN:-}"   # set METRICS_TOKEN in .env to require a bearer token
```

- **With `METRICS_TOKEN` set**, a scrape must send `Authorization: Bearer <token>`, from any
  address, this host included. A missing or wrong token gets `401`. The token is compared in
  constant time and never logged.
- **Without it**, only a client on the same host may scrape: a direct connection from a loopback
  address (`127.0.0.0/8`, `::1`). Anyone else gets `403`. A request that carries
  `X-Forwarded-For`, `Forwarded` or `X-Real-IP` came through a proxy and is refused too, even
  when the proxy connects from `127.0.0.1`. So a Prometheus on the same machine works with no
  setup, and anything across the network (another host, another container) needs the token.

The metrics hold no folder, file, user or query names (see [Labels](#labels)), but they show
traffic, error rates and the version. Keep `/metrics` off the public side of your reverse proxy
even with a token. With nginx:

```nginx
location = /metrics {
    allow 10.0.0.0/8;     # the Prometheus network
    deny  all;
    proxy_set_header X-Forwarded-For $remote_addr;   # without it every scrape looks local
    proxy_pass http://127.0.0.1:5039;
}
```

A proxy on the same host that forwards `/metrics` **without** any forwarding header makes every
request look local, so set `METRICS_TOKEN` whenever a proxy passes `/metrics` on. With Caddy,
`@metrics path /metrics` plus `respond @metrics 403` on the public site, and scrape Folio
directly on its own port from inside the network.

## Scrape config

```yaml
scrape_configs:
  - job_name: folio
    scrape_interval: 30s
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/folio-metrics-token   # the METRICS_TOKEN value
    static_configs:
      - targets: ["folio.internal:5039"]
```

Folio runs as one process (one uvicorn worker), so one target is the whole server. Counters
start from zero when the server restarts; `rate()` and `increase()` handle that.

## What is measured

Durations are in seconds. Histograms come with `_bucket`, `_sum` and `_count` series.

| Metric | Type | Labels | What |
|---|---|---|---|
| `folio_http_requests_total` | counter | `route`, `method`, `status` | Every HTTP request, by route template |
| `folio_http_request_duration_seconds` | histogram | `route`, `method` | Time until the response starts (headers sent). Streams (job progress, MCP streamable HTTP) count until they open, not while they stay open |
| `folio_retrieval_stage_duration_seconds` | histogram | `stage` | Each retrieval stage of a search: `embed`, `dense`, `segment`, `bm25`, `fuse`, `rerank`, `prefetch`, `merge`, `expand` (the same timings as the query trace). Dense and BM25 run concurrently, so stages do not add up to the total |
| `folio_retrieval_search_duration_seconds` | histogram | | One hybrid search: candidates, fusion and rerank |
| `folio_mcp_tool_calls_total` | counter | `mode`, `outcome` | MCP tool calls. `mode`: `search` / `list` / `read` (the folder tools), `find` / `find_list` (`folio_find` with / without a query). `outcome`: `ok`, `invalid_request` (e.g. a missing `file_id`), `unauthorized`, `forbidden`, `error` |
| `folio_mcp_tool_call_duration_seconds` | histogram | `mode` | MCP tool call duration |
| `folio_index_jobs_submitted_total` | counter | `kind` | Index jobs created, started at once or queued. `kind`: `index` or `rebuild_fts` |
| `folio_index_jobs_started_total` | counter | `kind` | Jobs that began running |
| `folio_index_jobs_finished_total` | counter | `kind`, `status` | Jobs that reached a final status: `succeeded`, `partial_success`, `failed`, `cancelled`, `interrupted` (stopped by a server restart, counted at the next start) |
| `folio_index_job_duration_seconds` | histogram | `kind`, `status` | From a job's creation to its final status, time queued included |
| `folio_index_jobs_active` | gauge | `kind`, `status` | Jobs `pending` or `running` now |
| `folio_index_files_total` | counter | `status` | Files processed by index jobs: `success`, `failed`, `skipped` |
| `folio_model_requests_total` | counter | `role`, `outcome` | HTTP requests to model endpoints. `role`: `embedding`, `llm` (answers), `contextual_llm` (contextual prefixes), `rerank`, `asr`, `vision`. `outcome`: `ok`, `client_error` (4xx), `server_error` (5xx), `error` (timeout, refused or reset connection). Each retry is one request |
| `folio_model_request_duration_seconds` | histogram | `role` | Model endpoint request duration |
| `folio_revocation_feed_enabled` | gauge | | 1 when the MCP Gate revocation feed is configured |
| `folio_revocation_feed_last_success_timestamp_seconds` | gauge | | Unix time of the last successful poll (0 = none yet) |
| `folio_revocation_feed_consecutive_failures` | gauge | | Failed polls since the last success |
| `folio_revocation_feed_stale` | gauge | | 1 when the revoked list is older than its stale limit |
| `folio_revocation_revoked_tokens` | gauge | | Revoked, unexpired tokens held in memory |
| `folio_db_pool_size`, `folio_db_pool_checked_out`, `folio_db_pool_checked_in`, `folio_db_pool_overflow` | gauge | `pool` | The shared connection pools: `sync` (ORM) and `async` (the query path) |
| `folio_build_info` | gauge | `version`, `profile` | Always 1; the labels carry the version and `rag.profile` |
| `process_*`, `python_info` | | | CPU, resident memory, open files and start time of the process; the Python version |

Not measured: models that run in-process (`rag.profile: local` / `hybrid`, e.g. a HuggingFace
embedding model or the CKIP segmenter) make no HTTP request, so `folio_model_*`
only covers endpoints; the console's model probes are not counted either. With
`database.vector_store_engines: per_folder` each folder has its own small pool, which is not
reported.

### Labels

No label carries a folder name or id, a file name, a user, a token or a query: those would grow
the number of series without limit and expose what people store and search for. Routes are
templates (`/api/admin/folders/{folder_id}`, never the id). A request no route matched is
`unmatched`, the console's static files are `/admin/*`, and MCP traffic is `/mcp`. Every other
label takes a fixed set of values; anything else is reported as `other`.

## Example queries

```promql
# Requests per second by route, and the share that failed (5xx)
sum by (route) (rate(folio_http_requests_total[5m]))
sum(rate(folio_http_requests_total{status=~"5.."}[5m])) / sum(rate(folio_http_requests_total[5m]))

# 95th percentile search latency, and per stage
histogram_quantile(0.95, sum by (le) (rate(folio_retrieval_search_duration_seconds_bucket[5m])))
histogram_quantile(0.95, sum by (stage, le) (rate(folio_retrieval_stage_duration_seconds_bucket[5m])))

# Model endpoint error ratio by role
sum by (role) (rate(folio_model_requests_total{outcome!="ok"}[5m]))
  / sum by (role) (rate(folio_model_requests_total[5m]))

# Index jobs that failed in the last day
sum by (kind) (increase(folio_index_jobs_finished_total{status="failed"}[1d]))
```

Alerts worth having:

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

**Grafana.** Add Prometheus as a data source and build panels from the queries above: a
time series of request rate by `route`, a heatmap of `folio_retrieval_search_duration_seconds_bucket`
(format "Heatmap", `sum by (le)`), a bar gauge of `folio_retrieval_stage_duration_seconds` p95 by
`stage`, a stat panel for `folio_index_jobs_active`, and a table of `folio_build_info` to see
which version is running.
