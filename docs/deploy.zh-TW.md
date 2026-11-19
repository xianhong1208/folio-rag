# 部署 Folio

[English](deploy.md)

正式上線要知道的事：Docker 映像檔與 compose 檔、模型端點、MCP Gate、反向代理與 HTTPS、備份和升級。

- [Docker 映像檔](#docker-映像檔)
- [Docker Compose](#docker-compose)
- [模型端點：Forgeyard 或 vLLM](#模型端點forgeyard-或-vllm)
- [MCP Gate：issuer 與 audience](#mcp-gateissuer-與-audience)
- [反向代理與 HTTPS](#反向代理與-https)
- [設定](#設定)
- [備份](#備份)
- [升級](#升級)
- [從原始碼執行](#從原始碼執行)

## Docker 映像檔

`tonyhuang1101/folio`（Docker Hub，`linux/amd64` 與 `linux/arm64`，標籤 `X.Y.Z`、`X.Y`、`latest`）是
**輕量版**映像檔：`rag.profile: cloud`，所有模型都是 HTTP 端點，容器裡不載入任何模型。沒有 torch、沒有
GPU 套件、不下載模型。

| | |
|---|---|
| 模型 | 嵌入、LLM、重排序、讀 PDF 與圖片的視覺模型，以及（選用的）語音轉文字：全部是 OpenAI 相容端點（[Forgeyard](https://github.com/xianhong1208/forgeyard)、vLLM、Ollama 或雲端服務） |
| 中文關鍵字搜尋 | `jieba` 斷詞（`FOLIO_SEGMENTER`；另一個不需模型的選擇是 `bigram`） |
| 監聽 | 容器內的 `5039`：`/mcp`、主控台 `/admin`、`/health` |
| 執行身分 | uid `10001`，不是 root |
| Volume | `/app/storage`：上傳的檔案與主控台的 session 金鑰 |
| 設定 | `/app/config` 裡的六個檔案，內建在映像檔中；其中 `models.yaml` 是 [`docker/config/models.yaml`](../docker/config/models.yaml)（端點、模型與金鑰都從環境變數來） |
| 日誌 | 只輸出到主控台：`docker logs`（容器裡不寫日誌檔） |
| Migration | 每次啟動都會執行，和 `python main.py` 一樣 |
| 健康檢查 | Docker `HEALTHCHECK` 打 `/health`；`/health/detailed` 檢查資料庫、pgvector 與撤銷清單 |

**映像檔裡沒有**：在程序內執行的模型（CKIP、HuggingFace 嵌入）。
自 1.5 起，任何安裝方式都不再在程序內執行 OCR、版面或語音轉文字模型。要用 `rag.profile: local` 或 `hybrid`，請從原始碼執行（[見下方](#從原始碼執行)）。
沒有 GPU 版映像檔。

**PDF 與圖片**在設定 `VISION_BASE_URL` 與 `VISION_MODEL` 時由視覺模型讀取（[升級到 1.5](migrate-1.5.zh-TW.md#pdf-與圖片)）。
沒有設定時，PDF 從文字層讀取（掃描頁會變成一個標記；完全沒有文字的 PDF 會索引失敗），上傳圖片則以 422 拒絕。

**音訊**由 OpenAI 相容端點轉成文字（[升級到 1.5](migrate-1.5.zh-TW.md#語音轉文字)）。Folio 會先用 `ffmpeg`
重新取樣並切段長音訊，而公開的映像檔為了保持輕量沒有裝 `ffmpeg`：沒有它時，24 MB 以內的檔案整份送出，更大的檔案
會失敗並提示安裝 `ffmpeg`。要處理長錄音，請自行建置含 `ffmpeg` 的映像檔：
`docker build --build-arg INSTALL_FFMPEG=true -t folio:audio .`。再設定 `ASR_ENABLED=true`、`ASR_BASE_URL`
與 `ASR_MODEL`。語音轉文字關閉或沒有端點時，上傳音訊會被拒絕（422）。若端點的憑證來自私有 CA（Forgeyard 自己的
CA），請把 CA 檔以唯讀方式掛載進容器，並把 `ASR_CA_FILE` 指向它。

## Docker Compose

[`docker-compose.yml`](../docker-compose.yml) 會啟動 Folio 和 PostgreSQL + pgvector
（`pgvector/pgvector:pg16`，有健康檢查與具名 volume），也可以選擇一併啟動 MCP Gate。

```bash
git clone https://github.com/xianhong1208/folio-rag.git && cd folio-rag   # 或只複製 docker-compose.yml 與 .env.example
cp .env.example .env
```

在 `.env` 裡取消註解並至少設定：

```bash
POSTGRES_PASSWORD=<一組夠長的隨機密碼>
# 模型（見下一節）
EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
LLM_BASE_URL=http://host.docker.internal:8001/v1
LLM_MODEL=Qwen/Qwen3-4B
RERANK_BASE_URL=http://host.docker.internal:8002/v1
RERANK_MODEL=BAAI/bge-reranker-v2-m3
VISION_BASE_URL=http://host.docker.internal:8003/v1   # 選用：不設時 PDF 從文字層讀取
VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct
# MCP Gate（見下方「MCP Gate」）
MCP_GATE_URL=http://localhost:4568
FOLIO_AUDIENCE=http://localhost:5039/mcp
```

```bash
docker compose pull
docker compose up -d
docker compose ps                              # folio 與 db：healthy
curl -s localhost:5039/health/detailed         # "status": "healthy"
```

主控台在 `http://localhost:5039/admin`，用 MCP Gate 發的個人存取權杖登入。

compose 檔替你處理的事：`DATABASE_URL` 指向 `db` 服務（`.env` 裡的 `DATABASE_URL` 不會被使用）、
`MCP_GATE_URL` 與 `FOLIO_AUDIENCE` 預設為 `localhost` 位址、`host.docker.internal` 可連到主機。
`FOLIO_VERSION` 用來固定版本（`FOLIO_VERSION=1.5.0`），`FOLIO_PORT` 改主機上的連接埠。資料放在具名
volume：`folio-storage`、`folio-db`，啟用 MCP Gate profile 時還有 `mcp-gate-data` 與 `mcp-gate-secrets`。

## 模型端點：Forgeyard 或 vLLM

每個角色都是 OpenAI 相容端點。相關變數（都寫在 `.env`）：

| 角色 | 端點 | 模型 | 金鑰 | 說明 |
|---|---|---|---|---|
| 嵌入 | `EMBEDDING_BASE_URL` | `EMBEDDING_MODEL`、`EMBEDDING_DIMENSION` | `EMBEDDING_API_KEY` | `POST /embeddings`。維度必須和模型一致；換模型或維度都要重新索引。e5 系列還需要 `EMBEDDING_QUERY_PREFIX="query: "` 與 `EMBEDDING_PASSAGE_PREFIX="passage: "`。 |
| LLM | `LLM_BASE_URL` | `LLM_MODEL` | `LLM_API_KEY` | `POST /chat/completions`：回答、評估，以及索引時替每個 chunk 寫的一句脈絡。 |
| 重排序 | `RERANK_BASE_URL` | `RERANK_MODEL` | `RERANK_API_KEY` | `RERANK_API`：`generic`（`POST /rerank`，vLLM、Forgeyard、Infinity；預設）、`score`（vLLM `/score`）、`cohere` 或 `jina`。`RERANK_ENABLED=false` 關閉。 |
| 視覺 | `VISION_BASE_URL` | `VISION_MODEL` | `VISION_API_KEY` | 以 `POST /chat/completions` 附上圖片，讀取每一頁 PDF 與每張圖片（例如 Qwen3-VL）。選用：沒有設定時 PDF 從文字層讀取，圖片上傳會被拒絕。私有 CA 用 `VISION_CA_FILE`。Office、Markdown、HTML 與純文字檔不需要模型。 |
| 語音轉文字 | `ASR_BASE_URL` | `ASR_MODEL` | `ASR_API_KEY` | `POST /audio/transcriptions`。預設關閉，`ASR_ENABLED=true` 才啟用；長音訊需要含 `ffmpeg` 的映像檔（見上方）。私有 CA 用 `ASR_CA_FILE`。中文轉錄結果會轉成繁體（`rag.asr.chinese_conversion`）。 |

在容器裡，`localhost` 指的是容器自己。同一台機器上的模型伺服器請用 `host.docker.internal`，其他機器
就用它的位址。

**[Forgeyard](https://github.com/xianhong1208/forgeyard)** 用一個 OpenAI 相容網址、一把 API 金鑰提供所有
角色：把每個 `*_BASE_URL` 都設成它（例如 `https://gpu-host:7860/v1`）、每個 `*_API_KEY` 都設成那把金鑰，
`*_MODEL` 則填部署名稱。Forgeyard 預設用自己的 CA 提供 HTTPS，請給容器一份包含它的 CA bundle：

```bash
cat "$(python3 -m certifi)" forgeyard-ca.pem > ca-bundle.pem    # 公開 CA + Forgeyard 的 CA
```

```yaml
# docker-compose.override.yml
services:
  folio:
    volumes:
      - ./ca-bundle.pem:/etc/ssl/folio-ca-bundle.pem:ro
    environment:
      SSL_CERT_FILE: /etc/ssl/folio-ca-bundle.pem
      REQUESTS_CA_BUNDLE: /etc/ssl/folio-ca-bundle.pem
```

**vLLM** 一個連接埠服務一個模型：每個角色各一個 `*_BASE_URL`，例如嵌入用
`http://host.docker.internal:8000/v1`。模型名稱是對外服務的名稱（`--served-model-name`，或 Hugging Face
id）。重排序模型名稱對不上時不會報錯，只會維持原本的順序，請在主控台 *Settings → Model roles → Test*
確認。

金鑰只從環境變數讀取：`profile: cloud` 不接受寫在設定檔裡的金鑰。

## MCP Gate：issuer 與 audience

Folio 是資源伺服器：它用 [MCP Gate](https://github.com/xianhong1208/MCP_Gate) 的簽章金鑰離線驗證 MCP Gate
發的權杖。有兩個值必須**完全一致**（以字串比較：scheme、主機、連接埠、路徑；`localhost` 和 `127.0.0.1`
不一樣），否則每個權杖都會得到 401，而且沒有任何地方會告訴你原因：

| Folio | 必須等於 | MCP Gate 端 |
|---|---|---|
| `MCP_GATE_URL`（`auth.issuer`） | 權杖的 `iss` | `OAUTH_ISSUER`，也就是用戶端連 MCP Gate 的位址 |
| `FOLIO_AUDIENCE`（`auth.audience`） | 權杖的 `aud` | 替 Folio 註冊的服務的 audience（*Services → Register Service*；預設是服務的 MCP 網址） |

兩者都是 MCP 用戶端使用的**對外**位址。Folio 也會從 `MCP_GATE_URL` 抓 MCP Gate 的簽章金鑰（以及撤銷
清單）。如果這個位址在 Folio 容器裡連不到，就把 `MCP_GATE_INTERNAL_URL` 設成連得到的位址：Folio 改從
那裡抓，但 `iss` 仍然拿 `MCP_GATE_URL` 比對。

**MCP Gate 跟 Folio 在同一個 stack**（`mcp-gate` profile，映像檔 `tonyhuang1101/mcp-gate:1.4.0`，有自己的
資料與 secrets volume）：

```bash
# .env
COMPOSE_PROFILES=mcp-gate
MCP_GATE_URL=http://localhost:4568              # MCP Gate 的 OAUTH_ISSUER 會設成同一個值
MCP_GATE_INTERNAL_URL=http://mcp-gate:4568      # 在 stack 內部，MCP Gate 就是 mcp-gate 這個服務
FOLIO_AUDIENCE=http://localhost:5039/mcp
MCP_GATE_ADMIN_EMAIL=you@example.com            # 它的擁有者帳號（也可以兩個都留空，改用
MCP_GATE_ADMIN_PASSWORD=<夠強的密碼>             #   `docker compose logs mcp-gate` 裡的 setup token）
```

接著 `docker compose up -d`，打開 `http://localhost:4568` 的 MCP Gate，在 *Services → Register Service*
註冊 Folio：主機 `folio`、連接埠 `5039`、路徑 `/mcp`（MCP Gate 用這個位址檢查服務並列出工具），
**OAuth audience** 填 `http://localhost:5039/mcp`，和 `FOLIO_AUDIENCE` 相同。替這個服務發一個個人存取
權杖，用它登入 Folio 的主控台。MCP Gate 不讀 Folio 的 `.env`，它的設定寫在 compose 檔裡
（`ALLOWED_HOSTS=mcp-gate` 讓 Folio 能用服務名稱抓金鑰）。

**MCP Gate 在別處**：把 `MCP_GATE_URL` 設成它的 `OAUTH_ISSUER`，`MCP_GATE_INTERNAL_URL` 留空，前提是容器
連得到那個位址。

**撤銷清單**：沒有它的話，在 MCP Gate 撤銷的權杖在 Folio 這邊會一直有效到過期為止。設定方式見
[驗證：權杖撤銷](auth.zh-TW.md)，並把 `FOLIO_REVOCATION_ENABLED`、`FOLIO_REVOCATION_CLIENT_ID` 和
`FOLIO_REVOCATION_CLIENT_SECRET` 寫進 `.env`。

## 反向代理與 HTTPS

在伺服器上，請把 Folio（和 MCP Gate）放在負責 HTTPS 的反向代理後面，容器的連接埠只對 `127.0.0.1` 開放
（compose 檔裡寫 `"127.0.0.1:5039:5039"`）。這時所有出現位址的地方都要改成對外位址：

```bash
MCP_GATE_URL=https://gate.example.com           # = MCP Gate 的 OAUTH_ISSUER
FOLIO_AUDIENCE=https://folio.example.com/mcp    # = 在 MCP Gate 註冊的 audience
```

MCP Gate 裡註冊的 audience 也要跟著改：替舊 audience 發的權杖會失效。主控台的 session cookie 只在請求
經由 HTTPS 進來時才會加上 `Secure`，所以代理必須傳 `X-Forwarded-Proto`（而且 Folio 要信任它，見下方），
或設定 `auth.console.secure_cookie: true`。

代理必須允許：最大 100 MB 的上傳，以及不經緩衝的長時間回應（MCP 的 streamable HTTP 與索引進度都用
server-sent events）。

**nginx**

```nginx
server {
    listen 443 ssl;
    http2 on;
    server_name folio.example.com;
    ssl_certificate     /etc/letsencrypt/live/folio.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/folio.example.com/privkey.pem;

    client_max_body_size 110m;              # Folio 限制上傳 100 MB

    location / {
        proxy_pass http://127.0.0.1:5039;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;                # server-sent events（MCP、索引進度）
        proxy_read_timeout 3600s;
    }
}
```

**Caddy**（自動申請與更新憑證；預設就不緩衝串流）：

```caddyfile
folio.example.com {
    request_body {
        max_size 110MB
    }
    reverse_proxy 127.0.0.1:5039
}

gate.example.com {
    reverse_proxy 127.0.0.1:4568
}
```

Folio 只信任 `FORWARDED_ALLOW_IPS` 裡的位址送來的 `X-Forwarded-Proto`（這是 uvicorn 的設定，預設
`127.0.0.1`）。在容器裡，代理的連線是從 Docker 網路進來的，所以請在 `.env` 設 `FORWARDED_ALLOW_IPS=*`，
而且只在連接埠只對 `127.0.0.1` 開放時這樣設，因為那時只有代理連得到它。不需驗證的
`/health*` 任何連得到 Folio 的人都能呼叫；請把它放在代理後面、可信任的網路裡。

## 設定

各台機器不同的值都來自環境變數（compose 時就是 `.env`）。所有變數都列在
[`.env.example`](../.env.example)；其中「Docker」一節列的是只有映像檔的 `models.yaml` 才會讀的變數。
各項設定的意義見[設定](configuration.zh-TW.md)。

要改變數沒涵蓋到的設定（chunk 大小、檢索參數、重排序模板），可以用主控台的 *Settings* 頁（存在
資料庫裡），或換掉設定檔：

```bash
docker compose cp folio:/app/config ./config    # 以映像檔的六個檔案為起點
# 編輯 ./config/*.yaml，再掛載：在 docker-compose.yml 取消註解 "- ./config:/app/config:ro"
docker compose up -d
```

`FOLIO_CONFIG_DIR` 可以改指 `/app` 底下的其他目錄（命令列的 `--config-dir` 效果相同）。`FOLIO_LOG_DIR`
為空（映像檔的預設）時日誌只輸出到主控台；設成某個目錄並在那裡掛 volume，就會寫日誌檔。

### 監控指標（Prometheus）

`/metrics` 隨 #11（Folio 1.5）推出。沒有設定 `METRICS_TOKEN` 時，它只回應同一個 loopback 介面上的用戶端；
在 Docker 裡，從主機或其他容器來的抓取都不是 loopback，所以請在 `.env` 設定一個 token（compose 會把它傳進
容器），並讓 Prometheus 帶同一個 token：

```bash
# .env
METRICS_TOKEN=<一串夠長的隨機字串>
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: folio
    authorization:
      credentials: <同一個 METRICS_TOKEN>
    static_configs:
      - targets: ["folio:5039"]     # 從 compose 網路外部則是 <host>:5039
```

設定了 `METRICS_TOKEN` 之後，每次抓取都要帶這個 bearer token，從 loopback 來的也一樣。

## 備份

資料庫和 `storage/` 是一體的：資料庫存資料夾、chunk 與向量，`storage/` 存產生它們的原始上傳檔。請在同
一個時間點一起備份，也一起還原。映像檔用一個指令就能做到（細節見[備份與還原](backup.zh-TW.md)）：

```bash
docker compose exec folio python main.py backup                 # → folio-backups volume（/app/backups）
docker compose exec folio ls -l /app/backups
docker compose cp folio:/app/backups/folio-backup-<time>.tar .  # 複製一份到機器外
```

`main.py backup` 會寫出一個封存檔（`folio-backup-<UTC 時間>.tar`：資料庫的 `pg_dump`、`storage/` 下的檔案，
以及含 checksum 的 manifest）到 `/app/backups`，也就是映像檔的 `database.backup.dir`，compose 把它放在獨立的
`folio-backups` volume。Folio 執行中也可以備份。映像檔內含 PostgreSQL 17 的客戶端工具，比內附的 pg16 伺服器新，
沒有問題。封存檔請另外複製一份到別處：這個 volume 和資料在同一顆磁碟上。

**排程備份：**在 `.env` 設 `FOLIO_BACKUP_ENABLED=true`（再 `docker compose up -d`）。Folio 會每
`FOLIO_BACKUP_INTERVAL_HOURS`（24）小時寫一個封存檔到同一個 volume，保留最新的 `FOLIO_BACKUP_KEEP`（7）份；
還沒有任何封存檔時，第一份在啟動兩分鐘後產生。`/health/detailed` 會顯示最新的封存檔與上一次的結果。

**還原**時 Folio 必須停止（還有其他連線連到資料庫時會拒絕），所以用相同 volume 的一次性容器執行：

```bash
docker compose stop folio
docker compose cp folio-backup-<time>.tar folio:/app/backups/   # 只有封存檔還不在 volume 裡時才需要
docker compose run --rm --no-deps folio python main.py restore /app/backups/folio-backup-<time>.tar --force
docker compose start folio
```

`--force` 會取代已經有資料表的資料庫，以及已經有資料夾的 `storage/`（還原到全新的 stack 時不需要）。原本的檔案
不會被刪除，而是移到 storage volume 裡的 `storage/.folio-work/before-restore-<時間>/`；確認還原後的內容沒問題再
刪掉那個目錄。`restore --check <archive>` 只驗證封存檔。

**不用指令**，手動做法如下：

```bash
docker compose stop folio                                       # 複製期間不要有寫入
docker compose exec -T db pg_dump -U folio -Fc folio > folio-db.dump
docker run --rm --entrypoint tar -v folio-rag_folio-storage:/data:ro tonyhuang1101/folio \
    czf - -C /data . > folio-storage.tgz
docker compose start folio
```

（`folio-rag_` 是 compose 專案名稱，預設是目錄名稱，`docker volume ls` 看得到。封存檔透過 Folio 映像檔本身
以串流傳出，不必另外下載映像檔，Docker daemon 在遠端也能用。）還原到一個全新的 stack：

```bash
docker compose up -d db
docker compose exec -T db pg_restore -U folio -d folio --clean --if-exists < folio-db.dump
docker run --rm -i --user 0 --entrypoint sh -v folio-rag_folio-storage:/data tonyhuang1101/folio \
    -c 'tar xzf - -C /data && chown -R 10001:10001 /data' < folio-storage.tgz
docker compose up -d     # 出現 volume "was not created by Docker Compose" 的警告不影響使用
```

啟用 MCP Gate profile 時，它的兩個 volume 也要備份，而且分開備份：`mcp-gate-secrets` 裡的金鑰用來解開
`mcp-gate-data` 裡的簽章金鑰（見 MCP Gate 的部署指南）。

### 少了其中一部分時

Folio 啟動時會檢查資料庫和 `storage/` 是不是同一套資料；少了其中一個就**拒絕啟動**，不會在沒有資料的情況下
看起來一切正常地跑起來：

| 發生了什麼 | Folio 的反應 |
|---|---|
| storage volume 不見或是空的（沒掛上、換成新的 volume、`FOLIO_STORAGE_ROOT` 指錯），但資料庫裡有檔案 | 拒絕啟動：「none of the N most recent [files] is in the storage directory」。請掛上或還原和這個資料庫同一套的 storage。 |
| 資料庫是新的（還沒有任何資料表：`folio-db` volume 不見、`DATABASE_URL` 指錯），但 `storage/` 裡已經有上傳檔 | 在 migration **之前**就拒絕啟動：「The database is new … already holds uploaded files」。請還原和這些檔案同一套的資料庫。真的要重新開始，先清空 storage 目錄。 |
| 最近的檔案少了一部分，不是全部 | 照常啟動，記一筆警告：那些檔案無法重新索引或下載。 |
| 第一次啟動後才改了 `.env` 的 `POSTGRES_PASSWORD` | 拒絕啟動：`password authentication failed`。PostgreSQL 沿用建立 volume 時的密碼；請改回原本的密碼（或在 PostgreSQL 裡用 `ALTER USER` 改）。 |
| 改了 `EMBEDDING_MODEL` 或 `EMBEDDING_DIMENSION` | 照常啟動；用舊模型索引的檔案會在主控台標成 **Re-index recommended**，維度不符時啟動記錄也會逐一列出資料夾的向量表。 |
| `storage/.console_session_secret` 不見 | 產生新的再啟動：主控台使用者要重新登入，沒有其他東西依賴它。 |
| 沒有掛上 `folio-backups` volume（override 或 `docker run` 沒帶） | 映像檔把 `/app/backups` 宣告為 volume，所以 Docker 會給它一個匿名 volume：封存檔不會隨容器消失，但名稱是隨機的（`docker volume ls`）。請掛上 `folio-backups`（或你自己的路徑），封存檔才會在你預期的地方。 |
| `database.backup.dir` 指向 Folio 無法寫入的目錄 | `main.py backup` 與排程會拒絕並說明「cannot write to the backup directory …」，其他不受影響。 |

啟用 `mcp-gate` profile 時，`mcp-gate-secrets` volume 和 `mcp-gate-data` 對不上，MCP Gate 會拒絕啟動。
`scripts/docker_smoke.sh` 會用實際的映像檔逐一檢查這些情況。

## 升級

1. 先讀新版本的 [CHANGELOG](../CHANGELOG.md)：標成 **Breaking** 的項目可能需要改 `.env` 或你自己的設定檔。
2. 備份（見上方）。Migration 在啟動時執行，而且只會往前；備份就是回頭的路。
3. 指定新版本並重建容器：
   ```bash
   # .env：FOLIO_VERSION=1.5.1
   docker compose pull
   docker compose up -d
   docker compose logs -f folio      # 看到 "Auto migration complete"，接著伺服器啟動
   ```
4. 檢查 `/health/detailed`，並在主控台看哪些檔案標著 **Re-index recommended**（新的切塊流程或嵌入模型）。

`docker compose down` 再 `up` 時 volume 都會保留；只有 `docker compose down -v` 會刪除它們。

## 從原始碼執行

要在程序內跑模型（`rag.profile: local` 或 `hybrid`，跑在 CPU 上），請用 uv 安裝後執行 `python main.py`，
見 README 的[安裝](../README.zh-TW.md#安裝)。本指南其餘部分同樣適用：PostgreSQL 加 pgvector（compose 檔
的 `db` 服務可以單獨使用：`docker compose up -d db`，並對外開放它的連接埠）、issuer 與 audience、反向
代理，以及資料庫和 `storage/` 一起備份。
