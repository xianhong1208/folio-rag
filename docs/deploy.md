# Deploying Folio

[繁體中文](deploy.zh-TW.md)

How to run Folio for real: the Docker image and compose file, the model endpoints, MCP Gate, a reverse
proxy with HTTPS, backups and upgrades.

- [The Docker image](#the-docker-image)
- [Docker Compose](#docker-compose)
- [Model endpoints: Forgeyard or vLLM](#model-endpoints-forgeyard-or-vllm)
- [MCP Gate: the issuer and the audience](#mcp-gate-the-issuer-and-the-audience)
- [Reverse proxy and HTTPS](#reverse-proxy-and-https)
- [Configuration](#configuration)
- [Backups](#backups)
- [Upgrades](#upgrades)
- [Running from source](#running-from-source)

## The Docker image

`tonyhuang1101/folio` (Docker Hub, `linux/amd64` and `linux/arm64`, tags `X.Y.Z`, `X.Y` and `latest`)
is the **lightweight** image: `rag.profile: cloud`, so every model is an HTTP endpoint and nothing is
loaded inside the container. No torch, no GPU stack, no model downloads.

| | |
|---|---|
| Models | embeddings, the LLM, the reranker, a vision model for PDFs and images, and optionally speech-to-text: all OpenAI-compatible endpoints ([Forgeyard](https://github.com/xianhong1208/forgeyard), vLLM, Ollama or a cloud provider) |
| Chinese keyword search | the `jieba` segmenter (`FOLIO_SEGMENTER`; `bigram` is the other model-free choice) |
| Listens on | `5039` inside the container: `/mcp`, the console at `/admin`, `/health` |
| Runs as | uid `10001`, not root |
| Volume | `/app/storage`: uploaded files and the console's session secret |
| Config | the six files in `/app/config`, baked in; `models.yaml` is [`docker/config/models.yaml`](../docker/config/models.yaml) (every endpoint, model and key from the environment) |
| Logs | the console only: `docker logs` (no log files in the container) |
| Migrations | run on every start, as with `python main.py` |
| Health | a Docker `HEALTHCHECK` on `/health`; `/health/detailed` checks the database, pgvector and the revocation feed |

**Not in the image:** the in-process models (CKIP, docling's layout and OCR models, HuggingFace
embeddings). Folio has no in-process speech-to-text in any install since 1.5. For `rag.profile: local` or `hybrid`, run Folio from source
([below](#running-from-source)). There is no GPU image.

**Audio** is transcribed by an OpenAI-compatible endpoint ([Moving speech-to-text to an
endpoint](migrate-asr.md)). Folio resamples and splits long audio with `ffmpeg` first, and the
published image leaves `ffmpeg` out to stay small: without it, a file up to 24 MB is sent whole and
a larger one fails with a message saying to install `ffmpeg`. For long recordings, build the image
with it: `docker build --build-arg INSTALL_FFMPEG=true -t folio:audio .`. Then set
`ASR_ENABLED=true`, `ASR_BASE_URL` and `ASR_MODEL`. While speech-to-text is off or has no endpoint,
audio uploads are refused (422). If the endpoint's certificate comes from a private CA (Forgeyard's
own), mount the CA file read-only and point `ASR_CA_FILE` at it.

## Docker Compose

[`docker-compose.yml`](../docker-compose.yml) runs Folio with PostgreSQL + pgvector
(`pgvector/pgvector:pg16`, with a health check and a named volume), and optionally MCP Gate.

```bash
git clone https://github.com/xianhong1208/folio-rag.git && cd folio-rag   # or copy docker-compose.yml and .env.example
cp .env.example .env
```

In `.env`, uncomment and set at least:

```bash
POSTGRES_PASSWORD=<a long random password>
# the models (next section)
EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
LLM_BASE_URL=http://host.docker.internal:8001/v1
LLM_MODEL=Qwen/Qwen3-4B
RERANK_BASE_URL=http://host.docker.internal:8002/v1
RERANK_MODEL=BAAI/bge-reranker-v2-m3
VISION_BASE_URL=http://host.docker.internal:8003/v1
VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct
# MCP Gate (see "MCP Gate" below)
MCP_GATE_URL=http://localhost:4568
FOLIO_AUDIENCE=http://localhost:5039/mcp
```

```bash
docker compose pull
docker compose up -d
docker compose ps                              # folio and db: healthy
curl -s localhost:5039/health/detailed         # "status": "healthy"
```

The console is at `http://localhost:5039/admin`. Sign in with a personal access token from MCP Gate.

What the compose file sets for you: `DATABASE_URL` points at the `db` service (a `DATABASE_URL` in
`.env` is not used), `MCP_GATE_URL` and `FOLIO_AUDIENCE` default to `localhost` addresses, and
`host.docker.internal` reaches the host machine. `FOLIO_VERSION` pins a release (`FOLIO_VERSION=1.5.0`)
and `FOLIO_PORT` changes the host port. Named volumes hold the data: `folio-storage`,
`folio-db`, and, with the MCP Gate profile, `mcp-gate-data` and `mcp-gate-secrets`.

## Model endpoints: Forgeyard or vLLM

Every role is an OpenAI-compatible endpoint. The variables (all in `.env`):

| Role | Endpoint | Model | Key | Notes |
|---|---|---|---|---|
| Embedding | `EMBEDDING_BASE_URL` | `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` | `EMBEDDING_API_KEY` | `POST /embeddings`. The dimension must match the model; changing either means re-indexing. e5 models also need `EMBEDDING_QUERY_PREFIX="query: "` and `EMBEDDING_PASSAGE_PREFIX="passage: "`. |
| LLM | `LLM_BASE_URL` | `LLM_MODEL` | `LLM_API_KEY` | `POST /chat/completions`: answers, evaluation and the one-line context written for each chunk at index time. |
| Reranker | `RERANK_BASE_URL` | `RERANK_MODEL` | `RERANK_API_KEY` | `RERANK_API`: `generic` (`POST /rerank`, vLLM, Forgeyard, Infinity; the default), `score` (vLLM `/score`), `cohere` or `jina`. `RERANK_ENABLED=false` turns it off. |
| Vision | `VISION_BASE_URL` | `VISION_MODEL` | `VISION_API_KEY` | Reads PDFs and images page by page (the image has no built-in OCR). Office, Markdown, HTML and text files need no model. |
| Speech-to-text | `ASR_BASE_URL` | `ASR_MODEL` | `ASR_API_KEY` | `POST /audio/transcriptions`. Off unless `ASR_ENABLED=true`; long audio needs an image with `ffmpeg` (above). `ASR_CA_FILE` for a private CA. Chinese transcripts are converted to Traditional (`rag.asr.chinese_conversion`). |

Inside a container, `localhost` is the container itself. Reach a model server on the same machine
as `host.docker.internal`, and one on another machine by its address.

**[Forgeyard](https://github.com/xianhong1208/forgeyard)** serves every role behind one
OpenAI-compatible URL, with one API key: set every `*_BASE_URL` to it (for example
`https://gpu-host:7860/v1`) and every `*_API_KEY` to the key, and each `*_MODEL` to a deployment
name. Forgeyard uses HTTPS with its own CA by default. Give the container a CA bundle that includes it:

```bash
cat "$(python3 -m certifi)" forgeyard-ca.pem > ca-bundle.pem    # public CAs + Forgeyard's
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

**vLLM** serves one model per port: one `*_BASE_URL` per role, for example
`http://host.docker.internal:8000/v1` for the embeddings. The model names are the served names
(`--served-model-name`, or the Hugging Face id). A reranker whose name does not match falls back to
the original order without an error, so check it in the console: *Settings → Model roles → Test*.

The keys are read from the environment only: `profile: cloud` refuses a key written in the config
files.

## MCP Gate: the issuer and the audience

Folio is a resource server: it verifies tokens issued by
[MCP Gate](https://github.com/xianhong1208/MCP_Gate) offline, against MCP Gate's signing keys. Two
values must match **exactly**, as strings (scheme, host, port and path; `localhost` and `127.0.0.1`
are different), or every token gets 401 and nothing says why:

| Folio | must equal | in MCP Gate |
|---|---|---|
| `MCP_GATE_URL` (`auth.issuer`) | the token's `iss` | `OAUTH_ISSUER`, the address clients use for MCP Gate |
| `FOLIO_AUDIENCE` (`auth.audience`) | the token's `aud` | the audience of the service registered for Folio (*Services → Register Service*; it defaults to the service's MCP URL) |

Both are the **public** addresses, the ones MCP clients use. Folio also fetches MCP Gate's signing
keys (and the revocation feed) from `MCP_GATE_URL`. When that address does not work from inside
Folio's container, set `MCP_GATE_INTERNAL_URL` to one that does: Folio then fetches from it and still
checks `iss` against `MCP_GATE_URL`.

**MCP Gate in the same stack** (the `mcp-gate` profile, image `tonyhuang1101/mcp-gate:1.4.0`, its
own data and secrets volumes):

```bash
# .env
COMPOSE_PROFILES=mcp-gate
MCP_GATE_URL=http://localhost:4568              # MCP Gate's OAUTH_ISSUER is set to the same value
MCP_GATE_INTERNAL_URL=http://mcp-gate:4568      # inside the stack, MCP Gate is the service mcp-gate
FOLIO_AUDIENCE=http://localhost:5039/mcp
MCP_GATE_ADMIN_EMAIL=you@example.com            # its owner account (or leave both empty and use
MCP_GATE_ADMIN_PASSWORD=<a strong password>     #   the setup token in `docker compose logs mcp-gate`)
```

Then `docker compose up -d`, open MCP Gate at `http://localhost:4568`, and register Folio under
*Services → Register Service*: host `folio`, port `5039`, path `/mcp` (the address MCP Gate uses to
check the service and list its tools), and **OAuth audience** `http://localhost:5039/mcp`, the same as
`FOLIO_AUDIENCE`. Issue a personal access token for the service and sign in to Folio's console with
it. MCP Gate does not read Folio's `.env`; its settings are in the compose file (`ALLOWED_HOSTS=mcp-gate`
lets Folio fetch the keys by the service name).

**MCP Gate elsewhere:** set `MCP_GATE_URL` to its `OAUTH_ISSUER` and leave `MCP_GATE_INTERNAL_URL`
empty, as long as that address is reachable from the container.

**Revocation feed:** without it, a token revoked in MCP Gate keeps working in Folio until it expires.
Set it up as in [Authentication, Token revocation](auth.md#token-revocation), and put
`FOLIO_REVOCATION_ENABLED`, `FOLIO_REVOCATION_CLIENT_ID` and `FOLIO_REVOCATION_CLIENT_SECRET` in `.env`.

## Reverse proxy and HTTPS

On a server, put Folio (and MCP Gate) behind a reverse proxy that terminates HTTPS, and publish the
containers' ports on `127.0.0.1` only (`"127.0.0.1:5039:5039"` in the compose file). Then the
addresses change to the public ones, in every place they appear:

```bash
MCP_GATE_URL=https://gate.example.com           # = MCP Gate's OAUTH_ISSUER
FOLIO_AUDIENCE=https://folio.example.com/mcp    # = the audience registered in MCP Gate
```

Change the registered audience in MCP Gate to match: tokens issued for the old audience stop working.
The console's session cookie is marked `Secure` when the request arrived over HTTPS, so the proxy
must pass `X-Forwarded-Proto` (and Folio must trust it, below), or set `auth.console.secure_cookie: true`.

What the proxy must allow: uploads up to 100 MB, and long-lived responses without buffering (MCP's
streamable HTTP and the indexing progress stream use server-sent events).

**nginx**

```nginx
server {
    listen 443 ssl;
    http2 on;
    server_name folio.example.com;
    ssl_certificate     /etc/letsencrypt/live/folio.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/folio.example.com/privkey.pem;

    client_max_body_size 110m;              # uploads are limited to 100 MB by Folio

    location / {
        proxy_pass http://127.0.0.1:5039;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;                # server-sent events (MCP, indexing progress)
        proxy_read_timeout 3600s;
    }
}
```

**Caddy** (gets and renews the certificates itself; it streams without buffering by default):

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

Folio trusts `X-Forwarded-Proto` only from the addresses in `FORWARDED_ALLOW_IPS` (uvicorn's
setting, `127.0.0.1` by default). In a container the proxy's connections arrive from the Docker
network instead, so set `FORWARDED_ALLOW_IPS=*` in `.env`, and only with the port published on
`127.0.0.1`, where nothing but the proxy can reach it. The unauthenticated `/v1` media endpoints and `/health*` are reachable by anyone who reaches
Folio; keep it behind the proxy on a trusted network, or block `/v1/` at the proxy.

## Configuration

Everything per machine comes from the environment (`.env` with compose). Every variable is listed
in [`.env.example`](../.env.example); the "Docker" section lists the ones only the image's
`models.yaml` reads. The settings themselves are described in [Configuration](configuration.md).

To change a setting the variables do not cover (chunk sizes, retrieval tuning, the reranker
templates), use the console's *Settings* page (stored in the database), or replace the config files:

```bash
docker compose cp folio:/app/config ./config    # the image's six files, as a starting point
# edit ./config/*.yaml, then mount it: uncomment "- ./config:/app/config:ro" in docker-compose.yml
docker compose up -d
```

`FOLIO_CONFIG_DIR` points at another directory under `/app` instead (`--config-dir` sets the same on
the command line). `FOLIO_LOG_DIR` empty (the image's default) keeps the logs on the console; set it to
a directory, and mount a volume there, for log files.

### Metrics (Prometheus)

`/metrics` arrives with #11 (Folio 1.5). Without `METRICS_TOKEN` it answers only a client on the same
loopback interface, so in Docker, where a scrape from the host or from another container is not
loopback, set a token in `.env` (compose passes it to the container) and give Prometheus the same
token:

```bash
# .env
METRICS_TOKEN=<a long random string>
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: folio
    authorization:
      credentials: <the same METRICS_TOKEN>
    static_configs:
      - targets: ["folio:5039"]     # or <host>:5039 from outside the compose network
```

With `METRICS_TOKEN` set, every scrape needs it as a bearer token, from loopback too.

## Backups

The database and `storage/` belong together: the database holds the folders, the chunks and the
vectors, and `storage/` the uploaded files they were made from. Back them up together, at the same
moment, and restore them together. The image does both in one command
([Backup and restore](backup.md) has the details):

```bash
docker compose exec folio python main.py backup                 # → the folio-backups volume (/app/backups)
docker compose exec folio ls -l /app/backups
docker compose cp folio:/app/backups/folio-backup-<time>.tar .  # a copy off the machine
```

`main.py backup` writes one archive (`folio-backup-<UTC time>.tar`: a `pg_dump` of the database, the
files under `storage/` and a manifest with checksums) into `/app/backups`, the image's
`database.backup.dir`, which compose keeps in its own volume, `folio-backups`. It runs while Folio
runs. The image has the PostgreSQL 17 client tools, newer than the bundled pg16 server, which is fine.
Keep copies of the archives somewhere else too: the volume is on the same disk as the data.

**Scheduled backups:** set `FOLIO_BACKUP_ENABLED=true` in `.env` (and `docker compose up -d`). Folio
then writes one archive every `FOLIO_BACKUP_INTERVAL_HOURS` (24) into the same volume and keeps the
newest `FOLIO_BACKUP_KEEP` (7); the first comes two minutes after the start when there is none yet.
`/health/detailed` shows the newest archive and the last result.

**Restore** needs Folio stopped (it refuses while anything else is connected to the database), so
run it in a one-off container with the same volumes:

```bash
docker compose stop folio
docker compose cp folio-backup-<time>.tar folio:/app/backups/   # only if the archive is not in the volume yet
docker compose run --rm --no-deps folio python main.py restore /app/backups/folio-backup-<time>.tar --force
docker compose start folio
```

`--force` replaces a database that already has tables and a `storage/` that already holds folders
(into a fresh stack it is not needed). The previous files are not deleted: they are moved to
`storage/.folio-work/before-restore-<time>/` in the storage volume; remove that directory once the
restored instance looks right. `restore --check <archive>` only verifies an archive.

**Without the command**, the same by hand:

```bash
docker compose stop folio                                       # nothing writes while you copy
docker compose exec -T db pg_dump -U folio -Fc folio > folio-db.dump
docker run --rm --entrypoint tar -v folio-rag_folio-storage:/data:ro tonyhuang1101/folio \
    czf - -C /data . > folio-storage.tgz
docker compose start folio
```

(`folio-rag_` is the compose project name, by default the directory's name: `docker volume ls` shows
it. The archive is streamed through the Folio image itself, so nothing else is pulled and it works with a
remote Docker daemon too.) To restore, into a fresh stack:

```bash
docker compose up -d db
docker compose exec -T db pg_restore -U folio -d folio --clean --if-exists < folio-db.dump
docker run --rm -i --user 0 --entrypoint sh -v folio-rag_folio-storage:/data tonyhuang1101/folio \
    -c 'tar xzf - -C /data && chown -R 10001:10001 /data' < folio-storage.tgz
docker compose up -d     # a warning that the volume "was not created by Docker Compose" is harmless
```

With the MCP Gate profile, back up its two volumes as well, separately: `mcp-gate-secrets` holds the
key that decrypts the signing keys in `mcp-gate-data` (see MCP Gate's deploy guide).

### When one piece is missing

Folio checks at startup that the database and `storage/` belong together, and **refuses to start**
when one of them is missing, rather than starting healthy without the data:

| What happened | What Folio does |
|---|---|
| The storage volume is missing or empty (not mounted, a new volume, a wrong `FOLIO_STORAGE_ROOT`), while the database lists files | Refuses: "none of the N most recent [files] is in the storage directory". Mount or restore the storage that belongs to this database. |
| The database is new (no tables yet: a missing `folio-db` volume, a wrong `DATABASE_URL`) while `storage/` already holds uploads | Refuses **before** migrating it: "The database is new … already holds uploaded files". Restore the database that belongs to these files. To really start over, empty the storage directory first. |
| Some of the most recent files are missing, not all | Starts, with a warning: those files cannot be re-indexed or downloaded. |
| `POSTGRES_PASSWORD` changed in `.env` after the first start | Refuses: `password authentication failed`. PostgreSQL keeps the password its volume was created with; set the old one back (or change it inside PostgreSQL with `ALTER USER`). |
| `EMBEDDING_MODEL` or `EMBEDDING_DIMENSION` changed | Starts; the files indexed with the old model are marked **Re-index recommended** in the console, and a dimension mismatch is also logged at startup for each folder table. |
| `storage/.console_session_secret` is missing | Starts with a new one: console users sign in again. Nothing else depends on it. |
| The `folio-backups` volume is not mounted (an override or a `docker run` without it) | The image declares `/app/backups` a volume, so Docker gives it an anonymous one: archives survive the container, but under a random name (`docker volume ls`). Mount `folio-backups` (or your own path) to keep them where you expect. |
| `database.backup.dir` points at a directory Folio cannot write | `main.py backup` and the schedule refuse with "cannot write to the backup directory …", nothing else changes. |

With the `mcp-gate` profile, MCP Gate refuses to start when its `mcp-gate-secrets` volume does not
match `mcp-gate-data`. `scripts/docker_smoke.sh` checks each of these cases against the real image.

## Upgrades

1. Read the [CHANGELOG](../CHANGELOG.md) for the new version: anything marked **Breaking** may need a
   change to `.env` or to your own config files.
2. Back up (above). Migrations run on start and only go forward; the backup is the way back.
3. Pin the new version and recreate the container:
   ```bash
   # .env: FOLIO_VERSION=1.5.1
   docker compose pull
   docker compose up -d
   docker compose logs -f folio      # "Auto migration complete", then the server starts
   ```
4. Check `/health/detailed`, and in the console the files marked **Re-index recommended** (a new
   chunking pipeline or embedding model).

The volumes are kept across `docker compose down` and `up`; only `docker compose down -v` deletes them.

## Running from source

For the in-process models (`rag.profile: local` or `hybrid`, GPU or CPU), install with uv and run
`python main.py`, as in the README's [Installation](../README.md#installation). The rest of this guide
applies the same way: PostgreSQL with pgvector (the compose file's `db` service works on its own:
`docker compose up -d db`, with its port published), the issuer and the audience, the reverse proxy,
and backups of the database and `storage/` together.
