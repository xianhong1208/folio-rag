# Folio — the lightweight image: rag.profile cloud, every model is an HTTP endpoint.
#
# Published as docker.io/tonyhuang1101/folio (amd64 + arm64) by .github/workflows/publish-image.yml.
#   docker compose up -d                  # Folio + PostgreSQL/pgvector, see docker-compose.yml and docs/deploy.md
#   docker build -t folio:local .         # build it from this checkout
#
# What is in it: the core install (`uv sync --no-dev` + the jieba extra), no torch, no GPU stack, no local models.
# Embeddings, the LLM, the reranker, the vision model (PDFs and images) and speech-to-text are OpenAI-compatible
# endpoints such as Forgeyard or vLLM, set through environment variables (.env.example, "Docker"). For the in-process
# models (rag.profile local / hybrid) run Folio from source instead (README, Installation).
#
#   /app/storage  a VOLUME: uploaded files and the console's session secret. Back it up together with the database.
#   /app/backups  a VOLUME: archives of `main.py backup` and of the backup schedule (database.backup.dir = backups).
#                 The PostgreSQL client tools are installed for them (Debian's postgresql-client-17: newer than the
#                 bundled pg16 server is fine).
#   /app/config   the six config files, baked in (docker/config/models.yaml is the cloud models.yaml). To change them,
#                 mount your own directory over /app/config, or anywhere under /app and point FOLIO_CONFIG_DIR at it.
# Logs go to the console only (`docker logs`). Migrations run on every start, as with `python main.py`.
#
# The console (static/console) is committed and checked against a fresh build in CI, so no Node stage is needed here.

# ---- Dependencies (uv, locked) --------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.9.24 AS uv

FROM python:3.12-slim-trixie AS deps
COPY --from=uv /uv /usr/local/bin/uv
# psycopg2 is built from source (it publishes no Linux wheels): a compiler and libpq's headers, in this stage only
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev libpq-dev \
 && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
# uv reads the vendored FireRedASR's metadata to check the lock; the `local` extra that uses it is not installed
COPY vendor/fireredasr_src/pyproject.toml vendor/fireredasr_src/pyproject.toml
# The application itself is not installed: main.py runs from /app. --no-default-groups keeps every torch group out;
# the jieba extra is the image's Chinese segmenter (no model). Nothing from the `local` extra.
RUN uv sync --locked --no-dev --no-default-groups --extra jieba --no-install-project --no-cache

# ---- Runtime ----------------------------------------------------------------------
FROM python:3.12-slim-trixie

# Set by the publish workflow; the version Folio reports is read from pyproject.toml either way
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="Folio" \
      org.opencontainers.image.description="Agentic RAG MCP server: hierarchical chunking, hybrid retrieval, rerank (cloud profile)" \
      org.opencontainers.image.source="https://github.com/xianhong1208/folio-rag" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.revision="$REVISION"

# libpq5: psycopg2 at run time. postgresql-client-17: pg_dump / pg_restore / psql for main.py backup / restore and
# the backup schedule (at least the server's major version; their directory goes on PATH below). INSTALL_FFMPEG=true adds ffmpeg, which audio transcription needs (it resamples and
# splits audio before uploading it); the published image leaves it out to stay small.
ARG INSTALL_FFMPEG=false
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 postgresql-client-17 \
 && if [ "$INSTALL_FFMPEG" = true ]; then apt-get install -y --no-install-recommends ffmpeg; fi \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --user-group --create-home --home-dir /home/folio --shell /usr/sbin/nologin folio \
 && mkdir -p /app/storage /app/reports /app/backups \
 && chown folio:folio /app/storage /app/reports /app/backups

WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml main.py app.py ./
COPY config/ config/
COPY docker/config/models.yaml config/models.yaml
COPY db/ db/
COPY versions/ versions/
COPY resources/ resources/
COPY src/ src/
COPY static/console/ static/console/
# The console's retrieval evaluation loads this script; its reports go to /app/reports (not kept across containers)
COPY scripts/rag_eval.py scripts/rag_eval.py

# FOLIO_LOG_DIR empty: console only, `docker logs` has everything. NO_COLOR: no colour codes in that log.
# FOLIO_SEGMENTER: jieba for Chinese keyword search (bigram is the other model-free choice).
# The container always listens on 5039; publish it on any host port.
ENV PATH=/app/.venv/bin:/usr/lib/postgresql/17/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FOLIO_LOG_DIR="" \
    NO_COLOR=1 \
    FOLIO_SEGMENTER=jieba \
    HOME=/home/folio

USER folio
VOLUME ["/app/storage", "/app/backups"]
EXPOSE 5039
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5039/health', timeout=4)"]

CMD ["python", "main.py", "--port", "5039"]
