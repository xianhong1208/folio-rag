#!/usr/bin/env bash
# End-to-end check of the Docker image and docker-compose.yml. From the repository root:
#
#   bash scripts/docker_smoke.sh                          # builds folio:smoke first
#   IMAGE=folio:local SKIP_BUILD=1 bash scripts/docker_smoke.sh
#
# Brings up docker-compose.yml as shipped, with the mcp-gate profile (pulls pgvector/pgvector:pg16 and
# tonyhuang1101/mcp-gate), plus a stub of the OpenAI-compatible model APIs (scripts/docker_smoke_models.py, run from
# the Folio image) that every model role points at. Then checks:
#   - the container turns healthy, runs as uid 10001, writes no log files; /health/detailed is healthy and the
#     database is at the newest migration;
#   - without a token /mcp and the REST API answer 401; a personal access token issued by that MCP Gate for the
#     Folio service (its audience) is accepted: /mcp initialize answers 200 (issuer + audience + key fetch via
#     MCP_GATE_INTERNAL_URL all agree);
#   - a folder, a Markdown upload that is indexed with the stub's embeddings and contextual prefixes, and a search
#     that finds it;
#   - after a restart, and after the containers are recreated, the folder, the file and the search hit are still there;
#   - split state: the storage volume missing, the database volume missing, POSTGRES_PASSWORD changed, and MCP Gate's
#     secrets volume missing each stop startup with a clear message, also when the container restarts (and the empty
#     database is left unmigrated); a changed EMBEDDING_DIMENSION starts and logs the mismatch; a deleted console
#     session secret is replaced; with every piece back the data is there again.
# Everything it creates (containers, volumes, network, the temporary override file) is removed on exit; the image
# is kept.
# SMOKE_PREFIX names the compose project (and so every container, volume and network); SMOKE_FOLIO_PORT and
# SMOKE_GATE_PORT are the host ports (published on 127.0.0.1).
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-folio:smoke}"
PREFIX="${SMOKE_PREFIX:-folio-smoke}"
FOLIO_PORT="${SMOKE_FOLIO_PORT:-15039}"
GATE_PORT="${SMOKE_GATE_PORT:-14568}"
FOLIO="http://localhost:$FOLIO_PORT"
GATE="http://localhost:$GATE_PORT"
AUDIENCE="$FOLIO/mcp"
EMAIL="smoke@example.com"
GATE_PASSWORD="Smoke-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')"
WORD="quokkadrome"   # a word only the test document contains
WORK="$(mktemp -d)"

export COMPOSE_PROJECT_NAME="$PREFIX"
export FOLIO_IMAGE="$IMAGE" FOLIO_PORT MCP_GATE_PORT="$GATE_PORT"
export POSTGRES_PASSWORD="smoke-$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')"
export MCP_GATE_URL="$GATE" FOLIO_AUDIENCE="$AUDIENCE" MCP_GATE_INTERNAL_URL="http://mcp-gate:4568"
export MCP_GATE_ADMIN_EMAIL="$EMAIL" MCP_GATE_ADMIN_PASSWORD="$GATE_PASSWORD"
# The stub's code, handed to its container through the environment (no bind mount: works with a remote daemon)
STUB_CODE="$(cat scripts/docker_smoke_models.py)"
export STUB_CODE

# Publish on 127.0.0.1 only, add the stub, and point every model role at it
cat > "$WORK/smoke.yml" <<EOF
services:
  folio:
    ports: !override
      - "127.0.0.1:$FOLIO_PORT:5039"
    environment:
      EMBEDDING_BASE_URL: http://models:8000/v1
      EMBEDDING_MODEL: stub-embedding
      EMBEDDING_DIMENSION: "256"
      LLM_BASE_URL: http://models:8000/v1
      LLM_MODEL: stub-llm
      RERANK_BASE_URL: http://models:8000/v1
      RERANK_MODEL: stub-rerank
      VISION_BASE_URL: http://models:8000/v1
      VISION_MODEL: stub-vision
  mcp-gate:
    ports: !override
      - "127.0.0.1:$GATE_PORT:4568"
  models:
    image: $IMAGE
    entrypoint: ["python", "-c", "import os; exec(os.environ['STUB_CODE'])"]
    environment:
      - STUB_CODE
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=4)"]
      interval: 2s
      retries: 30
EOF
compose() { docker compose -f docker-compose.yml -f "$WORK/smoke.yml" --profile mcp-gate "$@"; }
# up [what failed]: start (or restart) the stack and wait for every health check; progress goes to a file
up() { compose up -d --no-build --wait --wait-timeout 180 >"$WORK/up.log" 2>&1 || { cat "$WORK/up.log"; fail "$1"; }; }

cleanup() {
    status=$?
    if [ $status -ne 0 ]; then
        echo "---- logs (last 80 lines per service) ----"
        compose logs --tail 80 2>&1 || true
    fi
    compose down -v --remove-orphans >/dev/null 2>&1 || true
    # The empty volumes of the split-state cases are declared only in their override files
    for v in folio-storage-replaced folio-db-replaced mcp-gate-secrets-replaced; do
        docker volume rm -f "${PREFIX}_$v" >/dev/null 2>&1 || true
    done
    rm -rf "$WORK"
    exit $status
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {'json': json}, {'d': d}))" "$1"; }

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    docker build -t "$IMAGE" .
fi

compose down -v --remove-orphans >/dev/null 2>&1 || true
compose config --quiet
up "the stack did not come up healthy"
echo "ok: docker compose up (folio, db, mcp-gate, model stub): every service with a health check is healthy"

folio_cid="$(compose ps -q folio)"
[ "$(docker inspect -f '{{.State.Health.Status}}' "$folio_cid")" = healthy ] || fail "folio is not healthy"
[ "$(compose exec -T folio id -u)" = 10001 ] || fail "Folio does not run as uid 10001"
compose exec -T folio sh -c 'test ! -e /app/logs' || fail "/app/logs exists: file logging is on in the image"
compose exec -T folio python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('torch') is not None)" \
    || fail "torch is installed in the image"
echo "ok: healthy, runs as uid 10001, logs to the console only, no torch"

detailed="$(curl -fsS "$FOLIO/health/detailed")"
[ "$(echo "$detailed" | json "d['status']")" = healthy ] || fail "/health/detailed: $detailed"
[ "$(echo "$detailed" | json "d['checks']['pgvector']['status']")" = healthy ] || fail "pgvector is not installed"
head_rev="$(compose exec -T folio python -c "
from alembic.config import Config
from alembic.script import ScriptDirectory
print(ScriptDirectory.from_config(Config('db/migrate/alembic.ini')).get_current_head())")"
db_rev="$(compose exec -T db psql -U folio -d folio -tAc 'SELECT version_num FROM alembic_version')"
[ -n "$head_rev" ] && [ "$db_rev" = "$head_rev" ] || fail "the database is at '$db_rev', the newest migration is '$head_rev'"
echo "ok: /health/detailed healthy (database, pgvector); migrations ran on boot (at $db_rev)"

# ---- authentication against the MCP Gate of the stack ----
mcp_init='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"docker-smoke","version":"1"}}}'
mcp() {  # mcp [token] -> HTTP status of an initialize request
    curl -sS -o "$WORK/mcp.out" -w '%{http_code}' -X POST "$FOLIO/mcp" \
        -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
        ${1:+-H "Authorization: Bearer $1"} -d "$mcp_init"
}
code="$(mcp)"
[ "$code" = 401 ] || fail "/mcp without a token answered $code, expected 401"
code="$(curl -sS -o /dev/null -w '%{http_code}' "$FOLIO/api/folders/")"
[ "$code" = 401 ] || fail "/api/folders/ without a token answered $code, expected 401"
echo "ok: no token -> 401 on /mcp and /api/folders/"

# OAuth clients follow the 401 challenge to the protected-resource document: both must name the public address
challenge="$(curl -sS -o /dev/null -D - -X POST "$FOLIO/mcp" -H 'Content-Type: application/json' -d '{}' \
    | tr -d '\r' | grep -i '^www-authenticate:')"
case "$challenge" in
    *"resource_metadata=\"$FOLIO/.well-known/oauth-protected-resource/mcp\""*) ;;
    *) fail "the 401 challenge does not point at $FOLIO: $challenge" ;;
esac
resource="$(curl -fsS "$FOLIO/.well-known/oauth-protected-resource/mcp" | json "d['resource']")"
[ "$resource" = "$AUDIENCE" ] || fail "the protected-resource document names $resource, expected $AUDIENCE"
echo "ok: the 401 challenge and the protected-resource document name the public address ($AUDIENCE)"

TOKEN="$(python3 scripts/docker_smoke_token.py "$GATE" "$EMAIL" "$GATE_PASSWORD" "$AUDIENCE")" \
    || fail "could not get a personal access token from MCP Gate"
code="$(mcp "$TOKEN")"
[ "$code" = 200 ] || fail "/mcp with MCP Gate's token answered $code: $(cat "$WORK/mcp.out")"
grep -q '"serverInfo"' "$WORK/mcp.out" || fail "/mcp initialize returned no serverInfo"
echo "ok: a personal access token issued by MCP Gate (aud $AUDIENCE) -> /mcp initialize 200"
AUTH=(-H "Authorization: Bearer $TOKEN")

# ---- a folder, an upload, indexing and a search, on the stub models ----
folder_id="$(curl -fsS "${AUTH[@]}" -X POST "$FOLIO/api/folders/" -H 'Content-Type: application/json' \
    -d '{"name":"smoke","description":"docker smoke test"}' | json "d[0]['id']")" || fail "could not create a folder"
cat > "$WORK/smoke.md" <<EOF
# Docker smoke test

The $WORD is a small marsupial observatory that only exists in this test document.
It is indexed by the containerised Folio with stub embeddings, then found again by a search.
EOF
curl -fsS "${AUTH[@]}" -o /dev/null -X POST "$FOLIO/api/folders/$folder_id/file" \
    -F "file=@$WORK/smoke.md;type=text/markdown" -F auto_index=true || fail "the upload failed"

search() {  # prints how many results mention $WORD
    curl -fsS "${AUTH[@]}" -X POST "$FOLIO/api/rag/query" -H 'Content-Type: application/json' \
        -d "{\"query\":\"what is the $WORD\",\"folder_name\":\"smoke\"}" \
        | json "sum('$WORD' in json.dumps(r) for r in d['data']['results'])" 2>/dev/null || echo 0
}
hits=0
for _ in $(seq 1 90); do
    hits="$(search)"
    [ "$hits" -gt 0 ] && break
    sleep 2
done
[ "$hits" -gt 0 ] || fail "the uploaded file was not found by a search"
logs="$(compose logs models)"
for call in /v1/embeddings /v1/chat/completions; do
    grep -q "STUB POST $call" <<<"$logs" || fail "Folio never called the stub's $call"
done
echo "ok: folder created, markdown uploaded and indexed (stub embeddings + contextual prefixes), search finds it ($hits hit(s))"

# Files are stored by id under /app/storage/<owner>/<folder>/: find the upload by its content
storage_files() { compose exec -T folio sh -c "grep -rl $WORD /app/storage | wc -l"; }
[ "$(storage_files)" -ge 1 ] || fail "the uploaded file is not in /app/storage"

# ---- data survives a restart and recreated containers (named volumes) ----
compose restart folio >/dev/null 2>&1
up "not healthy after a restart"
[ "$(search)" -gt 0 ] || fail "the search hit is gone after a restart"
echo "ok: after a restart the search still finds the file"

compose down >/dev/null 2>&1   # keeps the volumes, like an upgrade
up "not healthy after recreating the containers"
TOKEN="$(python3 scripts/docker_smoke_token.py "$GATE" "$EMAIL" "$GATE_PASSWORD" "$AUDIENCE")" \
    || fail "MCP Gate lost its owner after recreating the containers"
AUTH=(-H "Authorization: Bearer $TOKEN")
[ "$(search)" -gt 0 ] || fail "the search hit is gone after recreating the containers"
[ "$(storage_files)" -ge 1 ] || fail "the uploaded file is gone after recreating the containers"
[ "$(mcp "$TOKEN")" = 200 ] || fail "/mcp rejects a new token after recreating the containers"
echo "ok: after recreating the containers the folder, the file and the search hit are still there"

# ---- main.py backup / restore inside the image (PostgreSQL client tools, the folio-backups volume) ----
compose exec -T folio python main.py backup >"$WORK/backup.log" 2>&1 || { cat "$WORK/backup.log"; fail "main.py backup failed in the container"; }
archive="$(compose exec -T folio sh -c 'ls -t /app/backups/folio-backup-*.tar | head -1' | tr -d '\r')"
[ -n "$archive" ] || fail "main.py backup wrote no archive into /app/backups"
# Change the data after the backup: a second document with a word of its own
WORD2="wombatorium"
printf '# After the backup\n\nThe %s exists only after the backup was taken.\n' "$WORD2" > "$WORK/after.md"
curl -fsS "${AUTH[@]}" -o /dev/null -X POST "$FOLIO/api/folders/$folder_id/file" \
    -F "file=@$WORK/after.md;type=text/markdown" -F auto_index=true || fail "the second upload failed"
search2() {
    curl -fsS "${AUTH[@]}" -X POST "$FOLIO/api/rag/query" -H 'Content-Type: application/json' \
        -d "{\"query\":\"what is the $WORD2\",\"folder_name\":\"smoke\"}" \
        | json "sum('$WORD2' in json.dumps(r) for r in d['data']['results'])" 2>/dev/null || echo 0
}
for _ in $(seq 1 90); do [ "$(search2)" -gt 0 ] && break; sleep 2; done
[ "$(search2)" -gt 0 ] || fail "the document uploaded after the backup was not indexed"
compose stop folio >/dev/null 2>&1
compose run --rm --no-deps -T folio python main.py restore "$archive" --force >"$WORK/restore.log" 2>&1 \
    || { cat "$WORK/restore.log"; fail "main.py restore failed in a one-off container"; }
up "not healthy after a restore"
[ "$(search)" -gt 0 ] || fail "the search hit is gone after the restore"
[ "$(search2)" = 0 ] || fail "the document uploaded after the backup is still there after the restore"
compose exec -T folio sh -c 'ls -d /app/storage/.folio-work/before-restore-*' >/dev/null \
    || fail "the previous storage contents were not kept in storage/.folio-work"
echo "ok: main.py backup into the folio-backups volume; after a new upload, main.py restore --force brings back exactly the backed-up data"

# ---- split state: each piece missing or replaced on its own must stop startup with a clear message ----
# The data lives in two volumes (folio-db, folio-storage) plus the settings in .env. A piece replaced by an empty
# volume is mounted through an extra override file, so the real volumes stay untouched for the next case.
swap() {  # swap <service> <volume:mount>...: run <service> with exactly these volumes; new names are created empty
    local service="$1"; shift
    {
        echo "services:"
        echo "  $service:"
        echo "    volumes: !override"
        for v in "$@"; do echo "      - $v"; done
        echo "volumes:"
        for v in "$@"; do echo "  ${v%%:*}:"; done
    } > "$WORK/swap.yml"
}
# refused <what> <pattern> [compose args...]: Folio must not turn healthy and its log must say <pattern>, again
# after restart: unless-stopped has started it a second time
refused() {
    local what="$1" pattern="$2" first=0 cid; shift 2
    compose "$@" up -d --no-build --no-deps folio >/dev/null 2>&1
    cid="$(compose "$@" ps -aq folio)"
    count() { docker logs "$cid" 2>&1 | grep -c -- "$pattern" || true; }
    for _ in $(seq 1 60); do first="$(count)"; [ "$first" -ge 1 ] && break; sleep 2; done
    [ "$first" -ge 1 ] || { docker logs --tail 40 "$cid"; fail "$what: Folio did not refuse with '$pattern'"; }
    for _ in $(seq 1 60); do
        [ "$(docker inspect -f '{{.RestartCount}}' "$cid")" -ge 1 ] && [ "$(count)" -gt "$first" ] && break
        sleep 2
    done
    [ "$(count)" -gt "$first" ] || fail "$what: Folio did not refuse again after restart: unless-stopped started it again"
    [ "$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$cid")" != healthy ] \
        || fail "$what: Folio turned healthy"
    compose "$@" rm -sf folio >/dev/null 2>&1
}
compose stop folio >/dev/null 2>&1

# 1. The storage volume missing (an empty one in its place): the database lists files that are not there
swap folio folio-storage-replaced:/app/storage
refused "storage volume missing" "none of the 1 most recent" -f "$WORK/swap.yml"
echo "ok: storage volume missing -> Folio refuses to start (the database lists files the storage does not hold)"

# 2. The database volume missing (an empty database in its place) next to the old uploads
compose stop db >/dev/null 2>&1
swap db folio-db-replaced:/var/lib/postgresql/data
compose -f "$WORK/swap.yml" up -d --no-build --wait --wait-timeout 120 db >/dev/null 2>&1 \
    || fail "the empty database did not start"
refused "database volume missing" "The database is new" -f "$WORK/swap.yml"
tables="$(compose -f "$WORK/swap.yml" exec -T db psql -U folio -d folio -tAc \
    "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")"
[ "$tables" = 0 ] || fail "database volume missing: Folio migrated the empty database ($tables tables) before refusing"
echo "ok: database volume missing -> Folio refuses to start, also after a restart, and leaves the empty database unmigrated"
compose -f "$WORK/swap.yml" rm -sf db >/dev/null 2>&1

# 3. POSTGRES_PASSWORD changed in .env: PostgreSQL keeps the password its volume was created with
compose up -d --no-build --wait --wait-timeout 120 db >/dev/null 2>&1 || fail "the database did not start again"
POSTGRES_PASSWORD="changed-$POSTGRES_PASSWORD" refused "POSTGRES_PASSWORD changed" "password authentication failed"
echo "ok: POSTGRES_PASSWORD changed -> Folio refuses to start: password authentication failed"

# 4. MCP Gate's secrets volume missing: the signing keys in its database cannot be decrypted, so it must not start
compose stop mcp-gate >/dev/null 2>&1
swap mcp-gate mcp-gate-data:/app/data mcp-gate-secrets-replaced:/app/secrets
compose -f "$WORK/swap.yml" up -d --no-build --no-deps mcp-gate >/dev/null 2>&1
gate_refused=0
for _ in $(seq 1 45); do
    if [ "$(compose -f "$WORK/swap.yml" logs mcp-gate 2>&1 | grep -c "The encryption key does not match the database")" -ge 1 ]; then gate_refused=1; break; fi
    sleep 2
done
gate_health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' \
    "$(compose -f "$WORK/swap.yml" ps -aq mcp-gate)")"
[ "$gate_refused" = 1 ] && [ "$gate_health" != healthy ] \
    || { compose -f "$WORK/swap.yml" logs --tail 40 mcp-gate; fail "MCP Gate started without its secrets volume"; }
compose -f "$WORK/swap.yml" rm -sf mcp-gate >/dev/null 2>&1
echo "ok: MCP Gate's secrets volume missing -> MCP Gate refuses to start (its stored signing keys cannot be decrypted)"

# 5. EMBEDDING_DIMENSION changed in .env: not a missing piece. Folio starts, logs the mismatch and the files are
#    marked for re-indexing (console: "Re-index recommended")
{
    echo "services:"
    echo "  folio:"
    echo "    environment:"
    echo '      EMBEDDING_DIMENSION: "128"'
} > "$WORK/dim.yml"
compose -f "$WORK/dim.yml" up -d --no-build --wait --wait-timeout 180 >"$WORK/up.log" 2>&1 \
    || { cat "$WORK/up.log"; fail "EMBEDDING_DIMENSION changed: Folio did not start"; }
[ "$(compose logs folio 2>&1 | grep -c "Vector dim mismatch")" -ge 1 ] || fail "EMBEDDING_DIMENSION changed: no mismatch in the log"
echo "ok: EMBEDDING_DIMENSION changed -> Folio starts and logs the dimension mismatch (files to re-index)"

# 6. The console session secret (in the storage volume) deleted: a new one, console users sign in again
compose exec -T folio rm -f /app/storage/.console_session_secret

# Every piece back: the same data again
compose up -d --no-build --force-recreate folio >/dev/null 2>&1
up "not healthy with every volume back"
compose exec -T folio test -s /app/storage/.console_session_secret || fail "no new console session secret"
TOKEN="$(python3 scripts/docker_smoke_token.py "$GATE" "$EMAIL" "$GATE_PASSWORD" "$AUDIENCE")" \
    || fail "no token from MCP Gate with every volume back"
AUTH=(-H "Authorization: Bearer $TOKEN")
[ "$(search)" -gt 0 ] || fail "the search hit is gone with every volume back"
echo "ok: with every volume back the same folder, file and search hit are there (and a new console session secret)"

echo "docker smoke test passed"
