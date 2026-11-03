# PostgreSQL operations

Folio stores everything relational, plus one pgvector table per folder
(`data_<folder_id>_<uuid>`), in PostgreSQL 15 with the `vector` extension.
Uploaded files live on disk under `storage/`, not in the database.

## Current local setup

The development server is the `db` service in `docker-compose.yml`:

| Item | Value |
|---|---|
| Container / port | `pg`, `127.0.0.1:5444` |
| Data | bind mount `/srv/pgdata` |
| Image | `pg15-pgvector:0.8.7`, built from `docker/pgvector/Dockerfile`: official `pgvector/pgvector:0.8.7-pg15-bookworm` plus pg_jieba 1.1.1 compiled from source (two legacy databases still use `jiebacfg` generated columns; Folio itself does not need it) |
| Server | PostgreSQL 15.19, `max_connections=200`, `shm_size: 4gb` |
| Extension | `folio` database: `vector` 0.8.7. Other databases on the server may still carry the 0.5.1 extension SQL, which works with the 0.8.7 library. |

Folio does not depend on pgvector 0.8 features. Anything that would (halfvec,
`hnsw.iterative_scan`) has to check `SELECT extversion FROM pg_extension WHERE
extname = 'vector'` first, because other deployments may run older versions.

## Connection budget

Folio opens two shared pools, sync (psycopg2) and async (asyncpg), each at most
`database.pool_size + database.max_overflow` connections: 2 × (10 + 10) = 40 with
the shipped config. Every folder's vector store uses these pools
(`database.vector_store_engines: shared`), so the count does not grow with the
number of folders. Add the other services sharing the server when sizing
`max_connections`.

## Server settings

For this workload (HNSW index builds, large per-folder tables, short OLTP queries):

| Setting | Value | Why |
|---|---|---|
| `shared_buffers` | ~25% of RAM (4GB here) | keeps hot HNSW graphs and lookup indexes cached |
| `effective_cache_size` | ~50-75% of RAM (16GB here) | planner estimate only; favours index scans |
| `maintenance_work_mem` | 1-2GB (2GB here) | an HNSW build that fits in it is far faster; above that it spills to disk |
| `max_parallel_maintenance_workers` | 2-4 (4 here) | parallel HNSW builds (pgvector >= 0.6) |
| `shm_size` (Docker) | >= `maintenance_work_mem` | parallel builds use dynamic shared memory |
| `jit` | `off` | JIT compile time outweighs the gain on short vector/BM25 queries |

Changing `command:` or `shm_size` in the compose file requires recreating the
container (`docker compose up -d db`).

## Upgrading pgvector

1. Bump `PGVECTOR_TAG` in the Dockerfile (e.g. `0.8.8-pg15-bookworm`; keep the
   `pg15` major) and `image:` in the compose file, then `docker compose build db`.
2. Stop the services that use the database, then `docker compose stop db`.
3. Back up `/srv/pgdata` (copy of the stopped directory, or a
   snapshot). This is the only rollback path once step 5 has run.
4. `docker compose up -d db`. The data directory is reused unchanged.
5. In each database that should get the new SQL-level features:
   `ALTER EXTENSION vector UPDATE;` then check `SELECT extversion FROM pg_extension
   WHERE extname = 'vector';`. Databases left alone keep working on the new library.
6. Optionally `REINDEX INDEX CONCURRENTLY` large HNSW indexes to pick up build
   improvements; not required.

Rollback: before step 5, switching `image:` back and recreating the container is
enough. After step 5 the extension SQL references C functions an older
library does not have, so restore the step-3 copy of `pgdata` with the old image.

## Backups

- Physical: `pgBackRest` (full + incremental, WAL archiving, point-in-time
  recovery) or, at minimum, a scheduled `pg_basebackup` of the whole cluster.
  Physical backups include every per-folder vector table and their HNSW indexes.
- Logical, per database: `pg_dump -Fc folio` is portable across versions but
  restoring rebuilds every HNSW index, which is slow for large folders.
- Back up `storage/` at the same time (filesystem snapshot or rsync right after
  the database backup). The database only references those files; a restore
  needs both from the same point in time, otherwise FileIndices rows point at
  missing files or files exist without rows.
