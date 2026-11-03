
"""DB Bootstrap — ensure the target DB exists and extensions are enabled.

Called from main.py's startup flow, **before Alembic migration**. It solves the
chicken-and-egg problem: CREATE DATABASE cannot run inside its own database and
must connect to the default `postgres` DB.

Flow:
1. Query pg_database to confirm the target DB exists (parameterized query).
2. If absent → connect to the default postgres DB → CREATE DATABASE (dbname
   allowlisted + escaped).
3. Connect to the target DB → CREATE EXTENSION IF NOT EXISTS vector (pgvector).

NOTE: Chinese word segmentation for BM25/FTS is now done in Python via CKIP
(ckip-transformers, see src/domain/rag/ckip_segmenter.py), NOT pg_jieba. CKIP
emits space-joined tokens tokenized by Postgres with the `simple` config, so no
Chinese FTS dictionary extension is needed.

Security design:
- SQLAlchemy throughout, no direct psycopg2 — credentials are handled by
  make_url (including special characters); the password is never split out of
  the URL, never lands in any dict/kwargs, and there is no intermediate
  credential-holding point in the code (CWE-259 Use of Hard-coded Password,
  CWE-522 Insufficiently Protected Credentials).
- The CREATE DATABASE identifier cannot be a bind param, so it is guarded by
  both the _validate_dbname allowlist and SQL-standard double-quote escaping
  (CWE-89 SQL Injection, including the second-order variant).
"""

from __future__ import annotations

import re

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ProgrammingError

# Postgres identifier limit is 63 bytes (NAMEDATALEN-1); anything longer is silently truncated.
_MAX_IDENTIFIER_BYTES = 63

# dbname allowlist — a fail-fast check on DATABASE_URL at startup.
# This is both a config sanity check and defense-in-depth: CREATE DATABASE also applies `"` doubling.
_DBNAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$-]*")


def _validate_dbname(dbname: str) -> str:
    """Validate the dbname parsed from DATABASE_URL; raise ValueError if invalid."""
    if not _DBNAME_RE.fullmatch(dbname):
        raise ValueError(
            f"Invalid database name in DATABASE_URL: {dbname!r}. "
            f"Only letters/digits/underscore/hyphen/$ are allowed, and it must "
            f"start with a letter or underscore"
        )
    if len(dbname.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
        raise ValueError(
            f"Database name in DATABASE_URL exceeds the Postgres limit of "
            f"{_MAX_IDENTIFIER_BYTES} bytes (would be silently truncated): {dbname!r}"
        )
    return dbname


def _ensure_database_exists(db_url: str, logger) -> None:
    """Create the target DB if it does not exist (idempotent, handles concurrent races).

    CREATE DATABASE requires connecting to the postgres system DB. make_url
    safely handles special characters in credentials and swaps the target dbname
    to postgres; AUTOCOMMIT is used because CREATE DATABASE cannot run inside a
    transaction.
    """
    target_dbname = _validate_dbname(make_url(db_url).database or "")

    admin_url = make_url(db_url).set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": target_dbname},
            ).scalar()
            if exists:
                logger.info(f"✅ Database '{target_dbname}' exists")
                return

            logger.warning(f"⚠️ Database '{target_dbname}' not found, creating...")
            # The identifier cannot be a bind param. SQL-standard escaping: wrap
            # in double quotes and double any embedded ". The dbname has also
            # passed the _validate_dbname allowlist (which contains no " at all);
            # this is defense-in-depth.
            safe_name = target_dbname.replace('"', '""')
            conn.execute(text(f'CREATE DATABASE "{safe_name}"'))
            logger.info(f"✅ Database '{target_dbname}' created")
    except ProgrammingError as e:
        # Another process created it first under concurrency (duplicate_database) → treat as success.
        if "already exists" in str(e).lower():
            logger.info(f"✅ Database '{target_dbname}' was created concurrently")
            return
        raise
    finally:
        admin_engine.dispose()


def _enable_extensions(db_url: str, logger) -> None:
    """Register pgvector in the DB (required). Chinese BM25 segmentation is now
    handled by Python CKIP, so pg_jieba is no longer needed.

    Uses CREATE EXTENSION IF NOT EXISTS; the OS-level .so must already be
    installed (expected to be provided by the Postgres Docker image).

    Args:
        db_url: PostgreSQL connection string.
        logger: logger for server-facing logs.
    """
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        # pgvector — required; nothing works without it
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("✅ pgvector extension ready")
        except Exception as e:
            err = str(e).lower()
            if "could not open extension control file" in err or "control file" in err:
                logger.error(
                    f"❌ pgvector NOT installed at OS level.\n"
                    f"   Check that your Postgres image ships pgvector, or install it on bare metal:\n"
                    f"     apt install postgresql-XX-pgvector\n"
                    f"   Original error: {e}"
                )
            else:
                logger.error(f"❌ pgvector setup failed: {e}")
            raise

        # NOTE: pg_jieba is no longer required. Chinese word segmentation for
        # BM25/full-text search is now done in Python via CKIP
        # (ckip-transformers, see src/domain/rag/ckip_segmenter.py): CKIP
        # produces a space-joined token string that Postgres tokenizes with the
        # `simple` config. Postgres therefore needs no Chinese FTS dictionary
        # extension.

    engine.dispose()


def ensure_hnsw_for_table(engine, table_name: str, logger=None) -> bool:
    """Ensure a single PGVector table has its HNSW index and lookup indexes (idempotent).

    Purpose: index a new folder's table right after its first chunk write, without
    waiting for the next restart. Near-instant when everything already exists.
    The lookup indexes (see ensure_lookup_indexes_for_table) are checked on every
    call so tables created before they existed pick them up too.

    Args:
        engine: SQLAlchemy engine connected to the target DB (avoids rebuilding the engine each time)
        table_name: PGVector table name, e.g. 'data_2_b09d9acf-d049-...'
        logger: optional

    Returns:
        True if a new HNSW index was created, False if it already existed.
    """
    created = _ensure_hnsw_index(engine, table_name, logger)
    ensure_lookup_indexes_for_table(engine, table_name, logger)
    return created


def _ensure_hnsw_index(engine, table_name: str, logger=None) -> bool:
    idx_name = table_name + "_hnsw_idx"
    sql = text(f'''
        CREATE INDEX IF NOT EXISTS "{idx_name}"
        ON "{table_name}"
        USING hnsw (embedding vector_cosine_ops)
        WITH (m=16, ef_construction=64)
    ''')
    try:
        with engine.begin() as conn:
            # Check whether the index already exists first (cheaper check).
            exists = conn.execute(text(
                "SELECT 1 FROM pg_indexes WHERE indexname = :idx LIMIT 1"
            ), {"idx": idx_name}).first()
            if exists:
                return False
            conn.execute(sql)
        if logger:
            logger.debug(f"✅ HNSW index created on {table_name}")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ HNSW index create failed for {table_name}: {e}")
        return False


# B-tree lookup indexes on every vector table. Neighbour expansion, read mode, chunk
# listing and per-file deletes filter by node_id or by (file_id, chunk_index); with only
# the HNSW index those were sequential scans (at 50k leaves: ~1.3 s → ~13 ms for
# neighbour expansion). The expressions must match chunk_lookup's WHERE clauses exactly.
# Short suffixes keep "data_<folder>_<uuid>" + suffix under Postgres' 63-char limit.
LOOKUP_INDEX_DDL = (
    ("_nid_idx", "(node_id)"),
    ("_fid_idx", "((metadata_->>'file_id'), ((metadata_->>'chunk_index')::int))"),
)


def ensure_lookup_indexes_for_table(engine, table_name: str, logger=None) -> int:
    """Create the node_id and (file_id, chunk_index) lookup indexes if missing (idempotent).

    Non-fatal: a failure (e.g. a non-integer chunk_index in legacy rows) is logged and
    lookups simply stay sequential scans. Returns how many indexes were created.
    """
    created = 0
    for suffix, cols in LOOKUP_INDEX_DDL:
        idx_name = table_name + suffix
        try:
            with engine.begin() as conn:
                exists = conn.execute(text(
                    "SELECT 1 FROM pg_indexes WHERE indexname = :idx LIMIT 1"
                ), {"idx": idx_name}).first()
                if exists:
                    continue
                conn.execute(text(
                    f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" {cols}'))
            created += 1
            if logger:
                logger.debug(f"Lookup index {idx_name} created")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Lookup index {idx_name} create failed (not fatal): {e}")
    return created


# Data tables in the public schema that have an embedding column but no HNSW index.
_TABLES_MISSING_HNSW = r"""
    SELECT c.relname
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind = 'r'
      AND c.relname LIKE 'data\_%' ESCAPE '\'
      AND EXISTS (
        SELECT 1 FROM pg_attribute a
        WHERE a.attrelid = c.oid AND a.attname = 'embedding' AND NOT a.attisdropped)
      AND NOT EXISTS (
        SELECT 1 FROM pg_index i
        JOIN pg_class ic ON ic.oid = i.indexrelid
        JOIN pg_am am ON am.oid = ic.relam
        WHERE i.indrelid = c.oid AND am.amname = 'hnsw')
"""

HNSW_SWEEP_SQL = r"""
    DO $$
    DECLARE
      t TEXT;
      new_cnt INT := 0;
    BEGIN
      FOR t IN """ + _TABLES_MISSING_HNSW + r"""
      LOOP
        EXECUTE format(
          'CREATE INDEX IF NOT EXISTS %I ON %I USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)',
          t || '_hnsw_idx',
          t
        );
        new_cnt := new_cnt + 1;
      END LOOP;
      RAISE NOTICE 'HNSW sweep done: % vector tables newly indexed', new_cnt;
    END $$;
"""


def _ensure_hnsw_indexes(db_url: str, logger) -> None:
    """Create HNSW indexes on all PGVector data_* tables (idempotent; works around LlamaIndex's quoting bug).

    LlamaIndex's own HNSW creation does not quote identifiers containing UUID
    hyphens → it fails silently; we run IF NOT EXISTS ourselves to backfill.
    Without HNSW, search degrades linearly past ~1k chunks.

    Args:
        db_url: PostgreSQL connection string.
        logger: logger for server-facing logs.
    """
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
    # One round trip, and one catalog query inside it: an anti-join over pg_index /
    # pg_am picks only the data_* tables that have no HNSW index yet, instead of
    # scanning the pg_indexes view (which renders every index definition) once per
    # table. Startup cost no longer grows with the number of already-indexed folders.
    sql = text(HNSW_SWEEP_SQL)
    try:
        with engine.connect() as conn:
            conn.execute(sql)
        logger.info("✅ HNSW index sweep complete (all PGVector tables checked)")
    except Exception as e:
        # Non-fatal — it still runs without HNSW, search is just slower.
        logger.warning(
            f"⚠️ HNSW index sweep failed: {e}. "
            "search will fall back to sequential scan (slower)."
        )
    finally:
        engine.dispose()


def check_vector_dims(db_url: str, expected_dim: int, logger) -> list:
    """At startup, scan all data_* vector tables and report those whose dimension differs from config.

    pgvector's dimension is frozen at CREATE TABLE (stored in atttypmod); later
    config changes do not alter the table — a mismatched table only blows up at
    insert time with "expected N dimensions". This surfaces it loudly at startup
    and points to the fix (reindex that folder).

    Args:
        db_url: PostgreSQL connection string.
        expected_dim: config's rag.embedding.dimension.
        logger: server logger.

    Returns:
        [(table_name, actual_dim), ...] list of mismatches; returns an empty list
        on scan failure (does not block startup).
    """
    mismatched = []
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(r"""
                SELECT c.relname, a.atttypmod
                FROM pg_class c
                JOIN pg_attribute a ON a.attrelid = c.oid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND a.attname = 'embedding'
                  AND c.relkind = 'r'
                  AND c.relname LIKE 'data\_%' ESCAPE '\'
            """)).fetchall()
        for table_name, dim in rows:
            if dim is not None and dim > 0 and dim != expected_dim:
                mismatched.append((table_name, dim))
        if mismatched:
            for table_name, dim in mismatched:
                logger.error(
                    f"❌ Vector dim mismatch: table \"{table_name}\" was created with "
                    f"vector({dim}) but config dimension is {expected_dim}. "
                    f"Indexing into this folder WILL fail — reindex the folder "
                    f"(POST /api/rag/files/{{folder_id}}/reindex) to rebuild the table."
                )
        else:
            logger.info(
                f"✅ Vector dim check passed: {len(rows)} table(s) all match "
                f"dimension {expected_dim}"
            )
    except Exception as e:
        logger.warning(f"⚠️ Vector dim check skipped (non-fatal): {e}")
    finally:
        engine.dispose()
    return mismatched


def ensure_database_ready(db_url: str, logger) -> None:
    """Ensure the DB exists + extensions enabled + HNSW indexes backfilled (idempotent — safe to run repeatedly).

    Full declarative provisioning:
      Step 1: DB absent → create it
      Step 2: pgvector extension absent → enable it (Chinese segmentation now uses Python CKIP, no pg_jieba)
      Step 3: PGVector data_* tables missing an HNSW index → backfill them

    This function covers every scenario: switching DB / switching machine /
    Docker volume reset / first deployment. Safe to run repeatedly (IF NOT
    EXISTS guarantees idempotence).

    Args:
        db_url: SQLAlchemy DB URL, e.g. postgresql://user:pwd@host/dbname
        logger: server logger instance

    Raises:
        sqlalchemy.exc.OperationalError: even connecting to the postgres system DB failed (wrong password / service down)
        sqlalchemy.exc.ProgrammingError: no CREATEDB privilege, etc.
        Exception: pgvector extension creation failed (required; nothing works without it)
    """
    logger.info(f"🔍 Bootstrapping database '{make_url(db_url).database}'...")

    # Step 1: ensure DB exists (SQLAlchemy throughout; see the module docstring's security design)
    _ensure_database_exists(db_url, logger)

    # Step 2: ensure extensions
    _enable_extensions(db_url, logger)

    # Step 3: ensure HNSW indexes on all existing PGVector tables
    # (LlamaIndex's own creation breaks on the UUID-hyphen quoting bug, so we take this on ourselves)
    _ensure_hnsw_indexes(db_url, logger)
