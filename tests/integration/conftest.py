
"""Integration-test harness against a throwaway PostgreSQL DB.

Opt-in via RAG_RUN_DB_ITESTS=1 (else the group is skipped); a set env but an
unreachable DB also skips rather than fails. Opt-in is deliberate so a machine
that happens to have a matching postgres never gets databases created/dropped
without consent. Each run creates a fresh agentic_rag_itest, applies the full
alembic chain, and DROPs it when done, never touching an existing DB.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

# Overridable so parallel checkouts can each use their own throwaway database.
_ITEST_DB = os.getenv("RAG_ITEST_DB", "agentic_rag_itest")
_REPO = Path(__file__).resolve().parents[2]
_TMP_CONFIG = _REPO / "config" / "_itest_config.yaml"


def _base_url():
    """Connection info from config/config.yaml (everything except dbname).

    Resolved exactly like the server does it: ${VAR:-default} expanded and
    database.port applied when the URL has no port. Reading only the raw url line
    used to connect to 5432, find nothing and skip the whole group silently.
    """
    from src.utils.db_url import db_url_from_config_file
    return make_url(db_url_from_config_file(_REPO / "config" / "config.yaml"))


def _admin_conn():
    """Connect to the postgres admin DB (short timeout; on failure -> skip the whole integration group)."""
    import psycopg2
    u = _base_url()
    return psycopg2.connect(
        host=u.host or "localhost", port=u.port or 5432,
        user=u.username, password=u.password,
        dbname="postgres", connect_timeout=2,
    )


def _db_available() -> bool:
    try:
        conn = _admin_conn()
        conn.close()
        return True
    except Exception:
        return False


if os.getenv("RAG_RUN_DB_ITESTS") != "1":
    pytest.skip(
        "integration tests are opt-in: set RAG_RUN_DB_ITESTS=1 (so a machine that happens "
        "to run postgres never gets databases created without consent)",
        allow_module_level=True,
    )

if not _db_available():
    pytest.skip(
        "PostgreSQL is unreachable — skipping integration tests (unit tests unaffected)",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def itest_db():
    """Throwaway integration DB: create -> full alembic chain -> yield -> DROP."""
    # 1. Recreate a fresh DB
    conn = _admin_conn()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS {_ITEST_DB}')
        cur.execute(f'CREATE DATABASE {_ITEST_DB}')
    conn.close()

    # 1b. Install extensions (pgvector is required for writes; Chinese
    # tokenization now uses Python CKIP, so pg_jieba is no longer needed)
    import psycopg2
    u = _base_url()
    econn = psycopg2.connect(
        host=u.host or "localhost", port=u.port or 5432,
        user=u.username, password=u.password,
        dbname=_ITEST_DB, connect_timeout=2,
    )
    econn.autocommit = True
    with econn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    econn.close()

    # 2. Temporary config (the config path safety check requires it to be
    # inside the repo) pointing at the throwaway DB
    # Write the resolved URL (port included) rather than patching the text: the
    # url line may be a ${DATABASE_URL:-...} placeholder, and an env override
    # would otherwise point the run at a real database.
    import yaml
    raw = yaml.safe_load((_REPO / "config" / "config.yaml").read_text(encoding="utf-8"))
    raw["database"]["url"] = _base_url().set(database=_ITEST_DB).render_as_string(
        hide_password=False)
    _TMP_CONFIG.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")

    from src.config.config_manager import Config
    Config.set_config(str(_TMP_CONFIG))

    # 3. Build the schema via the full alembic chain (revalidates the
    # migration chain on every integration round)
    from db.migrate import DatabaseMigrator
    assert DatabaseMigrator().upgrade("head"), "applying the full alembic chain failed"

    # 4. Bind the Session factory (get_engine internally does
    #    Session.configure(bind=...); the migrator uses its own engine and
    #    does not trigger this step)
    from db.db import get_engine
    get_engine()

    yield _ITEST_DB

    # 4. Cleanup: dispose the SQLAlchemy connection pool first, otherwise the DROP fails
    try:
        import db.db as dbdb
        if getattr(dbdb, "_engine", None) is not None:
            dbdb._engine.dispose()
    except Exception:
        pass
    conn = _admin_conn()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS {_ITEST_DB} WITH (FORCE)')
    conn.close()
    _TMP_CONFIG.unlink(missing_ok=True)


# ---- Shared data fixtures (FileIndex has an FK: Folder before File) ----------

@pytest.fixture()
def folder(itest_db):
    """A clean folder per test (unique name, to avoid cross-test interference)."""
    import uuid as _uuid
    from db.folderdb import FolderDB
    return FolderDB.create(
        name=f"itest-{_uuid.uuid4().hex[:8]}",
        description="integration test folder",
        user_token="itest-token",
    )


@pytest.fixture()
def file_row(itest_db, folder):
    import uuid as _uuid
    from db.filedb import FileDB
    return FileDB.create(
        folder_id=folder.id,
        file_name="doc.pdf",
        file_path=f"/tmp/itest/{_uuid.uuid4().hex}",
        file_size=1234,
        mime_type="application/pdf",
    )
