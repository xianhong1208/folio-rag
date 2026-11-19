"""Every database connection path resolves the URL the same way.

SQLAlchemy, psycopg2 and Alembic only read the URL; when it omits the port they
default to 5432 and ignore the separate `database.port` field. resolve_db_url
injects that port, and every connection path (runtime engines, startup bootstrap,
migrations, Alembic, the vector store, the integration-test harness) must use it.
Before this, only the runtime engines did: with `port: 5444` and a portless URL
the server's startup bootstrap connected to 5432 and failed authentication.
"""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.utils.db_url import alembic_safe_url, db_url_from_config_dir, resolve_db_url

_REPO = Path(__file__).resolve().parents[1]


def _cfg(url, port):
    return SimpleNamespace(url=url, port=port)


class TestResolveDbUrl:
    def test_injects_field_port_when_url_has_none(self):
        r = resolve_db_url(_cfg("postgresql://postgres:pw@localhost/db", 5444))
        assert ":5444/" in r

    def test_url_port_takes_precedence_over_field(self):
        r = resolve_db_url(_cfg("postgresql://postgres:pw@localhost:5432/db", 5444))
        assert ":5432/" in r and ":5444" not in r

    def test_no_port_anywhere_left_untouched(self):
        r = resolve_db_url(_cfg("postgresql://postgres:pw@localhost/db", None))
        assert "localhost/db" in r and ":54" not in r

    def test_password_preserved(self):
        # Used for connecting, so the password must not be masked.
        r = resolve_db_url(_cfg("postgresql://postgres:secret@localhost/db", 5444))
        assert "secret" in r


def test_alembic_safe_url_escapes_percent():
    url = "postgresql://u:p%40ss@h:5444/db"
    assert alembic_safe_url(url) == "postgresql://u:p%%40ss@h:5444/db"


class TestFromConfigDir:
    def _dir(self, tmp_path, database_yaml):
        from tests.config_helpers import write_config_dir
        d = write_config_dir(tmp_path / "cfg", {})
        (d / "database.yaml").write_text(database_yaml, encoding="utf-8")
        return d

    def test_applies_port_field(self, tmp_path):
        d = self._dir(tmp_path, "database:\n  url: postgresql://u:pw@localhost/rag\n  port: 5444\n")
        assert ":5444/rag" in db_url_from_config_dir(d)

    def test_expands_env_placeholder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ITEST_DB_URL_X", "postgresql://u:pw@dbhost/other")
        monkeypatch.setenv("ITEST_DB_PORT_X", "5444")
        d = self._dir(tmp_path, 'database:\n  url: "${ITEST_DB_URL_X:-postgresql://u:pw@localhost/rag}"\n'
                      '  port: "${ITEST_DB_PORT_X:-}"\n')
        assert "dbhost:5444/other" in db_url_from_config_dir(d)

    def test_empty_port_means_the_url_decides(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ITEST_DB_PORT_X", raising=False)
        d = self._dir(tmp_path, 'database:\n  url: postgresql://u:pw@h:6000/rag\n'
                      '  port: "${ITEST_DB_PORT_X:-}"\n')
        assert db_url_from_config_dir(d).endswith("@h:6000/rag")

    def test_missing_url_raises(self, tmp_path):
        d = self._dir(tmp_path, "database:\n  port: 5444\n")
        with pytest.raises(RuntimeError, match="no database.url"):
            db_url_from_config_dir(d)


def test_migrator_engine_uses_resolved_url():
    from db import migrate
    conf = _cfg("postgresql://postgres:pw@localhost/rag", 5444)
    with patch.object(migrate.Config, "get_database_config", return_value=conf), \
         patch.object(migrate, "create_engine") as create_engine:
        m = migrate.DatabaseMigrator()
    assert ":5444/rag" in create_engine.call_args.args[0]
    assert ":5444/rag" in m.db_url


def test_vector_store_uses_resolved_port():
    from src.domain.rag import vector_store_manager as vsm_mod
    fake_cfg = MagicMock()
    fake_cfg.database = _cfg("postgresql://postgres:pw@localhost/rag", 5444)
    fake_cfg.database.vector_store_engines = "per_folder"
    with patch.object(vsm_mod.Config, "get_config_model", return_value=fake_cfg):
        vsm = vsm_mod.VectorStoreManager(embed_dim=8)
    with patch.object(vsm_mod, "PGVectorStore") as ctor:
        vsm._create_vector_store("1_abc")
    # Private engines (per_folder mode) connect with the resolved URL, port applied.
    kw = ctor.call_args.kwargs
    assert kw["connection_string"].endswith("@localhost:5444/rag")
    assert kw["async_connection_string"].startswith("postgresql+asyncpg://")
    assert kw["async_connection_string"].endswith("@localhost:5444/rag")


def test_vector_store_shared_mode_uses_shared_engines():
    """Default mode: stores run on db.db's engines (which resolve the port themselves)."""
    from src.domain.rag import vector_store_manager as vsm_mod
    fake_cfg = MagicMock()
    fake_cfg.database = _cfg("postgresql://postgres:pw@localhost/rag", 5444)
    with patch.object(vsm_mod.Config, "get_config_model", return_value=fake_cfg):
        vsm = vsm_mod.VectorStoreManager(embed_dim=8)
    eng, aeng = MagicMock(), MagicMock()
    with patch.object(vsm_mod, "PGVectorStore") as ctor, \
         patch.object(vsm_mod, "get_engine", return_value=eng), \
         patch.object(vsm_mod, "get_async_engine", return_value=aeng):
        vsm._create_vector_store("1_abc")
    kw = ctor.call_args.kwargs
    assert kw["engine"] is eng and kw["async_engine"] is aeng
    assert "pw" not in kw["connection_string"]


# Files that open database connections outside db/db.py's engines.
_CONNECTION_PATHS = [
    "main.py",
    "db/db.py",
    "db/migrate.py",
    "db/migrate/alembic/env.py",
    "src/domain/rag/vector_store_manager.py",
    "src/utils/db_bootstrap.py",
    "tests/integration/conftest.py",
]
_RAW_URL = re.compile(r"\b(?:db_conf|database|db_config)\.url\b|getattr\([^)]*['\"]url['\"]\)")


@pytest.mark.parametrize("rel", _CONNECTION_PATHS)
def test_no_connection_path_reads_the_raw_url(rel):
    """Guard: reading database.url directly bypasses the port rule."""
    src = (_REPO / rel).read_text(encoding="utf-8")
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert not _RAW_URL.search(code), f"{rel} reads database.url directly; use resolve_db_url"
