
"""db/migrate.py runtime guards — chain integrity / multiple heads / advisory lock.

Three real risks addressed:
1. Revisions are parsed by regex, so a migration whose id contains characters
   outside [a-f0-9] was **silently skipped** (written but never applied, silent
   schema drift) → it must now fail loudly.
2. Multiple heads / bases / a broken chain used to be only a warning → only one
   chain was followed and the other silently went unapplied.
3. Multiple instances starting at once would run the same migration concurrently
   (no lock) → pg_advisory_lock.

Unit scope: _discover_revisions as a pure function (a tmp versions dir, no DB);
the advisory lock uses a fake engine to verify the lock/unlock/double-check
order. Concurrency is exercised in tests/integration/test_migrate_concurrency.py.
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from db.migrate import DatabaseMigrator, MigrationChainError


def _write_migration(d: Path, fname: str, rev, down):
    down_repr = "None" if down is None else f"'{down}'"
    (d / fname).write_text(
        f"revision: str = '{rev}'\ndown_revision = {down_repr}\n"
        f"def upgrade():\n    pass\n",
        encoding="utf-8",
    )


class TestDiscoverRevisions:
    def test_valid_linear_chain(self, tmp_path):
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        _write_migration(tmp_path, "002_b.py", "bbb222", "aaa111")
        _write_migration(tmp_path, "003_c.py", "ccc333", "bbb222")
        chain = DatabaseMigrator._discover_revisions(tmp_path)
        assert [rev for rev, _, _ in chain] == ["aaa111", "bbb222", "ccc333"]

    def test_empty_dir_ok(self, tmp_path):
        assert DatabaseMigrator._discover_revisions(tmp_path) == []

    def test_dunder_files_ignored(self, tmp_path):
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        (tmp_path / "__init__.py").write_text("", encoding="utf-8")
        assert len(DatabaseMigrator._discover_revisions(tmp_path)) == 1

    def test_unparseable_revision_id_fails_loud(self, tmp_path):
        """A revision id the regex cannot match (containing characters outside [a-f0-9]) must error, not be silently skipped — the root of "a migration written but never applied"."""
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        _write_migration(tmp_path, "002_bad.py", "20261001_add_xyz", "aaa111")  # contains an underscore
        with pytest.raises(MigrationChainError, match="002_bad.py"):
            DatabaseMigrator._discover_revisions(tmp_path)

    def test_multiple_heads_fail(self, tmp_path):
        """A fork (two files with the same down_revision) → two heads → error, not just following one."""
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        _write_migration(tmp_path, "002_b.py", "bbb222", "aaa111")
        _write_migration(tmp_path, "003_c.py", "ccc333", "aaa111")  # also chained after a → fork
        with pytest.raises(MigrationChainError, match="head"):
            DatabaseMigrator._discover_revisions(tmp_path)

    def test_multiple_bases_fail(self, tmp_path):
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        _write_migration(tmp_path, "002_b.py", "bbb222", None)  # a second base
        with pytest.raises(MigrationChainError, match="base"):
            DatabaseMigrator._discover_revisions(tmp_path)

    def test_dangling_down_revision_fails(self, tmp_path):
        """A down_revision pointing at a nonexistent rev: neither node is pointed to → manifests as multiple heads, and fails just as loudly (different classification, same outcome: never silent)."""
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        _write_migration(tmp_path, "002_b.py", "bbb222", "deadbeef")  # nonexistent
        with pytest.raises(MigrationChainError):
            DatabaseMigrator._discover_revisions(tmp_path)

    def test_orphan_cycle_fails(self, tmp_path):
        """An orphan cycle (c↔d pointing at each other): adds no base/head, but is unreachable from the main chain → broken-chain error."""
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        _write_migration(tmp_path, "002_b.py", "bbb222", "aaa111")
        _write_migration(tmp_path, "003_c.py", "ccc333", "ddd444")
        _write_migration(tmp_path, "004_d.py", "ddd444", "ccc333")
        with pytest.raises(MigrationChainError, match="orphan"):
            DatabaseMigrator._discover_revisions(tmp_path)


_LOCK_SQL = "pg_advisory_lock"
_UNLOCK_SQL = "pg_advisory_unlock"


def _bare_migrator(tmp_path) -> DatabaseMigrator:
    """Assemble manually, bypassing __init__ (which needs Config/DB), filling only the fields upgrade requires."""
    m = object.__new__(DatabaseMigrator)
    m._engine = MagicMock()
    m.versions_dir = tmp_path
    return m


def _lock_calls(conn) -> list:
    """Extract lock/unlock events from the fake connection's execute call sequence."""
    events = []
    for c in conn.execute.call_args_list:
        sql = str(c.args[0])
        if _LOCK_SQL in sql and _UNLOCK_SQL not in sql:
            events.append("lock")
        elif _UNLOCK_SQL in sql:
            events.append("unlock")
    return events


class TestAdvisoryLock:
    def test_noop_when_already_at_target_takes_no_lock(self, tmp_path):
        """Already at head: short-circuit return, never touching the advisory lock."""
        m = _bare_migrator(tmp_path)
        with patch.object(DatabaseMigrator, "_get_current_revision", return_value="aaa111"), \
             patch.object(DatabaseMigrator, "_get_head_revision", return_value="aaa111"):
            assert m.upgrade("head") is True
        m._engine.connect.assert_not_called()

    def test_apply_path_locks_and_unlocks(self, tmp_path):
        """An actual apply: acquire the lock → apply migrations → release (guaranteed in finally)."""
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        m = _bare_migrator(tmp_path)
        conn = MagicMock()
        m._engine.connect.return_value.__enter__.return_value = conn

        with patch.object(DatabaseMigrator, "_get_current_revision", return_value=None), \
             patch.object(DatabaseMigrator, "_get_head_revision", return_value="aaa111"), \
             patch.object(DatabaseMigrator, "_execute_upgrade", return_value=True) as ex:
            assert m.upgrade("head") is True
        assert _lock_calls(conn) == ["lock", "unlock"]
        ex.assert_called_once()

    def test_double_check_after_lock_skips_if_peer_already_upgraded(self, tmp_path):
        """Another instance upgraded to head while waiting for the lock: re-read current after acquiring → do not re-run."""
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        m = _bare_migrator(tmp_path)
        conn = MagicMock()
        m._engine.connect.return_value.__enter__.return_value = conn

        # Before the lock current=None (looks like it needs upgrading); after, current=head (a peer already finished)
        with patch.object(DatabaseMigrator, "_get_current_revision",
                          side_effect=[None, "aaa111"]), \
             patch.object(DatabaseMigrator, "_get_head_revision", return_value="aaa111"), \
             patch.object(DatabaseMigrator, "_execute_upgrade") as ex:
            assert m.upgrade("head") is True
        ex.assert_not_called()
        assert _lock_calls(conn) == ["lock", "unlock"]

    def test_unlock_even_when_migration_fails(self, tmp_path):
        _write_migration(tmp_path, "001_a.py", "aaa111", None)
        m = _bare_migrator(tmp_path)
        conn = MagicMock()
        m._engine.connect.return_value.__enter__.return_value = conn

        with patch.object(DatabaseMigrator, "_get_current_revision", return_value=None), \
             patch.object(DatabaseMigrator, "_get_head_revision", return_value="aaa111"), \
             patch.object(DatabaseMigrator, "_execute_upgrade", return_value=False):
            assert m.upgrade("head") is False
        assert _lock_calls(conn)[-1] == "unlock", "失敗路徑也必須解鎖"
