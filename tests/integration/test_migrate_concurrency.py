
"""Integration tests -- migration concurrency protection via pg_advisory_lock.

Two independent DatabaseMigrator instances upgrade concurrently (simulating a
rolling deploy); the advisory lock serializes them so both return success and
the version converges to head, rather than one crashing on concurrent DDL.
"""

import threading

from db.migrate import DatabaseMigrator


def test_concurrent_upgrade_serialized_by_advisory_lock(itest_db):
    """After a one-step downgrade, two threads upgrade('head') concurrently: both succeed, version converges."""
    m0 = DatabaseMigrator()
    head = m0._get_head_revision()
    assert m0.downgrade("-1"), "setup: step back one revision so there is a migration to apply"

    results = {}

    def _run(tag: str):
        try:
            # Each with its own migrator (independent engine/connection) -- simulating two processes
            results[tag] = DatabaseMigrator().upgrade("head")
        except Exception as e:  # must not blow each other up
            results[tag] = e

    t1 = threading.Thread(target=_run, args=("a",))
    t2 = threading.Thread(target=_run, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=60)
    t2.join(timeout=60)

    assert results.get("a") is True and results.get("b") is True, f"one of the concurrent upgrades failed: {results}"
    assert m0._get_current_revision() == head, "the revision must converge on head"


def test_real_chain_passes_new_guards(itest_db):
    """The real versions/ chain still passes under the new triple guard (parseable / single head / no orphans)."""
    chain = DatabaseMigrator._discover_revisions(DatabaseMigrator().versions_dir)
    assert len(chain) >= 3
    assert chain[-1][0] == "20261119f0b001"


def test_files_metadata_migration_round_trip(itest_db):
    """20261119f0b001: down removes Files.metadata and its GIN indexes, up adds them back;
    existing rows read '{}' and keep their tags; running up twice is harmless."""
    import importlib.util
    import uuid

    from sqlalchemy import text

    from db.db import get_engine
    from db.filedb import FileDB
    from db.folderdb import FolderDB

    m = DatabaseMigrator()
    folder = FolderDB.create(name=f"mig-{uuid.uuid4().hex[:6]}", user_token="sub:itest-mig")
    f = FileDB.create(folder_id=folder.id, file_name="a.txt", file_path=f"/tmp/{uuid.uuid4().hex}",
                      file_size=1, tags=["Keep", "me"], doc_metadata={"y": 1})
    eng = get_engine()

    def state():
        with eng.connect() as c:
            cols = c.execute(text("SELECT count(*) FROM information_schema.columns "
                                  "WHERE table_name = 'Files' AND column_name = 'metadata'")).scalar()
            idx = c.execute(text("SELECT count(*) FROM pg_indexes WHERE tablename = 'Files' "
                                 "AND indexname IN ('idx_files_tags_gin', 'idx_files_metadata_gin')")).scalar()
            tags = c.execute(text('SELECT tags FROM "Files" WHERE id = :i'), {"i": f.id}).scalar()
        return cols, idx, tags

    try:
        assert m.downgrade("20261113e0b001")
        assert state() == (0, 0, ["Keep", "me"])
        assert m.upgrade("head")
        assert state() == (1, 2, ["Keep", "me"])
        with eng.connect() as c:
            assert c.execute(text('SELECT metadata FROM "Files" WHERE id = :i'), {"i": f.id}).scalar() == {}
        path = m.versions_dir / "20261119_files_metadata.py"
        spec = importlib.util.spec_from_file_location("mig_f0b001", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from unittest.mock import patch
        with eng.begin() as c, patch.object(mod.op, "execute", side_effect=lambda sql: c.execute(text(sql))):
            mod.upgrade()                                    # idempotent
        assert state() == (1, 2, ["Keep", "me"])
    finally:
        with eng.begin() as c:
            c.execute(text('DELETE FROM "Files" WHERE folder_id = :i'), {"i": folder.id})
            c.execute(text('DELETE FROM "Folders" WHERE id = :i'), {"i": folder.id})
