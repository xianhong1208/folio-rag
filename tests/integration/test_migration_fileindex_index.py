"""Integration test -- 20261103d0b001 drops the duplicate FileIndices(file_id) index.

At head only the UNIQUE constraint's index covers file_id; one step down brings
idx_fileindices_file_id back; both directions are idempotent (re-running a step is
a no-op); and the upgrade keeps the index when no unique index exists.
"""

from sqlalchemy import text

REV = "20261103d0b001"


def _file_id_indexes(engine):
    with engine.connect() as c:
        return sorted(c.execute(text(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' "
            "AND tablename = 'FileIndices' AND indexdef LIKE '%(file_id)%'")).scalars().all())


def _run(engine, sql):
    with engine.begin() as c:
        c.execute(text(sql))


def test_duplicate_index_dropped_and_restored(itest_db):
    import importlib.util
    from db.db import get_engine
    from db.migrate import DatabaseMigrator

    m = DatabaseMigrator()
    eng = get_engine()
    # Later migrations are stepped back first, and re-applied at the end.
    assert m.downgrade(REV)
    assert m._get_current_revision() == REV
    assert _file_id_indexes(eng) == ["FileIndices_file_id_key"]

    assert m.downgrade("-1")
    assert "idx_fileindices_file_id" in _file_id_indexes(eng)
    assert m.upgrade(REV)
    assert m._get_current_revision() == REV
    assert _file_id_indexes(eng) == ["FileIndices_file_id_key"]

    # Re-running either direction's SQL is a no-op.
    path = m.versions_dir / "20261103_drop_redundant_fileindex_index.py"
    spec = importlib.util.spec_from_file_location("mig_d0b001", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _run(eng, mod.UPGRADE_SQL)
    assert _file_id_indexes(eng) == ["FileIndices_file_id_key"]
    down = 'CREATE INDEX IF NOT EXISTS idx_fileindices_file_id ON "FileIndices" (file_id)'
    _run(eng, down)
    _run(eng, down)
    assert "idx_fileindices_file_id" in _file_id_indexes(eng)

    # Without the unique constraint the plain index is the only one: keep it.
    _run(eng, 'ALTER TABLE "FileIndices" DROP CONSTRAINT "FileIndices_file_id_key"')
    try:
        _run(eng, mod.UPGRADE_SQL)
        assert _file_id_indexes(eng) == ["idx_fileindices_file_id"]
    finally:
        _run(eng, 'ALTER TABLE "FileIndices" ADD CONSTRAINT "FileIndices_file_id_key" UNIQUE (file_id)')
        _run(eng, mod.UPGRADE_SQL)
    assert _file_id_indexes(eng) == ["FileIndices_file_id_key"]
    assert m.upgrade("head")
