"""Integration test -- the embedding model and dimension a file was indexed with.

20261113e0b001 adds FileIndices.embedding_dimension and backfills it from each
folder's pgvector table; upsert_indexed records it; the console's folder count
flags files indexed with another embedding model or dimension.
"""

import importlib.util
import uuid
from unittest.mock import patch

from sqlalchemy import text

from src.domain.rag.chunk_identity import CHUNK_PIPELINE_VERSION

REV = "20261113e0b001"


def _index(file_row, folder, model="emb-a", dim=8, table="t"):
    from db.fileindexdb import FileIndexDB
    return FileIndexDB.upsert_indexed(
        file_id=file_row.id, folder_id=folder.id, index_id=f"file_{file_row.id}",
        vector_store_table=table, num_chunks=3, embedding_model=model,
        chunk_size=256, chunk_overlap=50, content_hash="h",
        pipeline_version=CHUNK_PIPELINE_VERSION, embedding_dimension=dim)


def _dim(eng, file_id):
    with eng.connect() as c:
        return c.execute(text('SELECT embedding_dimension FROM "FileIndices" WHERE file_id = :f'),
                         {"f": file_id}).scalar()


def test_upsert_records_the_dimension(itest_db, folder, file_row):
    from db.db import get_engine
    _index(file_row, folder, dim=1024)
    assert _dim(get_engine(), file_row.id) == 1024
    _index(file_row, folder, dim=768)  # re-index updates it
    assert _dim(get_engine(), file_row.id) == 768


def test_folder_count_flags_another_embedding(itest_db, folder, file_row):
    from db import admin_stats
    _index(file_row, folder, model="emb-a", dim=8)

    def stale(current):
        with patch("db.admin_stats.configured_embedding", return_value=current):
            row = next(f for f in admin_stats.folders_with_index_stats() if f["id"] == folder.id)
            me = next(r for r in admin_stats.files_with_index_status(folder.id)
                      if r["id"] == str(file_row.id))
        return row["reindex_recommended_files"], me["reindex_reason"], me["embedding_dimension"]

    assert stale(("emb-a", 8)) == (0, None, 8)
    assert stale(("emb-b", 8)) == (1, "embedding", 8)
    assert stale(("emb-a", 16)) == (1, "embedding", 8)
    assert stale((None, None)) == (0, None, 8)


def _migration():
    from db.migrate import DatabaseMigrator
    m = DatabaseMigrator()
    path = m.versions_dir / "20261113_file_index_embedding_dimension.py"
    spec = importlib.util.spec_from_file_location("mig_e0b001", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return m, mod


def test_migration_adds_and_backfills_the_dimension(itest_db, folder):
    from db.db import get_engine
    from db.filedb import FileDB
    eng = get_engine()
    m, mod = _migration()
    assert m._get_current_revision() == REV

    suffix = uuid.uuid4().hex[:8]
    with_prefix, bare = f"data_{folder.id}_{suffix}", f"{folder.id}_{suffix}x"
    with eng.begin() as c:
        c.execute(text(f'CREATE TABLE "{with_prefix}" (id serial, embedding vector(8))'))
        c.execute(text(f'CREATE TABLE "data_{bare}" (id serial, embedding vector(12))'))
    rows = {}
    for name, table in (("a", with_prefix), ("b", bare), ("gone", "data_no_such_table"), ("blank", "")):
        f = FileDB.create(folder_id=folder.id, file_name=f"{name}-{suffix}.txt",
                          file_path=f"/tmp/itest/{uuid.uuid4().hex}", file_size=1, mime_type="text/plain")
        _index(f, folder, dim=None, table=table)
        rows[name] = f.id
    try:
        # One step down removes the column; the upgrade adds it back and backfills.
        assert m.downgrade("-1")
        with eng.connect() as c:
            assert c.execute(text(
                "SELECT count(*) FROM information_schema.columns WHERE table_name = 'FileIndices' "
                "AND column_name = 'embedding_dimension'")).scalar() == 0
        assert m.upgrade("head")
        assert m._get_current_revision() == REV
        assert _dim(eng, rows["a"]) == 8
        assert _dim(eng, rows["b"]) == 12
        assert _dim(eng, rows["gone"]) is None and _dim(eng, rows["blank"]) is None

        # Idempotent: re-running changes nothing and never overwrites a recorded value.
        with eng.begin() as c:
            c.execute(text('UPDATE "FileIndices" SET embedding_dimension = 99 WHERE file_id = :f'),
                      {"f": rows["a"]})
            c.execute(text('ALTER TABLE "FileIndices" ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER'))
            c.execute(text(mod.BACKFILL_SQL))
        assert _dim(eng, rows["a"]) == 99 and _dim(eng, rows["b"]) == 12
    finally:
        with eng.begin() as c:
            c.execute(text(f'DROP TABLE IF EXISTS "{with_prefix}"'))
            c.execute(text(f'DROP TABLE IF EXISTS "data_{bare}"'))
