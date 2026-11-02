"""Index operations against a real database: the 20261102 migration (IndexJobs.kind,
vector-table lookup indexes), interrupted/cancelled file rows in the console stats,
and the per-file FTS rebuild's file listing.
"""

import importlib.util
import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

_MIGRATION = Path(__file__).resolve().parents[2] / "versions" / "20261102_vector_lookup_indexes.py"


def _migration():
    spec = importlib.util.spec_from_file_location("_mig_20261102", _MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def engine(itest_db):
    from db.db import get_engine
    return get_engine()


@pytest.fixture()
def vector_table(engine):
    """A minimal llama-index-shaped table (node_id + JSON metadata_) with 40 chunks of 2 files."""
    name = f"data_9{uuid.uuid4().hex[:4]}_{uuid.uuid4()}"
    with engine.begin() as c:
        c.execute(text(f'CREATE TABLE "{name}" (id BIGSERIAL PRIMARY KEY, node_id VARCHAR, '
                       f'text VARCHAR, metadata_ JSON)'))
        for i in range(40):
            md = {"file_id": "fa" if i < 20 else "fb", "chunk_index": i % 20}
            c.execute(text(f'INSERT INTO "{name}" (node_id, text, metadata_) '
                           f'VALUES (:n, :t, CAST(:m AS JSON))'),
                      {"n": f"n{i}", "t": f"t{i}", "m": json.dumps(md)})
    yield name
    with engine.begin() as c:
        c.execute(text(f'DROP TABLE IF EXISTS "{name}"'))


def _indexes(engine, table):
    with engine.connect() as c:
        return {r[0] for r in c.execute(text(
            "SELECT indexname FROM pg_indexes WHERE tablename = :t"), {"t": table})}


def _plan(engine, sql, params):
    with engine.begin() as c:
        c.execute(text("SET LOCAL enable_seqscan = off"))
        return "\n".join(r[0] for r in c.execute(text("EXPLAIN " + sql), params))


class TestMigration:
    def test_indexjobs_has_kind_with_default(self, engine):
        with engine.connect() as c:
            row = c.execute(text(
                "SELECT column_default, is_nullable FROM information_schema.columns "
                "WHERE table_name = 'IndexJobs' AND column_name = 'kind'")).first()
        assert row is not None and "index" in row[0] and row[1] == "NO"

    def test_lookup_indexes_on_existing_tables_are_used(self, engine, vector_table):
        with engine.begin() as c:
            c.execute(text(_migration().UPGRADE_LOOKUP_SQL))
            c.execute(text(_migration().UPGRADE_LOOKUP_SQL))  # idempotent
        assert {vector_table + "_nid_idx", vector_table + "_fid_idx"} <= _indexes(engine, vector_table)
        # The exact predicates chunk_lookup issues must be able to use them.
        p1 = _plan(engine, f'SELECT text FROM "{vector_table}" WHERE node_id = :n', {"n": "n3"})
        assert vector_table + "_nid_idx" in p1
        p2 = _plan(engine, f'SELECT text FROM "{vector_table}" WHERE metadata_->>\'file_id\' = :f '
                           f"AND (metadata_->>'chunk_index')::int BETWEEN :s AND :e",
                   {"f": "fa", "s": 2, "e": 5})
        assert vector_table + "_fid_idx" in p2

    def test_bad_table_is_skipped_not_fatal(self, engine, vector_table):
        with engine.begin() as c:
            c.execute(text(f'INSERT INTO "{vector_table}" (node_id, metadata_) '
                           "VALUES ('bad', CAST('{\"file_id\": \"fz\", \"chunk_index\": \"x\"}' AS JSON))"))
            c.execute(text(_migration().UPGRADE_LOOKUP_SQL))  # must not raise
        idx = _indexes(engine, vector_table)
        assert vector_table + "_fid_idx" not in idx
        assert vector_table + "_nid_idx" in idx  # the other index is unaffected

    def test_downgrade_drops_lookup_indexes(self, engine, vector_table):
        mig = _migration()
        with engine.begin() as c:
            c.execute(text(mig.UPGRADE_LOOKUP_SQL))
            c.execute(text(mig.DOWNGRADE_LOOKUP_SQL))
        idx = _indexes(engine, vector_table)
        assert not any(i.endswith(("_nid_idx", "_fid_idx")) for i in idx)


class TestRuntimeLookupIndexes:
    def test_new_table_gets_lookup_indexes(self, engine, vector_table):
        from src.utils.db_bootstrap import ensure_lookup_indexes_for_table
        assert ensure_lookup_indexes_for_table(engine, vector_table) == 2
        assert ensure_lookup_indexes_for_table(engine, vector_table) == 0
        assert {vector_table + "_nid_idx", vector_table + "_fid_idx"} <= _indexes(engine, vector_table)


class TestFtsRebuildListing:
    def test_lists_distinct_file_ids(self, engine, vector_table):
        from src.adapter.fts_rebuild import list_chunk_file_ids
        assert list_chunk_file_ids(engine, vector_table) == ["fa", "fb"]

    def test_missing_table_is_empty(self, engine):
        from src.adapter.fts_rebuild import list_chunk_file_ids
        assert list_chunk_file_ids(engine, f"data_0_{uuid.uuid4()}") == []


class TestInterruptedFiles:
    def test_interrupted_and_cancelled_rows_need_retry(self, itest_db, folder, file_row):
        from db import admin_stats
        from db.filedb import FileDB
        from db.fileindexdb import FileIndexDB
        other = FileDB.create(folder_id=folder.id, file_name="b.pdf",
                              file_path=f"/tmp/itest/{uuid.uuid4().hex}", file_size=1,
                              mime_type="application/pdf")
        FileIndexDB.mark_failed(file_row.id, "Interrupted by server restart — use Retry",
                                folder_id=folder.id, status="interrupted")
        FileIndexDB.mark_failed(other.id, "Cancelled by user", folder_id=folder.id,
                                status="cancelled")
        rows = {r["id"]: r for r in admin_stats.files_with_index_status(folder.id)}
        assert rows[str(file_row.id)]["status"] == "interrupted"
        assert rows[str(file_row.id)]["error"].startswith("Interrupted by server restart")
        assert rows[str(other.id)]["status"] == "cancelled"
        frow = next(f for f in admin_stats.folders_with_index_stats() if f["id"] == folder.id)
        assert frow["failed_files"] == 2 and frow["indexed_files"] == 0

    def test_job_kind_round_trips(self, itest_db, folder):
        from db import admin_stats
        from db.indexjobdb import IndexJobDB
        jid = str(uuid.uuid4())
        IndexJobDB.upsert({"job_id": jid, "folder_id": folder.id, "status": "interrupted",
                           "kind": "rebuild_fts", "last_updated_at": "2099-01-01T00:00:00"})
        assert IndexJobDB.get_by_id(jid)["kind"] == "rebuild_fts"
        js = {j["job_id"]: j for j in admin_stats.recent_jobs(50)}
        assert js[jid]["kind"] == "rebuild_fts" and js[jid]["status"] == "interrupted"


class TestRuntimeTablesMigration:
    """ref: the three runtime tables come from Alembic, matching the ORM."""

    @pytest.mark.parametrize("table", ["RuntimeSettings", "SettingsAudit", "QueryLogs"])
    def test_table_exists_after_chain(self, engine, table):
        with engine.connect() as c:
            assert c.execute(text("SELECT to_regclass(:t)"), {"t": f'"{table}"'}).scalar()

    def test_orm_columns_match_migration(self, engine):
        from sqlalchemy import inspect
        from db.db import QueryLog, RuntimeSetting, SettingsAudit
        insp = inspect(engine)
        for orm in (RuntimeSetting, SettingsAudit, QueryLog):
            db_cols = {c["name"] for c in insp.get_columns(orm.__tablename__)}
            assert db_cols == {c.name for c in orm.__table__.columns}, orm.__tablename__

    def test_orm_round_trip(self, itest_db):
        from db.query_log_db import QueryLogDB
        from db.settings_audit_db import SettingsAuditDB
        from db.runtime_settings_db import RuntimeSettingsDB
        RuntimeSettingsDB.upsert("rag.rerank.score_threshold", 0.3, updated_by="t")
        assert RuntimeSettingsDB.load_all()["rag.rerank.score_threshold"] == 0.3
        SettingsAuditDB.record("k", "set", 1, 2, "t")
        assert SettingsAuditDB.recent(5)[0]["key"] == "k"
        QueryLogDB.record("hello", 1, 3, 12, "tok…")
        assert QueryLogDB.analytics(days=1)["total"] >= 1
