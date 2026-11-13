
"""db/admin_stats aggregate queries -- real-DB verification (Control Center data plane).

Uses itest fixtures to create folder/file/index/job, then asserts the four
aggregates report the correct numbers.
"""

import uuid

import pytest


@pytest.fixture()
def seeded(itest_db, folder, file_row):
    """folder + 1 file + 1 indexed FileIndex + 1 failed job."""
    from db.fileindexdb import FileIndexDB
    from db.indexjobdb import IndexJobDB
    FileIndexDB.upsert_indexed(
        file_id=file_row.id, folder_id=folder.id,
        index_id=f"file_{file_row.id}", vector_store_table="t",
        num_chunks=42, embedding_model="test-embed",
        chunk_size=256, chunk_overlap=50, content_hash="h",
    )
    IndexJobDB.ensure_table()
    IndexJobDB.upsert({
        "job_id": str(uuid.uuid4()), "folder_id": folder.id, "status": "failed",
        "total_files": 3, "processed_files": 1, "error": "boom",
        "last_updated_at": "2026-09-04T00:00:00",
    })
    return folder, file_row


class TestAdminStats:
    def test_overview_counts(self, seeded):
        from db import admin_stats
        ov = admin_stats.overview()
        assert ov["folders"] >= 1 and ov["files"] >= 1
        assert ov["index"]["indexed_files"] >= 1
        assert ov["index"]["total_chunks"] >= 42
        assert ov["jobs"]["by_status"].get("failed", 0) >= 1

    def test_folders_with_index_stats(self, seeded):
        from db import admin_stats
        folder, _ = seeded
        row = next(f for f in admin_stats.folders_with_index_stats() if f["id"] == folder.id)
        assert row["indexed_files"] == 1 and row["chunks"] == 42
        assert row["vector_table"].startswith(f"data_{folder.id}_")

    def test_files_with_index_status(self, seeded):
        from db import admin_stats
        folder, file_row = seeded
        rows = admin_stats.files_with_index_status(folder.id)
        me = next(r for r in rows if r["id"] == str(file_row.id))
        assert me["status"] == "indexed" and me["chunks"] == 42
        assert me["embedding_model"] == "test-embed"

    def test_recent_jobs_includes_folder_name(self, seeded):
        from db import admin_stats
        folder, _ = seeded
        js = admin_stats.recent_jobs(50)
        mine = [j for j in js if j["folder_id"] == folder.id]
        assert mine and mine[0]["folder_name"] == folder.name
        assert mine[0]["status"] == "failed" and mine[0]["error"] == "boom"


class TestOwnerScope:
    """Console scoping: an owner key only counts / lists its own folders."""

    def test_overview_scoped_to_owner(self, seeded):
        from db import admin_stats
        mine = admin_stats.overview(owner="itest-token")
        assert mine["folders"] >= 1 and mine["index"]["total_chunks"] >= 42
        nobody = admin_stats.overview(owner="no-such-owner")
        assert nobody["folders"] == 0 and nobody["files"] == 0
        assert nobody["index"]["total_chunks"] == 0 and nobody["jobs"]["by_status"] == {}
        assert admin_stats.indexing_trend(7, owner="no-such-owner")[-1]["count"] == 0

    def test_folder_list_scoped_to_owner(self, seeded):
        from db import admin_stats
        folder, _ = seeded
        assert any(f["id"] == folder.id for f in admin_stats.folders_with_index_stats(owner="itest-token"))
        assert admin_stats.folders_with_index_stats(owner="no-such-owner") == []

    def test_jobs_scoped_and_flagged(self, seeded):
        from db import admin_stats
        folder, _ = seeded
        assert admin_stats.recent_jobs(50, owner="no-such-owner") == []
        everyone = admin_stats.recent_jobs(50, viewer="someone-else")
        mine = [j for j in everyone if j["folder_id"] == folder.id]
        assert mine and mine[0]["owned"] is False
        own = admin_stats.recent_jobs(50, owner="itest-token", viewer="itest-token")
        assert own and all(j["owned"] for j in own)

    def test_chunks_require_file_in_folder(self, seeded):
        from db import admin_stats
        folder, file_row = seeded
        out = admin_stats.chunks_for_file(folder.id + 100000, str(file_row.id))
        assert out["file_name"] is None and out["chunks"] == []


class TestPipelineVersionAndWarning:
    @pytest.fixture(autouse=True)
    def configured_model_is_m(self):
        """These rows are indexed with model "m": make it the configured one, so only
        the pipeline rule is exercised here (the embedding rule has its own tests)."""
        from unittest.mock import patch
        with patch("db.admin_stats.configured_embedding", return_value=("m", None)):
            yield

    def test_file_list_reports_stale_pipeline_and_parse_warning(self, itest_db, folder, file_row):
        from db import admin_stats
        from db.fileindexdb import FileIndexDB
        from src.domain.rag.chunk_identity import CHUNK_PIPELINE_VERSION
        FileIndexDB.upsert_indexed(
            file_id=file_row.id, folder_id=folder.id, index_id="x", vector_store_table="t",
            num_chunks=1, embedding_model="m", chunk_size=256, chunk_overlap=0,
            index_warning="Partial parse: page 3 could not be parsed",
        )
        me = next(r for r in admin_stats.files_with_index_status(folder.id) if r["id"] == str(file_row.id))
        assert me["warning"].startswith("Partial parse") and me["reindex_recommended"] is True
        FileIndexDB.upsert_indexed(
            file_id=file_row.id, folder_id=folder.id, index_id="x", vector_store_table="t",
            num_chunks=1, embedding_model="m", chunk_size=256, chunk_overlap=0,
            pipeline_version=CHUNK_PIPELINE_VERSION,
        )
        me = next(r for r in admin_stats.files_with_index_status(folder.id) if r["id"] == str(file_row.id))
        assert me["warning"] is None and me["reindex_recommended"] is False


    def test_folder_list_counts_files_needing_reindex(self, itest_db, folder):
        """reindex_recommended_files: indexed files from an older (or unrecorded) pipeline only."""
        import uuid as _uuid
        from db import admin_stats
        from db.filedb import FileDB
        from db.fileindexdb import FileIndexDB
        from db.folderdb import FolderDB
        from src.domain.rag.chunk_identity import CHUNK_PIPELINE_VERSION

        def add(f, version, failed=False):
            row = FileDB.create(folder_id=f.id, file_name="x.txt",
                                file_path=f"/tmp/itest/{_uuid.uuid4().hex}",
                                file_size=1, mime_type="text/plain")
            if failed:
                FileIndexDB.mark_failed(row.id, "boom", folder_id=f.id)
                return
            FileIndexDB.upsert_indexed(
                file_id=row.id, folder_id=f.id, index_id="x", vector_store_table="t",
                num_chunks=1, embedding_model="m", chunk_size=256, chunk_overlap=0,
                pipeline_version=version)

        add(folder, None)                    # before versions were recorded -> stale
        add(folder, "2")                     # older pipeline -> stale
        add(folder, CHUNK_PIPELINE_VERSION)  # current -> fine
        add(folder, None, failed=True)       # not indexed -> needs a retry, not a re-index
        other = FolderDB.create(name=f"itest-{_uuid.uuid4().hex[:8]}", description="",
                                user_token="someone-else")
        add(other, "1")

        def row(owner):
            return {f["id"]: f for f in admin_stats.folders_with_index_stats(owner=owner)}

        mine = row("itest-token")
        assert mine[folder.id]["reindex_recommended_files"] == 2
        assert mine[folder.id]["indexed_files"] == 3 and mine[folder.id]["failed_files"] == 1
        assert other.id not in mine  # never another owner's folder
        assert row("someone-else")[other.id]["reindex_recommended_files"] == 1

    def test_folder_without_index_rows_counts_zero(self, itest_db, folder):
        from db import admin_stats
        me = next(f for f in admin_stats.folders_with_index_stats(owner="itest-token")
                  if f["id"] == folder.id)
        assert me["reindex_recommended_files"] == 0

class TestSystemResources:
    def test_vector_table_sizes_in_one_query(self, itest_db):
        """Counts and sizes every data_* table (the per-table loop became one query)."""
        from sqlalchemy import text
        from db import admin_stats
        from db.db import get_engine
        names = ["data_990001_res-a", "data_990002_res-b"]
        eng = get_engine()
        before = admin_stats.system_resources()
        with eng.begin() as c:
            for n in names:
                c.execute(text(f'CREATE TABLE "{n}" (id int primary key, t text)'))
                c.execute(text(f'INSERT INTO "{n}" SELECT g, repeat(\'x\', 500) FROM generate_series(1, 200) g'))
        try:
            with eng.connect() as c:
                expected = sum(c.execute(text("SELECT pg_total_relation_size(to_regclass(:t))"),
                                         {"t": f'"{n}"'}).scalar() for n in names)
            res = admin_stats.system_resources()
            assert res["vector_tables"] == before["vector_tables"] + 2
            assert res["vector_bytes"] == before["vector_bytes"] + expected
            assert res["db_size_bytes"] > 0
        finally:
            with eng.begin() as c:
                for n in names:
                    c.execute(text(f'DROP TABLE IF EXISTS "{n}"'))
