
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
