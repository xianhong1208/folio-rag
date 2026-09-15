
"""Integration tests -- IndexJobDB / FileIndexDB real-DB round-trips.

Covers IndexJobDB.upsert round-trip, update semantics, JSONB fields, and
load_active filtering; and FileIndexDB.upsert_indexed reindex idempotency,
content_hash persistence, and the mark_failed paths.
"""

import uuid

from db.fileindexdb import FileIndexDB
from db.indexjobdb import IndexJobDB

# The folder / file_row fixtures live in conftest.py (shared with test_index_pipeline)


def _job_state(folder_id: int, **over):
    base = {
        "job_id": str(uuid.uuid4()),
        "folder_id": folder_id,
        "status": "running",
        "total_files": 3,
        "processed_files": 1,
        "current_index": 1,
        "current_file_name": "doc.pdf",
        "message": "processing",
        "skip_existing": True,
        "started_at": "2026-08-19T00:00:00+00:00",
        "last_updated_at": "2026-08-19T00:01:00+00:00",
        "scope_file_ids": ["f-1", "f-2"],
        "file_timings": [{"file_id": "f-1", "total_ms": 1200}],
        # Keys not in _PERSISTED_FIELDS must be safely dropped (forward-compat)
        "current_file_eta_seconds": 42,
        "eta_seconds": 99,
    }
    base.update(over)
    return base


class TestIndexJobDB:
    def test_upsert_roundtrip_including_jsonb(self, itest_db, folder):
        state = _job_state(folder.id)
        IndexJobDB.upsert(state)

        row = IndexJobDB.get_by_id(state["job_id"])
        assert row is not None, "upsert 靜默失敗(它吞例外只 log — roundtrip 才驗得出)"
        assert row["status"] == "running"
        assert row["total_files"] == 3
        # JSONB fields round-trip verbatim
        assert row["scope_file_ids"] == ["f-1", "f-2"]
        assert row["file_timings"] == [{"file_id": "f-1", "total_ms": 1200}]

    def test_upsert_updates_existing(self, itest_db, folder):
        state = _job_state(folder.id)
        IndexJobDB.upsert(state)
        IndexJobDB.upsert({**state, "status": "succeeded", "processed_files": 3})

        row = IndexJobDB.get_by_id(state["job_id"])
        assert row["status"] == "succeeded"
        assert row["processed_files"] == 3

    def test_load_active_filters_terminal(self, itest_db, folder):
        running = _job_state(folder.id)
        done = _job_state(folder.id, status="succeeded")
        pending = _job_state(folder.id, status="pending")
        for s in (running, done, pending):
            IndexJobDB.upsert(s)

        active_ids = {str(r["job_id"]) for r in IndexJobDB.load_active()}
        assert running["job_id"] in active_ids
        assert pending["job_id"] in active_ids
        assert done["job_id"] not in active_ids, "terminal job 不該被 startup 回收撈到"

    def test_delete_for_folder(self, itest_db, folder):
        state = _job_state(folder.id)
        IndexJobDB.upsert(state)
        assert IndexJobDB.delete_for_folder(folder.id) >= 1
        assert IndexJobDB.get_by_id(state["job_id"]) is None


class TestFileIndexDB:
    def test_upsert_indexed_roundtrip_with_content_hash(self, itest_db, folder, file_row):
        FileIndexDB.upsert_indexed(
            file_id=file_row.id, folder_id=folder.id,
            index_id=f"file_{file_row.id}", vector_store_table="data_x_y",
            chunk_size=256, chunk_overlap=50,
            embedding_model="e5-large", num_chunks=7,
            content_hash="a" * 64,
        )
        rec = FileIndexDB.get_by_file(file_row.id)
        assert rec.status == "indexed"
        assert rec.num_chunks == 7
        assert rec.content_hash == "a" * 64

    def test_upsert_indexed_reindex_is_idempotent(self, itest_db, folder, file_row):
        """Second upsert of the same file: no UniqueViolation, values updated in place (reindex scenario)."""
        for num_chunks, chash in ((7, "a" * 64), (9, "b" * 64)):
            FileIndexDB.upsert_indexed(
                file_id=file_row.id, folder_id=folder.id,
                index_id=f"file_{file_row.id}", vector_store_table="data_x_y",
                chunk_size=256, chunk_overlap=50,
                embedding_model="e5-large", num_chunks=num_chunks,
                content_hash=chash,
            )
        rec = FileIndexDB.get_by_file(file_row.id)
        assert rec.num_chunks == 9
        assert rec.content_hash == "b" * 64

    def test_mark_failed_updates_existing_row(self, itest_db, folder, file_row):
        FileIndexDB.upsert_indexed(
            file_id=file_row.id, folder_id=folder.id,
            index_id=f"file_{file_row.id}", vector_store_table="data_x_y",
            chunk_size=256, chunk_overlap=50,
            embedding_model="e5-large", num_chunks=7,
        )
        FileIndexDB.mark_failed(file_row.id, "boom")
        rec = FileIndexDB.get_by_file(file_row.id)
        assert rec.status == "failed"
        assert "boom" in (rec.error_message or "")

    def test_mark_failed_creates_row_when_absent(self, itest_db, folder, file_row):
        """No prior record + folder_id given -> create a failed row (the key path for timeout tagging)."""
        FileIndexDB.mark_failed(
            file_row.id, "timeout", folder_id=folder.id,
            embedding_model="e5-large", chunk_size=256, chunk_overlap=50,
        )
        rec = FileIndexDB.get_by_file(file_row.id)
        assert rec is not None, "無先前記錄時必須建 failed row,前端才有失敗 tag"
        assert rec.status == "failed"
