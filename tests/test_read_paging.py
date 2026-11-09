"""mode='read' pages through long files; mode='list' estimates tokens from the
stored text (ChunkLookup.fetch_file_full / file_token_estimates, handle_read /
handle_list)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.domain.rag import agentic_handlers as ah
from src.domain.rag.chunk_lookup import ChunkLookup

FOLDER = SimpleNamespace(id=3, name="docs", description="d", vector_table_uuid="uu")


class _SyncConn:
    def __init__(self, results):
        self.results = list(results)   # one entry per execute call
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        self.calls.append((str(stmt), dict(params or {})))
        res = self.results.pop(0)
        if isinstance(res, Exception):
            raise res
        return SimpleNamespace(fetchall=lambda: res, scalar=lambda: res)


def _engine(*results):
    conn = _SyncConn(results)
    eng = MagicMock()
    eng.connect.return_value = conn
    return eng, conn


def _rows(start, n, chars=400, total=10):
    return [("字" * chars, i, total) for i in range(start, start + n)]


class TestFetchFilePage:
    def _fetch(self, rows, **kw):
        eng, conn = _engine(rows)
        vs = MagicMock(table_name="data_3_uu", _engine=eng)
        return ChunkLookup.fetch_file_full(vs, "f1", **kw), conn

    def test_page_stops_at_budget_and_points_to_next(self):
        out, conn = self._fetch(_rows(0, 10), max_tokens=1000)          # 280 tokens / chunk
        assert out["chunk_count"] == 3 and out["truncated"] is True
        assert out["next_offset"] == 3 and out["total_chunks"] == 10
        assert out["tokens_estimated"] == 840 and out["offset"] == 0
        assert conn.calls[0][1] == {"file_id": "f1", "offset": 0}

    def test_offset_continues_and_last_page_has_no_next(self):
        out, conn = self._fetch(_rows(8, 2), max_tokens=1000, offset=8)
        assert conn.calls[0][1]["offset"] == 8
        assert out["chunk_count"] == 2 and out["next_offset"] is None and not out["truncated"]

    def test_first_chunk_always_included_so_paging_advances(self):
        out, _ = self._fetch(_rows(0, 3, chars=5000), max_tokens=100)
        assert out["chunk_count"] == 1 and out["next_offset"] == 1

    def test_empty_and_failure(self):
        out, _ = self._fetch([], offset=99)
        assert out["chunk_count"] == 0 and out["next_offset"] is None and out["offset"] == 99
        out, _ = self._fetch(RuntimeError("db"))
        assert out["text"] == "" and out["truncated"] is False


class TestTokenEstimates:
    def test_estimate_from_stored_text(self):
        eng, conn = _engine(True, [("f1", 1000, 600), ("f2", 400, 0)])
        with patch("db.db.get_engine", return_value=eng):
            out = ChunkLookup.file_token_estimates("data_3_uu")
        assert out == {"f1": round(600 * 0.7 + 400 * 0.25), "f2": 100}
        sql, params = conn.calls[1]
        assert "GROUP BY" in sql and '"data_3_uu"' in sql and params["non_cjk"].startswith("[^")

    def test_missing_table_and_failure(self):
        eng, _ = _engine(False)
        with patch("db.db.get_engine", return_value=eng):
            assert ChunkLookup.file_token_estimates("data_3_uu") == {}
        eng, _ = _engine(RuntimeError("db"))
        with patch("db.db.get_engine", return_value=eng):
            assert ChunkLookup.file_token_estimates("data_3_uu") == {}


class TestHandleList:
    def _list(self, estimates):
        files = [SimpleNamespace(id="f1", file_name="a.pdf", file_size=2048, upload_time=None),
                 SimpleNamespace(id="f2", file_name="b.pdf", file_size=0, upload_time=None),
                 SimpleNamespace(id="f3", file_name="c.pdf", file_size=0, upload_time=None)]
        idx = [SimpleNamespace(file_id="f1", status="indexed", num_chunks=10),
               SimpleNamespace(file_id="f2", status="indexed", num_chunks=4)]
        with patch.object(ah.FileDB, "get", return_value=files), \
             patch.object(ah.FileIndexDB, "get_by_folder", return_value=idx), \
             patch.object(ChunkLookup, "file_token_estimates", return_value=estimates) as est:
            out = ah.handle_list(folder=FOLDER)
        return out, est

    def test_uses_stored_sizes_and_falls_back_per_file(self):
        out, est = self._list({"f1": 3456})
        assert est.call_args.args[0] == "data_3_uu"
        by_id = {f["file_id"]: f for f in out["files"]}
        assert by_id["f1"]["estimated_tokens"] == 3456         # real stored size
        assert by_id["f2"]["estimated_tokens"] == 800          # fallback 200 / chunk
        assert by_id["f3"]["estimated_tokens"] is None         # not indexed
        assert "next_offset" in out["_hint"]


class TestHandleRead:
    async def _read(self, full, offset=None, record_folder=3):
        rec = SimpleNamespace(folder_id=record_folder, file_name="a.pdf")
        with patch.object(ah.FileDB, "get_by_ids", return_value={"f1": rec}), \
             patch.object(ChunkLookup, "fetch_file_full", return_value=full) as fetch:
            out = await ah.handle_read(file_id="f1", folder=FOLDER, vector_store=MagicMock(),
                                       offset=offset)
        return out, fetch

    async def test_truncated_page_returns_next_offset(self):
        full = {"text": "x", "tokens_estimated": 10, "truncated": True, "chunk_count": 3,
                "total_chunks": 10, "offset": 0, "next_offset": 3}
        out, fetch = await self._read(full)
        assert out["next_offset"] == 3 and out["total_chunks"] == 10
        assert "offset=3" in out["_hint"]
        assert fetch.call_args.kwargs["offset"] == 0

    async def test_offset_is_passed_and_end_is_reported(self):
        full = {"text": "x", "tokens_estimated": 10, "truncated": False, "chunk_count": 2,
                "total_chunks": 10, "offset": 8, "next_offset": None}
        out, fetch = await self._read(full, offset=8)
        assert fetch.call_args.kwargs["offset"] == 8
        assert out["next_offset"] is None and out["_hint"] == "End of file."

    async def test_past_the_end_and_whole_file(self):
        full = {"text": "", "tokens_estimated": 0, "truncated": False, "chunk_count": 0,
                "total_chunks": 0, "offset": 50, "next_offset": None}
        out, _ = await self._read(full, offset=50)
        assert "past the end" in out["_hint"]
        full = {"text": "all", "tokens_estimated": 1, "truncated": False, "chunk_count": 1}
        out, _ = await self._read(full)
        assert out["_hint"] == "Full content loaded." and out["total_chunks"] == 1

    async def test_a_file_without_indexed_content_says_so(self):
        # QA 2026-11-09: a file that failed to index read back empty with "Full content loaded."
        full = {"text": "", "tokens_estimated": 0, "truncated": False, "chunk_count": 0,
                "total_chunks": 0, "offset": 0, "next_offset": None}
        out, _ = await self._read(full)
        assert "no indexed content" in out["_hint"] and "Full content loaded" not in out["_hint"]

    async def test_bad_offset_missing_file_and_cross_folder(self):
        out = await ah.handle_read(file_id="f1", folder=FOLDER, vector_store=None, offset="x")
        assert "offset" in out["error"]
        out = await ah.handle_read(file_id="", folder=FOLDER, vector_store=None)
        assert "required" in out["error"]
        with patch.object(ah.FileDB, "get_by_ids", return_value={}):
            out = await ah.handle_read(file_id="f1", folder=FOLDER, vector_store=None)
        assert "not found" in out["error"]
        out, _ = await self._read({}, record_folder=99)
        assert "does not belong" in out["error"]
