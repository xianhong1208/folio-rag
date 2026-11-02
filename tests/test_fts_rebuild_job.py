"""Rebuild-FTS split per file (src/adapter/fts_rebuild.py): progress, cancel, summary."""

from unittest.mock import MagicMock, patch

from src.adapter import fts_rebuild


def _run(fids, stop_after=None, rows_per_file=3, stamps=None):
    calls, progress = [], []
    stamps = [] if stamps is None else stamps

    def rebuild_one(engine, table, seg, file_id):
        calls.append(file_id)
        return rows_per_file

    def should_stop():
        return stop_after is not None and len(calls) >= stop_after

    with patch.object(fts_rebuild, "list_chunk_file_ids", return_value=fids):
        out = fts_rebuild.rebuild_fts_by_file(
            "ENGINE", "data_1_u", "SEG",
            progress=lambda *a: progress.append(a), should_stop=should_stop,
            rebuild_one=rebuild_one, stamp=lambda *a: stamps.append(a))
    return out, calls, progress


def test_rebuilds_each_file_scoped_with_progress():
    out, calls, progress = _run(["a", "b", "c"])
    assert calls == ["a", "b", "c"]
    assert progress == [(0, 3), (1, 3, "a"), (2, 3, "b"), (3, 3, "c")]
    assert out == {"files": 3, "files_done": 3, "rows_updated": 9, "stopped": False,
                   "message": "Rebuilt full-text search for 9 chunk(s) in 3 file(s)"}


def test_stops_between_files_when_cancelled():
    out, calls, _ = _run(["a", "b", "c"], stop_after=1)
    assert calls == ["a"]
    assert out["stopped"] is True and out["files_done"] == 1
    assert out["message"].startswith("Stopped after 1/3")


def test_empty_table_reports_zero():
    out, calls, progress = _run([])
    assert calls == [] and progress == [(0, 0)]
    assert out["rows_updated"] == 0 and not out["stopped"]


def test_default_per_file_rebuild_is_the_vector_store_one():
    with patch.object(fts_rebuild, "list_chunk_file_ids", return_value=["a"]), \
         patch("src.domain.rag.vector_store_manager.stamp_tsv_segmenter") as st, \
         patch("src.domain.rag.vector_store_manager.rebuild_text_search_tsv",
               return_value=2) as one:
        out = fts_rebuild.rebuild_fts_by_file("E", "t", "S", lambda *a: None, lambda: False)
    one.assert_called_once_with("E", "t", "S", file_id="a")
    st.assert_called_once_with("E", "t", None)
    assert out["rows_updated"] == 2


def test_list_ids_missing_table_is_empty():
    conn = MagicMock()
    conn.execute.return_value.scalar.return_value = None
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    assert fts_rebuild.list_chunk_file_ids(engine, "data_9_x") == []
    assert conn.execute.call_count == 1  # never selects from a table that isn't there


def test_list_ids_reads_distinct_file_ids():
    conn = MagicMock()
    exists = MagicMock()
    exists.scalar.return_value = "data_9_x"
    rows = MagicMock()
    rows.fetchall.return_value = [("a",), ("b",)]
    conn.execute.side_effect = [exists, rows]
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    assert fts_rebuild.list_chunk_file_ids(engine, "data_9_x") == ["a", "b"]
    sql = str(conn.execute.call_args_list[1].args[0])
    assert "DISTINCT metadata_->>'file_id'" in sql and '"data_9_x"' in sql


def test_completed_rebuild_stamps_the_segmenter():
    seg = MagicMock()
    seg.name = "bigram"
    stamps = []
    with patch.object(fts_rebuild, "list_chunk_file_ids", return_value=["a"]):
        fts_rebuild.rebuild_fts_by_file(
            "E", "t", seg, lambda *a: None, lambda: False,
            rebuild_one=lambda *a, **k: 1, stamp=lambda *a: stamps.append(a))
    assert stamps == [("E", "t", "bigram")]


def test_stopped_rebuild_keeps_the_old_stamp():
    stamps = []
    _run(["a", "b"], stop_after=1, stamps=stamps)
    assert stamps == []


def test_stamp_skips_missing_table():
    from src.domain.rag import vector_store_manager as vsm
    conn = MagicMock()
    conn.execute.return_value.scalar.return_value = False
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    vsm.stamp_tsv_segmenter(engine, "data_1_x", "jieba")
    assert conn.execute.call_count == 1 and not conn.commit.called


def test_stamp_writes_column_comment():
    from src.domain.rag import vector_store_manager as vsm
    conn = MagicMock()
    conn.execute.return_value.scalar.return_value = True
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    vsm.stamp_tsv_segmenter(engine, "data_1_x", "jieba")
    assert "segmenter=jieba" in str(conn.execute.call_args_list[1].args[0])
    conn.commit.assert_called_once()
