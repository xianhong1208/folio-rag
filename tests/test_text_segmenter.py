"""Pluggable Chinese segmentation for keyword search (rag.retrieval.segmenter).

One function (text_segmenter.get_segmenter) serves the tsvector writer, Rebuild
FTS and the query side; each folder's table records which segmenter built it so
the console can flag a needed Rebuild FTS after a switch.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.domain.rag import text_segmenter as ts
from src.domain.rag import vector_store_manager as vsm


@pytest.fixture(autouse=True)
def _fresh_instances():
    ts._instances.clear()
    yield
    ts._instances.clear()


def _cfg(segmenter="ckip", tsc="simple"):
    return SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(
        segmenter=segmenter, text_search_config=tsc)))


class TestBigram:
    def test_cjk_pairs_and_whole_words(self):
        assert ts.bigram_tokens("退貨政策 30天, GPT-4o!") == [
            "退貨", "貨政", "政策", "30", "天", "GPT", "4o"]

    def test_lone_character_stays(self):
        assert ts.bigram_tokens("第3條") == ["第", "3", "條"]

    def test_segment_and_batch(self):
        seg = ts.BigramSegmenter()
        assert seg.segment("退貨政策") == "退貨 貨政 政策"
        assert seg.segment("   ") == ""
        assert seg.segment("!!") == "!!"
        assert seg.segment_batch(["你好", ""]) == ["你好", ""]
        assert seg.name == "bigram"


class TestJieba:
    def test_uses_jieba_lazily(self):
        fake = types.ModuleType("jieba")
        fake.lcut = lambda t: ["退貨", "政策", " ", "，"]
        fake.setLogLevel = lambda lvl: None
        seg = ts.JiebaSegmenter()
        with patch.dict(sys.modules, {"jieba": fake}):
            assert seg.segment("退貨政策，") == "退貨 政策 ，"
            assert seg.segment_batch(["退貨政策", " "]) == ["退貨 政策 ，", ""]

    def test_missing_jieba_returns_raw_text(self):
        seg = ts.JiebaSegmenter()
        with patch.dict(sys.modules, {"jieba": None}):
            assert seg.segment("退貨政策") == "退貨政策"
            assert seg.segment("再一次") == "再一次"      # no retry storm

    def test_runtime_error_returns_raw_text(self):
        fake = types.ModuleType("jieba")
        fake.setLogLevel = lambda lvl: None

        def boom(t):
            raise RuntimeError("bad dict")

        fake.lcut = boom
        with patch.dict(sys.modules, {"jieba": fake}):
            assert ts.JiebaSegmenter().segment("退貨") == "退貨"


class TestSelection:
    def test_configured_name(self):
        with patch("src.config.config_manager.Config.get_config_model", return_value=_cfg("bigram")):
            assert ts.configured_segmenter_name() == "bigram"
            assert ts.get_segmenter().name == "bigram"
        with patch("src.config.config_manager.Config.get_config_model", return_value=_cfg("nope")):
            assert ts.configured_segmenter_name() == "ckip"
        with patch("src.config.config_manager.Config.get_config_model", side_effect=RuntimeError):
            assert ts.configured_segmenter_name() == "ckip"

    def test_unset_follows_the_profile(self):
        def cfg(profile):
            return SimpleNamespace(rag=SimpleNamespace(
                profile=profile, retrieval=SimpleNamespace(segmenter=None)))
        for profile, want in (("local", "ckip"), ("hybrid", "ckip"), ("cloud", "bigram")):
            with patch("src.config.config_manager.Config.get_config_model",
                       return_value=cfg(profile)):
                assert ts.configured_segmenter_name() == want

    def test_one_instance_per_kind_and_ckip_is_lazy(self):
        sentinel = MagicMock(name="ckip")
        with patch("src.domain.rag.ckip_segmenter.get_segmenter", return_value=sentinel) as ck:
            assert ts.get_segmenter("ckip") is sentinel
            assert ts.get_segmenter("CKIP") is sentinel
            assert ts.get_segmenter("unknown") is sentinel
        assert ck.call_count == 1
        assert ts.get_segmenter("jieba") is ts.get_segmenter("jieba")

    def test_ckip_segmenter_has_a_name(self):
        from src.domain.rag.ckip_segmenter import CkipSegmenter
        assert CkipSegmenter.name == "ckip"

    async def test_query_side_uses_the_configured_segmenter(self):
        from src.domain.rag.hybrid_search import HybridSearcher, SearchSettings
        s = HybridSearcher(MagicMock(), index=MagicMock())
        with patch("src.config.config_manager.Config.get_config_model", return_value=_cfg("bigram")):
            assert await s.segment_query("退貨政策", SearchSettings()) == "退貨 貨政 政策"

    def test_config_rejects_unknown_segmenter(self):
        from src.config.model import RAGRetrievalConfig
        base = dict(default_top_k=5, default_similarity_cutoff=0.1, hybrid_search=True)
        # Unset = the profile's default (ckip; bigram under profile: cloud).
        assert RAGRetrievalConfig(**base).segmenter is None
        with pytest.raises(Exception):
            RAGRetrievalConfig(**base, segmenter="pg_jieba")

    def test_segmenter_is_live_editable(self):
        from src.config.runtime_overrides import EDITABLE
        assert EDITABLE["rag.retrieval.segmenter"] == "live"


# ---------------------------------------------------------------- stamps

class _Conn:
    """Sync connection stub keyed on SQL fragments."""

    def __init__(self, comment=None, exists=True, has_rows=True, others=False, rows=()):
        self.comment, self.exists, self.has_rows, self.others = comment, exists, has_rows, others
        self.rows = list(rows)
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def execute(self, stmt, params=None):
        q = str(stmt)
        self.sql.append((q, params))
        r = MagicMock()
        if "col_description" in q:
            r.first.return_value = (self.comment,)
        elif "to_regclass(:t) IS NOT NULL" in q:
            r.scalar.return_value = self.exists
        elif "IS DISTINCT FROM :fid" in q:
            r.scalar.return_value = self.others
        elif "SELECT EXISTS" in q:
            r.scalar.return_value = self.has_rows
        elif q.startswith("SELECT id,"):
            r.fetchall.return_value = self.rows
        elif q.startswith("COMMENT ON COLUMN"):
            self.comment = q.split("IS '")[1].rstrip("'")
        return r


def _engine(conn):
    e = MagicMock()
    e.connect.return_value = conn
    return e


class TestStamp:
    def test_read_stamp(self):
        assert vsm.read_tsv_segmenter(_Conn(comment="segmenter=jieba"), "t") == "jieba"
        assert vsm.read_tsv_segmenter(_Conn(comment=None), "t") is None
        assert vsm.read_tsv_segmenter(_Conn(comment="something else"), "t") is None

    def test_status_flags_a_switch(self):
        st = vsm.fts_status(_engine(_Conn(comment="segmenter=ckip")), "t", "bigram")
        assert st == {"indexed_with": "ckip", "configured": "bigram", "needs_rebuild": True}
        st = vsm.fts_status(_engine(_Conn(comment="segmenter=bigram")), "t", "bigram")
        assert st["needs_rebuild"] is False

    def test_unstamped_table_with_rows_is_legacy_ckip(self):
        st = vsm.fts_status(_engine(_Conn(comment=None, has_rows=True)), "t", "jieba")
        assert st["indexed_with"] == "ckip" and st["needs_rebuild"] is True
        st = vsm.fts_status(_engine(_Conn(comment=None, has_rows=False)), "t", "jieba")
        assert st["indexed_with"] is None and st["needs_rebuild"] is False

    def test_missing_table_needs_nothing(self):
        st = vsm.fts_status(_engine(_Conn(exists=False)), "t", "ckip")
        assert st["needs_rebuild"] is False and st["indexed_with"] is None

    def test_full_rebuild_stamps_the_table(self):
        conn = _Conn(comment="segmenter=ckip", rows=[(1, "退貨政策")])
        n = vsm.rebuild_text_search_tsv(_engine(conn), "t", ts.BigramSegmenter())
        assert n == 1 and conn.comment == "segmenter=bigram"
        update = [p for q, p in conn.sql if q.startswith("UPDATE") and "to_tsvector" in q][0]
        assert update == [{"id": 1, "seg": "退貨 貨政 政策"}]

    def test_rebuild_matches_the_indexer_leaves_only(self):
        """Same rows and text as chunk_store: leaf retrieval text in, parents cleared."""
        conn = _Conn(comment="segmenter=ckip", rows=[(1, "x")])
        vsm.rebuild_text_search_tsv(_engine(conn), "t", ts.BigramSegmenter(), file_id="f1")
        select = next(q for q, _ in conn.sql if q.startswith("SELECT id,"))
        assert "concat_ws" in select and "IS DISTINCT FROM 'parent'" in select
        assert "metadata_->>'file_id' = :fid" in select
        clear = next(q for q, _ in conn.sql if "= NULL" in q)
        assert "NOT (metadata_->>'node_role' IS DISTINCT FROM 'parent')" in clear

    def test_full_rebuild_of_empty_table_still_stamps(self):
        conn = _Conn(comment=None, rows=[])
        assert vsm.rebuild_text_search_tsv(_engine(conn), "t", ts.BigramSegmenter()) == 0
        assert conn.comment == "segmenter=bigram"

    def test_first_file_of_a_new_table_stamps_it(self):
        conn = _Conn(comment=None, others=False, rows=[(1, "x")])
        vsm.rebuild_text_search_tsv(_engine(conn), "t", ts.BigramSegmenter(), file_id="f1")
        assert conn.comment == "segmenter=bigram"

    def test_file_added_to_an_old_folder_keeps_its_stamp(self):
        conn = _Conn(comment=None, others=True, rows=[(1, "x")])
        vsm.rebuild_text_search_tsv(_engine(conn), "t", ts.BigramSegmenter(), file_id="f1")
        assert conn.comment is None                       # still flagged as legacy ckip
        conn = _Conn(comment="segmenter=ckip", rows=[(1, "x")])
        vsm.rebuild_text_search_tsv(_engine(conn), "t", ts.BigramSegmenter(), file_id="f1")
        assert conn.comment == "segmenter=ckip"


# ------------------------------------------------------------ console API

@pytest.fixture()
def client(as_identity, make_identity):
    from src.api.router.admin_settings import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(as_identity(app, make_identity("owner-tok", admin=False)))


_DB = "db.cached_folderdb.CachedFolderDB.get_by_id"


def _folder():
    return SimpleNamespace(id=7, name="f", user_token="owner-tok", vector_table_uuid="u")


class TestConsole:
    def test_status_endpoint(self, client):
        status = {"indexed_with": "ckip", "configured": "bigram", "needs_rebuild": True}
        with patch(_DB, return_value=_folder()), \
             patch("src.config.config_manager.Config.get_config_model", return_value=_cfg("bigram")), \
             patch("db.db.get_engine"), \
             patch("src.domain.rag.vector_store_manager.fts_status", return_value=status) as st:
            r = client.get("/api/admin/manage/folders/7/fts")
        assert r.status_code == 200 and r.json()["data"]["needs_rebuild"] is True
        assert st.call_args.args[1] == "data_7_u" and st.call_args.args[2] == "bigram"

    def test_status_is_advisory_on_db_error(self, client):
        with patch(_DB, return_value=_folder()), \
             patch("src.config.config_manager.Config.get_config_model", return_value=_cfg()), \
             patch("db.db.get_engine"), \
             patch("src.domain.rag.vector_store_manager.fts_status", side_effect=RuntimeError):
            r = client.get("/api/admin/manage/folders/7/fts")
        assert r.status_code == 200 and r.json()["data"]["needs_rebuild"] is False

    def test_status_outside_segmented_mode(self, client):
        with patch(_DB, return_value=_folder()), \
             patch("src.config.config_manager.Config.get_config_model",
                   return_value=_cfg(tsc="english")):
            r = client.get("/api/admin/manage/folders/7/fts")
        assert r.json()["data"]["text_search_config"] == "english"

    def test_rebuild_uses_the_configured_segmenter(self, client):
        """Rebuild FTS is a background job; its work re-segments with the configured segmenter."""
        jm = MagicMock()
        jm.start_maintenance_job = AsyncMock(return_value={"job_id": "j1", "status": "running"})
        with patch(_DB, return_value=_folder()), \
             patch("src.config.config_manager.Config.get_config_model", return_value=_cfg("bigram")), \
             patch("db.db.get_engine", return_value="ENGINE"), \
             patch("src.domain.rag.index_job_manager.IndexingJobManager.get_instance",
                   return_value=jm), \
             patch("src.adapter.fts_rebuild.rebuild_fts_by_file",
                   return_value={"rows_updated": 3}) as rb:
            r = client.post("/api/admin/manage/folders/7/rebuild-fts", json={})
            assert r.status_code == 202 and r.json()["data"]["job_id"] == "j1"
            work = jm.start_maintenance_job.await_args.kwargs["work"]
            work(lambda *a: None, lambda: False)
        assert rb.call_args.args[1] == "data_7_u" and rb.call_args.args[2].name == "bigram"

    def test_rebuild_409_outside_segmented_mode(self, client):
        with patch(_DB, return_value=_folder()), \
             patch("src.config.config_manager.Config.get_config_model",
                   return_value=_cfg(tsc="english")):
            assert client.post("/api/admin/manage/folders/7/rebuild-fts", json={}).status_code == 409

    def test_rebuild_conflict_is_409(self, client):
        from src.domain.exceptions import ConflictError
        jm = MagicMock()
        jm.start_maintenance_job = AsyncMock(
            side_effect=ConflictError(message="busy", resource="indexing_job:7"))
        with patch(_DB, return_value=_folder()), \
             patch("src.config.config_manager.Config.get_config_model", return_value=_cfg()), \
             patch("db.db.get_engine"), \
             patch("src.domain.rag.index_job_manager.IndexingJobManager.get_instance",
                   return_value=jm):
            assert client.post("/api/admin/manage/folders/7/rebuild-fts", json={}).status_code == 409
