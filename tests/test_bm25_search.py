"""BM25 (Postgres full-text search) on the shared retrieval path.

- leaves only, filtered in SQL (parents' long text used to win ts_rank and fill
  the LIMIT before a Python-side filter threw them away);
- one statement: rows matching all query terms first, then any term;
- question / function words dropped; bigram queries skip particle-straddling pairs;
- the folder's stamped segmenter tokenises its queries;
- depth = max(k, sparse_top_k) — sparse_top_k used to be ignored by llama-index;
- dense and BM25 run concurrently.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from src.domain.rag import hybrid_search as hs
from src.domain.rag.hybrid_search import HybridSearcher, SearchSettings
from tests.test_hybrid_search import nws


class TestLexemes:
    def test_drops_question_and_function_words(self):
        assert hs.query_lexemes("退貨 政策 是 什麼 ?") == ["退貨", "政策"]
        assert hs.query_lexemes("What is the refund policy ?") == ["refund", "policy"]

    def test_keeps_everything_when_only_stop_words(self):
        assert hs.query_lexemes("what is it") == ["what", "is", "it"]

    def test_punctuation_dropped_digits_kept_deduped(self):
        assert hs.query_lexemes("第 3 條 , 第 3 條 !") == ["第", "3", "條"]
        assert hs.query_lexemes("") == []

    def test_tsquery_quotes_terms(self):
        assert hs.to_tsquery_text(["a", "it's", "x\\y"], "&") == "'a' & 'it''s' & 'x\\\\y'"


class _Conn:
    """Async connection stub: returns canned rows per OR-query, records calls.

    ``run_sync`` answers the segmenter-stamp lookup with ``stamp``.
    """

    def __init__(self, rows_by_query, stamp=None):
        self.rows_by_query = rows_by_query
        self.stamp = stamp
        self.calls = []
        self.stamp_reads = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def run_sync(self, fn):
        self.stamp_reads += 1
        return self.stamp

    async def execute(self, stmt, params):
        self.calls.append((str(stmt), dict(params)))
        rows = self.rows_by_query.get(params["q_any"], [])
        return SimpleNamespace(fetchall=lambda: rows[: params["k"]])


def _row(nid, rank, role="leaf"):
    return (nid, f"text {nid}", {"node_role": role, "file_id": "f"}, rank)


def _searcher_with_db(rows_by_query, segmented="退貨 政策 是 什麼", stamp=None):
    conn = _Conn(rows_by_query, stamp)
    engine = MagicMock()
    engine.connect.return_value = conn
    s = HybridSearcher(MagicMock(table_name="data_1_x"), index=MagicMock())
    s.segment_query = AsyncMock(return_value=segmented)
    return s, conn, engine


class TestSparseSql:
    async def test_one_statement_all_terms_first_leaf_only_in_sql(self):
        rows = {"'退貨' | '政策'": [_row("a", 0.9), _row("b", 0.5), _row("c", 0.4)]}
        s, conn, engine = _searcher_with_db(rows)
        with patch("db.db.get_async_engine", return_value=engine):
            out = await s.sparse("退貨政策是什麼", 3, SearchSettings())
        assert [n.node.node_id for n in out] == ["a", "b", "c"]
        assert len(conn.calls) == 1                              # one round trip
        sql, p = conn.calls[0]
        assert p["q_any"] == "'退貨' | '政策'" and p["q_all"] == "'退貨' & '政策'"
        assert p["cfg"] == "simple" and p["k"] == 3
        assert "ORDER BY (text_search_tsv @@ q_all) DESC, rank DESC" in sql
        assert "IS DISTINCT FROM 'parent'" in sql and '"data_1_x"' in sql
        assert out[0].score == 0.9 and out[0].node.metadata["file_id"] == "f"

    async def test_single_term_runs_one_query(self):
        rows = {"'退貨'": [_row("a", 0.9)]}
        s, conn, engine = _searcher_with_db(rows, segmented="退貨")
        with patch("db.db.get_async_engine", return_value=engine):
            out = await s.sparse("q", 5, SearchSettings())
        assert len(conn.calls) == 1 and [n.node.node_id for n in out] == ["a"]
        assert conn.calls[0][1]["q_all"] == "'退貨'"

    async def test_no_terms_no_query(self):
        s, conn, engine = _searcher_with_db({}, segmented="? !")
        with patch("db.db.get_async_engine", return_value=engine):
            assert await s.sparse("?!", 5, SearchSettings()) == []
        assert conn.calls == []

    async def test_db_failure_degrades_to_dense_only(self):
        s, _, _ = _searcher_with_db({})
        with patch("db.db.get_async_engine", side_effect=RuntimeError("down")):
            assert await s.sparse("q", 5, SearchSettings()) == []

    async def test_legacy_rows_without_node_content_still_load(self):
        rows = {"'退貨'": [("a", "raw text", None, 0.3)]}
        s, _, engine = _searcher_with_db(rows, segmented="退貨")
        with patch("db.db.get_async_engine", return_value=engine):
            out = await s.sparse("q", 5, SearchSettings())
        assert out[0].node.get_content() == "raw text"


class TestFolderSegmenter:
    async def test_query_uses_the_folders_stamped_segmenter(self):
        s, conn, engine = _searcher_with_db({}, stamp="jieba")
        with patch("db.db.get_async_engine", return_value=engine), \
             patch("src.domain.rag.text_segmenter.configured_segmenter_name",
                   return_value="ckip"):
            await s.sparse("退貨政策", 5, SearchSettings())
        assert s.segment_query.await_args.args[2] == "jieba"

    async def test_unstamped_folder_uses_the_configured_segmenter(self):
        s, _, engine = _searcher_with_db({}, stamp=None)
        with patch("db.db.get_async_engine", return_value=engine):
            await s.sparse("退貨政策", 5, SearchSettings())
        assert s.segment_query.await_args.args[2] is None

    async def test_stamp_is_cached_per_table(self):
        s, conn, engine = _searcher_with_db({}, stamp="bigram")
        with patch("db.db.get_async_engine", return_value=engine):
            await s.sparse("a b", 5, SearchSettings())
            await s.sparse("a b", 5, SearchSettings())
        assert conn.stamp_reads == 1

    async def test_non_simple_config_skips_the_stamp(self):
        s, conn, engine = _searcher_with_db({}, stamp="bigram")
        with patch("db.db.get_async_engine", return_value=engine):
            await s.sparse("a b", 5, SearchSettings(text_search_config="english"))
        assert conn.stamp_reads == 0 and s.segment_query.await_args.args[2] is None

    async def test_unreadable_stamp_means_configured(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("down")
        assert await s.folder_segmenter(engine, "data_1_x") is None

    def test_usable_segmenter_rules(self):
        cfg_cloud = SimpleNamespace(rag=SimpleNamespace(profile="cloud"))
        cfg_local = SimpleNamespace(rag=SimpleNamespace(profile="local"))
        with patch("src.domain.rag.text_segmenter.configured_segmenter_name",
                   return_value="bigram"):
            assert hs._usable_segmenter(None) is None
            assert hs._usable_segmenter("nonsense") is None
            assert hs._usable_segmenter("jieba") == "jieba"
            with patch("src.config.config_manager.Config.get_config_model",
                       return_value=cfg_cloud):
                assert hs._usable_segmenter("ckip") is None     # no CKIP model in cloud
            with patch("src.config.config_manager.Config.get_config_model",
                       return_value=cfg_local):
                assert hs._usable_segmenter("ckip") == "ckip"
            with patch("src.config.config_manager.Config.get_config_model",
                       side_effect=RuntimeError):
                assert hs._usable_segmenter("ckip") == "ckip"


class TestBigramQueryTerms:
    def test_particle_straddling_pairs_are_dropped(self):
        assert hs.bigram_query_terms("請問公司的年假規定是什麼？") == ["公司", "年假", "假規", "規定"]

    def test_orphaned_content_char_keeps_a_pair(self):
        assert hs.bigram_query_terms("現在幾點") == ["現在", "幾點"]      # 在 is a particle
        assert hs.bigram_query_terms("和平與發展") == ["和平", "發展"]

    def test_plain_runs_and_latin_words_unchanged(self):
        assert hs.bigram_query_terms("退貨政策") == ["退貨", "貨政", "政策"]
        assert hs.bigram_query_terms("AMD AI 395 的價格") == ["AMD", "AI", "395", "價格"]

    def test_all_particles_or_single_char_fall_back(self):
        assert hs.bigram_query_terms("的") == ["的"]
        assert hs.bigram_query_terms("是什麼") == ["是什", "什麼"]
        assert hs.bigram_query_terms("") == []

    def test_pruned_query_is_a_subset_of_the_indexed_bigrams(self):
        from src.domain.rag.text_segmenter import bigram_tokens

        q = "請問公司的年假規定是什麼？現在幾點"
        assert set(hs.bigram_query_terms(q)) <= set(bigram_tokens(q))


class TestDepthAndConcurrency:
    async def test_sparse_depth_honours_sparse_top_k(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        s.dense = AsyncMock(return_value=[])
        s.sparse = AsyncMock(return_value=[])
        await s.candidates("q", SearchSettings(top_k=5, sparse_top_k=40))
        assert s.sparse.await_args.args[1] == 40
        await s.candidates("q", SearchSettings(top_k=50, sparse_top_k=4))
        assert s.sparse.await_args.args[1] == 50

    async def test_dense_and_sparse_run_concurrently(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        started = []

        async def dense(q, k):
            started.append("dense")
            await asyncio.sleep(0.05)
            return [nws("a")]

        async def sparse(q, k, st):
            started.append("sparse")
            await asyncio.sleep(0.05)
            return [nws("b")]

        s.dense, s.sparse = dense, sparse
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        fused, _, _ = await s.candidates("q", SearchSettings(top_k=2))
        assert loop.time() - t0 < 0.09
        assert {n.node.node_id for n in fused} == {"a", "b"}


class TestSegmentQuery:
    async def test_bigram_segmenter_prunes_particle_pairs(self):
        from src.domain.rag.text_segmenter import BigramSegmenter

        s = HybridSearcher(MagicMock(), index=MagicMock())
        with patch("src.domain.rag.text_segmenter.get_segmenter",
                   return_value=BigramSegmenter()) as gs:
            out = await s.segment_query("公司的年假", SearchSettings(), "bigram")
        assert out == "公司 年假" and gs.call_args.args[0] == "bigram"

    async def test_non_simple_config_uses_raw_words(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        assert await s.segment_query("refund policies", SearchSettings(
            text_search_config="english")) == "refund policies"

    async def test_simple_config_segments_in_python(self):
        seg = MagicMock()
        seg.segment.return_value = "退貨 政策"
        s = HybridSearcher(MagicMock(), index=MagicMock())
        with patch("src.domain.rag.text_segmenter.get_segmenter", return_value=seg):
            assert await s.segment_query("退貨政策", SearchSettings()) == "退貨 政策"

    async def test_segmentation_failure_falls_back_to_raw(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        with patch("src.domain.rag.text_segmenter.get_segmenter", side_effect=RuntimeError):
            assert await s.segment_query("raw text", SearchSettings()) == "raw text"

    async def test_segmentation_runs_on_one_dedicated_thread(self):
        """Every query is segmented on the same thread (one torch/OpenMP team)."""
        import threading

        threads = set()

        def segment(text):
            threads.add(threading.current_thread().name)
            return text

        seg = MagicMock()
        seg.segment.side_effect = segment
        s = HybridSearcher(MagicMock(), index=MagicMock())
        with patch("src.domain.rag.text_segmenter.get_segmenter", return_value=seg):
            await asyncio.gather(*(s.segment_query(f"查詢{i}", SearchSettings())
                                   for i in range(6)))
        assert len(threads) == 1 and next(iter(threads)).startswith("folio-query-segment")
        assert hs._segment_executor() is hs._segment_executor()
