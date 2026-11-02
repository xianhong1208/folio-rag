
"""Control Center management-plane API contract.

Pins down: (1) existing endpoint functions are always called with the signed-in
token's owner key (bypassing none of the existing security/cleanup logic);
(2) an ownerless or foreign folder is 404 (409 for ownerless in open mode);
(3) a nonexistent folder is 404; (4) a created folder is owned by the session.
The underlying endpoint functions are mocked -- their behavior is guarded by
their own tests.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router.admin_settings import router


@pytest.fixture()
def client(as_identity, make_identity):
    # Signed in as the owner of the folders the tests use ("owner-tok").
    app = FastAPI()
    app.include_router(router)
    return TestClient(as_identity(app, make_identity("owner-tok", admin=False)))


def _folder(token="owner-tok"):
    return SimpleNamespace(id=7, name="f", user_token=token)


_DB = "db.cached_folderdb.CachedFolderDB.get_by_id"
_MGR = "src.domain.rag.index_job_manager.IndexingJobManager.get_instance"


class TestActAsOwner:
    def test_delete_folder_uses_owner_token(self, client):
        with patch(_DB, return_value=_folder()), \
             patch("src.api.router.folder_api.delete_folder",
                   new=AsyncMock(return_value={"ok": True})) as ep:
            r = client.delete("/api/admin/manage/folders/7")
        assert r.status_code == 200
        ep.assert_awaited_once_with(7, user_token="owner-tok")

    def test_missing_folder_404(self, client):
        with patch(_DB, return_value=None):
            assert client.delete("/api/admin/manage/folders/999").status_code == 404

    def test_unowned_folder_is_404(self, client):
        # Fail-closed: an ownerless folder is nobody's, so a token session cannot see it.
        with patch(_DB, return_value=_folder(token=None)):
            r = client.delete("/api/admin/manage/folders/7")
        assert r.status_code == 404

    def test_unowned_folder_409_in_open_mode(self, as_identity):
        from src.auth.console_session import ConsoleIdentity
        app = FastAPI()
        app.include_router(router)
        c = TestClient(as_identity(app, ConsoleIdentity(mode="open", via="open", is_admin=True)))
        with patch(_DB, return_value=_folder(token=None)):
            r = c.delete("/api/admin/manage/folders/7")
        assert r.status_code == 409
        assert "owner token" in r.json()["detail"]

    def test_create_is_owned_by_the_session(self, client):
        # Any owner_token in the body is ignored: the owner is always the session's jti.
        with patch("src.api.router.folder_api.create_folder",
                   new=AsyncMock(return_value={"id": 1})) as ep:
            r = client.post("/api/admin/manage/folders",
                            json={"name": "docs", "owner_token": "tok-a"})
        assert r.status_code == 200
        assert ep.await_args.kwargs["user_token"] == "owner-tok"
        assert ep.await_args.args[0].name == "docs"

    def test_create_in_open_mode_uses_console_owner(self, as_identity):
        from src.auth.console_session import ConsoleIdentity
        app = FastAPI()
        app.include_router(router)
        c = TestClient(as_identity(app, ConsoleIdentity(mode="open", via="open", is_admin=True)))
        with patch("src.api.router.folder_api.create_folder",
                   new=AsyncMock(return_value={"id": 1})) as ep:
            assert c.post("/api/admin/manage/folders", json={"name": "docs"}).status_code == 200
        assert ep.await_args.kwargs["user_token"] == "console"

    def test_incremental_index_uses_index_endpoint(self, client):
        f = _folder()
        with patch(_DB, return_value=f), \
             patch("src.api.router.rag_indexing.index_folder_endpoint",
                   new=AsyncMock(return_value={"data": {}})) as ep:
            r = client.post("/api/admin/manage/folders/7/index",
                            json={"skip_existing": True})
        assert r.status_code == 200
        kw = ep.await_args.kwargs
        assert kw["skip_existing"] is True and kw["token"] == "owner-tok"
        assert kw["folder"] is f

    def test_full_rebuild_uses_reindex_endpoint(self, client):
        # A full rebuild must go through reindex (deletes index records first) --
        # index(skip_existing=False) is short-circuited by content_hash, making
        # the rebuild a no-op
        f = _folder()
        with patch(_DB, return_value=f), \
             patch("src.api.router.rag_indexing.reindex_folder_endpoint",
                   new=AsyncMock(return_value={"data": {}})) as re_ep, \
             patch("src.api.router.rag_indexing.index_folder_endpoint",
                   new=AsyncMock()) as idx_ep:
            r = client.post("/api/admin/manage/folders/7/index",
                            json={"skip_existing": False})
        assert r.status_code == 200
        re_ep.assert_awaited_once()
        idx_ep.assert_not_awaited()

    def test_cancel_job_delegates(self, client):
        f = _folder()
        with patch(_DB, return_value=f), \
             patch("src.api.router.rag_indexing.cancel_index_job",
                   new=AsyncMock(return_value={"data": {"cancelled": True}})) as ep:
            r = client.post("/api/admin/manage/folders/7/jobs/j1/cancel")
        assert r.status_code == 200
        ep.assert_awaited_once_with(7, "j1", token="owner-tok", folder=f)

    def test_delete_file_delegates(self, client):
        fid = "9e68c18e-3209-44d3-b7e9-2211b3660d66"
        with patch(_DB, return_value=_folder()), \
             patch("src.api.router.file_api.delete_file",
                   new=AsyncMock(return_value={"ok": True})) as ep:
            r = client.delete(f"/api/admin/manage/folders/7/files/{fid}")
        assert r.status_code == 200
        assert str(ep.await_args.args[1]) == fid
        assert ep.await_args.kwargs["user_token"] == "owner-tok"


class TestRemainingDelegations:
    def test_create_empty_name_422(self, client):
        r = client.post("/api/admin/manage/folders",
                        json={"name": "  ", "owner_token": "t"})
        assert r.status_code == 422 and "name" in str(r.json()["detail"])

    def test_update_folder_delegates(self, client):
        with patch(_DB, return_value=_folder()), \
             patch("src.api.router.folder_api.update_folder_api",
                   new=AsyncMock(return_value={"ok": True})) as ep:
            r = client.patch("/api/admin/manage/folders/7",
                             json={"name": "new", "description": "d"})
        assert r.status_code == 200
        assert ep.await_args.kwargs["user_token"] == "owner-tok"
        assert ep.await_args.args[1].name == "new"

    def test_upload_delegates_with_owner_token(self, client):
        with patch(_DB, return_value=_folder()), \
             patch("src.api.router.file_api.upload_file",
                   new=AsyncMock(return_value={"ok": True})) as ep:
            r = client.post("/api/admin/manage/folders/7/files",
                            files={"file": ("a.txt", b"hello")},
                            data={"auto_index": "true"})
        assert r.status_code == 200
        kw = ep.await_args.kwargs
        assert kw["user_token"] == "owner-tok" and kw["auto_index"] is True

    def test_download_delegates(self, client):
        fid = "9e68c18e-3209-44d3-b7e9-2211b3660d66"
        with patch(_DB, return_value=_folder()), \
             patch("src.api.router.file_api.download_file",
                   new=AsyncMock(return_value={"ok": True})) as ep:
            r = client.get(f"/api/admin/manage/folders/7/files/{fid}/download")
        assert r.status_code == 200
        assert ep.await_args.kwargs["user_token"] == "owner-tok"


class TestQueryPlayground:
    def test_query_delegates_with_owner_token(self, client):
        f = _folder()
        with patch(_DB, return_value=f), \
             patch("src.api.router.rag_query.query_rag",
                   new=AsyncMock(return_value={"data": {"results": [], "total_results": 0}})) as ep:
            r = client.post("/api/admin/manage/folders/7/query", json={"query": "hello world"})
        assert r.status_code == 200
        assert ep.await_args.kwargs["token"] == "owner-tok"
        assert ep.await_args.args[1].folder_name == "f"

    def test_query_too_short_422(self, client):
        with patch(_DB, return_value=_folder()):
            r = client.post("/api/admin/manage/folders/7/query", json={"query": "x"})
        assert r.status_code == 422

    @pytest.mark.parametrize("body", [
        {"query": 123},                                   # used to crash on .strip()
        {"query": "hello", "top_k": 500},                 # out of range
        {"query": "hello", "top_k": "lots"},
        {"query": "hello", "similarity_cutoff": 7},
        {"query": "hello", "trace": "maybe"},
        {"query": "   x  "},                              # too short once stripped
        {"query": "x" * 2001},                            # query length cap
        {},
    ])
    def test_bad_params_are_422_not_500(self, client, body):
        with patch(_DB, return_value=_folder()), \
             patch("src.api.router.rag_query.query_rag", new=AsyncMock()) as ep:
            r = client.post("/api/admin/manage/folders/7/query", json=body)
            rs = client.post("/api/admin/manage/folders/7/query/stream", json=body)
        assert r.status_code == 422, r.text
        assert rs.status_code == 422, rs.text
        ep.assert_not_awaited()

    def test_query_is_stripped_and_params_forwarded(self, client):
        with patch(_DB, return_value=_folder()), \
             patch("src.api.router.rag_query.query_rag",
                   new=AsyncMock(return_value={"data": {"results": []}})) as ep:
            r = client.post("/api/admin/manage/folders/7/query",
                            json={"query": "  hello  ", "top_k": "5", "similarity_cutoff": 0})
        assert r.status_code == 200
        req = ep.await_args.args[1]
        assert req.query == "hello" and req.top_k == 5 and req.similarity_cutoff == 0

    def test_trace_context_overflow_is_422(self, client):
        adapter = MagicMock()
        adapter.query_trace = AsyncMock(side_effect=RuntimeError("maximum context length is 512"))
        with patch(_DB, return_value=_folder()), \
             patch("src.adapter.rag.get_rag_adapter", return_value=adapter):
            r = client.post("/api/admin/manage/folders/7/query", json={"query": "hello", "trace": True})
        assert r.status_code == 422

    def test_trace_keeps_explicit_zero_cutoff(self, client):
        adapter = MagicMock()
        adapter.query_trace = AsyncMock(return_value={"results": []})
        with patch(_DB, return_value=_folder()), \
             patch("src.adapter.rag.get_rag_adapter", return_value=adapter):
            client.post("/api/admin/manage/folders/7/query",
                        json={"query": "hello", "trace": True, "similarity_cutoff": 0})
        assert adapter.query_trace.await_args.kwargs["similarity_cutoff"] == 0

    def test_trace_uses_query_trace_and_maps_results(self, client):
        """trace=true routes to query_trace and builds results from the trace's final hits (does not re-run a normal query)."""
        f = _folder()
        trace = {"reranked": True, "candidates": 30, "results": [
            {"text": "seg1", "final_score": 0.9, "reranked": True, "rerank_score": 0.9,
             "hybrid_score": 0.5, "hybrid_rank": 3, "vector_score": 0.6, "vector_rank": 1,
             "bm25_score": None, "bm25_rank": None,
             "metadata": {"file_name": "a.pdf", "page": 2, "headings": ["H"], "node_id": "n1"}},
        ]}
        adapter = MagicMock()
        adapter.query_trace = AsyncMock(return_value=trace)
        with patch(_DB, return_value=f), \
             patch("src.adapter.rag.get_rag_adapter", return_value=adapter):
            r = client.post("/api/admin/manage/folders/7/query",
                            json={"query": "hello world", "trace": True})
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["trace"]["reranked"] is True and d["trace"]["candidates"] == 30
        # results mapped from the trace: score = final_score
        assert d["total_results"] == 1
        assert d["results"][0] == {"text": "seg1", "score": 0.9,
            "metadata": {"file_name": "a.pdf", "page": 2, "headings": ["H"], "node_id": "n1"}}
        assert adapter.query_trace.await_args.kwargs["token"] == "owner-tok"


class TestAnswerAndRetry:
    def test_query_with_answer_flag(self, client):
        f = _folder()
        qresp = {"data": {"results": [{"text": "ctx", "score": 0.9,
                 "metadata": {"file_name": "a.pdf"}}], "total_results": 1}}
        gen = {"answer": "Answer [1]", "confidence": "high", "kept": 1, "dropped": 0,
               "note": "Answered from 1 relevant chunk(s)"}
        with patch(_DB, return_value=f), \
             patch("src.api.router.rag_query.query_rag",
                   new=AsyncMock(return_value=dict(data=dict(qresp["data"])))), \
             patch("src.api.router.admin_manage._generate_answer", return_value=gen):
            r = client.post("/api/admin/manage/folders/7/query",
                            json={"query": "what", "answer": True})
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["answer"] == "Answer [1]"
        assert d["rag_gate"] == {"confidence": "high", "kept": 1, "dropped": 0,
                                 "note": "Answered from 1 relevant chunk(s)"}

    def _frow(self, folder_id=7, fid="9e68c18e-3209-4111-8111-000000000001"):
        return SimpleNamespace(id=fid, file_name="a.pdf", file_path="p", folder_id=folder_id,
                               mime_type="x", file_size=1, content_hash="h")

    def test_retry_queues_a_forced_background_job(self, client):
        # Retry goes through the job manager (folder lock, progress, cancel) instead of
        # indexing inline in the request; force bypasses the content_hash short-circuit.
        mgr = MagicMock()
        mgr.start_indexing_files = AsyncMock(return_value={"job_id": "j1", "status": "running"})
        with patch(_DB, return_value=_folder()), \
             patch("db.filedb.FileDB.get", return_value=[self._frow()]), \
             patch("src.adapter.rag.get_rag_adapter", return_value="ADAPTER"), \
             patch(_MGR, return_value=mgr):
            r = client.post("/api/admin/manage/folders/7/files/9e68c18e-3209-4111-8111-000000000001/retry")
        assert r.status_code == 202, r.text
        assert r.json()["data"]["job_id"] == "j1"
        kw = mgr.start_indexing_files.await_args.kwargs
        assert kw["force"] is True and kw["token"] == "owner-tok" and kw["folder_id"] == 7
        assert kw["file_ids"] == ["9e68c18e-3209-4111-8111-000000000001"]
        assert mgr.start_indexing_files.await_args.args == ("ADAPTER",)

    def test_retry_rejects_file_from_another_folder(self, client):
        # Folder 7 is owned by the session, but the file lives in folder 8 (someone else's).
        mgr = MagicMock()
        mgr.start_indexing_files = AsyncMock()
        with patch(_DB, return_value=_folder()), \
             patch("db.filedb.FileDB.get", return_value=[self._frow(folder_id=8)]), \
             patch(_MGR, return_value=mgr):
            r = client.post("/api/admin/manage/folders/7/files/9e68c18e-3209-4111-8111-000000000001/retry")
        assert r.status_code == 404
        mgr.start_indexing_files.assert_not_awaited()

    def test_retry_missing_file_404(self, client):
        with patch(_DB, return_value=_folder()), patch("db.filedb.FileDB.get", return_value=[]):
            r = client.post("/api/admin/manage/folders/7/files/9e68c18e-3209-4111-8111-000000000001/retry")
        assert r.status_code == 404

    def test_bulk_retry_is_one_job(self, client):
        a, b = "9e68c18e-3209-4111-8111-00000000000a", "9e68c18e-3209-4111-8111-00000000000b"
        mgr = MagicMock()
        mgr.start_indexing_files = AsyncMock(return_value={"job_id": "j1", "queued": True})
        rows = {a: [self._frow(fid=a)], b: [self._frow(fid=b)]}
        with patch(_DB, return_value=_folder()), \
             patch("db.filedb.FileDB.get", side_effect=lambda id: rows[str(id)]), \
             patch("src.adapter.rag.get_rag_adapter"), patch(_MGR, return_value=mgr):
            r = client.post("/api/admin/manage/folders/7/retry", json={"file_ids": [a, b, a]})
        assert r.status_code == 202
        assert mgr.start_indexing_files.await_count == 1
        assert mgr.start_indexing_files.await_args.kwargs["file_ids"] == [a, b]
        assert "queued" in r.json()["message"]

    @pytest.mark.parametrize("body", [{}, {"file_ids": []}, {"file_ids": "x"}, {"file_ids": ["not-a-uuid"]}])
    def test_bulk_retry_validates_ids(self, client, body):
        with patch(_DB, return_value=_folder()):
            r = client.post("/api/admin/manage/folders/7/retry", json=body)
        assert r.status_code == 422

    def test_bulk_retry_rejects_foreign_file(self, client):
        a = "9e68c18e-3209-4111-8111-00000000000a"
        mgr = MagicMock()
        mgr.start_indexing_files = AsyncMock()
        with patch(_DB, return_value=_folder()), \
             patch("db.filedb.FileDB.get", return_value=[self._frow(fid=a, folder_id=8)]), \
             patch(_MGR, return_value=mgr):
            r = client.post("/api/admin/manage/folders/7/retry", json={"file_ids": [a]})
        assert r.status_code == 404
        mgr.start_indexing_files.assert_not_awaited()

    @pytest.mark.parametrize("outcome, code", [("dequeued", 200), ("flagged", 200),
                                               ("cancelled_job", 200), (None, 409)])
    def test_cancel_file(self, client, outcome, code):
        mgr = MagicMock()
        mgr.cancel_file = AsyncMock(return_value=outcome)
        with patch(_DB, return_value=_folder()), \
             patch("db.filedb.FileDB.get", return_value=[self._frow()]), \
             patch(_MGR, return_value=mgr):
            r = client.post("/api/admin/manage/folders/7/files/9e68c18e-3209-4111-8111-000000000001/cancel")
        assert r.status_code == code
        mgr.cancel_file.assert_awaited_once_with("9e68c18e-3209-4111-8111-000000000001", 7)
        if outcome:
            assert r.json()["data"]["outcome"] == outcome

    def test_rebuild_fts_starts_background_job(self, client):
        mgr = MagicMock()
        mgr.start_maintenance_job = AsyncMock(return_value={"job_id": "j2", "kind": "rebuild_fts"})
        cfg = SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(text_search_config="simple")))
        with patch(_DB, return_value=SimpleNamespace(id=7, name="f", user_token="owner-tok",
                                                     vector_table_uuid="u")), \
             patch("src.config.config_manager.Config.get_config_model", return_value=cfg), \
             patch(_MGR, return_value=mgr):
            r = client.post("/api/admin/manage/folders/7/rebuild-fts")
        assert r.status_code == 202 and r.json()["data"]["job_id"] == "j2"
        kw = mgr.start_maintenance_job.await_args.kwargs
        assert kw["folder_id"] == 7 and kw["kind"] == "rebuild_fts" and callable(kw["work"])

    def test_rebuild_fts_busy_folder_is_409(self, client):
        from src.domain.exceptions import ConflictError
        mgr = MagicMock()
        mgr.start_maintenance_job = AsyncMock(side_effect=ConflictError("busy"))
        cfg = SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(text_search_config="simple")))
        with patch(_DB, return_value=SimpleNamespace(id=7, name="f", user_token="owner-tok",
                                                     vector_table_uuid="u")), \
             patch("src.config.config_manager.Config.get_config_model", return_value=cfg), \
             patch(_MGR, return_value=mgr):
            r = client.post("/api/admin/manage/folders/7/rebuild-fts")
        assert r.status_code == 409

    def test_rebuild_fts_not_ckip_mode_is_409(self, client):
        cfg = SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(text_search_config="english")))
        with patch(_DB, return_value=_folder()), \
             patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
            r = client.post("/api/admin/manage/folders/7/rebuild-fts")
        assert r.status_code == 409

    def test_job_status_live_and_persisted(self, client):
        mgr = MagicMock()
        mgr.get_status = AsyncMock(return_value={"job_id": "j1", "folder_id": 7, "status": "running"})
        with patch(_DB, return_value=_folder()), patch(_MGR, return_value=mgr):
            r = client.get("/api/admin/manage/folders/7/jobs/j1")
        assert r.status_code == 200 and r.json()["data"]["status"] == "running"
        # Not in memory (evicted / restarted) → the DB projection answers.
        mgr.get_status = AsyncMock(return_value=None)
        with patch(_DB, return_value=_folder()), patch(_MGR, return_value=mgr), \
             patch("db.indexjobdb.IndexJobDB.get_by_id",
                   return_value={"job_id": "j1", "folder_id": 7, "status": "interrupted"}):
            r = client.get("/api/admin/manage/folders/7/jobs/j1")
        assert r.json()["data"]["status"] == "interrupted"

    def test_job_status_of_other_folder_is_404(self, client):
        mgr = MagicMock()
        mgr.get_status = AsyncMock(return_value={"job_id": "j1", "folder_id": 8})
        with patch(_DB, return_value=_folder()), patch(_MGR, return_value=mgr), \
             patch("db.indexjobdb.IndexJobDB.get_by_id", return_value={"job_id": "j1", "folder_id": 8}):
            r = client.get("/api/admin/manage/folders/7/jobs/j1")
        assert r.status_code == 404


class TestGenerateAnswer:
    def test_empty_results_returns_none_answer(self):
        from src.api.router.admin_manage import _generate_answer
        out = _generate_answer("q", [])
        assert out["answer"] is None and out["confidence"] == "none"

    def _cfg(self):
        return SimpleNamespace(rag=SimpleNamespace(llm=SimpleNamespace(
            provider="vllm", model="m", base_url="http://x", api_key=None,
            azure_deployment=None)))

    def test_crag_relevant_generates_answer(self):
        """grade returns all-relevant -> generate from the relevant segments, confidence by relevant count."""
        from src.api.router import admin_manage as am
        fake = MagicMock()
        # 1st call = grading (returns a JSON array), 2nd = generation
        fake.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[true, true]"))]),
            # gpt-oss-style citations, should be normalized to [1][2]
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Ans【1†L1-L3】【2†L2】"))]),
        ]
        with patch("src.config.config_manager.Config.get_config_model", return_value=self._cfg()), \
             patch("src.domain.rag.context_generator.ContextGenerator._create_sync_client",
                   return_value=fake):
            out = am._generate_answer("what files?", [
                {"text": "ctx one", "score": 0.9}, {"text": "ctx two", "score": 0.8}])
        assert out["answer"] == "Ans[1][2]" and out["confidence"] == "high" and out["kept"] == 2
        # The generation prompt (2nd call) carries context and question
        prompt = fake.chat.completions.create.call_args_list[1].kwargs["messages"][0]["content"]
        assert "ctx one" in prompt and "what files?" in prompt

    def test_empty_generation_returns_note_not_silent_none(self):
        """grade passes but generation returns empty (reasoning consumed the budget) -> give an explicit truncation message, do not silently return None."""
        from src.api.router import admin_manage as am
        fake = MagicMock()
        fake.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[true]"),
                                                     finish_reason="stop")]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None),
                                                     finish_reason="length")]),
        ]
        with patch("src.config.config_manager.Config.get_config_model", return_value=self._cfg()), \
             patch("src.domain.rag.context_generator.ContextGenerator._create_sync_client",
                   return_value=fake):
            out = am._generate_answer("q?", [{"text": "ctx", "score": 0.9}])
        assert out["answer"] and "did not complete" in out["answer"]
        assert out["confidence"] == "low" and "finish=length" in out["note"]

    def test_crag_gate_blocks_when_nothing_relevant(self):
        """grade all-irrelevant -> the CRAG gate blocks, no forced answer (generation not called)."""
        from src.api.router import admin_manage as am
        fake = MagicMock()
        fake.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[false]"))])
        with patch("src.config.config_manager.Config.get_config_model", return_value=self._cfg()), \
             patch("src.domain.rag.context_generator.ContextGenerator._create_sync_client",
                   return_value=fake):
            out = am._generate_answer("unrelated?", [{"text": "irrelevant", "score": 0.2}])
        assert out["confidence"] == "low" and out["kept"] == 0 and out["dropped"] == 1
        assert "no sufficient basis" in out["answer"]
        assert fake.chat.completions.create.call_count == 1  # graded only, no generation
