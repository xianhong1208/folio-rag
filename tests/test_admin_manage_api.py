
"""Control Center management-plane API contract (act-as-owner mode).

Pins down: (1) existing endpoint functions are always called with the folder
owner's token (bypassing none of the existing security/cleanup logic);
(2) an ownerless folder cannot be managed (409); (3) a nonexistent folder is
404; (4) creating a folder requires owner_token. The underlying endpoint
functions are mocked -- their behavior is guarded by their own tests.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router.admin_settings import router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _folder(token="owner-tok"):
    return SimpleNamespace(id=7, name="f", user_token=token)


_DB = "db.cached_folderdb.CachedFolderDB.get_by_id"


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

    def test_unowned_folder_409(self, client):
        with patch(_DB, return_value=_folder(token=None)):
            r = client.delete("/api/admin/manage/folders/7")
        assert r.status_code == 409
        assert "owner token" in r.json()["detail"]

    def test_create_requires_owner_token(self, client):
        r = client.post("/api/admin/manage/folders", json={"name": "x"})
        assert r.status_code == 422
        assert "owner_token" in r.json()["detail"]

    def test_create_passes_token_through(self, client):
        with patch("src.api.router.folder_api.create_folder",
                   new=AsyncMock(return_value={"id": 1})) as ep:
            r = client.post("/api/admin/manage/folders",
                            json={"name": "docs", "owner_token": "tok-a"})
        assert r.status_code == 200
        assert ep.await_args.kwargs["user_token"] == "tok-a"
        assert ep.await_args.args[0].name == "docs"

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
        assert r.status_code == 422 and "name" in r.json()["detail"]

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

    def test_retry_forces_reindex(self, client):
        f = _folder()
        frow = SimpleNamespace(id="9e68c18e-3209-4111-8111-000000000001", file_name="a.pdf", file_path="p",
                               folder_id=7, mime_type="x", file_size=1, content_hash="h")
        adapter = MagicMock()
        adapter.index_document = AsyncMock(
            return_value=SimpleNamespace(model_dump=lambda: {"status": "indexed"}))
        with patch(_DB, return_value=f), \
             patch("db.filedb.FileDB.get", return_value=[frow]), \
             patch("src.adapter.rag.get_rag_adapter", return_value=adapter):
            r = client.post("/api/admin/manage/folders/7/files/9e68c18e-3209-4111-8111-000000000001/retry")
        assert r.status_code == 200
        assert adapter.index_document.await_args.kwargs["force"] is True


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
        assert out["answer"] and "生成未完成" in out["answer"]
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
        assert "找不到足夠依據" in out["answer"]
        assert fake.chat.completions.create.call_count == 1  # graded only, no generation
