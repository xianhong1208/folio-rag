
"""Control Center data-plane API contract (admin_overview).

The aggregate queries themselves (db/admin_stats) are verified against a real
DB by the integration tests; here we pin the HTTP face: routing, response
shape, models-summary assembly, and limit validation.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router.admin_settings import router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


_STATS = "db.admin_stats"


class TestOverviewApi:
    def test_overview_shape_with_models_summary(self, client):
        base = {"folders": 2, "files": 5, "total_size_bytes": 100,
                "index": {"indexed_files": 4, "failed_files": 1,
                          "unindexed_files": 0, "total_chunks": 99},
                "jobs": {"by_status": {"succeeded": 3}, "active": []}}
        eff = {"rag.embedding.model": "m", "rag.embedding.dimension": 1024,
               "rag.embedding.base_url": "http://e", "rag.llm.model": "l",
               "rag.llm.base_url": "http://l", "rag.rerank.enabled": True,
               "rag.rerank.model": "r", "rag.rerank.base_url": "http://r",
               "rag.asr.enabled": True, "rag.asr.provider": "fireredasr"}
        with patch(f"{_STATS}.overview", return_value=dict(base)), \
             patch("src.config.runtime_overrides.get_effective", return_value=eff), \
             patch("src.config.config_manager.Config.get_config_model",
                   return_value=SimpleNamespace()):
            r = client.get("/api/admin/overview")
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["folders"] == 2 and d["index"]["total_chunks"] == 99
        assert d["models"]["embedding"] == {"model": "m", "dimension": 1024,
                                            "base_url": "http://e"}
        assert d["models"]["asr"]["provider"] == "fireredasr"

    def test_folders_passthrough(self, client):
        rows = [{"id": 1, "name": "f", "chunks": 10}]
        with patch(f"{_STATS}.folders_with_index_stats", return_value=rows):
            r = client.get("/api/admin/folders")
        assert r.json()["data"]["folders"] == rows

    def test_folder_files_passthrough(self, client):
        rows = [{"id": "x", "name": "a.pdf", "status": "failed", "error": "boom"}]
        with patch(f"{_STATS}.files_with_index_status", return_value=rows) as m:
            r = client.get("/api/admin/folders/7/files")
        m.assert_called_once_with(7)
        assert r.json()["data"] == {"folder_id": 7, "files": rows}

    def test_jobs_limit_validated(self, client):
        with patch(f"{_STATS}.recent_jobs", return_value=[]) as m:
            assert client.get("/api/admin/jobs?limit=5").status_code == 200
        m.assert_called_once_with(5)
        assert client.get("/api/admin/jobs?limit=0").status_code == 422
        assert client.get("/api/admin/jobs?limit=999").status_code == 422


class TestChunkViewer:
    def test_chunks_endpoint_shape(self, client):
        data = {"chunks": [{"index": 0, "text": "hi", "chars": 2,
                            "headings": ["H"], "page": 1}],
                "total": 1, "file_name": "a.pdf"}
        with patch(f"{_STATS}.chunks_for_file", return_value=data) as m:
            r = client.get("/api/admin/folders/7/files/fid-1/chunks?limit=100")
        assert r.status_code == 200
        assert r.json()["data"]["chunks"][0]["headings"] == ["H"]
        m.assert_called_once_with(7, "fid-1", 100)

    def test_chunks_limit_validated(self, client):
        assert client.get("/api/admin/folders/7/files/x/chunks?limit=0").status_code == 422
        assert client.get("/api/admin/folders/7/files/x/chunks?limit=9999").status_code == 422


class TestActiveStages:
    def test_active_stages_endpoint(self, client):
        stages = {"fid-1": {"stage": "embedding", "done": 3, "total": 9, "eta": 12}}
        from src.domain.rag.index_job_manager import IndexingJobManager
        mgr = MagicMock()
        mgr.active_file_stages.return_value = stages
        with patch.object(IndexingJobManager, "get_instance", return_value=mgr):
            r = client.get("/api/admin/active-stages")
        assert r.status_code == 200
        assert r.json()["data"]["stages"]["fid-1"]["stage"] == "embedding"


class TestIndexingTrend:
    def test_overview_includes_trend(self, client):
        base = {"folders": 1, "files": 1, "total_size_bytes": 1,
                "index": {"indexed_files": 1, "failed_files": 0, "unindexed_files": 0, "total_chunks": 1},
                "jobs": {"by_status": {}, "active": []}}
        trend = [{"date": "2026-09-07", "count": 8}]
        with patch(f"{_STATS}.overview", return_value=dict(base)), \
             patch(f"{_STATS}.indexing_trend", return_value=trend), \
             patch("src.config.runtime_overrides.get_effective", return_value={}), \
             patch("src.config.config_manager.Config.get_config_model", return_value=SimpleNamespace()):
            r = client.get("/api/admin/overview")
        assert r.json()["data"]["trend"] == trend


class TestAuditAndResources:
    def test_audit_endpoint(self, client):
        entries = [{"key": "rag.rerank.enabled", "action": "set", "old": False,
                    "new": True, "by": "console", "at": "2026-09-07T00:00:00"}]
        with patch("src.adapter.runtime_settings_service.audit_log", return_value=entries):
            r = client.get("/api/admin/audit?limit=50")
        assert r.status_code == 200 and r.json()["data"]["entries"] == entries

    def test_resources_endpoint(self, client):
        res = {"db_size_bytes": 100, "vector_bytes": 50, "vector_tables": 2,
               "disk_used_bytes": 10, "disk_total_bytes": 100}
        with patch(f"{_STATS}.system_resources", return_value=res):
            r = client.get("/api/admin/resources")
        assert r.json()["data"]["vector_tables"] == 2


_OV = "src.api.router.admin_overview"


class TestAnalytics:
    def test_analytics_passthrough_and_params(self, client):
        payload = {"total": 12, "zero_result": 2, "zero_rate": 0.1667,
                   "avg_latency_ms": 88, "top_queries": [{"query": "q", "count": 5, "avg_hits": 3}],
                   "zero_queries": [{"query": "gap", "count": 2}], "by_day": [], "days": 7,
                   "folder_id": 3}
        with patch("db.query_log_db.QueryLogDB.analytics", return_value=payload) as m:
            r = client.get("/api/admin/analytics?days=7&folder_id=3")
        assert r.status_code == 200
        assert r.json()["data"]["total"] == 12
        assert m.call_args.kwargs == {"folder_id": 3, "days": 7}

    def test_analytics_days_validated(self, client):
        assert client.get("/api/admin/analytics?days=0").status_code == 422
        assert client.get("/api/admin/analytics?days=999").status_code == 422


class TestEval:
    def _reset(self):
        from src.api.router import admin_overview as ov
        ov._EVAL.update(running=False, folder=None, done=0, total=0,
                        result=None, error=None, started_at=None)

    def test_status_returns_state(self, client):
        self._reset()
        r = client.get("/api/admin/eval/status")
        assert r.status_code == 200 and r.json()["data"]["running"] is False

    def test_run_requires_folder(self, client):
        self._reset()
        assert client.post("/api/admin/eval/run", json={}).status_code == 422

    def test_run_rejected_when_already_running(self, client):
        from src.api.router import admin_overview as ov
        ov._EVAL["running"] = True
        r = client.post("/api/admin/eval/run", json={"folder_id": 1})
        assert r.status_code == 409
        self._reset()

    def test_run_starts_background_task(self, client):
        self._reset()
        with patch("asyncio.create_task") as T:
            r = client.post("/api/admin/eval/run", json={"folder_id": 1, "n": 5})
        assert r.status_code == 200 and r.json()["data"]["started"] is True
        T.assert_called_once()
        from src.api.router import admin_overview as ov
        assert ov._EVAL["running"] is True and ov._EVAL["total"] == 5
        self._reset()

    async def test_run_bg_success_and_failure(self):
        from unittest.mock import AsyncMock
        from src.api.router import admin_overview as ov
        self._reset()
        fake_mod = MagicMock()
        fake_mod.run_eval_async = AsyncMock(return_value={"folder": "f", "summary": {}})
        with patch.object(ov, "_load_eval_mod", return_value=fake_mod):
            ov._EVAL["running"] = True
            await ov._run_eval_bg("1", 5, 10, False)
        assert ov._EVAL["running"] is False and ov._EVAL["result"]["folder"] == "f"
        self._reset()
        with patch.object(ov, "_load_eval_mod", side_effect=RuntimeError("boom")):
            ov._EVAL["running"] = True
            await ov._run_eval_bg("1", 5, 10, False)
        assert ov._EVAL["running"] is False and ov._EVAL["error"] == "boom"
        self._reset()

    def test_report_no_folder_404(self, client):
        with patch("db.cached_folderdb.CachedFolderDB.get_by_id", return_value=None):
            assert client.get("/api/admin/eval/report?folder_id=9").status_code == 404

    def test_report_missing_file_returns_null(self, client):
        with patch("db.cached_folderdb.CachedFolderDB.get_by_id",
                   return_value=SimpleNamespace(name="nope")), \
             patch("os.path.exists", return_value=False):
            r = client.get("/api/admin/eval/report?folder_id=1")
        assert r.status_code == 200 and r.json()["data"] is None

    def test_report_existing_file_returns_data(self, client):
        from unittest.mock import mock_open
        rep = '{"folder": "f", "summary": {"vector": {}}}'
        with patch("db.cached_folderdb.CachedFolderDB.get_by_id",
                   return_value=SimpleNamespace(name="f")), \
             patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=rep)):
            r = client.get("/api/admin/eval/report?folder_id=1")
        assert r.status_code == 200 and r.json()["data"]["folder"] == "f"

    def test_load_eval_mod_imports_run_eval(self):
        from src.api.router.admin_overview import _load_eval_mod
        mod = _load_eval_mod()
        assert callable(mod.run_eval) and callable(mod.evaluate)


class TestHealth:
    def _cfg(self):
        return SimpleNamespace(database=SimpleNamespace(url="postgresql://u:p@host:5444/db"))

    def test_all_operational(self, client):
        from unittest.mock import AsyncMock
        eff = {"rag.embedding.base_url": "http://e", "rag.embedding.api_key": "SECRETKEY123",
               "rag.llm.base_url": "http://l", "rag.llm.api_key": "SECRETKEY123",
               "rag.rerank.enabled": True, "rag.rerank.base_url": "http://r", "rag.rerank.api_key": "SECRETKEY123",
               "rag.asr.enabled": True, "rag.asr.provider": "fireredasr"}
        with patch("src.config.config_manager.Config.get_config_model", return_value=self._cfg()), \
             patch("src.config.runtime_overrides.get_effective", return_value=eff), \
             patch(f"{_OV}._probe_db_health", return_value=("ok", "verified", 3)), \
             patch(f"{_OV}._probe_endpoint_health", new=AsyncMock(return_value=("ok", "HTTP 200", 40))), \
             patch(f"{_OV}._asr_available", return_value=True):
            r = client.get("/api/admin/health")
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["overall"] == "ok"
        names = {c["name"]: c for c in d["checks"]}
        assert names["Database"]["status"] == "ok" and names["Database"]["latency_ms"] == 3
        # The endpoint returns only base_url, never api_key (does not leak keys)
        assert "SECRETKEY123" not in str(d)
        assert names["Speech-to-Text"]["status"] == "ok"

    def test_rerank_disabled_and_db_down_makes_overall_down(self, client):
        from unittest.mock import AsyncMock
        eff = {"rag.embedding.base_url": "http://e", "rag.llm.base_url": "http://l",
               "rag.rerank.enabled": False, "rag.asr.enabled": False}
        with patch("src.config.config_manager.Config.get_config_model", return_value=self._cfg()), \
             patch("src.config.runtime_overrides.get_effective", return_value=eff), \
             patch(f"{_OV}._probe_db_health", return_value=("down", "ConnectError", 5)), \
             patch(f"{_OV}._probe_endpoint_health", new=AsyncMock(return_value=("ok", "HTTP 200", 40))), \
             patch(f"{_OV}._asr_available", return_value=False):
            r = client.get("/api/admin/health")
        d = r.json()["data"]
        names = {c["name"]: c for c in d["checks"]}
        assert names["Reranker"]["status"] == "disabled"
        assert names["Speech-to-Text"]["status"] == "disabled"
        assert names["Database"]["status"] == "down"
        assert d["overall"] == "down"

    def test_probe_db_health_ok_and_fail(self):
        from src.api.router.admin_overview import _probe_db_health
        eng = MagicMock()
        eng.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        eng.connect.return_value.__exit__ = MagicMock(return_value=False)
        with patch("db.db.get_engine", return_value=eng):
            status, detail, ms = _probe_db_health()
        assert status == "ok" and isinstance(ms, int)
        with patch("db.db.get_engine", side_effect=RuntimeError("boom")):
            status, detail, ms = _probe_db_health()
        assert status == "down" and detail == "RuntimeError"

    @pytest.mark.asyncio
    async def test_probe_endpoint_health_unset_and_ok(self):
        from src.api.router.admin_overview import _probe_endpoint_health
        # Empty base_url -> unset, no network call
        status, detail, ms = await _probe_endpoint_health("", None)
        assert status == "unset" and ms is None
        # Reachable endpoint -> ok (mock httpx client)
        from unittest.mock import AsyncMock
        resp = SimpleNamespace(status_code=200)
        client_cm = MagicMock()
        client_cm.__aenter__ = AsyncMock(return_value=SimpleNamespace(get=AsyncMock(return_value=resp)))
        client_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client_cm):
            status, detail, ms = await _probe_endpoint_health("http://x", "key")
        assert status == "ok" and detail == "HTTP 200"
