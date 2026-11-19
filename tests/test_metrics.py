"""Prometheus metrics at /metrics: names and labels, the bearer-token guard, the route-template
middleware, metered model clients, index-job transitions, scrape-time gauges, and the
promise that no folder, file, user or query ever becomes a label value."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from src import metrics
from src.api.router.metrics import include_metrics
from src.middleware.http_metrics import HttpMetricsMiddleware, route_template
from src.utils.metered_http import metered_async_client, metered_client, openai_sdk_clients

SECRET_FOLDER = "payroll-secret-folder"
SECRET_USER = "sub:alice@example.com"


def value(name, labels=None):
    return metrics.REGISTRY.get_sample_value(name, labels or {}) or 0.0


def scrape_text() -> str:
    body, content_type = metrics.render()
    assert content_type.startswith("text/plain")
    return body.decode()


def families() -> dict:
    return {f.name: f for f in text_string_to_metric_families(scrape_text())}


# -- exposition ----------------------------------------------------------------------------

EXPECTED_FAMILIES = {
    "folio_http_requests": "counter",
    "folio_http_request_duration_seconds": "histogram",
    "folio_retrieval_stage_duration_seconds": "histogram",
    "folio_retrieval_search_duration_seconds": "histogram",
    "folio_mcp_tool_calls": "counter",
    "folio_mcp_tool_call_duration_seconds": "histogram",
    "folio_index_jobs_submitted": "counter",
    "folio_index_jobs_started": "counter",
    "folio_index_jobs_finished": "counter",
    "folio_index_job_duration_seconds": "histogram",
    "folio_index_files": "counter",
    "folio_model_requests": "counter",
    "folio_model_request_duration_seconds": "histogram",
    "folio_index_jobs_active": "gauge",
    "folio_revocation_feed_enabled": "gauge",
    "folio_db_pool_size": "gauge",
    "folio_db_pool_checked_out": "gauge",
    "folio_build_info": "gauge",
    "process_resident_memory_bytes": "gauge",
}


class TestExposition:
    def test_every_family_is_exposed_with_its_type_and_parses(self):
        metrics.observe_http("/health", "GET", 200, 0.01)
        fams = families()  # the official parser accepts the whole body
        for name, kind in EXPECTED_FAMILIES.items():
            assert name in fams, name
            assert fams[name].type == kind, name

    def test_label_names(self):
        metrics.observe_mcp_call("search", "ok", 0.1)
        metrics.observe_model_call("rerank", 200, 0.1)
        fams = families()

        def labels_of(name):
            return {k for s in fams[name].samples for k in s.labels} - {"le"}

        assert labels_of("folio_http_requests") == {"route", "method", "status"}
        assert labels_of("folio_mcp_tool_calls") == {"mode", "outcome"}
        assert labels_of("folio_model_requests") == {"role", "outcome"}
        assert labels_of("folio_model_request_duration_seconds") == {"role"}

    def test_build_info_carries_the_version(self):
        from src.version import get_version
        sample = next(s for s in families()["folio_build_info"].samples)
        assert sample.labels["version"] == str(get_version()) and sample.value == 1.0


# -- bounded labels ------------------------------------------------------------------------

class TestBoundedLabels:
    def test_unknown_values_fall_into_other(self):
        before = value("folio_retrieval_stage_duration_seconds_count", {"stage": "other"})
        metrics.observe_search({SECRET_FOLDER: 5.0}, 0.01)
        assert value("folio_retrieval_stage_duration_seconds_count", {"stage": "other"}) == before + 1

        before = value("folio_mcp_tool_calls_total", {"mode": "other", "outcome": "other"})
        metrics.observe_mcp_call(SECRET_FOLDER, SECRET_USER, 0.1)
        assert value("folio_mcp_tool_calls_total", {"mode": "other", "outcome": "other"}) == before + 1

        before = value("folio_model_requests_total", {"role": "other", "outcome": "ok"})
        metrics.observe_model_call(SECRET_FOLDER, 200, 0.1)
        assert value("folio_model_requests_total", {"role": "other", "outcome": "ok"}) == before + 1

        before = value("folio_index_jobs_submitted_total", {"kind": "other"})
        metrics.job_submitted(SECRET_FOLDER)
        assert value("folio_index_jobs_submitted_total", {"kind": "other"}) == before + 1

        before = value("folio_http_requests_total", {"route": "/x", "method": "other", "status": "200"})
        metrics.observe_http("/x", "BREW", 200, 0.1)
        assert value("folio_http_requests_total", {"route": "/x", "method": "other", "status": "200"}) == before + 1

    def test_no_folder_user_or_query_ever_reaches_the_output(self):
        metrics.observe_search({SECRET_FOLDER: 1.0}, 0.01)
        metrics.observe_mcp_call(SECRET_FOLDER, SECRET_USER, 0.1)
        metrics.observe_model_call(SECRET_USER, 200, 0.1)
        metrics.job_submitted(SECRET_USER)
        metrics.file_finished(SECRET_FOLDER)
        text = scrape_text()
        assert SECRET_FOLDER not in text and SECRET_USER not in text

    def test_recording_never_raises(self):
        metrics.observe_http("/x", None, "not-a-status", 0.1)
        metrics.observe_search({"dense": "nan?"}, 0.1)
        metrics.job_finished("index", "succeeded", "not-a-date")

    @pytest.mark.parametrize("code,outcome", [(200, "ok"), (302, "ok"), (404, "client_error"),
                                              (429, "client_error"), (500, "server_error"),
                                              (None, "error")])
    def test_model_outcome(self, code, outcome):
        assert metrics.model_outcome(code) == outcome


# -- HTTP middleware ------------------------------------------------------------------------

def _app(tmp_path):
    from fastapi import APIRouter
    app = FastAPI()
    sub = APIRouter()

    @sub.post("/query")
    async def query():
        return {}

    @sub.get("/files/{folder_id}/index/jobs/{job_id}")
    async def job(folder_id: int, job_id: str):
        return {}

    app.include_router(sub, prefix="/api/rag")

    @app.get("/api/folders/{folder_id}")
    async def get_folder(folder_id: str):
        return {"id": folder_id}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    (tmp_path / "index.html").write_text("ok")
    app.mount("/admin", StaticFiles(directory=str(tmp_path), html=True))
    return HttpMetricsMiddleware(app)


class TestHttpMiddleware:
    def test_the_route_template_is_the_label_never_the_path(self, tmp_path):
        client = TestClient(_app(tmp_path))
        labels = {"route": "/api/folders/{folder_id}", "method": "GET", "status": "200"}
        before = value("folio_http_requests_total", labels)
        assert client.get(f"/api/folders/{SECRET_FOLDER}").status_code == 200
        assert value("folio_http_requests_total", labels) == before + 1
        assert value("folio_http_request_duration_seconds_count",
                     {"route": "/api/folders/{folder_id}", "method": "GET"}) >= 1
        assert SECRET_FOLDER not in scrape_text()

    def test_an_included_router_keeps_its_prefix(self, tmp_path):
        client = TestClient(_app(tmp_path))
        q = {"route": "/api/rag/query", "method": "POST", "status": "200"}
        before = value("folio_http_requests_total", q)
        client.post("/api/rag/query")
        assert value("folio_http_requests_total", q) == before + 1
        j = {"route": "/api/rag/files/{folder_id}/index/jobs/{job_id}", "method": "GET", "status": "200"}
        before = value("folio_http_requests_total", j)
        client.get("/api/rag/files/7/index/jobs/" + SECRET_FOLDER)
        assert value("folio_http_requests_total", j) == before + 1
        assert SECRET_FOLDER not in scrape_text()

    def test_prefix_reconstruction_guards(self):
        r = SimpleNamespace(path_format="/x/{id}")
        assert route_template({"route": r, "path": "/api/x/5", "path_params": {"id": "5"}}) == "/api/x/{id}"
        # a prefix segment that is a parameter value is never used
        assert route_template({"route": r, "path": "/5/x/5", "path_params": {"id": "5"}}) == "/x/{id}"
        # segments that do not line up with the template
        assert route_template({"route": r, "path": "/api/y/5", "path_params": {"id": "5"}}) == "/x/{id}"
        # path convertors span segments
        p = SimpleNamespace(path_format="/f/{rest:path}")
        assert route_template({"route": p, "path": "/api/f/a/b/c"}) == "/f/{rest:path}"
        assert route_template({"route": p, "path": "/api/f/x/f/y",
                               "path_params": {"rest": "x/f/y"}}) == "/f/{rest:path}"
        assert route_template({"route": r, "path": "/x/5"}) == "/x/{id}"

    def test_unmatched_and_mounted_paths(self, tmp_path):
        client = TestClient(_app(tmp_path))
        un = {"route": "unmatched", "method": "GET", "status": "404"}
        before = value("folio_http_requests_total", un)
        client.get("/no/such/path/" + SECRET_FOLDER)
        assert value("folio_http_requests_total", un) == before + 1

        mounted = {"route": "/admin/*", "method": "GET", "status": "200"}
        before = value("folio_http_requests_total", mounted)
        assert client.get("/admin/index.html").status_code == 200
        assert value("folio_http_requests_total", mounted) == before + 1

    def test_an_exception_counts_as_500(self, tmp_path):
        client = TestClient(_app(tmp_path), raise_server_exceptions=False)
        labels = {"route": "/boom", "method": "GET", "status": "500"}
        before = value("folio_http_requests_total", labels)
        client.get("/boom")
        assert value("folio_http_requests_total", labels) == before + 1

    async def test_an_error_raised_before_any_response_counts_as_500(self):
        async def raw(scope, receive, send):
            raise RuntimeError("wrapper failed")

        labels = {"route": "unmatched", "method": "POST", "status": "500"}
        before = value("folio_http_requests_total", labels)
        with pytest.raises(RuntimeError):
            await HttpMetricsMiddleware(raw)({"type": "http", "method": "POST", "path": "/x"},
                                             AsyncMock(), AsyncMock())
        assert value("folio_http_requests_total", labels) == before + 1

    async def test_non_http_scopes_pass_through(self):
        inner = AsyncMock()
        await HttpMetricsMiddleware(inner)({"type": "lifespan"}, AsyncMock(), AsyncMock())
        inner.assert_awaited_once()

    def test_route_template_shapes(self):
        assert route_template({}) == "unmatched"
        assert route_template({"route": Mount("", app=MagicMock())}) == "unmatched"
        assert route_template({"route": Mount("/admin", app=MagicMock())}) == "/admin/*"
        assert route_template({"route": SimpleNamespace(path_format="/a/{b}")}) == "/a/{b}"
        assert route_template({"root_path": "/admin"}) == "/admin/*"
        assert route_template({"root_path": "/folio/admin"}, "/folio") == "/admin/*"
        assert route_template({"root_path": "/folio"}, "/folio") == "unmatched"


# -- /metrics and its token -----------------------------------------------------------------

LOOPBACK = ("127.0.0.1", 50000)


def _metrics_app(token=None, enabled=True, client=LOOPBACK):
    app = FastAPI()
    app.state.config = SimpleNamespace(metrics=SimpleNamespace(enabled=enabled, token=token))
    added = include_metrics(app, app.state.config)
    return TestClient(app, client=client), added


class TestEndpoint:
    @pytest.mark.parametrize("client", [LOOPBACK, ("127.0.0.9", 1), ("::1", 1), ("::ffff:127.0.0.1", 1)])
    def test_this_host_scrapes_without_a_token(self, client):
        c, added = _metrics_app(client=client)
        assert added
        r = c.get("/metrics")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
        assert "folio_build_info" in r.text

    @pytest.mark.parametrize("client,headers", [
        (("10.0.0.5", 1), {}),
        (("203.0.113.7", 1), {}),
        (("testclient", 1), {}),
        (("::ffff:10.0.0.5", 1), {}),
        (LOOPBACK, {"X-Forwarded-For": "203.0.113.7"}),
        (LOOPBACK, {"Forwarded": "for=203.0.113.7"}),
        (LOOPBACK, {"X-Real-IP": "203.0.113.7"}),
    ])
    def test_without_a_token_anyone_else_is_refused(self, client, headers):
        c, _ = _metrics_app(client=client)
        r = c.get("/metrics", headers=headers)
        assert r.status_code == 403
        assert r.json()["error"] == "forbidden" and "METRICS_TOKEN" in r.json()["detail"]
        assert "folio_" not in r.text

    def test_disabled_is_not_served(self):
        client, added = _metrics_app(enabled=False)
        assert not added
        assert client.get("/metrics").status_code == 404

    @pytest.mark.parametrize("headers", [
        {}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic czNjcmV0"},
        {"Authorization": "Bearer "}, {"Authorization": "s3cret-metrics"},
    ])
    @pytest.mark.parametrize("client", [LOOPBACK, ("10.0.0.5", 1)])
    def test_a_token_is_required_when_configured(self, headers, client):
        # with a token, loopback is no exception
        c, _ = _metrics_app(token="s3cret-metrics", client=client)
        r = c.get("/metrics", headers=headers)
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Bearer")
        assert "folio_" not in r.text and "s3cret" not in r.text

    def test_the_right_token_scrapes_from_anywhere(self):
        client, _ = _metrics_app(token="s3cret-metrics", client=("203.0.113.7", 1))
        r = client.get("/metrics", headers={"Authorization": "Bearer s3cret-metrics",
                                            "X-Forwarded-For": "198.51.100.1"})
        assert r.status_code == 200 and "folio_build_info" in r.text


class TestConfig:
    def test_defaults_and_token_hidden_from_repr(self):
        from src.config.model import MetricsConfig
        assert MetricsConfig().enabled is True and MetricsConfig().token is None
        assert MetricsConfig(token="").token is None
        conf = MetricsConfig(token="tok-123")
        assert "tok-123" not in repr(conf)

    def test_committed_system_yaml_reads_the_token_from_the_environment(self):
        from pathlib import Path
        import yaml
        raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "system.yaml").read_text())
        assert raw["metrics"] == {"enabled": True, "token": "${METRICS_TOKEN:-}"}


# -- metered model clients ------------------------------------------------------------------

def _transport(status=200, exc=None):
    def handler(request):
        if exc is not None:
            raise exc
        return httpx.Response(status, json={})
    return handler


class TestMeteredClients:
    def test_sync_ok_server_error_and_transport_error(self):
        def count(outcome):
            return value("folio_model_requests_total", {"role": "rerank", "outcome": outcome})
        ok, five, err = count("ok"), count("server_error"), count("error")
        with metered_client("rerank", transport=httpx.MockTransport(_transport(200))) as c:
            c.post("http://m/rerank", json={})
        with metered_client("rerank", transport=httpx.MockTransport(_transport(503))) as c:
            c.post("http://m/rerank", json={})
        with metered_client("rerank", transport=httpx.MockTransport(
                _transport(exc=httpx.ConnectError("refused")))) as c:
            with pytest.raises(httpx.ConnectError):
                c.post("http://m/rerank", json={})
        assert (count("ok"), count("server_error"), count("error")) == (ok + 1, five + 1, err + 1)

    async def test_async_client(self):
        before = value("folio_model_requests_total", {"role": "embedding", "outcome": "client_error"})
        async with metered_async_client("embedding", transport=httpx.MockTransport(_transport(401))) as c:
            await c.post("http://m/embeddings", json={})
        assert value("folio_model_requests_total",
                     {"role": "embedding", "outcome": "client_error"}) == before + 1

    def test_openai_sdk_clients_keep_the_sdk_defaults(self):
        from openai import DefaultAsyncHttpxClient, DefaultHttpxClient
        clients = openai_sdk_clients("embedding")
        assert isinstance(clients["http_client"], DefaultHttpxClient)
        assert isinstance(clients["async_http_client"], DefaultAsyncHttpxClient)
        assert clients["http_client"]._folio_role == "embedding"

    def test_the_openai_sdk_requests_are_counted(self):
        from src.domain.rag.llm_client import create_openai_client
        cfg = SimpleNamespace(provider="vllm", base_url="http://llm.test/v1", api_key="k",
                              timeout=5, max_retries=0)
        client = create_openai_client(cfg, metrics_role="llm")
        before = value("folio_model_requests_total", {"role": "llm", "outcome": "ok"})
        reply = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": "hi"}}]}
        with patch.object(httpx.HTTPTransport, "handle_request",
                          return_value=httpx.Response(200, json=reply)):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        assert value("folio_model_requests_total", {"role": "llm", "outcome": "ok"}) == before + 1

    def test_probes_are_not_metered(self):
        from src.domain.rag.llm_client import create_openai_client
        cfg = SimpleNamespace(provider="vllm", base_url="http://llm.test/v1", api_key="k")
        assert not hasattr(create_openai_client(cfg)._client, "_folio_role")


# -- retrieval and MCP ----------------------------------------------------------------------

class TestRetrievalAndMcp:
    async def test_a_search_records_its_stages_and_total(self):
        from src.domain.rag.hybrid_search import HybridSearcher, SearchSettings, record_stage
        import time

        async def fake_retrieve(self, query_text, settings, depth):
            record_stage("dense", time.perf_counter())
            record_stage("bm25", time.perf_counter())
            return [], [], [], [], []

        dense = value("folio_retrieval_stage_duration_seconds_count", {"stage": "dense"})
        total = value("folio_retrieval_search_duration_seconds_count")
        with patch.object(HybridSearcher, "_retrieve", new=fake_retrieve):
            await HybridSearcher(MagicMock(table_name="t"), index=MagicMock()).search(
                "q", SearchSettings(top_k=3))
        assert value("folio_retrieval_stage_duration_seconds_count", {"stage": "dense"}) == dense + 1
        assert value("folio_retrieval_search_duration_seconds_count") == total + 1

    def test_merge_and_expand_are_recorded(self):
        import time
        from src.domain.rag.auto_merging import _timing
        before = value("folio_retrieval_stage_duration_seconds_count", {"stage": "merge"})
        outcome = SimpleNamespace(timings={})
        _timing(outcome, "merge", time.perf_counter())
        assert "merge" in outcome.timings
        assert value("folio_retrieval_stage_duration_seconds_count", {"stage": "merge"}) == before + 1

    @pytest.mark.parametrize("effect,outcome", [
        ({"mode": "search", "results": []}, "ok"),
        ({"mode": "read", "error": "file_id is required for read mode"}, "invalid_request"),
        (RuntimeError("db down"), "error"),
    ])
    async def test_mcp_tool_outcomes(self, effect, outcome):
        from src.fastmcp_tools import agentic_tools as at
        adapter = MagicMock()
        if isinstance(effect, Exception):
            adapter.query = AsyncMock(side_effect=effect)
        else:
            adapter.query = AsyncMock(return_value=effect)
        mode = "read" if outcome == "invalid_request" else "search"
        before = value("folio_mcp_tool_calls_total", {"mode": mode, "outcome": outcome})
        with patch.object(at, "extract_owner_key", AsyncMock(return_value=SECRET_USER)), \
             patch.object(at, "get_rag_adapter", return_value=adapter), \
             patch.object(at, "_retrieval_config", return_value=None):
            await at._query(folder_name=SECRET_FOLDER, mode=mode, query="q", file_id=None,
                            top_k=3, similarity_cutoff=None, expand_context=False)
        assert value("folio_mcp_tool_calls_total", {"mode": mode, "outcome": outcome}) == before + 1
        assert SECRET_FOLDER not in scrape_text()

    async def test_mcp_auth_failures(self):
        from src.domain.exceptions import UnauthorizedAccessError
        from src.fastmcp_tools import agentic_tools as at
        adapter = MagicMock()
        adapter.query = AsyncMock(side_effect=UnauthorizedAccessError(
            resource_type="folder", resource_id="f", reason="no"))
        before = value("folio_mcp_tool_calls_total", {"mode": "list", "outcome": "forbidden"})
        with patch.object(at, "extract_owner_key", AsyncMock(return_value="k")), \
             patch.object(at, "get_rag_adapter", return_value=adapter):
            await at._query(folder_name="f", mode="list", query="", file_id=None,
                            top_k=3, similarity_cutoff=None, expand_context=False)
        assert value("folio_mcp_tool_calls_total", {"mode": "list", "outcome": "forbidden"}) == before + 1


# -- index jobs -----------------------------------------------------------------------------

@pytest.fixture()
def manager():
    from src.domain.rag.index_job_manager import IndexingJobManager
    with patch("db.indexjobdb.IndexJobDB.upsert"), \
         patch("db.indexjobdb.IndexJobDB.ensure_table"), \
         patch("db.fileindexdb.FileIndexDB.ensure_content_hash_columns"):
        m = IndexingJobManager()
        m._persist = lambda d: None
        yield m


class TestIndexJobs:
    def test_each_transition_is_counted_once(self, manager):
        from src.domain.rag.index_job_manager import IndexJobState, JobStatus
        started_at = (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat()
        state = IndexJobState(job_id="j1", folder_id=1, started_at=started_at)
        manager._jobs["j1"] = state
        fin = {"kind": "index", "status": "succeeded"}
        b_started = value("folio_index_jobs_started_total", {"kind": "index"})
        b_fin = value("folio_index_jobs_finished_total", fin)
        b_sum = value("folio_index_job_duration_seconds_sum", fin)

        manager._update_job_state("j1", status=JobStatus.RUNNING)
        manager._update_job_state("j1", status=JobStatus.RUNNING, message="still running")
        manager._update_job_state("j1", status=JobStatus.SUCCEEDED)
        manager._update_job_state("j1", status=JobStatus.FAILED)  # ignored: already terminal

        assert value("folio_index_jobs_started_total", {"kind": "index"}) == b_started + 1
        assert value("folio_index_jobs_finished_total", fin) == b_fin + 1
        assert 85 <= value("folio_index_job_duration_seconds_sum", fin) - b_sum <= 600

    def test_live_counts_feed_the_active_gauge(self, manager):
        from src.domain.rag.index_job_manager import IndexingJobManager, IndexJobState, JobStatus
        manager._jobs["a"] = IndexJobState(job_id="a", folder_id=1)
        manager._jobs["b"] = IndexJobState(job_id="b", folder_id=2, status=JobStatus.RUNNING)
        manager._jobs["c"] = IndexJobState(job_id="c", folder_id=3, status=JobStatus.SUCCEEDED,
                                           kind="rebuild_fts")
        assert manager.live_job_counts() == {("index", "pending"): 1, ("index", "running"): 1}
        with patch.object(IndexingJobManager, "_instance", manager):
            assert value("folio_index_jobs_active", {"kind": "index", "status": "running"}) == 1
            assert value("folio_index_jobs_active", {"kind": "rebuild_fts", "status": "running"}) == 0

    async def test_submitted_started_and_finished_for_a_real_job(self, manager):
        kind = {"kind": "rebuild_fts"}
        fin = {"kind": "rebuild_fts", "status": "succeeded"}
        b_sub = value("folio_index_jobs_submitted_total", kind)
        b_start = value("folio_index_jobs_started_total", kind)
        b_fin = value("folio_index_jobs_finished_total", fin)

        def work(progress, should_stop):
            return {"message": "done"}
        job = await manager.start_maintenance_job(folder_id=9, kind="rebuild_fts", work=work)
        for _ in range(250):
            st = await manager.get_status(job["job_id"])
            if st and st["status"] == "succeeded":
                break
            await asyncio.sleep(0.02)
        assert value("folio_index_jobs_submitted_total", kind) == b_sub + 1
        assert value("folio_index_jobs_started_total", kind) == b_start + 1
        assert value("folio_index_jobs_finished_total", fin) == b_fin + 1

    def test_files_by_status(self, manager):
        from src.domain.rag.index_job_manager import IndexJobState, IndexProgressTracker
        manager._jobs["f"] = IndexJobState(job_id="f", folder_id=1)
        before = value("folio_index_files_total", {"status": "skipped"})
        IndexProgressTracker(manager, "f").on_file_completed(1, "skipped", "already indexed")
        assert value("folio_index_files_total", {"status": "skipped"}) == before + 1


# -- revocation feed and DB pools -----------------------------------------------------------

class TestStateGauges:
    def test_revocation_feed(self):
        from src.auth import revocation as rv
        assert value("folio_revocation_feed_enabled") == 0
        fake = SimpleNamespace(metrics_snapshot=lambda: {
            "last_success": 1_700_000_000.0, "consecutive_failures": 3, "stale": True, "revoked": 4})
        rv.set_revocation_list(fake)
        try:
            assert value("folio_revocation_feed_enabled") == 1
            assert value("folio_revocation_feed_last_success_timestamp_seconds") == 1_700_000_000.0
            assert value("folio_revocation_feed_consecutive_failures") == 3
            assert value("folio_revocation_feed_stale") == 1
            assert value("folio_revocation_revoked_tokens") == 4
        finally:
            rv.set_revocation_list(None)

    def test_real_revocation_list_snapshot(self):
        from src.auth.revocation import RevocationList, RevocationSettings
        rl = RevocationList(RevocationSettings(
            issuer="http://gate.test", client_id="c", client_secret="s", poll_interval=30.0,
            fail_mode="open", stale_after_intervals=4), http_client=MagicMock())
        snap = rl.metrics_snapshot()
        assert snap == {"last_success": None, "consecutive_failures": 0, "stale": True, "revoked": 0}

    def test_db_pools_are_read_without_creating_engines(self):
        import db.db as dbmod
        from sqlalchemy import create_engine
        engine = create_engine("postgresql://u:p@127.0.0.1:1/x", pool_size=3, max_overflow=2)
        with patch.object(dbmod, "_engine", None), patch.object(dbmod, "_async_engine", None):
            assert dbmod.existing_engine_pools() == {}
            assert value("folio_db_pool_size", {"pool": "sync"}) == 0
        with patch.object(dbmod, "_engine", engine), patch.object(dbmod, "_async_engine", None):
            assert value("folio_db_pool_size", {"pool": "sync"}) == 3
            assert value("folio_db_pool_checked_out", {"pool": "sync"}) == 0
        engine.dispose()


def test_no_created_series_in_the_output():
    metrics.observe_http("/health", "GET", 200, 0.01)
    assert "_created" not in scrape_text()
