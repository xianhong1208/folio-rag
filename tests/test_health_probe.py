"""_probe_endpoint_health: OpenAI-compatible endpoint health probe.

A vLLM reranker exposes its model list under /v1/ while being configured with a
base_url that omits it, so GET {base}/models 404s even though the service is fine.
The probe must fall back to {base}/v1/models before reporting.
"""
import httpx
import pytest

from src.api.router import admin_overview as ov


class _Resp:
    def __init__(self, code):
        self.status_code = code


def _client_returning(url_to_code):
    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            for suffix, code in url_to_code:
                if url.endswith(suffix):
                    return _Resp(code)
            return _Resp(404)
    return _Client


async def test_falls_back_to_v1_models_when_models_404s(monkeypatch):
    # /models -> 404, /v1/models -> 200 : should report the healthy 200.
    monkeypatch.setattr(httpx, "AsyncClient",
                        _client_returning([("/v1/models", 200), ("/models", 404)]))
    status, detail, _ = await ov._probe_endpoint_health("http://localhost:8787", None)
    assert status == "ok" and detail == "HTTP 200"


async def test_direct_models_200_no_fallback(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _client_returning([("/models", 200)]))
    status, detail, _ = await ov._probe_endpoint_health("http://host/v1", None)
    assert status == "ok" and detail == "HTTP 200"


async def test_no_double_v1_when_base_already_v1(monkeypatch):
    # base already ends with /v1 -> do not probe /v1/v1/models; a 404 stays 404.
    monkeypatch.setattr(httpx, "AsyncClient", _client_returning([("/models", 404)]))
    status, detail, _ = await ov._probe_endpoint_health("http://host/v1", None)
    assert status == "ok" and detail == "HTTP 404"


async def test_5xx_is_down(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _client_returning([("/models", 503)]))
    status, detail, _ = await ov._probe_endpoint_health("http://host/v1", None)
    assert status == "down" and detail == "HTTP 503"


async def test_empty_base_is_unset(monkeypatch):
    status, detail, ms = await ov._probe_endpoint_health("", None)
    assert status == "unset" and ms is None
