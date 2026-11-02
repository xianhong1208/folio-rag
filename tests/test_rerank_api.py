"""Rerank API styles (rerank_api.py), the shared retry helper (utils/http_retry.py) and the
Reranker over each style. HTTP is mocked (httpx.MockTransport) or served by the in-process
fake server — no network, no keys.
"""

from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from src.domain.rag.rerank_api import RerankApi
from src.utils.http_retry import (
    RetryPolicy,
    arequest_with_retry,
    compute_delay,
    join_endpoint,
    request_with_retry,
    retry_after_seconds,
)
from tests.fake_model_server import FakeModelServer


class TestJoinEndpoint:
    @pytest.mark.parametrize("base,path,want", [
        ("http://h", "/v1/score", "http://h/v1/score"),
        ("http://h/", "/v1/score", "http://h/v1/score"),
        ("http://h/v1", "/v1/score", "http://h/v1/score"),
        ("http://h/v1/", "/v1/rerank", "http://h/v1/rerank"),
        ("https://api.cohere.com/v2", "/v2/rerank", "https://api.cohere.com/v2/rerank"),
        ("http://h/v1", "/v2/rerank", "http://h/v1/v2/rerank"),  # different versions stay
        ("http://h/api", "rerank", "http://h/api/rerank"),
        ("http://h/v1", "/models", "http://h/v1/models"),
        ("http://h/vx", "/vx/a", "http://h/vx/vx/a"),  # not a version segment
    ])
    def test_join(self, base, path, want):
        assert join_endpoint(base, path) == want


class TestRetryHelpers:
    def test_retry_after_seconds_and_date(self):
        assert retry_after_seconds(None) is None
        assert retry_after_seconds(httpx.Response(429)) is None
        assert retry_after_seconds(httpx.Response(429, headers={"Retry-After": "3"})) == 3.0
        assert retry_after_seconds(httpx.Response(429, headers={"Retry-After": "-2"})) == 0.0
        assert retry_after_seconds(httpx.Response(429, headers={"Retry-After": "soon"})) is None
        when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
        got = retry_after_seconds(httpx.Response(429, headers={"Retry-After": when}))
        assert 25 <= got <= 31

    def test_compute_delay(self):
        p = RetryPolicy(backoff_base=1, backoff_max=5, retry_after_max=10)
        assert compute_delay(p, 0, rng=lambda: 1.0) == 1
        assert compute_delay(p, 4, rng=lambda: 1.0) == 5  # capped window
        assert compute_delay(p, 4, rng=lambda: 0.5) == 2.5  # jitter
        hinted = httpx.Response(429, headers={"Retry-After": "120"})
        assert compute_delay(p, 0, hinted) == 10  # server hint, capped

    def _client(self, statuses, exc_first=None):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if exc_first and calls["n"] == 1:
                raise exc_first
            status = statuses[min(calls["n"] - 1 - (1 if exc_first else 0), len(statuses) - 1)]
            return httpx.Response(status, json={"n": calls["n"]})
        return httpx.Client(transport=httpx.MockTransport(handler)), calls

    def test_retries_retryable_status_then_succeeds(self):
        client, calls = self._client([503, 429, 200])
        slept, retried = [], []
        r = request_with_retry(client, "GET", "http://h/x", policy=RetryPolicy(max_retries=3),
                               sleep=slept.append, on_retry=lambda n, why: retried.append(why))
        assert r.status_code == 200 and calls["n"] == 3 and len(slept) == 2
        assert retried == ["HTTP 503", "HTTP 429"]

    def test_gives_up_and_returns_last_response(self):
        client, calls = self._client([500])
        r = request_with_retry(client, "GET", "http://h/x", policy=RetryPolicy(max_retries=1),
                               sleep=lambda s: None)
        assert r.status_code == 500 and calls["n"] == 2

    def test_does_not_retry_client_errors(self):
        client, calls = self._client([401])
        r = request_with_retry(client, "GET", "http://h/x", policy=RetryPolicy(max_retries=3),
                               sleep=lambda s: None)
        assert r.status_code == 401 and calls["n"] == 1

    def test_retries_transport_errors(self):
        client, calls = self._client([200], exc_first=httpx.ConnectError("refused"))
        r = request_with_retry(client, "GET", "http://h/x", policy=RetryPolicy(max_retries=1),
                               sleep=lambda s: None, on_retry=lambda n, why: None)
        assert r.status_code == 200 and calls["n"] == 2

    def test_transport_error_exhausted_raises(self):
        def handler(request):
            raise httpx.ReadTimeout("slow")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(httpx.ReadTimeout):
            request_with_retry(client, "GET", "http://h/x", policy=RetryPolicy(max_retries=2),
                               sleep=lambda s: None)

    def test_non_transport_exception_is_not_retried(self):
        def handler(request):
            raise RuntimeError("bug")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(RuntimeError):
            request_with_retry(client, "GET", "http://h/x", policy=RetryPolicy(max_retries=2),
                               sleep=lambda s: pytest.fail("must not sleep"))

    async def test_async_variant(self):
        n = {"c": 0}

        def handler(request):
            n["c"] += 1
            if n["c"] == 1:
                raise httpx.ConnectError("x")
            return httpx.Response(429 if n["c"] == 2 else 200)

        async def nosleep(s):
            return None
        retried = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            r = await arequest_with_retry(c, "GET", "http://h", policy=RetryPolicy(max_retries=3),
                                          sleep=nosleep, on_retry=lambda a, w: retried.append(w))
        assert r.status_code == 200 and retried == ["ConnectError", "HTTP 429"]

    async def test_async_exhausted_raises(self):
        def handler(request):
            raise httpx.ConnectTimeout("x")

        async def nosleep(s):
            return None
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            with pytest.raises(httpx.ConnectTimeout):
                await arequest_with_retry(c, "GET", "http://h", policy=RetryPolicy(max_retries=1),
                                          sleep=nosleep)


class TestRerankApi:
    def test_unknown_style(self):
        with pytest.raises(ValueError):
            RerankApi("voyage", "http://h", "m")

    def test_score_style(self):
        api = RerankApi("score", "http://h:8787", "bge")
        assert api.url == "http://h:8787/v1/score"
        assert api.build_payload("q", ["a", "b"]) == {"model": "bge", "text_1": "q", "text_2": ["a", "b"]}
        assert api.parse_scores({"data": [{"index": 1, "score": 0.2}, {"index": 0, "score": 0.9}]}, 2) == [0.9, 0.2]

    def test_cohere_style(self):
        api = RerankApi("cohere", "https://api.cohere.com/v2", "rerank-v3.5", endpoint_path="/v2/rerank")
        assert api.url == "https://api.cohere.com/v2/rerank"
        p = api.build_payload("q", ["a", "b"])
        assert p == {"model": "rerank-v3.5", "query": "q", "documents": ["a", "b"], "top_n": 2}
        assert api.parse_scores({"results": [{"index": 1, "relevance_score": 0.7},
                                             {"index": 0, "relevance_score": 0.1}]}, 2) == [0.1, 0.7]

    def test_jina_style(self):
        api = RerankApi("jina", "https://api.jina.ai", "jina-reranker-v2-base-multilingual")
        assert api.url == "https://api.jina.ai/v1/rerank"
        assert api.build_payload("q", ["a"])["return_documents"] is False
        assert api.parse_scores({"results": [{"index": 0, "relevance_score": 0.3,
                                              "document": {"text": "a"}}]}, 1) == [0.3]

    def test_generic_style_tolerant_parsing(self):
        api = RerankApi("generic", "http://h/v1", "m")
        assert api.url == "http://h/v1/rerank"
        assert api.parse_scores({"data": [{"index": 0, "score": 0.4}]}, 1) == [0.4]
        assert api.parse_scores([{"index": 0, "score": 0.5}], 1) == [0.5]

    @pytest.mark.parametrize("data", [
        {}, {"results": "x"}, "nope",
        {"results": [{"index": 5, "relevance_score": 1}]},
        {"results": [{"index": 0}]},
        {"results": [{"index": "0", "relevance_score": 1}]},
    ])
    def test_malformed_responses_raise(self, data):
        with pytest.raises(ValueError):
            RerankApi("cohere", "http://h", "m").parse_scores(data, 1)

    def test_partial_scores_raise(self):
        with pytest.raises(ValueError, match="unscored"):
            RerankApi("cohere", "http://h", "m").parse_scores(
                {"results": [{"index": 0, "relevance_score": 1}]}, 2)


def _nodes(*texts):
    return [SimpleNamespace(text=t, score=0.0) for t in texts]


class TestRerankerOverStyles:
    @pytest.mark.parametrize("style,base_suffix", [
        ("score", ""), ("cohere", "/v1"), ("jina", ""), ("generic", "/v1"),
    ])
    def test_sync_rerank_reorders_by_api_scores(self, style, base_suffix):
        from src.domain.rag.reranker import Reranker
        with FakeModelServer() as srv:
            r = Reranker(base_url=srv.url + base_suffix, model="m", api_key="k", api=style,
                         query_template="query: {text}")
            out = r.rerank("hello", _nodes("first", "second", "third"))
            # the fake scores index 0 highest (1/(i+1)); ordering must follow the index mapping
            assert [n.text for n in out] == ["first", "second", "third"]
            assert out[0].score == 1.0
            body = srv.requests[-1][3]
            assert (body.get("text_1") or body.get("query")) == "query: hello"
            assert srv.requests[-1][2]["Authorization"] == "Bearer k"

    async def test_async_rerank_with_retry(self):
        from src.domain.rag.reranker import Reranker
        with FakeModelServer() as srv:
            srv.fail_next = [(503, "0")]
            r = Reranker(base_url=srv.url, model="m", api="cohere", max_retries=1, score_threshold=0.4)
            out = await r.arerank("q", _nodes("a", "b", "c"))
            assert [n.text for n in out] == ["a", "b"]  # 1/3 < 0.4 filtered
            assert len(srv.posts("/rerank")) == 2
            await r._async_client.aclose()

    def test_failure_falls_back_to_original_order(self):
        from src.domain.rag.reranker import Reranker
        with FakeModelServer() as srv:
            srv.fail_next = [(500, None), (500, None)]
            r = Reranker(base_url=srv.url, model="m", api="generic", max_retries=1)
            nodes = _nodes("x", "y")
            assert r.rerank("q", nodes) is nodes

    def test_from_config_passes_style_and_tuning(self, monkeypatch):
        from src.config.model import RerankConfig
        from src.domain.rag import reranker as mod
        monkeypatch.setattr(mod.Reranker, "verify_connection", lambda self: None)
        r = mod.Reranker.from_config(RerankConfig(enabled=True, api="jina", base_url="https://api.jina.ai",
                                                  timeout=3, max_retries=0, endpoint_path="/v1/rerank"))
        assert r._api.style == "jina" and r._timeout == 3 and r._retry.max_retries == 0

    def test_verify_connection_checks_model_list(self, caplog):
        from src.domain.rag.reranker import Reranker
        with FakeModelServer(models=["other"]) as srv:
            r = Reranker(base_url=srv.url + "/v1", model="m", api="generic")
            r.verify_connection()  # logs an error (model not served); must not raise
            assert srv.requests[-1][1] == "/v1/models"

    def test_verify_connection_skips_hosted_styles(self):
        from src.domain.rag.reranker import Reranker
        r = Reranker(base_url="http://127.0.0.1:9", model="m", api="cohere")
        r.verify_connection()  # no request attempted (would fail fast otherwise)

    def test_verify_connection_unreachable_only_warns(self):
        from src.domain.rag.reranker import Reranker
        Reranker(base_url="http://127.0.0.1:9", model="m").verify_connection()

