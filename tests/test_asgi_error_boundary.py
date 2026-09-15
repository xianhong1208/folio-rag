
"""Outermost ASGI error boundary.

Middleware runs outside FastAPI, so an exception there bubbles past FastAPI's
handlers to the ASGI server as a bare 500. ASGIErrorBoundary is the outermost
catch-all. Contract: normal requests pass through untouched; an exception
before the response starts becomes a 500 with unified JSON; an exception after
the response started re-raises (headers already sent); non-http scopes
(lifespan/websocket) pass through unwrapped.
"""

import pytest
from starlette.testclient import TestClient

from src.middleware.asgi_error_boundary import ASGIErrorBoundary


def _ok_app():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})
    return app


def _boom_before_start():
    async def app(scope, receive, send):
        raise RuntimeError("middleware bug")
    return app


def _boom_after_start():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/plain")]})
        raise RuntimeError("boom mid-stream")
    return app


def test_passthrough_on_success():
    client = TestClient(ASGIErrorBoundary(_ok_app()))
    r = client.get("/anything")
    assert r.status_code == 200
    assert r.text == "ok"


def test_unified_json_500_when_exception_before_response_start():
    client = TestClient(ASGIErrorBoundary(_boom_before_start()), raise_server_exceptions=False)
    r = client.get("/api/some/path")
    assert r.status_code == 500
    body = r.json()
    # Format aligned with error_handler.generic_exception_handler
    assert body["error"] == "INTERNAL_SERVER_ERROR"
    assert "message" in body and "timestamp" in body
    assert body["path"] == "/api/some/path"
    assert "middleware bug" not in r.text, "內部錯誤細節不得外洩"


def test_reraise_when_response_already_started():
    """Headers already sent, unrecoverable -- must re-raise, must not attempt a second response."""
    client = TestClient(ASGIErrorBoundary(_boom_after_start()))
    with pytest.raises(RuntimeError, match="boom mid-stream"):
        client.get("/x")


async def test_non_http_scope_not_wrapped():
    """Non-http scopes like lifespan pass through as-is (exceptions still bubble, not wrapped into an http response)."""
    captured = {}

    async def inner(scope, receive, send):
        captured["scope_type"] = scope["type"]
        raise RuntimeError("lifespan boom")

    boundary = ASGIErrorBoundary(inner)
    with pytest.raises(RuntimeError, match="lifespan boom"):
        await boundary({"type": "lifespan"}, None, None)
    assert captured["scope_type"] == "lifespan"


async def test_non_http_scope_passthrough_on_success():
    """Non-http scope completes normally: passed through as-is, no processing."""
    seen = {}

    async def inner(scope, receive, send):
        seen["type"] = scope["type"]

    await ASGIErrorBoundary(inner)({"type": "lifespan"}, None, None)
    assert seen["type"] == "lifespan"
