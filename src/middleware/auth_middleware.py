"""Pure-ASGI authentication middleware.

initialize / notifications/initialized require no authentication; other MCP
methods do. Implemented as pure ASGI (not BaseHTTPMiddleware) to avoid response
buffering so SSE streaming works correctly.
"""

import json as _json

from fastapi import HTTPException
from src.auth.dependencies import authenticate_request_remote


class SelectiveAuthMiddleware:
    """Pure-ASGI authentication middleware."""

    PUBLIC_METHODS = {"initialize", "notifications/initialized"}
    PROTECTED_PATHS = ["/mcp", "/messages", "/sse"]

    def __init__(self, app, config):
        self.app = app
        self.config = config
        auth_config = getattr(config, 'auth', None)
        self.auth_enabled = getattr(auth_config, 'enabled', False) if auth_config else False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        is_protected = any(path.startswith(p) for p in self.PROTECTED_PATHS)

        if not (is_protected and self.auth_enabled):
            await self.app(scope, receive, send)
            return

        # Read and cache the body
        body_chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return

        body = b"".join(body_chunks)

        method = ""
        try:
            if body:
                data = _json.loads(body.decode("utf-8"))
                method = data.get("method", "")
        except (_json.JSONDecodeError, UnicodeDecodeError):
            pass

        if method in self.PUBLIC_METHODS:
            await self.app(scope, self._make_receive(body, receive), send)
            return

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode("utf-8")

        if not auth_header or not auth_header.startswith("Bearer "):
            await self._send_json_error(
                send,
                status_code=401,
                content={"error": "missing_token", "error_description": "Authorization header required"},
                extra_headers=[(b"www-authenticate", b"Bearer")],
            )
            return

        # Reject an empty token early to avoid a pointless remote request
        token = auth_header[len("Bearer "):].strip()
        if not token:
            await self._send_json_error(
                send,
                status_code=401,
                content={"error": "missing_token", "error_description": "Bearer token is empty"},
                extra_headers=[(b"www-authenticate", b"Bearer")],
            )
            return

        from starlette.requests import Request as StarletteRequest
        fake_scope = dict(scope)
        request = StarletteRequest(fake_scope, self._make_receive(body, receive))

        try:
            await authenticate_request_remote(request)
        except HTTPException as exc:
            extra_headers = []
            if exc.headers:
                for k, v in exc.headers.items():
                    extra_headers.append((k.lower().encode(), v.encode()))
            await self._send_json_error(send, exc.status_code, exc.detail, extra_headers)
            return

        await self.app(scope, self._make_receive(body, receive), send)

    def _make_receive(self, body: bytes, original_receive):
        """Build a receive callable that returns the cached body, then proxies the original receive."""
        body_sent = False

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            # After the body is sent, await the original receive so the app can
            # still detect a client disconnect (matters for SSE streaming).
            return await original_receive()

        return receive

    async def _send_json_error(self, send, status_code: int, content: dict, extra_headers: list = None):
        """Send a JSON error response."""
        body = _json.dumps(content).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        if extra_headers:
            headers.extend(extra_headers)

        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": headers,
        })
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        })
