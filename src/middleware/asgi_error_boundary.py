"""Outermost ASGI error boundary.

The FastAPI exception handlers are registered on the inner app, but the auth /
request-id middleware wrap outside FastAPI, so an unexpected exception from the
middleware itself bubbles past FastAPI to the ASGI server, leaving the client
with a bare 500 and no consistent JSON format. This wrapper sits at the very
outermost layer as the final catch-all:

- Response not yet started: log the traceback and return a consistent JSON 500
  (aligned with error_handler.generic_exception_handler; no internal details).
- Response already started: unrecoverable, re-raise so the server drops the connection.
- Non-http scope (lifespan / websocket): passed through untouched.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone

from src.log import get_api_logger

logger = get_api_logger()


class ASGIErrorBoundary:
    """Outermost catch-all for the entire ASGI stack (assembled in app.py)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            # lifespan / websocket exceptions must reach the server as-is
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_tracking(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_tracking)
        except Exception as exc:
            path = scope.get("path", "?")
            logger.error(
                f"Unhandled exception escaped middleware stack: {exc}\n"
                f"Path: {path}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            if response_started:
                # Headers already sent: re-raise to let the server drop the connection
                raise
            body = json.dumps({
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": path,
            }).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
