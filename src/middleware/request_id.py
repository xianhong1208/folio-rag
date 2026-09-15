"""Request ID middleware.

Generates a unique request_id per HTTP request and injects it into contextvars,
so every subsequent logger call (at any layer) carries it for tracing a whole
request chain.
"""

from src.log import set_request_id, generate_request_id


class RequestIdMiddleware:
    """Generate a unique request_id for each HTTP request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            rid = generate_request_id()
            set_request_id(rid)
        await self.app(scope, receive, send)
