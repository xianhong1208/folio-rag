"""HTTP request metrics (pure ASGI).

Counts every HTTP request by route *template* (``/api/admin/folders/{folder_id}``, never
the raw path), method and status, and times it until the response starts. Timing to the
first response message keeps long-lived streams (job progress SSE, MCP's streamable HTTP)
from swamping the histogram with their open time.

The template is the route the router matched (``scope["route"]``, set on the shared scope
dict, also inside the mounted FastMCP app), with the prefix of the router it was
included from. A request no route matched is ``unmatched``;
a mount without a closer route is ``<mount path>/*`` (e.g. ``/admin/*`` for the console's
static files). This keeps the label set as small as the route table.
"""

from __future__ import annotations

import time

from starlette.routing import Mount

from src import metrics

UNMATCHED = "unmatched"


def route_template(scope, entry_root_path: str = "") -> str:
    route = scope.get("route")
    if route is None:
        # FastAPI's router records only API routes in scope["route"]; a Mount it matched
        # (the console's static files) shows as the root_path it added.
        root = scope.get("root_path") or ""
        if root != entry_root_path and root.startswith(entry_root_path):
            return f"{root[len(entry_root_path):]}/*"
        return UNMATCHED
    if isinstance(route, Mount):
        return f"{route.path}/*" if route.path else UNMATCHED
    template = getattr(route, "path_format", None) or getattr(route, "path", None)
    if not isinstance(template, str) or not template:
        return UNMATCHED
    return _with_router_prefix(template, scope)


def _with_router_prefix(template: str, scope) -> str:
    """FastAPI records an included router's route without the router's prefix
    (``/query`` for ``/api/rag/query``); put the prefix back from the request path.

    The prefix is the leading path segments the template does not cover. It is used only
    when the remaining segments line up with the template, and never when a prefix
    segment is a path parameter's value, so a label never holds a request value.
    """
    if ":path}" in template:  # a path convertor spans segments: they cannot be aligned
        return template
    parts = (scope.get("path") or "").split("/")
    fmt = template.split("/")
    extra = len(parts) - len(fmt)
    if extra <= 0:
        return template
    for seg, pat in zip(parts[extra + 1:], fmt[1:]):
        if not (pat.startswith("{") and pat.endswith("}")) and seg != pat:
            return template
    prefix = parts[1:extra + 1]
    values = {str(v) for v in (scope.get("path_params") or {}).values()}
    if any(not seg or seg in values for seg in prefix):
        return template
    return "/" + "/".join(prefix) + template


class HttpMetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        entry_root = scope.get("root_path") or ""
        recorded = False

        async def send_wrapper(message):
            nonlocal recorded
            if message.get("type") == "http.response.start" and not recorded:
                recorded = True
                metrics.observe_http(route_template(scope, entry_root), scope.get("method", ""),
                                     message.get("status", 0), time.perf_counter() - started)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            if not recorded:
                recorded = True
                # The error boundary outside turns this into a 500.
                metrics.observe_http(route_template(scope, entry_root), scope.get("method", ""), 500,
                                     time.perf_counter() - started)
            raise
