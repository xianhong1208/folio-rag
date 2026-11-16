"""``GET /metrics``: Prometheus text exposition (docs/metrics.md).

Registered only when ``metrics.enabled`` is true. With ``metrics.token`` set
(``METRICS_TOKEN``) a scrape must send ``Authorization: Bearer <token>``: no or a wrong
token gets 401 with the same body either way. The token is compared in constant time.

Without a token only a direct loopback client may scrape (403 for anyone else), so an
anonymous request from the network never reads the metrics. A request carrying a
forwarding header (``X-Forwarded-For``, ``Forwarded``, ``X-Real-IP``) came through a
proxy and is not treated as local, even when the proxy itself connects from 127.0.0.1.
"""

from __future__ import annotations

import hmac
import ipaddress
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from src import metrics
from src.log import get_api_logger

logger = get_api_logger()
router = APIRouter(tags=["Health"])


def _configured_token(request: Request) -> Optional[str]:
    conf = getattr(getattr(request.app.state, "config", None), "metrics", None)
    token = getattr(conf, "token", None)
    return token or None


def _presented_token(request: Request) -> Optional[str]:
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


_FORWARDING_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip")


def _is_direct_loopback(request: Request) -> bool:
    """True for a client on this host that did not come through a proxy."""
    if any(request.headers.get(h) for h in _FORWARDING_HEADERS):
        return False
    host = getattr(request.client, "host", None)
    if not host:
        return False
    try:
        # is_loopback also covers IPv4-mapped addresses (::ffff:127.0.0.1)
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:  # e.g. a unix-socket peer or "testclient"
        return False


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request):
    expected = _configured_token(request)
    if expected is not None:
        presented = _presented_token(request) or ""
        if not hmac.compare_digest(presented.encode(), expected.encode()):
            return JSONResponse(
                {"error": "unauthorized", "detail": "GET /metrics needs the metrics bearer token"},
                status_code=401, headers={"WWW-Authenticate": 'Bearer realm="metrics"'})
    elif not _is_direct_loopback(request):
        return JSONResponse(
            {"error": "forbidden",
             "detail": "GET /metrics without METRICS_TOKEN is served to this host only; "
                       "set METRICS_TOKEN to scrape from another host"},
            status_code=403)
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


def include_metrics(app, config) -> bool:
    """Add ``/metrics`` to ``app`` unless ``metrics.enabled`` is false; True when added."""
    conf = getattr(config, "metrics", None)
    if not getattr(conf, "enabled", True):
        logger.info("Metrics disabled (metrics.enabled: false): /metrics is not served")
        return False
    app.include_router(router)
    if getattr(conf, "token", None):
        logger.info("Metrics router loaded: /metrics (bearer token required)")
    else:
        logger.info("Metrics router loaded: /metrics (this host only: set METRICS_TOKEN "
                    "to scrape from another host)")
    return True
