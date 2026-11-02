
"""Admin settings API (runtime hot-reconfiguration) — the admin module's root router.

- GET  /api/admin/settings        — whitelisted settings view (effective value/level/override status)
- PUT  /api/admin/settings        — hot update ({"settings": {path: value}}); all-or-nothing validation
- DELETE /api/admin/settings/{path} — remove one override, restoring the config.yaml value
- POST /api/admin/probe           — test model-service connectivity (GET {base_url}/models)
- GET  /                          — redirect to the React console at /admin/

Auth: reading settings needs a console session (secrets are always masked); changing
them, and probing arbitrary URLs, needs the admin scope (``rag:admin``). Changes are
audited under the session's ``sub``. See src/auth/console_session.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import RedirectResponse

from src.auth.console_session import (
    ConsoleIdentity,
    require_console_admin,
    require_console_session,
)
from src.log import get_api_logger

logger = get_api_logger()

router = APIRouter(tags=["Admin: Runtime Settings"])

# Control Center data plane (read-only aggregates) and management plane (CRUD/index ops) mount on the same module
from src.api.router.admin_overview import router as _overview_router  # noqa: E402
from src.api.router.admin_manage import router as _manage_router  # noqa: E402
from src.api.router.console_session import router as _session_router  # noqa: E402
router.include_router(_overview_router)
router.include_router(_manage_router)
router.include_router(_session_router)


@router.get("/api/admin/settings", dependencies=[Depends(require_console_session)])
async def get_settings():
    """The currently effective hot-reconfigurable settings (full whitelist; secrets masked)."""
    from src.adapter import runtime_settings_service as svc
    return {"data": svc.get_settings_view(), "message": "ok"}


@router.put("/api/admin/settings")
async def put_settings(
    body: Dict[str, Any] = Body(..., examples=[{"settings": {"rag.rerank.score_threshold": 0.3}}]),
    ident: ConsoleIdentity = Depends(require_console_admin),
):
    """Apply a batch of settings (written only if all validation passes; takes effect and persists immediately, surviving restarts)."""
    patch = body.get("settings") or {}
    if not isinstance(patch, dict) or not patch:
        raise HTTPException(422, "body must be {\"settings\": {\"<path>\": <value>, ...}}")
    from src.adapter import runtime_settings_service as svc
    try:
        applied, warnings = svc.apply_settings(patch, updated_by=ident.actor)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {
        "data": {"applied": applied, "warnings": warnings},
        "message": f"Applied {len(applied)} setting(s)" + (" (with warnings)" if warnings else ""),
    }


@router.delete("/api/admin/settings/{path:path}")
async def delete_setting(path: str, ident: ConsoleIdentity = Depends(require_console_admin)):
    """Remove one override -> restore the config.yaml factory value (takes effect immediately)."""
    from src.adapter import runtime_settings_service as svc
    try:
        removed = svc.reset_setting(path, updated_by=ident.actor)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not removed:
        raise HTTPException(404, f"{path} has no runtime override")
    return {"data": {"reset": path}, "message": f"{path} restored to factory value"}


@router.post("/api/admin/probe", dependencies=[Depends(require_console_admin)])
async def probe_endpoint(
    body: Dict[str, Any] = Body(..., examples=[{"base_url": "http://localhost:7075/v1", "api_key": "..."}]),
):
    """Test model-service connectivity: GET {base_url}/models (OpenAI-compatible; works for vLLM/rerank services).

    Reports only reachability and the model list — not a guarantee that inference will succeed, but
    it catches wrong URLs / services that aren't up.

    SSRF note: this endpoint's purpose is to probe internal model services (localhost vLLM, etc.),
    so blocking loopback/private ranges would break the feature; no IP-range filtering is applied.
    Mitigations in place: (1) fixed GET /models, never echoing the raw body (only the model id is
    extracted); (2) no redirect following; (3) errors report only the exception class name, never
    the raw message. Residual risk = HTTP GET reachability probing of the internal network,
    which is why the endpoint is admin-only (``rag:admin``).
    """
    base_url = (body.get("base_url") or "").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise HTTPException(422, "base_url must be an http(s) URL")
    headers = {}
    if body.get("api_key"):
        headers["Authorization"] = f"Bearer {body['api_key']}"
    try:
        async with httpx.AsyncClient(
            timeout=5.0, headers=headers, follow_redirects=False,
        ) as client:
            resp = await client.get(f"{base_url}/models")
        ok = resp.status_code < 500
        models = []
        try:
            models = [m.get("id") for m in resp.json().get("data", [])][:10]
        except Exception:
            pass
        return {"data": {"reachable": ok, "status_code": resp.status_code,
                         "models": models}, "message": "probe done"}
    except Exception as e:
        # Return only the exception class name (ConnectError / ConnectTimeout…) — enough to point
        # at the problem without echoing the raw message (which may contain internal address/path details)
        return {"data": {"reachable": False, "error": type(e).__name__},
                "message": "probe failed"}


@router.get("/", include_in_schema=False)
async def root_redirect():
    """Front door: send the root URL to the React console at /admin/.

    The module router registers before the index router, so this "/" overrides the
    original API-docs landing page (now at /docs). 302 (not 301/308) keeps browsers
    from hard-caching the redirect while the console front door is still settling.
    """
    return RedirectResponse("/admin/", status_code=302)
