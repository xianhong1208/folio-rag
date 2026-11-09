"""Admin-console session endpoints (public: they are how you sign in).

- POST   /api/console/session  {token}  — verify an MCP Gate access token and set the
  HttpOnly session cookie. The token itself is never stored or echoed back.
- GET    /api/console/session           — who am I (from the cookie or a bearer PAT)
- DELETE /api/console/session           — sign out (clear the cookie)

See src/auth/console_session.py for the session format and guards.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, Request, Response

from src.auth.console_session import (
    COOKIE_NAME,
    check_csrf,
    claim_legacy,
    get_console_auth,
    resolve_identity,
)
from src.log import get_api_logger

logger = get_api_logger()

router = APIRouter(prefix="/api/console/session", tags=["Admin: Console Session"])


@router.post("")
async def create_session(
    request: Request,
    response: Response,
    body: Dict[str, Any] = Body(..., examples=[{"token": "<MCP Gate access token>"}]),
):
    """Sign in with an MCP Gate access token issued for this service."""
    auth = get_console_auth()
    if not auth.enabled:
        return {"data": auth.open_identity().public_view(),
                "message": "Authentication is disabled; the console is open"}
    # Login CSRF: only this console may create a session in the browser.
    check_csrf(request)
    ident = await auth.verify_pat(str(body.get("token") or ""))
    # Folders this token still owns under its pre-1.5.0 per-token key move to the person now.
    await claim_legacy(ident)
    value, exp = auth.issue(ident)
    auth.set_cookie(response, request, value, exp)
    session = auth.read_session(value)
    logger.info(f"Console sign-in: actor={session.actor} admin={session.is_admin}")
    return {"data": session.public_view(), "message": "Signed in"}


@router.get("")
async def get_session(request: Request):
    """The current console identity: {mode, via, jti_prefix, sub, scopes, is_admin, expires_at}."""
    ident = await resolve_identity(request)
    return {"data": ident.public_view(), "message": "ok"}


@router.delete("")
async def delete_session(request: Request, response: Response):
    """Sign out: the session ends on the server too, so a copy of the cookie stops
    working. Idempotent; ending your own session carries no authority, and the cookie
    is SameSite=Strict, so it needs no CSRF check."""
    auth = get_console_auth()
    if auth.enabled:
        cookie = request.cookies.get(COOKIE_NAME)
        if cookie:
            auth.end_session(cookie)
        auth.clear_cookie(response, request)
    return {"data": {"signed_out": True}, "message": "Signed out"}
