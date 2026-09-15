
"""FastAPI authentication dependencies

Provides the authenticate_request dependency used by API routers.
This module is the single source of truth for authentication state,
so both app.py and src/api/router/ import from here.
"""

from typing import Optional

from fastapi import Depends, Request, HTTPException
from fastapi.security import HTTPBearer

from src.log import get_api_logger
from .remote_auth import RemoteTokenVerifier

logger = get_api_logger()
security = HTTPBearer()

_remote_verifier: Optional[RemoteTokenVerifier] = None


def set_remote_verifier(verifier: RemoteTokenVerifier) -> None:
    global _remote_verifier
    _remote_verifier = verifier


def get_remote_verifier_instance() -> Optional[RemoteTokenVerifier]:
    return _remote_verifier


async def authenticate_request_remote(request: Request) -> dict:
    """Authenticate via remote Token Server (supports JWT and Opaque Token)"""
    global _remote_verifier

    if _remote_verifier is None:
        logger.error("Remote verifier not initialized")
        raise HTTPException(
            status_code=500,
            detail={"error": "auth_config_error", "error_description": "Remote authentication not configured"},
        )

    auth_header = request.headers.get('Authorization')

    if not auth_header or not auth_header.startswith('Bearer '):
        logger.warning("Authentication failed: missing Authorization header")
        raise HTTPException(
            status_code=401,
            detail={"error": "missing_token", "error_description": "Authorization header required"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.removeprefix('Bearer ').strip()

    if not token:
        logger.warning("Authentication failed: empty token")
        raise HTTPException(
            status_code=401,
            detail={"error": "missing_token", "error_description": "Bearer token is empty"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await _remote_verifier.verify(token)

    if not result.valid:
        error_msg = result.error or "invalid_token"
        logger.warning(f"Authentication failed (remote): {error_msg} - {token[:8]}...")
        raise HTTPException(
            status_code=401,
            detail={"error": error_msg, "error_description": f"Token validation failed: {error_msg}"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info(f"Token validation succeeded (remote): {token[:8]}... (path: {request.url.path})")
    return {
        "user_name": result.user_name,
        "scopes": result.scopes,
        "expires_at": result.expires_at
    }


async def authenticate_request(request: Request, credentials=Depends(security)):
    """Unified authentication dependency for FastAPI routers"""
    return await authenticate_request_remote(request)
