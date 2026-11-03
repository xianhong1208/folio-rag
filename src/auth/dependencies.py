
"""FastAPI authentication dependencies

Provides the authenticate_request dependency used by API routers.
This module is the single source of truth for authentication state,
so both app.py and src/api/router/ import from here.
"""

from typing import Optional

from fastapi import Depends, Request, HTTPException
from fastapi.security import HTTPBearer

from src.log import get_api_logger
from .mcp_gate_auth import McpGateTokenVerifier

logger = get_api_logger()
security = HTTPBearer()

_token_verifier: Optional[McpGateTokenVerifier] = None


def set_token_verifier(verifier: McpGateTokenVerifier) -> None:
    global _token_verifier
    _token_verifier = verifier


def get_token_verifier() -> Optional[McpGateTokenVerifier]:
    return _token_verifier


async def authenticate_bearer_token(request: Request) -> dict:
    """Verify the request's bearer token with the registered MCP Gate verifier (401 on failure)."""
    global _token_verifier

    if _token_verifier is None:
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

    result = await _token_verifier.verify(token)

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
    return await authenticate_bearer_token(request)
