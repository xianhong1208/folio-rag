
"""error_handler — unified error format (domain / generic) and the registration function."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException

from src.domain.exceptions import RAGFileNotFoundError, DomainException
from src.middleware.error_handler import (
    domain_exception_handler,
    generic_exception_handler,
    register_error_handlers,
)


def _req(path="/api/x"):
    r = MagicMock()
    r.url.path = path
    return r


async def test_domain_exception_maps_status_and_body():
    """DomainException maps to the corresponding HTTP status plus a structured body (error_code/message)."""
    exc = RAGFileNotFoundError(file_id="f-123")
    resp = await domain_exception_handler(_req(), exc)
    assert resp.status_code == 404
    body = json.loads(resp.body)
    assert body.get("error") or body.get("error_code")


async def test_generic_exception_returns_500_without_leaking():
    """Unexpected exception returns a 500 generic body without leaking internal details."""
    resp = await generic_exception_handler(_req("/api/y"), RuntimeError("secret detail"))
    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["error"] == "INTERNAL_SERVER_ERROR"
    assert "secret detail" not in resp.body.decode()
    assert body["path"] == "/api/y"


async def test_generic_handler_reraises_http_exception():
    """HTTPException is explicitly re-raised so FastAPI's native handler owns the status (not swallowed into a 500)."""
    with pytest.raises(HTTPException):
        await generic_exception_handler(_req(), HTTPException(status_code=418))


def test_register_error_handlers_wires_both():
    """register_error_handlers wires both the domain and generic handlers onto the app."""
    app = FastAPI()
    register_error_handlers(app)
    assert app.exception_handlers.get(DomainException) is domain_exception_handler
    assert app.exception_handlers.get(Exception) is generic_exception_handler
