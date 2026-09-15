"""FastAPI error handlers that convert domain exceptions into consistent HTTP responses."""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import traceback

from src.domain.exceptions import DomainException, get_http_status_for_exception
from src.log import get_api_logger

logger = get_api_logger()


async def domain_exception_handler(request: Request, exc: DomainException) -> JSONResponse:
    """Convert a domain exception into a JSON error response."""
    status_code = get_http_status_for_exception(exc)

    error_response = {
        "error": exc.error_code,
        "message": exc.message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path": str(request.url.path)
    }

    if exc.details:
        error_response["details"] = exc.details

    log_message = f"[{exc.error_code}] {exc.message}"
    if status_code >= 500:
        logger.error(f"{log_message} | path={request.url.path} | details={exc.details}")
    elif status_code >= 400:
        logger.warning(f"{log_message} | path={request.url.path}")

    return JSONResponse(status_code=status_code, content=error_response)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for non-DomainException errors: log the traceback and return a generic 500.

    HTTPException is re-raised so FastAPI's native handler sets the status code;
    otherwise this catch-all would swallow it into a 500.
    """
    if isinstance(exc, HTTPException):
        raise exc

    error_traceback = traceback.format_exc()
    logger.error(
        f"Unhandled exception: {str(exc)}\n"
        f"Path: {request.url.path}\n"
        f"Traceback:\n{error_traceback}"
    )

    # Generic response, no internal details exposed
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path)
        }
    )


def register_error_handlers(app) -> None:
    """Register the domain and generic exception handlers on the FastAPI app."""
    app.add_exception_handler(DomainException, domain_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Error handlers registered successfully")
