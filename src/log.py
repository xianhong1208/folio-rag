#!/usr/bin/env python3

"""Standalone logging configuration (loguru).

Provides colored console output, per-module file logging, automatic request_id
injection via contextvars, and structured grep-friendly log helpers.
"""

import os
import sys
import json
import uuid
from contextvars import ContextVar
from typing import Dict, Any, Literal, Optional
from loguru import logger
from fastmcp.server.dependencies import get_http_request
from starlette.requests import Request


# Request ID: auto-generated per HTTP request, carried across layers via contextvars.
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

def get_request_id() -> str:
    """Return the current request's request_id (callable from any layer)."""
    return _request_id_var.get()

def set_request_id(rid: str) -> None:
    """Set the current request's request_id (called by middleware)."""
    _request_id_var.set(rid)

def generate_request_id() -> str:
    """Generate a short request_id (first 8 hex chars of a UUID)."""
    return uuid.uuid4().hex[:8]


LogModule = Literal["adapter", "api", "auth", "db", "mcptools", "server"]
LOG_MODULES: list[LogModule] = ["adapter", "api", "auth", "db", "mcptools", "server"]


def _format_file_log(record) -> str:
    """File log format, appending the request_id.

    Reads the contextvar first, falling back to extra["rid"]. Order matters:
    get_logger usually runs at module import when the contextvar is still "-",
    and once bound into extra it is frozen; so at format time the contextvar
    must be read first to get the current request's real value.
    """
    rid = _request_id_var.get()
    if rid == "-":
        rid = record["extra"].get("rid", "-")
    return (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        f"rid={rid} | "
        "{function}:{line} | {message}\n"
    )

def _format_console_log(record) -> str:
    """Console log format: colored + request_id (see _format_file_log for the ordering note).

    The full date is kept because overnight long runs (indexing, transcription,
    cross-day SSE streams) would otherwise blur different days together, and it
    aligns with the file logs for easier grep.
    """
    rid = _request_id_var.get()
    if rid == "-":
        rid = record["extra"].get("rid", "-")
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | "
        f"<dim>rid={rid}</dim> | "
        "<cyan>[{extra[module]}]</cyan> | {message}\n"
    )


def setup_logger(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_base_dir: str = "logs",
    rotation: str = "5 MB",
    encoding: str = "utf-8"
) -> None:
    """Configure the loguru logging system.

    Creates a separate log directory per module, split into info/error files.
    Every log automatically carries a request_id for correlating request chains.
    """
    logger.remove()

    LOG_LEVELS = {
        "info": lambda record: record["level"].name in ["DEBUG", "INFO", "SUCCESS", "WARNING"],
        "error": lambda record: record["level"].name in ["ERROR", "CRITICAL"],
    }

    for module in LOG_MODULES:
        module_log_dir = os.path.join(log_base_dir, module)
        os.makedirs(module_log_dir, exist_ok=True)

        for level_name, level_filter in LOG_LEVELS.items():
            logger.add(
                f"{module_log_dir}/{level_name}_{{time:YYYY-MM-DD}}.log",
                rotation=rotation,
                level=file_level,
                encoding=encoding,
                format=_format_file_log,
                filter=lambda record, m=module, lf=level_filter: (
                    record["extra"].get("module") == m and lf(record)
                )
            )

    # Console: colored output to stderr, kept separate from uvicorn's stdout.
    logger.add(
        sys.stderr,
        level=console_level,
        format=_format_console_log,
        filter=lambda record: record["extra"].get("module") in LOG_MODULES,
        colorize=True
    )


def get_logger(module: LogModule):
    """Get the loguru logger for a module (binds module only, not rid).

    Binding rid would freeze it to "-" at import time; instead the format
    function reads the contextvar live on each call.
    """
    return logger.bind(module=module)


def get_adapter_logger():
    return get_logger("adapter")

def get_api_logger():
    return get_logger("api")

def get_auth_logger():
    return get_logger("auth")

def get_db_logger():
    return get_logger("db")

def get_mcptools_logger():
    return get_logger("mcptools")

def get_server_logger():
    return get_logger("server")


def mask_token(token: Optional[str]) -> str:
    """Mask a token, showing only the first 8 characters."""
    if not token:
        return "-"
    return f"{token[:8]}..." if len(token) > 8 else token


def _build_context(**kwargs) -> str:
    """Join key=value pairs into a log context string, skipping None values."""
    parts = []
    for key, value in kwargs.items():
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def log_op(log, op: str, *, folder_id=None, file_id=None, token=None, msg: str = "", **extra):
    """Log an INFO-level operation event in a uniform, grep-friendly format.

    Output: ``[OP] fid=N file=UUID tok=abcdefgh... | msg``
    """
    ctx = _build_context(
        fid=folder_id,
        file=str(file_id) if file_id else None,
        tok=mask_token(token),
        **extra,
    )
    log.info(f"[{op}] {ctx} | {msg}" if msg else f"[{op}] {ctx}")


def log_op_debug(log, op: str, *, folder_id=None, file_id=None, token=None, msg: str = "", **extra):
    """Like log_op but at DEBUG level, for detailed events not needed day-to-day."""
    ctx = _build_context(
        fid=folder_id,
        file=str(file_id) if file_id else None,
        tok=mask_token(token),
        **extra,
    )
    log.debug(f"[{op}] {ctx} | {msg}" if msg else f"[{op}] {ctx}")


def log_err(log, op: str, error, *, folder_id=None, file_id=None, token=None, **extra):
    """Log an ERROR-level event (like log_op, plus the exception class + message).

    Output: ``[OP] fid=N file=UUID tok=abcdefgh... | ExceptionName: message``
    """
    ctx = _build_context(
        fid=folder_id,
        file=str(file_id) if file_id else None,
        tok=mask_token(token),
        **extra,
    )
    log.error(f"[{op}] {ctx} | {error}")


def log_warn(log, op: str, *, folder_id=None, file_id=None, token=None, msg: str = "", **extra):
    """Log a warning (WARNING level)."""
    ctx = _build_context(
        fid=folder_id,
        file=str(file_id) if file_id else None,
        tok=mask_token(token),
        **extra,
    )
    log.warning(f"[{op}] {ctx} | {msg}" if msg else f"[{op}] {ctx}")


async def log_request_info(
    tool_name: str = "",
    module: LogModule = "mcptools",
    request: Request = None
) -> Dict[str, Any]:
    """Log and return the current HTTP request info.

    Supports two calling styles: FastAPI (pass ``request``) and FastMCP (request
    is obtained automatically from the context).
    """
    module_logger = get_logger(module)

    try:
        mcp_session_id = None

        if request:
            module_logger.info(f"[API_CALL] tool={tool_name}")
        else:
            request = get_http_request()
            mcp_session_id = request.headers.get("mcp-session-id", "N/A")
            module_logger.info(f"[MCP_CALL] tool={tool_name} session={mcp_session_id}")

        json_data = None
        content_type = request.headers.get("content-type", "")

        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                if "application/json" in content_type:
                    json_data = await request.json()
            except Exception as e:
                module_logger.warning(f"[REQ_PARSE] JSON parse failed: {e}")

        request_info = {"json": json_data}
        if mcp_session_id:
            request_info["mcp_session_id"] = mcp_session_id

        if json_data:
            module_logger.debug(f"[REQ_BODY] {json.dumps(json_data, ensure_ascii=False)}")

        return request_info

    except Exception as e:
        module_logger.warning(f"[REQ_INFO] unavailable: {e}")
        return {}
