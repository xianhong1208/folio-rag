"""Module registry: how each module is wired into the app.

``modules.enabled`` in config/mcp.yaml says which modules run; this table says what
each one loads (a REST router and/or an MCP tool registration function). The wiring
lived in config/config.yaml up to Folio 1.4; it is code, not configuration, so it lives here.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class RouterConfig(BaseModel):
    """API router wiring."""
    module: str
    router_name: str
    prefix: str


class McpToolsConfig(BaseModel):
    """MCP tools wiring."""
    module: str
    function_name: str


class ModuleConfig(BaseModel):
    """How one module is wired into the app."""
    api_router: Optional[RouterConfig]
    mcp_tools: Optional[McpToolsConfig]
    description: str


def _router(module: str, prefix: str) -> RouterConfig:
    return RouterConfig(module=module, router_name="router", prefix=prefix)


MODULES: Dict[str, ModuleConfig] = {
    "folder": ModuleConfig(
        api_router=_router("src.api.router.folder_api", "/api"), mcp_tools=None,
        description="Folder CRUD REST API"),
    "file": ModuleConfig(
        api_router=_router("src.api.router.file_api", "/api"), mcp_tools=None,
        description="File upload/download/management REST API"),
    "rag_indexing": ModuleConfig(
        api_router=_router("src.api.router.rag_indexing", "/api/rag"), mcp_tools=None,
        description="RAG indexing: index / reindex / delete-index / job status"),
    "rag_query": ModuleConfig(
        api_router=_router("src.api.router.rag_query", "/api/rag"), mcp_tools=None,
        description="RAG query: hybrid retrieval REST endpoint"),
    "rag": ModuleConfig(
        api_router=None,
        mcp_tools=McpToolsConfig(module="src.fastmcp_tools.agentic_tools",
                                 function_name="register_agentic_tools"),
        description="Folio MCP tools (per-folder Folio_{name} with auto-merging)"),
    # The router carries its own /api/admin and /admin paths.
    "admin": ModuleConfig(
        api_router=_router("src.api.router.admin_settings", ""), mcp_tools=None,
        description="Console and runtime settings (/admin UI and /api/admin)"),
    # The router already prefixes /v1 itself.
    "media": ModuleConfig(
        api_router=_router("src.api.router.media", ""), mcp_tools=None,
        description="OpenAI-style speech-to-text / OCR REST API (/v1)"),
}

# What config/mcp.yaml enables out of the box.
DEFAULT_ENABLED = ("folder", "file", "rag_indexing", "rag_query", "rag", "media", "admin")
