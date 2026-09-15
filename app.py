"""Agentic RAG MCP Server — FastAPI + FastMCP application assembly.

Exposes MCP RAG queries (no file/folder REST API), wires auth and request-id
middleware, and registers a per-folder MCP tool for each (folder, user_token) pair.
"""

from __future__ import annotations

import importlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import BaseModel

from db.folderdb import FolderDB
from src.api.startup_state import record_load_failure
from src.auth.init_auth import initialize_auth_system
from src.auth.mcp_center_auth import build_auth_provider, settings_from_config
from src.config.config_manager import Config
from src.domain.rag.index_job_manager import IndexingJobManager
from src.fastmcp_tools.agentic_tools import register_folder_tool
from src.fastmcp_tools.auth_filtered_mcp import AuthFilteredFastMCP
from src.log import get_api_logger
from src.middleware.request_id import RequestIdMiddleware
from src.utils.runtime_paths import resolve_external_dir
from src.version import get_version

api_logger = get_api_logger()


def _resolve_instructions_path() -> Path:
    config_dir = resolve_external_dir(
        "config",
        dev_root=Path(__file__).resolve().parent,
    )
    if config_dir is None:
        return Path(__file__).resolve().parent / "config" / "instructions.md"
    return config_dir / "instructions.md"


def _load_instructions() -> str:
    path = _resolve_instructions_path()
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        api_logger.warning(f"Instructions file not found: {path}, using default")
        return "Agentic RAG — Hierarchical RAG with auto-merging retrieval"


def create_app(config: BaseModel, transport: str) -> FastAPI:
    if not isinstance(config, BaseModel):
        raise RuntimeError("create_app requires a Pydantic ConfigModel instance.")

    app_config = getattr(config, "app", None)
    app_name = getattr(app_config, "name", "Agentic RAG")
    app_version = get_version()  # single source of truth: pyproject.toml (see src/version.py)
    app_title = getattr(app_config, "title", "Agentic RAG MCP Server")
    app_description = getattr(app_config, "description", "")
    instructions = _load_instructions()

    auth_config = getattr(config, "auth", None)
    dynamic_tools_config = getattr(auth_config, "dynamic_tools", None) if auth_config else None
    dynamic_tools_enabled = getattr(dynamic_tools_config, "enabled", False) if dynamic_tools_config else False

    # FastMCP auth provider: governs the /mcp transport (offline RS256 JWT verification
    # against MCP Center's JWKS) and publishes /.well-known/oauth-protected-resource/mcp
    # so OAuth-capable MCP clients can discover where to sign in. None when auth disabled.
    auth_provider = build_auth_provider(config)

    if dynamic_tools_enabled:
        mcp = AuthFilteredFastMCP(
            name=app_name, version=app_version, instructions=instructions, auth=auth_provider
        )
        api_logger.info("Using AuthFilteredFastMCP (per-token tool filtering)")
    else:
        mcp = FastMCP(
            name=app_name, version=app_version, instructions=instructions, auth=auth_provider
        )
        api_logger.info("Using standard FastMCP")

    mcp_app = mcp.http_app(transport=transport)

    # Wrap (not replace) MCP's own lifespan: the watchdog needs a running event
    # loop to spawn its background task, and restart cleanup must run at startup.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp_app.lifespan(app):
            manager = IndexingJobManager.get_instance()
            try:
                flipped = manager.mark_all_running_as_restart_failed()
                if flipped:
                    api_logger.info(f"Startup: marked {flipped} stale jobs as failed (server_restart)")
            except Exception as e:
                api_logger.warning(f"Startup restart-cleanup hook failed (continuing): {e}")
            try:
                manager.start_watchdog()
            except Exception as e:
                api_logger.warning(f"Startup watchdog start failed (continuing): {e}")
            # Terminal job rows have no other reclamation path (the TTLCache only
            # governs memory), so purge expired ones once at startup to keep the
            # IndexJobs table from growing without bound.
            try:
                from db.indexjobdb import IndexJobDB
                purged = IndexJobDB.purge_terminal_older_than(days=30)
                if purged:
                    api_logger.info(f"Startup: purged {purged} old terminal IndexJob rows")
            except Exception as e:
                api_logger.warning(f"Startup IndexJob purge failed (continuing): {e}")
            yield

    # openapi_tags sets Swagger tag order and group descriptions.
    openapi_tags = [
        {"name": "Folders",      "description": "Folder CRUD"},
        {"name": "Files",        "description": "File upload / download / list / attribute update / delete"},
        {"name": "Media",        "description": "Media-specific endpoints (transcription / screenshot, etc.)"},
        {"name": "RAG: Indexing","description": "Start background indexing jobs (folder / single file / reindex)"},
        {"name": "RAG: Jobs",    "description": "Query / list / cancel / SSE realtime progress"},
        {"name": "RAG: Status",  "description": "Indexed list + per-file index status"},
        {"name": "RAG: Query",   "description": "Hybrid retrieval + rerank query"},
        {"name": "RAG: Cleanup", "description": "Delete an entire folder index / delete a single file index"},
        {"name": "Health",       "description": "Health check + cache / DB statistics"},
        {"name": "Index",        "description": "Landing page (internal)"},
    ]

    fastapi_app = FastAPI(
        title=app_title,
        description=app_description,
        version=app_version,
        lifespan=lifespan,
        openapi_tags=openapi_tags,
    )
    fastapi_app.state.config = config
    # Stash instructions on app.state so the landing page (/) can render them
    # without re-reading the file. Matches the template's index router contract.
    fastapi_app.state.instructions = instructions
    # Resolved auth settings (issuer/audience/scopes) — consumed by the OAuth discovery
    # router. None when auth is disabled.
    fastapi_app.state.auth_settings = settings_from_config(config)

    # Auth system: build + register the offline MCP Center token verifier (used by the
    # REST middleware and the MCP tool filter via set_remote_verifier).
    initialize_auth_system(config)

    # Wire folder event hooks so the MCP tool surface stays in sync on folder
    # create/update/delete. The adapter only fires hooks (no fastmcp knowledge);
    # attach the MCP-layer register/unregister here. Must run before any folder CRUD.
    from src.adapter.folder_events import set_folder_tool_hooks
    from src.fastmcp_tools.agentic_tools import unregister_folder_tool
    set_folder_tool_hooks(register=register_folder_tool, unregister=unregister_folder_tool)

    modules_conf = getattr(config, "modules", None)
    if modules_conf is None:
        raise RuntimeError("Modules configuration is missing in Pydantic ConfigModel.")
    enabled_modules = modules_conf.enabled or []

    for module_name in enabled_modules:
        module_config = Config.get_module_model(module_name)
        if not module_config:
            api_logger.warning(f"No configuration found for module '{module_name}'")
            continue
        try:
            api_router_config = getattr(module_config, "api_router", None)
            if api_router_config:
                module_path = getattr(api_router_config, "module", None)
                router_name = getattr(api_router_config, "router_name", None)
                prefix = getattr(api_router_config, "prefix", None)
                if module_path and router_name:
                    try:
                        module = importlib.import_module(module_path)
                        router = getattr(module, router_name)
                        fastapi_app.include_router(router, prefix=prefix or "")
                        api_logger.info(f"API router loaded: {module_name} (prefix={prefix})")
                    except (ImportError, AttributeError) as e:
                        api_logger.error(f"Failed to load router {module_path}: {e}")
                        record_load_failure(f"router:{module_name}", e)

            mcp_tools_config = getattr(module_config, "mcp_tools", None)
            if mcp_tools_config:
                module_path = getattr(mcp_tools_config, "module", None)
                function_name = getattr(mcp_tools_config, "function_name", None)
                if module_path and function_name:
                    module = importlib.import_module(module_path)
                    tools_function = getattr(module, function_name)
                    tools_function(mcp)
                    api_logger.info(f"MCP tools loaded: {module_name}")
        except Exception as e:
            api_logger.error(f"Failed to load module '{module_name}': {e}")
            record_load_failure(f"module:{module_name}", e)

    try:
        api_logger.info("Registering per-folder MCP tools for all known tokens...")
        all_folders = FolderDB.get()
        registered = 0
        for folder in all_folders:
            if folder.user_token:
                try:
                    register_folder_tool(folder.name, folder.description, folder.user_token)
                    registered += 1
                except Exception as e:
                    api_logger.warning(f"Failed to register tool for {folder.name}: {e}")
        api_logger.info(f"✅ Registered {registered} folder-specific MCP tools")
    except Exception as e:
        api_logger.warning(f"Per-folder tool registration encountered errors: {e}")

    try:
        from src.api.router.health import router as health_router
        fastapi_app.include_router(health_router)
        api_logger.info("Health check router loaded: /health")
    except Exception as e:
        api_logger.error(f"Failed to load health router: {e}")
        record_load_failure("router:health", e)

    # Serves root-form / legacy well-known endpoints for MCP clients that probe them
    # (FastMCP only publishes the path-specific /.well-known/oauth-protected-resource/mcp).
    if getattr(fastapi_app.state, "auth_settings", None) is not None:
        try:
            from src.api.router.oauth_discovery import router as oauth_discovery_router
            fastapi_app.include_router(oauth_discovery_router)
            api_logger.info("OAuth discovery router loaded: /.well-known/*")
        except Exception as e:
            api_logger.error(f"Failed to load OAuth discovery router: {e}")
            record_load_failure("router:oauth_discovery", e)

    try:
        from src.api.router.index import router as index_router
        fastapi_app.include_router(index_router)
        api_logger.info("Index router loaded: GET /")
    except Exception as e:
        api_logger.error(f"Failed to load index router: {e}")
        record_load_failure("router:index", e)

    try:
        from src.middleware.error_handler import register_error_handlers
        register_error_handlers(fastapi_app)
    except Exception as e:
        api_logger.warning(f"Error handler registration skipped: {e}")

    # Serve the built React console SPA at /admin (index.html + hashed assets), with a
    # client-route fallback so deep links like /admin/folders return index.html. Must be
    # registered before the catch-all FastMCP mount below. Falls back silently to the
    # legacy single-file console at /admin-classic when the SPA has not been built.
    try:
        from fastapi.staticfiles import StaticFiles
        from starlette.exceptions import HTTPException as _StarletteHTTPException
        _console_dir = resolve_external_dir("static/console", dev_root=Path(__file__).resolve().parent)
        if _console_dir is None:
            _console_dir = Path(__file__).resolve().parent / "static" / "console"
        if (_console_dir / "index.html").exists():
            class _SPAStaticFiles(StaticFiles):
                async def get_response(self, path, scope):
                    try:
                        return await super().get_response(path, scope)
                    except _StarletteHTTPException as exc:
                        if exc.status_code == 404:
                            return await super().get_response("index.html", scope)
                        raise
            fastapi_app.mount("/admin", _SPAStaticFiles(directory=str(_console_dir), html=True), name="console")
            api_logger.info(f"React console SPA mounted at /admin (dir={_console_dir})")
        else:
            api_logger.info("React console not built (static/console/index.html missing); /admin-classic serves the legacy console")
    except Exception as e:
        api_logger.warning(f"Console SPA mount skipped: {e}")

    # Mount the FastMCP app LAST. Mounting (not copying its routes) keeps FastMCP's own
    # middleware stack — RequestContext / Authentication / AuthContext — which is where the
    # bearer token is verified and the per-request auth context for tools is set up. Copying
    # only the routes left that middleware behind, so /mcp saw no authenticated user and
    # rejected every request (valid token included) with 401, which also blocked MCP Center's
    # initialize handshake from ever reading the server name / instructions.
    # Starlette matches routes in registration order, so the FastAPI routes included above
    # (/, /health, /docs, /.well-known/* root form) still win; everything else goes to FastMCP.
    fastapi_app.mount("/", mcp_app)

    # ASGIErrorBoundary must be the outermost layer: unexpected exceptions raised by
    # the pure-ASGI wrappers (auth / request_id) do not pass through the inner FastAPI
    # exception handlers. Without this layer the client would get a bare 500 or a
    # dropped connection instead of the unified JSON error response.
    from src.middleware.asgi_error_boundary import ASGIErrorBoundary
    from src.api.router.oauth_discovery import WellKnownCORSMiddleware
    # WellKnownCORSMiddleware adds permissive CORS to /.well-known/* responses so browser
    # -based MCP hosts can read the public discovery documents. It sits inside auth so its
    # OPTIONS short-circuit / header injection is never gated by the bearer check.
    cors_wrapped = WellKnownCORSMiddleware(fastapi_app)
    # /mcp auth is unified on FastMCP's RemoteAuthProvider (see build_auth_provider).
    # The legacy SelectiveAuthMiddleware was removed: it re-buffered the request body and
    # interfered with FastMCP's streamable-HTTP bearer auth, rejecting otherwise-valid tokens.
    final_app = ASGIErrorBoundary(RequestIdMiddleware(app=cors_wrapped))

    auth_enabled = getattr(getattr(config, "auth", None), "enabled", False)
    if auth_enabled:
        api_logger.info("MCP authentication: FastMCP RemoteAuthProvider on /mcp (offline RS256 JWT)")

    return final_app
