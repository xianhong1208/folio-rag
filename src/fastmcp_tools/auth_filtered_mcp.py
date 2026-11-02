
"""Custom FastMCP subclass implementing per-token tool permission filtering.

FastMCP v3.x list_tools mechanism:
  - public method: `async def list_tools(*, run_middleware: bool = True)`
  - run_middleware=True (default, on a real client call): control passes to the
        middleware chain, and call_next eventually invokes
        `self.list_tools(run_middleware=False)` to fetch the real underlying
        tool list.
  - run_middleware=False: skips middleware and actually fetches tools from the
        provider.

Filtering happens in the run_middleware=False stage — the innermost point that
middleware cannot see, and the last step before FastMCP wraps the list into the
response.

Note: FastMCP v2's `_list_tools` API does not exist in v3, so overriding it has
no effect and every token sees all tools; overriding the public method here is
the correct v3 approach.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import traceback

from cachetools import TTLCache
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request

from db.folderdb import FolderDB
from src.log import get_api_logger, mask_token
from src.auth.dependencies import get_remote_verifier_instance
from src.auth.owner_key import owner_key_from_bearer
from src.fastmcp_tools.dynamic_tool_manager import DynamicToolManager

if TYPE_CHECKING:
    from mcp.types import Tool

api_logger = get_api_logger()

# Cache filtered tool lists per token — every MCP list_tools call previously did
# token-verify + FolderDB.get round-trip. With many MCP clients polling list_tools
# this dominated request latency. TTL matches auth.cache_ttl (60s) — if it gets
# out of sync slightly that's OK, worst case is a stale tool list for <1min.
_FILTER_CACHE_TTL = 60
_FILTER_CACHE_MAX = 1024
_filter_cache: "TTLCache[str, list]" = TTLCache(maxsize=_FILTER_CACHE_MAX, ttl=_FILTER_CACHE_TTL)


class AuthFilteredFastMCP(FastMCP):
    """Extend FastMCP to support per-token tool filtering.

    Each token only sees the MCP tools for folders it is authorized to access.
    """

    def _is_dynamic_folder_tool(self, tool_name: str) -> bool:
        """Determine whether a tool is a dynamically registered per-folder tool.

        Primary: check DynamicToolManager.registered_tools. Fallback: check the
        name prefix (Agentic_).

        Args:
            tool_name: Tool name.

        Returns:
            True = dynamic tool (should go through the ACL filter); False = global tool.
        """
        try:
            manager = DynamicToolManager.get_instance()
            if tool_name in manager.get_registered_tools():
                return True
            # Fallback: the prefix may be inconsistent in some agentic_rag deployments
            return tool_name.startswith("Agentic_")
        except Exception:
            return tool_name.startswith("Agentic_")

    # Correct override point for FastMCP v3.x is the public method `list_tools`
    async def list_tools(self, *, run_middleware: bool = True) -> "Sequence[Tool]":
        """list_tools override — apply the per-token ACL at the inner middleware
        layer (to avoid double filtering).

        FastMCP calls in two layers: run_middleware=True is the client entry
        point, which recurses into a run_middleware=False call. The ACL is
        applied only in the False layer so middleware sees the already-filtered
        result and the filter does not run twice.

        Args:
            run_middleware: True = entry layer (runs the middleware chain);
                False = the actual fetch.

        Returns:
            The filtered tool list (what the token can see).
        """
        # Let FastMCP finish its middleware / transforms / visibility logic first
        tools = await super().list_tools(run_middleware=run_middleware)

        # Apply the ACL only in the innermost fetch stage to avoid filtering twice
        if not run_middleware:
            tools = await self._apply_token_filter(list(tools))

        return tools

    async def _apply_token_filter(self, all_tools: list) -> list:
        """Filter the tool list by the current request's token (with a TTL cache).

        No token -> return only static tools. Valid token -> check the ACL for
        each dynamic tool and inject the folder description. The cache is
        _filter_cache (60s TTL; a DynamicToolManager registration change bumps
        the version and auto-invalidates it).

        Args:
            all_tools: The list of all registered tools.

        Returns:
            The filtered tool list with descriptions injected.
        """
        _remote_verifier = get_remote_verifier_instance()

        try:
            request = get_http_request()
            auth_header = request.headers.get("Authorization", "")

            if not auth_header or not auth_header.startswith("Bearer "):
                api_logger.warning(
                    "list_tools without valid token, hiding all per-folder tools"
                )
                return [t for t in all_tools if not self._is_dynamic_folder_tool(t.name)]

            parts = auth_header.split(" ")
            token = parts[1] if len(parts) > 1 else ""

            # Cache key includes registry version so register/unregister auto-busts
            cache_key = f"{token}:{DynamicToolManager.get_registry_version()}"
            cached = _filter_cache.get(cache_key)
            if cached is not None:
                return cached

            if _remote_verifier is None:
                api_logger.warning(
                    "Remote verifier not initialized, hiding all per-folder tools"
                )
                return [t for t in all_tools if not self._is_dynamic_folder_tool(t.name)]

            result = await _remote_verifier.verify(token)
            if not result.valid:
                api_logger.warning(f"list_tools invalid token: {mask_token(token)}")
                return [t for t in all_tools if not self._is_dynamic_folder_tool(t.name)]

            # Ownership is keyed on the token's owner key (jti), not the raw token —
            # folders and the tool mapping are stored under that key (see acl.py /
            # register_folder_tool), so the lookups below must use it too.
            owner_key = owner_key_from_bearer(token)

            # Determine the per-folder tools this token can see
            try:
                manager = DynamicToolManager.get_instance()
                user_folders = FolderDB.get(user_token=owner_key)
                allowed: set = set()
                if user_folders:
                    for folder in user_folders:
                        tool_name = manager._token_folder_to_tool.get(
                            f"{owner_key}:{folder.name}"
                        )
                        if tool_name:
                            allowed.add(tool_name)
            except Exception as e:
                api_logger.error(f"Error getting allowed tools for token: {e}")
                allowed = set()

            # Filter and inject the per-token folder description
            filtered: list = []
            for tool in all_tools:
                if not self._is_dynamic_folder_tool(tool.name):
                    filtered.append(tool)
                    continue
                if tool.name not in allowed:
                    continue

                # Dynamically inject this token's folder description
                folder_name = manager.get_folder_name_from_tool(tool.name)
                if folder_name:
                    folder_desc = manager.get_folder_description_for_token(folder_name, owner_key)
                    if folder_desc:
                        new_desc = f"{tool.description}\n\nFolder notes: {folder_desc}"
                        tool = tool.model_copy(update={"description": new_desc})
                filtered.append(tool)

            api_logger.info(
                f"list_tools: token {mask_token(token)} sees {len(filtered)}/{len(all_tools)} tools "
                f"({len(allowed)} per-folder)"
            )
            _filter_cache[cache_key] = filtered
            return filtered

        except Exception as e:
            api_logger.error(f"Error in list_tools filtering: {e}")
            traceback.print_exc()
            # Fail-safe: on any exception, hide all dynamic tools
            return [t for t in all_tools if not self._is_dynamic_folder_tool(t.name)]
