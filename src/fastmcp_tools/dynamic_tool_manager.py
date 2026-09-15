
"""Dynamic Tool Manager — registers a single tool per folder accessible to a token.

The tool carries a mode parameter (search/list/read) plus expand_context.
"""

from typing import Any, Callable, Dict, Literal, Optional

from fastmcp import FastMCP

from src.config.config_manager import Config
from src.log import get_mcptools_logger, mask_token
from db.folderdb import FolderDB

logger = get_mcptools_logger()


class DynamicToolManager:
    """Singleton that manages per-folder MCP tool registration."""

    _instance: Optional["DynamicToolManager"] = None
    _mcp: Optional[FastMCP] = None
    _registered_tools: Dict[str, Callable] = {}                # tool_name -> func
    _token_folder_to_tool: Dict[str, str] = {}                 # "{token}:{folder}" -> tool_name
    _token_folder_descriptions: Dict[str, str] = {}            # "{token}:{folder}" -> description
    # Monotonic version — incremented on every register/unregister so external
    # caches (e.g., AuthFilteredFastMCP._filter_cache) can detect registry changes
    # without a separate invalidation channel.
    _registry_version: int = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, mcp: FastMCP):
        instance = cls()
        instance._mcp = mcp
        logger.info("DynamicToolManager initialized")
        return instance

    @classmethod
    def get_instance(cls) -> "DynamicToolManager":
        if cls._instance is None or cls._instance._mcp is None:
            raise RuntimeError("DynamicToolManager not initialized. Call initialize(mcp) first.")
        return cls._instance

    def get_registered_tools(self) -> Dict[str, Callable]:
        """Return the registry of dynamically registered per-folder tools.

        Used by AuthFilteredFastMCP to distinguish dynamic (folder-scoped) tools
        from static ones — name-prefix heuristics are unreliable when tool_prefix
        changes via config.
        """
        return self._registered_tools

    @classmethod
    def get_registry_version(cls) -> int:
        """Monotonic counter — bumped on every register/unregister.

        External caches keyed by token alone go stale when folders are added or
        removed for that token. Include this version in the cache key so cache
        entries auto-invalidate without an out-of-band purge call.
        """
        return cls._registry_version

    def register_folder_tool(
        self,
        folder_name: str,
        query_function: Callable,
        folder_description: Optional[str] = None,
        user_token: Optional[str] = None,
    ) -> bool:
        """Register a per-token, per-folder tool.

        Args:
            folder_name: Folder name.
            query_function: Async function with signature (folder_name, mode, query,
                file_id, top_k, expand_context, similarity_cutoff).
            folder_description: Description written into the tool description to
                guide the agent.
            user_token: Token (required) — used for ACL filtering (only this token
                can see the tool).

        Returns:
            True if registered (or already exists for this token).
        """
        if self._mcp is None:
            raise RuntimeError("FastMCP instance not set")
        if not user_token:
            logger.error("user_token is required")
            return False

        # ACL pre-check: confirm the token actually has access to this folder
        user_folders = FolderDB.get(user_token=user_token)
        if not any(f.name == folder_name for f in user_folders):
            logger.warning(
                f"Token {mask_token(user_token)} has no access to folder '{folder_name}', skip"
            )
            return False

        token_folder_key = f"{user_token}:{folder_name}"

        if folder_description:
            self._token_folder_descriptions[token_folder_key] = folder_description

        if token_folder_key in self._token_folder_to_tool:
            return True  # already registered

        # Tool name: take the prefix from config (default Agentic_)
        config = Config.get_config_model()
        prefix = getattr(getattr(config, "app", None), "tool_prefix", "Agentic")
        tool_name = f"{prefix}_{folder_name}"

        # Tool name already registered by another token? Share the same tool function
        if tool_name in self._registered_tools:
            self._token_folder_to_tool[token_folder_key] = tool_name
            logger.info(
                f"Token {mask_token(user_token)} shares existing tool: {tool_name}"
            )
            return True

        # Build the tool description (the interface doc the agent sees)
        config = Config.get_config_model()
        retrieval = config.rag.retrieval
        default_top_k = retrieval.default_top_k
        default_cutoff = retrieval.default_similarity_cutoff
        default_expand = getattr(retrieval, "expand_context_default", True)

        desc_parts = [
            f"**Smart retrieval tool for the {folder_name} folder**",
            "",
        ]
        if folder_description:
            desc_parts.append(f"Folder purpose: {folder_description}")
            desc_parts.append("")

        desc_parts.extend([
            "## Three modes (each has its own use case; do not mix them up)",
            "",
            "### 🔍 `mode=\"search\"` (default, **use this 90% of the time**)",
            "Natural-language retrieval; hits are auto-merged within a section and neighbouring chunks are expanded",
            "to get the full surrounding context.",
            "**When to use**:",
            "- The user asks 'what is XXX?', 'what does the rule say?', 'is YYY mentioned?'",
            "- Any need to find answers, clues or passages in the folder",
            "- You are not sure which file holds the answer",
            "",
            "### 📋 `mode=\"list\"` (list the folder)",
            "Lists every file in this folder with file_id, estimated_tokens and indexed_status.",
            "**When to use**:",
            "- The user asks 'which files are in this folder?', 'show me the index'",
            "- You need a file_id for a later read (rare)",
            "- ❌ Do not call it just to 'look first and decide what to search' — just search",
            "",
            "### 📖 `mode=\"read\"` (load a whole file, **use sparingly**)",
            "Loads the full text of one file (up to ~30000 tokens).",
            "**Use only when**:",
            "- ✅ The user **explicitly asks** for 'the full content', 'the transcript', 'the whole report/document'",
            "- ✅ The user asks to translate or summarise the whole file line by line",
            "- ✅ The file is an **audio transcript** (Whisper output) and the user wants the whole conversation",
            "",
            "**❌ Never use read in these cases**:",
            "- To answer a specific question → use search, not read",
            "- Vague requests like 'tell me more about this file' → use search with a relevant query",
            "- Unhappy with search results → change the query / adjust cutoff, do not jump to read",
            "- Exploratory browsing → use list to see the index, not read on whole files",
            "",
            "**Why so strict**: read puts ~3000-30000 tokens into your context at once,",
            "Most of it is unrelated to the question, wasting token budget and diluting your attention.",
            "search mode already applies auto-merge + context expansion; what it returns is usually enough.",
            "",
            "## Parameters",
            "- `query` (str): required for mode=search; ignored for list/read",
            "- `mode` (str): search / list / read, default search",
            "- `file_id` (str): required for mode=read (take it from list or search results)",
            f"- `top_k` (int): for mode=search, default {default_top_k}",
            f"- `similarity_cutoff` (float): for mode=search, default {default_cutoff}",
            f"- `expand_context` (bool): for mode=search, default {str(default_expand).lower()}",
            "",
            "Response: JSON with mode, results and _hint (suggested next step)",
        ])
        description = "\n".join(desc_parts)

        # Build the tool function
        def create_tool_func(top_k, similarity_cutoff, expand_context):
            async def agentic_tool(
                query: str = "",
                mode: Literal["search", "list", "read"] = "search",
                file_id: Optional[str] = None,
                top_k: int = top_k,
                similarity_cutoff: float = similarity_cutoff,
                expand_context: bool = expand_context,
            ) -> list:
                return await query_function(
                    folder_name=folder_name,
                    mode=mode,
                    query=query,
                    file_id=file_id,
                    top_k=top_k,
                    similarity_cutoff=similarity_cutoff,
                    expand_context=expand_context,
                )
            return agentic_tool

        agentic_tool = create_tool_func(default_top_k, default_cutoff, default_expand)

        agentic_tool.__name__ = tool_name
        agentic_tool.__doc__ = f"Agentic RAG tool for folder '{folder_name}' (search / list / read modes)"
        tool_func = agentic_tool

        self._mcp.tool(
            name=tool_name,
            description=description,
            output_schema=None,
        )(tool_func)

        self._registered_tools[tool_name] = tool_func
        self._token_folder_to_tool[token_folder_key] = tool_name
        DynamicToolManager._registry_version += 1  # bust external caches

        logger.info(
            f"✅ Registered {tool_name} (token={mask_token(user_token)}, folder={folder_name})"
        )
        return True

    def unregister_folder_tool(self, folder_name: str, user_token: str):
        """Remove a user_token's access to a folder's tool.

        If that token is the last user, unregister the tool itself as well.

        Args:
            folder_name: Folder name (corresponding to a registered tool).
            user_token: The token to remove.
        """
        if self._mcp is None:
            raise RuntimeError("FastMCP instance not set")
        if not user_token:
            return

        token_folder_key = f"{user_token}:{folder_name}"
        tool_name = self._token_folder_to_tool.get(token_folder_key)
        if not tool_name:
            return

        self._token_folder_to_tool.pop(token_folder_key, None)
        self._token_folder_descriptions.pop(token_folder_key, None)
        DynamicToolManager._registry_version += 1  # bust external caches

        # Any other tokens still using it?
        still_in_use = any(
            t == tool_name for t in self._token_folder_to_tool.values()
        )
        if not still_in_use:
            try:
                # FastMCP v3: mcp.remove_tool() is deprecated; use local_provider instead
                if hasattr(self._mcp, "local_provider") and hasattr(
                    self._mcp.local_provider, "remove_tool"
                ):
                    self._mcp.local_provider.remove_tool(tool_name)
                else:
                    # Fallback: older FastMCP still uses the top-level API
                    self._mcp.remove_tool(tool_name)
            except Exception as e:
                logger.warning(f"Failed to remove tool {tool_name} from MCP: {e}")
            self._registered_tools.pop(tool_name, None)
            logger.info(f"Tool '{tool_name}' fully removed (no remaining users)")
        else:
            logger.info(f"Token {mask_token(user_token)} removed from '{tool_name}' (others still use it)")

    def get_tools_by_token(self, user_token: str) -> list[str]:
        return [
            tool_name
            for key, tool_name in self._token_folder_to_tool.items()
            if key.startswith(f"{user_token}:")
        ]

    def get_folder_description_for_token(
        self, folder_name: str, user_token: str
    ) -> Optional[str]:
        return self._token_folder_descriptions.get(f"{user_token}:{folder_name}")

    def get_folder_name_from_tool(self, tool_name: str) -> Optional[str]:
        config = Config.get_config_model()
        prefix = getattr(getattr(config, "app", None), "tool_prefix", "Agentic")
        full_prefix = f"{prefix}_"
        if tool_name.startswith(full_prefix):
            return tool_name[len(full_prefix):]
        return None
