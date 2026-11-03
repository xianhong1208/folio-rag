
"""Dynamic Tool Manager — registers a single tool per folder accessible to a token.

The tool carries a mode parameter (search/list/read) plus expand_context.
"""

import hashlib
import re
import unicodedata
from typing import Callable, Dict, Literal, Optional

from fastmcp import FastMCP

from src.config.config_manager import Config
from src.log import get_mcptools_logger, mask_token
from db.folderdb import FolderDB

logger = get_mcptools_logger()

# MCP clients (Claude, OpenAI) accept tool names matching ^[a-zA-Z0-9_-]{1,64}$ only;
# one invalid name can make a client reject the whole tool list.
_TOOL_NAME_MAX = 64
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _slug(text: str) -> str:
    """ASCII-only, underscore-joined form (accents stripped; CJK and symbols dropped)."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return _UNSAFE.sub("_", ascii_text).strip("_")


DEFAULT_TOOL_PREFIX = "Folio"


def configured_tool_prefix() -> str:
    """The client-safe tool prefix from ``app.tool_prefix`` (default ``Folio``)."""
    try:
        config = Config.get_config_model()
        prefix = getattr(getattr(config, "app", None), "tool_prefix", None)
    except Exception:
        prefix = None
    return _slug(prefix or "") or DEFAULT_TOOL_PREFIX


def tool_name_for(prefix: str, folder_name: str) -> str:
    """Deterministic, client-safe MCP tool name for a folder.

    ``<prefix>_<folder>`` when the folder name is already safe and short enough;
    otherwise ``<prefix>_<readable ascii part>_<8-hex sha1 of the real name>``. The hash
    keeps names distinct when the readable part collides (CJK names, "a b" vs "a_b",
    long names that only differ after the cut) and the name stable across restarts,
    so tokens with a same-named folder keep sharing one tool.
    """
    pre = _slug(prefix or "") or DEFAULT_TOOL_PREFIX
    plain = f"{pre}_{folder_name}"
    if _UNSAFE.search(folder_name) is None and folder_name and len(plain) <= _TOOL_NAME_MAX:
        return plain
    digest = hashlib.sha1(folder_name.encode("utf-8")).hexdigest()[:8]
    room = _TOOL_NAME_MAX - len(pre) - len(digest) - 2  # two underscores
    readable = _slug(folder_name)[:max(room, 0)].rstrip("_")
    name = f"{pre}_{readable}_{digest}" if readable else f"{pre}_{digest}"
    return name[:_TOOL_NAME_MAX]


class DynamicToolManager:
    """Singleton that manages per-folder MCP tool registration."""

    _instance: Optional["DynamicToolManager"] = None
    _mcp: Optional[FastMCP] = None
    _registered_tools: Dict[str, Callable] = {}                # tool_name -> func
    _token_folder_to_tool: Dict[str, str] = {}                 # "{token}:{folder}" -> tool_name
    _token_folder_descriptions: Dict[str, str] = {}            # "{token}:{folder}" -> description
    _tool_to_folder: Dict[str, str] = {}                       # tool_name -> folder name
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

        # Tool name: config prefix (default Folio) + a client-safe form of the folder name
        tool_name = tool_name_for(configured_tool_prefix(), folder_name)

        # Tool name already registered by another token? Share the same tool function
        if tool_name in self._registered_tools:
            self._token_folder_to_tool[token_folder_key] = tool_name
            self._tool_to_folder[tool_name] = folder_name
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
            f"**Search tool for the '{folder_name}' folder**",
            "",
        ]
        if folder_description:
            desc_parts.append(f"Folder purpose: {folder_description}")
            desc_parts.append("")

        desc_parts.extend([
            "## Three modes (each has its own use; do not mix them up)",
            "",
            "### `mode=\"search\"` (default; the right choice almost every time)",
            "Natural-language retrieval. Hits are auto-merged with sibling chunks of the same",
            "section and expanded with neighbouring chunks for full context.",
            "**Use when**:",
            "- the user asks what something is, what a rule says, or whether X is mentioned",
            "- you need an answer, a lead or a passage from this folder",
            "- you do not know which file holds the answer",
            "",
            "### `mode=\"list\"` (table of contents)",
            "Lists every file with its file_id, estimated_tokens and indexed_status.",
            "**Use when**:",
            "- the user asks which files the folder contains",
            "- you need a file_id for a follow-up read (rare)",
            "- do NOT call it just to look around before searching — search directly",
            "",
            "### `mode=\"read\"` (whole file; use sparingly)",
            "Loads the full text of one file (up to ~30000 tokens).",
            "**Only when**:",
            "- the user explicitly asks for the complete content, a transcript or the whole document",
            "- the user wants the whole file translated or summarised line by line",
            "- the file is an audio transcript and the user wants the full conversation",
            "",
            "**Never use read**:",
            "- to answer a specific question — use search",
            "- for vague \"tell me more about this file\" requests — search for what matters",
            "- because search results were weak — rephrase the query or lower the cutoff",
            "- to browse — use list",
            "",
            "**Why**: one read puts ~3000-30000 tokens into your context, mostly unrelated to",
            "the question. search already merges and expands context and is usually enough.",
            "",
            "## Parameters",
            "- `query` (str): required for mode=search; ignored by list/read",
            "- `mode` (str): search / list / read, default search",
            "- `file_id` (str): required for mode=read (from list or search results)",
            f"- `top_k` (int): mode=search, default {default_top_k}",
            f"- `similarity_cutoff` (float): mode=search, default {default_cutoff}",
            f"- `expand_context` (bool): mode=search, default {str(default_expand).lower()}",
            "",
            "Response: JSON with mode, results and _hint (suggested next step)",
        ])
        description = "\n".join(desc_parts)

        # Build the tool function
        def create_tool_func(top_k, similarity_cutoff, expand_context):
            async def folder_tool(
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
            return folder_tool

        tool_func = create_tool_func(default_top_k, default_cutoff, default_expand)
        tool_func.__name__ = tool_name
        tool_func.__doc__ = f"Folio tool for folder '{folder_name}' (search / list / read modes)"

        self._mcp.tool(
            name=tool_name,
            description=description,
            output_schema=None,
        )(tool_func)

        self._registered_tools[tool_name] = tool_func
        self._token_folder_to_tool[token_folder_key] = tool_name
        self._tool_to_folder[tool_name] = folder_name
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
            self._tool_to_folder.pop(tool_name, None)
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
        """Folder name behind a registered tool (tool names are slugged, so this is a lookup)."""
        return self._tool_to_folder.get(tool_name)
