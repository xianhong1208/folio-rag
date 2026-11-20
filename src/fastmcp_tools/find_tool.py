"""The person-wide MCP tool ``folio_find``: which documents, across all the caller's folders.

One static tool (not one per folder): every caller sees it, and what it returns is
resolved from the caller's owner key (the token's ``sub``) on every call — the
arguments can narrow the scope, never widen it. The logic is in
src/domain/rag/doc_find.py; this module only resolves the caller, wires the
dependencies and turns refusals into ``{"error": ...}`` blocks like the folder tools.

Open mode (``auth.enabled: false``) behaves like the REST endpoints: every folder,
no token needed. A token without a ``sub`` is refused (``unauthorized``); a token
whose ``sub`` owns nothing (the MCP Gate scanner, a client_credentials client) gets
an empty result.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from mcp.types import TextContent
from pydantic import Field

from src import metrics
from src.domain.exceptions import InvalidTokenError
from src.domain.rag import doc_filter as DF
from src.domain.rag import doc_find as FIND
from src.fastmcp_tools.dynamic_tool_manager import configured_tool_prefix, tool_name_for
from src.log import get_mcptools_logger, mask_token

logger = get_mcptools_logger()

FIND_TOOL_NAME = "folio_find"

FIND_DESCRIPTION = "\n".join([
    "**Find documents across ALL of the user's folders** (one call), or list them by tags,",
    "metadata, type and upload date.",
    "",
    "Use it first when you do not know which folder or document holds the answer, or when",
    "the user asks for documents (\"my contracts tagged signed from 2024\", \"which files",
    "mention the warranty?\"). Then search INSIDE the documents found: call the folder tool",
    "named in each result's `tool` with mode=\"search\" and `file_ids` set to their `file_id`s.",
    "To answer a question about one known folder, use that folder's tool directly.",
    "",
    "## Two ways to call it",
    "- with `query`: documents ranked by their best matching passage, each with up to 2",
    "  short snippets, `hits` (matching passages) and `score`. `skipped_folders` lists",
    "  folders that could not be searched (nothing indexed, another embedding model, too",
    "  many folders) — say so if it matters.",
    "- without `query`: a listing of the documents matching `filters`, newest upload first",
    "  (`total`, `next_offset` for the next page via `offset`). Called with nothing at all",
    "  it also returns `vocabulary`: the user's tags and metadata keys.",
    "",
    "## Parameters",
    "- `query` (str, optional): what the documents should be about",
    "- `filters` (object, optional), every condition must hold:",
    "  `tags_any` [..] (at least one), `tags_all` [..], `folders` [names or ids],",
    "  `file_type` [\"pdf\", \".docx\", \"image/*\"], `uploaded_after` / `uploaded_before`",
    "  (YYYY-MM-DD; after inclusive, before exclusive), `name_contains` \"q3\",",
    "  `metadata` {key: value | {\"in\": [..]} | {\"gte\": x} | {\"lte\": x} | {\"exists\": true}}",
    "  (gte/lte take numbers or YYYY-MM-DD dates). Tags match case-insensitively; metadata",
    "  string values match exactly. Unknown keys are refused with their name.",
    "- `folders` (list, optional): only these folders (names or ids)",
    "- `limit` (int, 1-50, default 10): documents to return",
    "- `offset` (int): listing only, the next page's start (`next_offset`)",
    "",
    "Response: JSON with mode (find / list), documents [{file_id, name, folder, tool, tags,",
    "metadata, score, hits, snippets}], _hint (suggested next step). Large answers are cut",
    "to a token budget (omitted_results).",
])


def _json_block(payload: dict) -> TextContent:
    """The JSON payload as one MCP text block (as the folder tools return it)."""
    return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))


def tool_for_folder(folder_name: str) -> str:
    """The folder tool's name, as registered (src/fastmcp_tools/dynamic_tool_manager.py)."""
    return tool_name_for(configured_tool_prefix(), folder_name)


def auth_enabled() -> bool:
    """``auth.enabled`` (live); True when unreadable (fail closed: a token is needed)."""
    try:
        from src.config.config_manager import Config

        auth = getattr(Config.get_config_model(), "auth", None)
        return bool(getattr(auth, "enabled", True)) if auth is not None else True
    except Exception:  # noqa: BLE001
        return True


async def resolve_owner() -> DF.Owner:
    """The caller's owner key; ALL_FOLDERS in open mode. InvalidTokenError (incl.
    MissingSubjectError) when the token is missing, malformed or has no ``sub``."""
    if not auth_enabled():
        return DF.ALL_FOLDERS
    from src.fastmcp_tools.acl import extract_owner_key

    return await extract_owner_key()


def _connect():
    from db.db import get_engine
    return get_engine().connect()


async def _search_across(req, *, owner):
    from src.adapter.rag import get_rag_adapter
    return await get_rag_adapter().search_across(req, owner=owner)


async def folio_find(
    query: Annotated[Optional[str], Field(
        description="What the documents should be about; omit to list documents by filters")] = None,
    filters: Annotated[Optional[Dict[str, Any]], Field(
        description="Document filter: tags_any, tags_all, folders, file_type, uploaded_after, "
                    "uploaded_before, name_contains, metadata (see the tool description)")] = None,
    folders: Annotated[Optional[List[Union[str, int]]], Field(
        description="Only these folders (names or ids); omit for all your folders")] = None,
    limit: Annotated[Optional[int], Field(
        description=f"Documents to return, 1-{FIND.MAX_LIMIT} (default {FIND.DEFAULT_LIMIT})")] = None,
    offset: Annotated[Optional[int], Field(
        description="Listing (no query) only: start of the page, from next_offset")] = None,
) -> list:
    """Find documents across the caller's folders (see FIND_DESCRIPTION)."""
    from src.domain.rag.agentic_handlers import configured_response_budget

    started = time.perf_counter()
    mode = "find" if (query or "").strip() else "find_list"
    owner_label = "(none)"
    try:
        owner = await resolve_owner()
        owner_label = owner if isinstance(owner, str) else "(all folders)"
        response = await FIND.find(
            owner=owner, query=query, filters=filters, folders=folders, limit=limit,
            offset=offset, search_across=_search_across, connect=_connect,
            tool_of=tool_for_folder, budget=configured_response_budget(),
            run_sync=asyncio.to_thread)
        outcome = "ok"
    except InvalidTokenError as e:
        response, outcome = {"error": "unauthorized", "detail": str(e)}, "unauthorized"
    except FIND.FindError as e:
        response = e.payload()
        outcome = "forbidden" if e.code == "folder_not_found" else "invalid_request"
    except Exception as e:  # noqa: BLE001
        from src.domain.rag.query_limits import QUERY_TOO_LONG_DETAIL, is_input_too_long_error
        if is_input_too_long_error(e):
            response, outcome = {"error": "invalid_request", "detail": QUERY_TOO_LONG_DETAIL}, \
                "invalid_request"
        else:
            logger.error(f"[FIND_FAIL] {type(e).__name__}: {e}", exc_info=True)
            response, outcome = {"error": "internal_error", "detail": f"Find failed: {e}"}, "error"
    elapsed = time.perf_counter() - started
    metrics.observe_mcp_call(mode, outcome, elapsed)
    logger.info(
        f"[FIND_DONE] owner={mask_token(owner_label)} mode={mode} outcome={outcome} "
        f"documents={response.get('total_results', 0)} "
        f"omitted={response.get('omitted_results', 0)} "
        f"tokens~{response.get('response_tokens_estimated')} elapsed={elapsed * 1000:.0f}ms")
    return [_json_block(response)]


def register_find_tool(mcp: FastMCP) -> None:
    """Add ``folio_find`` to the server (static: visible to every caller)."""
    from fastmcp.tools import Tool

    tool = Tool.from_function(folio_find, name=FIND_TOOL_NAME, description=FIND_DESCRIPTION,
                              output_schema=None)
    mcp.add_tool(tool)
    logger.info(f"Registered {FIND_TOOL_NAME}")


def find_description_for(owner: DF.Owner) -> Optional[str]:
    """FIND_DESCRIPTION plus the owner's tags and metadata keys (blocking DB read);
    None when they have none."""
    with _connect() as conn:
        vocab = FIND.vocabulary(conn, owner=owner)
    text = FIND.vocabulary_text(vocab)
    return f"{FIND_DESCRIPTION}\n\n{text}" if text else None
