
"""Agentic Tools — MCP entry point.

config.modules.rag.mcp_tools.function_name points at this module's
register_agentic_tools. It does two things:
1. Initializes DynamicToolManager (bound to the FastMCP instance).
2. Provides the _query function — injected as the backend when
   dynamic_tool_manager registers a tool.

When config.rag.retrieval.return_resource_files is True and mode='search',
the returned list appends the raw binary File objects of the hit files after
the first JSON text block, so an MCP client (e.g. an agent builder) can turn
them into artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.utilities.types import File
from mcp.types import Annotations

from collections import Counter

from db.filedb import FileDB
from src.adapter.rag import get_rag_adapter
from src.config.config_manager import Config
from src.domain.exceptions import (
    InvalidTokenError,
    UnauthorizedAccessError,
)
from src.fastmcp_tools.acl import extract_token
from src.fastmcp_tools.dynamic_tool_manager import DynamicToolManager
from src.log import get_mcptools_logger, log_err, log_op, log_warn, mask_token
from src.storage.file_storage import FileStorage


# Pretty-print summary box emitted at the end of each MCP tool call, gathering the
# key info (mode/folder/token, params, result stats, attachments, elapsed) in one place.

# Visual constants
_BOX_WIDTH = 90
_BOX_TOP = "═" * _BOX_WIDTH
_BOX_MID = "─" * _BOX_WIDTH
_MODE_ICONS = {"search": "🔍", "list": "📋", "read": "📖"}


def _truncate(s: str, max_len: int) -> str:
    """Truncate an over-long string and append an ellipsis."""
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _visual_width(s: str) -> int:
    """Estimate a string's terminal column width (CJK / fullwidth / emoji = 2, else 1)."""
    import unicodedata
    width = 0
    for ch in s:
        # East Asian Width property: W=Wide, F=Fullwidth -> 2 columns
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        # Emoji typically fall in 0x1F000-0x1FFFF / 0x2600-0x27FF / 0x1F900+
        elif 0x1F000 <= ord(ch) <= 0x1FFFF or 0x2600 <= ord(ch) <= 0x27FF:
            width += 2
        else:
            width += 1
    return width


def _center_line(text: str, width: int = _BOX_WIDTH) -> str:
    """Center text within width (accounting for CJK / emoji width)."""
    vw = _visual_width(text)
    if vw >= width:
        return text
    pad = (width - vw) // 2
    return " " * pad + text


def _format_summary_box(
    *,
    mode: str,
    folder_name: str,
    token: str,
    response: dict,
    attached_files: int,
    attached_bytes: int,
    elapsed_ms: float,
    error: Optional[str] = None,
) -> str:
    """Render the box summary for an entire tool call based on response."""
    icon = _MODE_ICONS.get(mode, "🔧")
    status_icon = "❌" if error else "✅"
    # Center the header content within _BOX_WIDTH (CJK/emoji widths included)
    header_inner = f"{icon}  {mode.upper()}   folder={folder_name}   token={mask_token(token)}"
    header = _center_line(header_inner)

    # Blank lines before and after the box to separate it from surrounding detail logs
    lines = ["", "", _BOX_TOP, header, _BOX_MID]

    # mode-specific body
    if mode == "search":
        lines.extend(_format_search_body(response))
    elif mode == "list":
        lines.extend(_format_list_body(response))
    elif mode == "read":
        lines.extend(_format_read_body(response))

    # attachment info
    if attached_files > 0:
        kb = attached_bytes / 1024
        lines.append(f"  Attach    : {attached_files} files (~{kb:.0f} KB)")

    # error / done
    lines.append(_BOX_MID)
    if error:
        lines.append(f"  {status_icon} FAIL    : {_truncate(error, 70)}   elapsed={elapsed_ms:.0f}ms")
    else:
        blocks_count = 1 + attached_files  # 1 text + N files
        block_breakdown = f"{blocks_count} blocks (1 text + {attached_files} files)" if attached_files else "1 text block"
        lines.append(f"  {status_icon} DONE    : {block_breakdown}   elapsed={elapsed_ms:.0f}ms")
    lines.append(_BOX_TOP)
    lines.append("")

    return "\n".join(lines)


def _format_search_body(response: dict) -> List[str]:
    """search mode body: query / params / results distribution / files / scores / hint."""
    lines = []
    query = response.get("query", "")
    lines.append(f"  Query     : \"{_truncate(query, 70)}\"")

    # Infer params from the response (handle_search does not echo them explicitly,
    # but they are present in the results)
    results = response.get("results", []) or []
    total = response.get("total_results", 0)

    lines.append(f"  Results   : {total}")
    if results:
        # role distribution
        roles = Counter(r.get("node_role", "?") for r in results)
        merged_total = sum(
            r.get("merged_from_leaves", 0) for r in results if r.get("node_role") == "parent"
        )
        role_parts = []
        if roles.get("leaf"):
            role_parts.append(f"leaf:{roles['leaf']}")
        if roles.get("parent"):
            role_parts.append(f"parent:{roles['parent']}(+{merged_total} merged)")
        if roles.get("expanded"):
            role_parts.append(f"expanded:{roles['expanded']}")
        lines.append(f"    Roles   : {'  '.join(role_parts)}")

        # file distribution
        files = Counter(r.get("file_name", "?") for r in results)
        files_str = ", ".join(f"{name}:{cnt}" for name, cnt in files.most_common(3))
        if len(files) > 3:
            files_str += f", +{len(files) - 3} more"
        lines.append(f"    Files   : {_truncate(files_str, 70)}")

        # score distribution
        scores = [r.get("score", 0) for r in results]
        if scores:
            lines.append(
                f"    Scores  : min={min(scores):.3f}  max={max(scores):.3f}  avg={sum(scores)/len(scores):.3f}"
            )

    hint = response.get("_hint", "")
    if hint:
        # _hint can be long; truncate it
        lines.append(f"  Hint      : {_truncate(hint, 70)}")

    return lines


def _format_list_body(response: dict) -> List[str]:
    """list mode body: file count / index status distribution."""
    lines = []
    files = response.get("files", []) or []
    total = response.get("total_files", 0)

    lines.append(f"  Files     : {total}")

    if files:
        status_counts = Counter(f.get("indexed_status", "?") for f in files)
        status_parts = [f"{k}:{v}" for k, v in status_counts.most_common()]
        lines.append(f"    Status  : {'  '.join(status_parts)}")

        total_chunks = sum(f.get("chunk_count", 0) for f in files)
        lines.append(f"    Chunks  : {total_chunks} (total across all files)")

    return lines


def _format_read_body(response: dict) -> List[str]:
    """read mode body: file name / chunks / tokens / truncated."""
    lines = []
    file_name = response.get("file_name", "")
    file_id = response.get("file_id", "")
    chunk_count = response.get("chunk_count", 0)
    tokens = response.get("tokens_estimated", 0)
    truncated = response.get("truncated", False)

    lines.append(f"  File      : {_truncate(file_name, 60)}")
    lines.append(f"  File ID   : {file_id}")
    lines.append(f"  Loaded    : {chunk_count} chunks  ~{tokens} tokens"
                 + ("  ⚠️ TRUNCATED" if truncated else ""))
    return lines

logger = get_mcptools_logger()


# Extension -> standard MIME type mapping (aligned with the coreagent convention).
# Maintained here because Python's mimetypes.guess_type is imprecise for OOXML types
# such as .docx / .xlsx, returning 'application/octet-stream' which the artifacts UI
# cannot recognize. These are the IANA standard values.
_EXT_TO_MIME: Dict[str, str] = {
    "pdf":  "application/pdf",
    "doc":  "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls":  "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv":  "text/csv",
    "ppt":  "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "txt":  "text/plain",
    "md":   "text/markdown",
    "json": "application/json",
    "html": "text/html",
    "png":  "image/png",
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "tif":  "image/tiff",
    "webp": "image/webp",
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
}


class RagFile(File):
    """File subclass: keeps a clean extension in the URI but forces mimeType to the
    IANA standard value.

    FastMCP's File derives both the URI extension and the MIME type from format,
    so format='docx' yields mime='application/docx' (not the standard OOXML mime).
    This subclass keeps the URI's format-derived extension but replaces mimeType
    with the standard value at response time, so client-side artifacts can
    correctly identify the file format.
    """

    def __init__(
        self,
        data: bytes,
        format: str,
        name: str,
        mime_type: Optional[str] = None,
        annotations: Optional[Annotations] = None,
    ):
        super().__init__(data=data, format=format, name=name, annotations=annotations)
        self._standard_mime_type = mime_type

    def to_resource_content(self, mime_type=None, annotations=None):
        resource = super().to_resource_content(mime_type=mime_type, annotations=annotations)
        if self._standard_mime_type and not mime_type:
            resource.resource.mimeType = self._standard_mime_type
        return resource


def _build_resource_files(response: dict) -> List[File]:
    """Extract unique files from a search response's results, read their raw bytes,
    and wrap them as RagFile objects.

    Args:
        response: The dict produced by handle_search; must contain results[],
            each with file_id/file_name.

    Returns:
        A deduplicated list of File objects, sorted by hit count (most-hit files first).
    """
    results = response.get("results", []) if isinstance(response, dict) else []
    if not results:
        return []

    # Count hits per file_id (used for sorting), preserving order of appearance
    hits: Dict[str, int] = {}
    for r in results:
        fid = r.get("file_id")
        if fid:
            hits[fid] = hits.get(fid, 0) + 1

    if not hits:
        return []

    # Most-hit first, so the artifact order the agent sees reflects relevance
    sorted_fids = sorted(hits.keys(), key=lambda x: hits[x], reverse=True)

    file_records = FileDB.get_by_ids(sorted_fids)

    files: List[File] = []
    for fid in sorted_fids:
        record = file_records.get(fid)
        if not record:
            logger.warning(f"File record missing for id={fid}, skipping artifact")
            continue
        try:
            file_bytes = FileStorage.read_file(record.file_path)
            p = Path(record.file_name)
            ext = p.suffix.lstrip('.').lower()
            name_without_ext = p.stem
            standard_mime = _EXT_TO_MIME.get(ext)

            files.append(RagFile(
                data=file_bytes,
                format=ext or "bin",
                name=name_without_ext,
                mime_type=standard_mime,
            ))
            logger.debug(
                f"Attached artifact: {record.file_name} "
                f"({len(file_bytes)} bytes, ext={ext}, hits={hits[fid]})"
            )
        except Exception as e:
            logger.warning(f"Failed to read file {record.file_name} (id={fid}): {e}")

    return files


def register_agentic_tools(mcp: FastMCP) -> None:
    """Module entry point — called by app.create_app after dynamic loading."""
    DynamicToolManager.initialize(mcp)
    logger.info("✅ Agentic tools module initialized")


async def _query(
    *,
    folder_name: str,
    mode: str,
    query: str,
    file_id: Optional[str],
    top_k: int,
    similarity_cutoff: float,
    expand_context: bool,
) -> list:
    """Shared backend query for all per-folder MCP tools.

    Emits a multi-line box summary at the end of each call; detailed logs live in
    the domain layer's agentic_handlers.py.

    Args:
        folder_name: The target RAG folder.
        mode: One of ``search`` / ``list`` / ``read``.
        query: Query string (may be empty for mode=list).
        file_id: Required for mode=read; ignored for other modes.
        top_k: Return the top K results.
        similarity_cutoff: Similarity threshold (ignored when the reranker is enabled).
        expand_context: True -> for mode=search, unmerged leaves expand their neighbors.

    Returns:
        ``list[str | RagFile]`` — the first element is a JSON string (metadata +
        hits); subsequent elements are RagFile objects (only for mode=search with
        return_resource_files enabled in config).
    """
    start = time.perf_counter()
    token: Optional[str] = None
    response: dict = {}
    attached_files = 0
    attached_bytes = 0
    error_msg: Optional[str] = None

    try:
        token = extract_token()
        adapter = get_rag_adapter()
        response = await adapter.query(
            folder_name=folder_name,
            mode=mode,
            query=query,
            file_id=file_id,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
            expand_context=expand_context,
            token=token,
        )

        content_blocks: list = [json.dumps(response, ensure_ascii=False)]

        # Only search mode attaches artifacts (list/read do not need them)
        if mode == "search":
            rag_config = Config.get_config_model().rag
            if rag_config and getattr(rag_config.retrieval, "return_resource_files", False):
                resource_files = _build_resource_files(response)
                content_blocks.extend(resource_files)
                attached_files = len(resource_files)
                attached_bytes = sum(len(f.data) for f in resource_files if hasattr(f, "data"))

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Box summary for this tool call
        logger.info(_format_summary_box(
            mode=mode,
            folder_name=folder_name,
            token=token or "",
            response=response,
            attached_files=attached_files,
            attached_bytes=attached_bytes,
            elapsed_ms=elapsed_ms,
        ))

        return content_blocks

    except InvalidTokenError as exc:
        error_msg = f"AUTH_FAIL: {exc}"
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Still show the box when there is no token, but mark it as an error
        logger.warning(_format_summary_box(
            mode=mode, folder_name=folder_name, token=token or "(missing)",
            response={}, attached_files=0, attached_bytes=0,
            elapsed_ms=elapsed_ms, error=error_msg,
        ))
        return [json.dumps({"error": "unauthorized", "detail": str(exc)}, ensure_ascii=False)]

    except UnauthorizedAccessError as exc:
        error_msg = f"ACL_DENY: {exc}"
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(_format_summary_box(
            mode=mode, folder_name=folder_name, token=token or "",
            response={}, attached_files=0, attached_bytes=0,
            elapsed_ms=elapsed_ms, error=error_msg,
        ))
        return [json.dumps({"error": "forbidden", "detail": str(exc)}, ensure_ascii=False)]

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(_format_summary_box(
            mode=mode, folder_name=folder_name, token=token or "",
            response={}, attached_files=0, attached_bytes=0,
            elapsed_ms=elapsed_ms, error=error_msg,
        ))
        logger.error(f"Tool exception traceback:", exc_info=True)
        return [json.dumps(
            {"error": "internal_error", "detail": f"Query failed: {exc}"},
            ensure_ascii=False,
        )]


def register_folder_tool(folder_name: str, folder_description: Optional[str], user_token: str) -> bool:
    """Public API — called once per (token, folder) at app.py startup."""
    manager = DynamicToolManager.get_instance()
    return manager.register_folder_tool(
        folder_name=folder_name,
        query_function=_query,
        folder_description=folder_description,
        user_token=user_token,
    )


def unregister_folder_tool(folder_name: str, user_token: str) -> None:
    """Public API — called when a folder is deleted."""
    manager = DynamicToolManager.get_instance()
    manager.unregister_folder_tool(folder_name, user_token)
