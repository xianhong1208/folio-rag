
"""Agentic three-mode query handlers — the actual search / list / read logic (domain layer).

These handlers live in the domain layer because they depend only on db / domain / log and carry no
fastmcp dependency; tool_handlers.py remains as a re-export shim for backward compatibility. The
logger keeps using the mcptools sink, so logging behavior is unchanged.

Mode design:
- search: default mode; hybrid retrieval + auto-merge + optional expand_context
- list:   metadata-only query, no retrieval; returns file_id + estimated_tokens
- read:   given a file_id, concatenates all leaf chunks in order and returns the whole document

Each handler receives an already-validated token + folder and performs no ACL itself (folder_acl.py
handles that). Returns a JSON-serializable dict, which the caller wraps into an MCP content block.

Each mode logs [<MODE>_START] on entry and [<MODE>_DONE] with result stats on completion, to aid
RAG-quality debugging.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import List

from db.filedb import FileDB
from db.fileindexdb import FileIndexDB
from src.domain.rag.auto_merging import AutoMergingRetriever, MergedResult
from src.domain.rag.chunk_lookup import ChunkLookup
from src.log import get_mcptools_logger, log_err, log_op_debug, log_warn

logger = get_mcptools_logger()


async def handle_search(
    *,
    query: str,
    folder,
    retriever: AutoMergingRetriever,
    top_k: int,
    similarity_cutoff: float,
    expand_context: bool,
) -> dict:
    """mode='search' — primary hybrid + auto-merge (async: retrieval runs off the event loop)."""
    start = time.perf_counter()

    log_op_debug(
        logger,
        "SEARCH_START",
        folder_id=folder.id,
        msg=(
            f"folder='{folder.name}' query='{query[:50]}{'...' if len(query) > 50 else ''}' "
            f"top_k={top_k} cutoff={similarity_cutoff} expand_context={expand_context}"
        ),
    )

    if not query or len(query.strip()) < 2:
        log_warn(
            logger,
            "SEARCH_INVALID_QUERY",
            folder_id=folder.id,
            msg=f"query too short (len={len(query.strip()) if query else 0})",
        )
        return {
            "mode": "search",
            "error": "Query too short (min 2 characters)",
            "results": [],
        }

    try:
        results: List[MergedResult] = await retriever.aquery(
            query_text=query,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
            expand_context=expand_context,
        )
    except Exception as exc:
        log_err(
            logger,
            "SEARCH_RETRIEVER_FAIL",
            exc,
            folder_id=folder.id,
            msg=f"query='{query[:30]}'",
        )
        raise

    role_counts = Counter(r.node_role for r in results)
    file_counts = Counter(r.file_name for r in results)
    scores = [r.score for r in results]
    score_stats = (
        f"min={min(scores):.3f} max={max(scores):.3f} avg={sum(scores)/len(scores):.3f}"
        if scores
        else "n/a"
    )

    merged_leaves_total = sum(
        len(r.merged_from_leaves) for r in results if r.node_role == "parent"
    )

    response = {
        "mode": "search",
        "query": query,
        "folder": folder.name,
        "total_results": len(results),
        "results": [
            {
                "node_id": r.node_id,
                "file_id": r.file_id,
                "file_name": r.file_name,
                "score": round(r.score, 4),
                "node_role": r.node_role,
                "chunk_index_range": r.chunk_index_range,
                "text": r.text,
                # Citation provenance: included only when present (older indexed data lacks this metadata)
                **({"page": r.page} if r.page is not None else {}),
                **({"headings": r.headings} if r.headings else {}),
                # content_type ("table"/"picture") lets an MCP agent know a hit's text
                # is a serialized table/figure, not prose — parity with the REST path.
                **({"content_type": getattr(r, "content_type", None)}
                   if getattr(r, "content_type", None) else {}),
                **(
                    {"merged_from_leaves": len(r.merged_from_leaves)}
                    if r.node_role == "parent"
                    else {}
                ),
            }
            for r in results
        ],
    }

    hint_reason = "normal"
    if not results:
        response["_hint"] = (
            "No results above similarity_cutoff. Try: lower cutoff, broaden query, "
            "or use mode='list' to see what's available."
        )
        hint_reason = "no_results"
    elif len(results) < top_k:
        response["_hint"] = (
            f"Only {len(results)} results above threshold. "
            "Consider lowering similarity_cutoff or using mode='read' on most relevant file."
        )
        hint_reason = "below_top_k"
    else:
        response["_hint"] = (
            "If results seem fragmented, try mode='read' with a relevant file_id "
            "to load full document content."
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    log_op_debug(
        logger,
        "SEARCH_DONE",
        folder_id=folder.id,
        msg=(
            f"results={len(results)} "
            f"roles={{leaf:{role_counts.get('leaf', 0)},"
            f"parent:{role_counts.get('parent', 0)}(merged_from_{merged_leaves_total}),"
            f"expanded:{role_counts.get('expanded', 0)}}} "
            f"unique_files={len(file_counts)} "
            f"scores=[{score_stats}] "
            f"hint={hint_reason} "
            f"elapsed={elapsed_ms:.0f}ms"
        ),
    )

    if file_counts:
        logger.debug(
            f"[SEARCH_FILES] fid={folder.id} | "
            + ", ".join(f"'{name}':{cnt}" for name, cnt in file_counts.most_common())
        )

    return response


def handle_list(*, folder) -> dict:
    """mode='list' — list all files in the folder + index status + token estimate."""
    start = time.perf_counter()

    log_op_debug(
        logger,
        "LIST_START",
        folder_id=folder.id,
        msg=f"folder='{folder.name}'",
    )

    files = FileDB.get(folder_id=folder.id)

    indexed_map = {}
    chunk_count_map = {}
    try:
        index_records = FileIndexDB.get_by_folder(folder.id)
        for r in index_records:
            indexed_map[str(r.file_id)] = r.status
            chunk_count_map[str(r.file_id)] = r.num_chunks
    except Exception as e:
        log_warn(
            logger,
            "LIST_INDEX_LOOKUP_FAIL",
            folder_id=folder.id,
            msg=f"fallback to 'not_indexed' for all files: {e}",
        )

    status_counts = Counter()
    total_chunks = 0
    file_list = []
    for f in files:
        chunk_count = chunk_count_map.get(str(f.id), 0)
        status = indexed_map.get(str(f.id), "not_indexed")
        status_counts[status] += 1
        total_chunks += chunk_count

        # Rough estimate: ~200 tokens per leaf chunk (aligned with leaf_chunk_size 256).
        # This is for the agent to judge whether it can read the whole file; precision isn't needed.
        estimated_tokens = chunk_count * 200 if chunk_count > 0 else None
        file_list.append({
            "file_id": str(f.id),
            "file_name": f.file_name,
            "size_kb": round(f.file_size / 1024, 1) if f.file_size else 0,
            "upload_time": f.upload_time.isoformat() if f.upload_time else None,
            "indexed_status": status,
            "chunk_count": chunk_count,
            "estimated_tokens": estimated_tokens,
        })

    response = {
        "mode": "list",
        "folder": folder.name,
        "description": folder.description,
        "total_files": len(file_list),
        "files": file_list,
        "_hint": (
            "Files with estimated_tokens < 30000 can be loaded fully via mode='read'. "
            "For others, use mode='search' to find relevant sections."
        ),
    }

    elapsed_ms = (time.perf_counter() - start) * 1000

    status_str = ",".join(f"{k}:{v}" for k, v in status_counts.items()) or "empty"
    log_op_debug(
        logger,
        "LIST_DONE",
        folder_id=folder.id,
        msg=(
            f"files={len(file_list)} status={{{status_str}}} "
            f"total_chunks={total_chunks} elapsed={elapsed_ms:.0f}ms"
        ),
    )

    return response


async def handle_read(
    *,
    file_id: str,
    folder,
    vector_store,
    max_tokens: int = 30000,
) -> dict:
    """mode='read' handler — load and return an entire file's content (use sparingly).

    Verifies the file belongs to the current folder (prevents cross-folder access) and enforces a
    max_tokens cap.

    Args:
        file_id: UUID of the file to read.
        folder: The permission-validated Folder ORM.
        vector_store: PGVectorStore (taken from ctx, used for chunk lookup).
        max_tokens: Content cap; truncates and marks truncated=True when exceeded.

    Returns:
        dict ``{content, truncated, file_name, ...}``.
    """
    start = time.perf_counter()

    log_op_debug(
        logger,
        "READ_START",
        folder_id=folder.id,
        file_id=file_id,
        msg=f"folder='{folder.name}' max_tokens={max_tokens}",
    )

    if not file_id:
        log_warn(
            logger,
            "READ_MISSING_FILE_ID",
            folder_id=folder.id,
            msg="read mode requires file_id parameter",
        )
        return {"mode": "read", "error": "file_id is required for read mode"}

    file_records = FileDB.get_by_ids([file_id])
    file_record = file_records.get(file_id) if file_records else None
    if not file_record:
        log_warn(
            logger,
            "READ_FILE_NOT_FOUND",
            folder_id=folder.id,
            file_id=file_id,
            msg="no file record in DB",
        )
        return {"mode": "read", "error": f"File {file_id} not found"}

    if file_record.folder_id != folder.id:
        # Deliberate: don't reveal which folder this file belongs to, to avoid enumeration attacks
        log_warn(
            logger,
            "READ_CROSS_FOLDER",
            folder_id=folder.id,
            file_id=file_id,
            msg=f"file actually belongs to folder_id={file_record.folder_id}, denied",
        )
        return {
            "mode": "read",
            "error": f"File does not belong to folder '{folder.name}'",
        }

    try:
        # Offload the sync SQL fetch (hundreds of ms for large files) to a thread to keep the loop free
        import asyncio
        full = await asyncio.to_thread(
            ChunkLookup.fetch_file_full,
            vector_store=vector_store,
            file_id=file_id,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        log_err(
            logger,
            "READ_FETCH_FAIL",
            exc,
            folder_id=folder.id,
            file_id=file_id,
        )
        raise

    response = {
        "mode": "read",
        "folder": folder.name,
        "file_id": file_id,
        "file_name": file_record.file_name,
        "content": full["text"],
        "tokens_estimated": full["tokens_estimated"],
        "truncated": full["truncated"],
        "chunk_count": full["chunk_count"],
        "_hint": (
            "Truncated content — consider mode='search' for specific sections."
            if full["truncated"]
            else "Full content loaded."
        ),
    }

    elapsed_ms = (time.perf_counter() - start) * 1000

    truncated_marker = " ⚠️TRUNCATED" if full["truncated"] else ""
    log_op_debug(
        logger,
        "READ_DONE",
        folder_id=folder.id,
        file_id=file_id,
        msg=(
            f"file_name='{file_record.file_name}' "
            f"chunks={full['chunk_count']} tokens={full['tokens_estimated']}"
            f"{truncated_marker} elapsed={elapsed_ms:.0f}ms"
        ),
    )

    return response
