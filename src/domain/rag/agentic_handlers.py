
"""Agentic three-mode query handlers — the actual search / list / read logic (domain layer).

These handlers live in the domain layer because they depend only on db / domain / log and carry no
fastmcp dependency. The
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
from typing import List, Optional

from db.filedb import FileDB
from db.fileindexdb import FileIndexDB
from src.domain.rag.auto_merging import AutoMergingRetriever, MergedResult
from src.domain.rag.chunk_lookup import ChunkLookup
from src.log import get_mcptools_logger, log_err, log_op_debug, log_warn

logger = get_mcptools_logger()


# MCP search returns at most this many results (the REST API caps top_k the same way).
MAX_TOP_K = 20

_HINT_NO_RESULTS = (
    "No matching passages. Rephrase the query with other keywords or synonyms "
    "(e.g. the document's own terms), or use mode='list' to see which files exist.")
_HINT_FEW = (
    "Only {n} passage(s) matched. If they do not answer the question, rephrase the "
    "query with more specific terms; do not read whole files to make up for it.")
_HINT_OK = (
    "Results are ranked best first (score_type={score_type}). Cite file_name and page. "
    "Use mode='read' only when the user asks for a whole document.")
_HINT_LOW_RELEVANCE = (
    "Low relevance: the best passage scored {top:.2f}, below min_relevance {cut:.2f}. "
    "This folder most likely does not answer the question - tell the user so instead "
    "of answering from these passages, or retry once with the document's own terms.")
_HINT_SOME_LOW = (
    " {n} result(s) marked low_relevance scored below min_relevance {cut:.2f}; treat "
    "them as weak matches.")
_HINT_OMITTED = (
    " {omitted} lower-ranked result(s) were left out to keep this response under "
    "~{budget} tokens; ask a narrower query to surface them.")


def clamp_top_k(top_k) -> int:
    """top_k bounded to 1..MAX_TOP_K (a missing or invalid value counts as 1)."""
    try:
        return max(1, min(MAX_TOP_K, int(top_k)))
    except (TypeError, ValueError):
        return 1


def score_type_of(retriever) -> str:
    """``rerank`` / ``rrf`` / ``vector`` for the retriever's last search."""
    outcome = getattr(retriever, "last_outcome", None)
    st = getattr(outcome, "score_type", None)
    return st if isinstance(st, str) else "rrf"


def configured_response_budget() -> int:
    """``rag.retrieval.mcp_response_token_budget`` (live); the default when unreadable."""
    try:
        from src.config.config_manager import Config

        retrieval = Config.get_config_model().rag.retrieval
        value = getattr(retrieval, "mcp_response_token_budget", DEFAULT_RESPONSE_BUDGET)
        return int(value if value is not None else DEFAULT_RESPONSE_BUDGET)
    except Exception:  # noqa: BLE001
        return DEFAULT_RESPONSE_BUDGET


DEFAULT_RESPONSE_BUDGET = 6000


def configured_min_relevance() -> float:
    """``rag.retrieval.min_relevance`` (live), clamped to 0..1; 0.0 (check off) when
    unset or unreadable."""
    try:
        from src.config.config_manager import Config

        value = getattr(Config.get_config_model().rag.retrieval, "min_relevance", 0.0)
        return min(1.0, max(0.0, float(value or 0.0)))
    except Exception:  # noqa: BLE001
        return 0.0


def apply_relevance(response: dict, *, reranked: bool, top_score: Optional[float],
                    min_relevance: float) -> dict:
    """Add the relevance signal to a search response (in place) and return it.

    Top level: ``top_relevance`` - the cross-encoder's best score over every
    candidate, before ``rag.rerank.score_threshold`` (None when the results were
    not reranked); ``min_relevance`` - the configured cutoff (0 = off); and
    ``low_relevance`` - True when nothing came back, or when the best score is
    below the cutoff. Per result (reranked only): ``relevance`` and, below the
    cutoff, ``low_relevance: true``.

    Results are flagged, never dropped: the eval shows the cutoff separates
    off-topic questions cleanly but not questions that are near misses of the
    corpus, so the caller decides (dropping is what rerank.score_threshold does).
    """
    results = response.get("results") or []
    # Compared at the precision the caller sees (scores are rounded to 4 places), so a
    # value shown as equal to the cutoff is never flagged.
    top = round(float(top_score), 4) if reranked and top_score is not None else None
    if reranked:
        for r in results:
            score = r.get("score")
            if score is None:
                continue
            r["relevance"] = round(float(score), 4)
            if min_relevance > 0 and r["relevance"] < min_relevance:
                r["low_relevance"] = True
    response["top_relevance"] = top
    response["min_relevance"] = min_relevance
    response["low_relevance"] = (not results) or (
        top is not None and min_relevance > 0 and top < min_relevance)
    return response


def result_dict(r: MergedResult) -> dict:
    """One search hit as the agent sees it."""
    out = {
        "node_id": r.node_id,
        "file_id": r.file_id,
        "file_name": r.file_name,
        "score": round(r.score, 4),
        "node_role": r.node_role,
        "chunk_index_range": r.chunk_index_range,
        "text": r.text,
    }
    # Citation provenance: included only when present (older indexed data lacks this metadata)
    if r.page is not None:
        out["page"] = r.page
    if r.headings:
        out["headings"] = r.headings
    # content_type ("table"/"picture") lets an MCP agent know a hit's text
    # is a serialized table/figure, not prose — parity with the REST path.
    content_type = getattr(r, "content_type", None)
    if content_type:
        out["content_type"] = content_type
    merged = getattr(r, "merged_from_leaves", None) or []
    if r.node_role == "parent":
        out["merged_from_leaves"] = len(merged)
    elif len(merged) > 1:
        out["merged_hits"] = len(merged)      # one window covering several hits
    return out


def fit_to_budget(response: dict, budget: int) -> dict:
    """Drop the lowest-ranked results until ``response`` fits ``budget`` estimated
    tokens; when even the best result alone is too big, cut its text. Records
    ``omitted_results`` / ``response_tokens_estimated`` on the response.
    A budget of 0 (or less) disables the check.
    """
    import json

    from src.domain.rag.token_budget import estimate_tokens, truncate_to_tokens

    def size(obj) -> int:
        return estimate_tokens(json.dumps(obj, ensure_ascii=False))

    results = list(response.get("results") or [])
    if budget <= 0 or not results:
        response["response_tokens_estimated"] = size(response)
        return response
    # Room for the envelope and a hint that may grow by one sentence.
    envelope = {k: v for k, v in response.items() if k != "results"}
    used = size(envelope) + 80
    kept = []
    for r in results:
        cost = size(r)
        if used + cost > budget:
            if not kept:
                # The best hit alone does not fit: keep it, cut to what is left.
                room = max(budget - used - size({**r, "text": ""}), 0)
                text, _ = truncate_to_tokens(r.get("text") or "", room)
                r = {**r, "text": text, "text_truncated": True}
                kept.append(r)
                used += size(r)
            break
        kept.append(r)
        used += cost
    omitted = len(results) - len(kept)
    response["results"] = kept
    response["total_results"] = len(kept)
    if omitted:
        response["omitted_results"] = omitted
    response["response_tokens_estimated"] = size(response)
    return response


async def handle_search(
    *,
    query: str,
    folder,
    retriever: AutoMergingRetriever,
    top_k: int,
    similarity_cutoff: Optional[float] = None,
    expand_context: bool,
    token_budget: Optional[int] = None,
) -> dict:
    """mode='search' — primary hybrid + auto-merge (async: retrieval runs off the event loop).

    ``similarity_cutoff`` is accepted for backward compatibility and ignored: the
    ranking is rank fusion (+ rerank), whose scores a similarity threshold does
    not apply to. ``token_budget`` (default ``rag.retrieval.mcp_response_token_budget``)
    caps the estimated size of the response.
    """
    start = time.perf_counter()
    top_k = clamp_top_k(top_k)

    log_op_debug(
        logger,
        "SEARCH_START",
        folder_id=folder.id,
        msg=(
            f"folder='{folder.name}' query='{query[:50]}{'...' if len(query) > 50 else ''}' "
            f"top_k={top_k} expand_context={expand_context}"
            + (f" similarity_cutoff={similarity_cutoff} (ignored)"
               if similarity_cutoff is not None else "")
        ),
    )

    from src.domain.rag.query_limits import MAX_QUERY_CHARS
    if query and len(query) > MAX_QUERY_CHARS:
        log_warn(logger, "SEARCH_INVALID_QUERY", folder_id=folder.id,
                 msg=f"query too long (len={len(query)})")
        return {
            "mode": "search",
            "error": f"Query too long (max {MAX_QUERY_CHARS} characters)",
            "results": [],
        }

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
    score_type = score_type_of(retriever)

    response = {
        "mode": "search",
        "query": query,
        "folder": folder.name,
        "top_k": top_k,
        # What "score" means: rerank = cross-encoder relevance (0..1);
        # rrf = rank fusion (~0.01-0.03, compare only within one response).
        "score_type": score_type,
        "total_results": len(results),
        "results": [result_dict(r) for r in results],
    }
    outcome = getattr(retriever, "last_outcome", None)
    min_relevance = configured_min_relevance()
    apply_relevance(response, reranked=score_type == "rerank",
                    top_score=getattr(outcome, "top_score", None),
                    min_relevance=min_relevance)

    if not results:
        response["_hint"] = _HINT_NO_RESULTS
        hint_reason = "no_results"
    elif response["low_relevance"]:
        response["_hint"] = _HINT_LOW_RELEVANCE.format(
            top=response["top_relevance"], cut=min_relevance)
        hint_reason = "low_relevance"
    else:
        if len(results) < top_k:
            response["_hint"] = _HINT_FEW.format(n=len(results))
            hint_reason = "below_top_k"
        else:
            response["_hint"] = _HINT_OK.format(score_type=score_type)
            hint_reason = "normal"
        weak = sum(1 for r in response["results"] if r.get("low_relevance"))
        if weak:
            response["_hint"] += _HINT_SOME_LOW.format(n=weak, cut=min_relevance)
            hint_reason += "+weak"

    budget = configured_response_budget() if token_budget is None else int(token_budget)
    fit_to_budget(response, budget)
    if response.get("omitted_results"):
        response["_hint"] += _HINT_OMITTED.format(
            omitted=response["omitted_results"], budget=budget)
        hint_reason += "+budget"

    elapsed_ms = (time.perf_counter() - start) * 1000
    timings = getattr(getattr(retriever, "last_outcome", None), "timings", None)

    log_op_debug(
        logger,
        "SEARCH_DONE",
        folder_id=folder.id,
        msg=(
            f"results={len(results)} "
            f"returned={response['total_results']} "
            f"roles={{leaf:{role_counts.get('leaf', 0)},"
            f"parent:{role_counts.get('parent', 0)}(merged_from_{merged_leaves_total}),"
            f"expanded:{role_counts.get('expanded', 0)}}} "
            f"unique_files={len(file_counts)} "
            f"scores=[{score_stats}] score_type={score_type} "
            f"top_relevance={response.get('top_relevance')} "
            f"low_relevance={response.get('low_relevance')} "
            f"tokens~{response.get('response_tokens_estimated')} "
            f"hint={hint_reason} "
            f"elapsed={elapsed_ms:.0f}ms"
            + (f" stages_ms={timings}" if isinstance(timings, dict) else "")
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

    # Token estimates from the stored chunk text (one aggregate query); a file the
    # query does not cover falls back to ~200 tokens per leaf chunk.
    token_map = {}
    if getattr(folder, "vector_table_uuid", None):
        token_map = ChunkLookup.file_token_estimates(
            f"data_{folder.id}_{folder.vector_table_uuid}")

    status_counts = Counter()
    total_chunks = 0
    file_list = []
    for f in files:
        chunk_count = chunk_count_map.get(str(f.id), 0)
        status = indexed_map.get(str(f.id), "not_indexed")
        status_counts[status] += 1
        total_chunks += chunk_count

        estimated_tokens = token_map.get(str(f.id))
        if estimated_tokens is None:
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
            f"mode='read' returns up to ~{READ_PAGE_TOKENS} tokens per call; a longer file "
            "comes in pages (pass next_offset back as offset). To answer a question, "
            "use mode='search' instead."
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


READ_PAGE_TOKENS = 30000


async def handle_read(
    *,
    file_id: str,
    folder,
    vector_store,
    max_tokens: int = READ_PAGE_TOKENS,
    offset: Optional[int] = None,
) -> dict:
    """mode='read' handler — return a file's content, one page at a time (use sparingly).

    Verifies the file belongs to the current folder (prevents cross-folder access).
    A page holds up to ``max_tokens`` (estimated); when the file continues, the
    response carries ``next_offset`` — pass it back as ``offset`` for the next page.

    Args:
        file_id: UUID of the file to read.
        folder: The permission-validated Folder ORM.
        vector_store: PGVectorStore (taken from ctx, used for chunk lookup).
        max_tokens: Page size in estimated tokens.
        offset: Chunk index to start from (0 / None = the beginning).

    Returns:
        dict ``{content, truncated, next_offset, file_name, ...}``.
    """
    start = time.perf_counter()

    log_op_debug(
        logger,
        "READ_START",
        folder_id=folder.id,
        file_id=file_id,
        msg=f"folder='{folder.name}' max_tokens={max_tokens} offset={offset or 0}",
    )

    if not file_id:
        log_warn(
            logger,
            "READ_MISSING_FILE_ID",
            folder_id=folder.id,
            msg="read mode requires file_id parameter",
        )
        return {"mode": "read", "error": "file_id is required for read mode"}

    import asyncio

    try:
        offset = max(0, int(offset or 0))
    except (TypeError, ValueError):
        return {"mode": "read", "error": "offset must be a non-negative integer"}

    file_records = await asyncio.to_thread(FileDB.get_by_ids, [file_id])
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
        full = await asyncio.to_thread(
            ChunkLookup.fetch_file_full,
            vector_store=vector_store,
            file_id=file_id,
            max_tokens=max_tokens,
            offset=offset,
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
        "offset": full.get("offset", offset),
        "next_offset": full.get("next_offset"),
        "chunk_count": full["chunk_count"],
        "total_chunks": full.get("total_chunks", full["chunk_count"]),
        "_hint": _read_hint(full, offset),
    }

    elapsed_ms = (time.perf_counter() - start) * 1000

    truncated_marker = (f" ⚠️TRUNCATED next_offset={full.get('next_offset')}"
                        if full["truncated"] else "")
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


def _read_hint(full: dict, offset: int) -> str:
    if full.get("truncated"):
        return (f"Page {offset}..{(full.get('next_offset') or 0) - 1} of "
                f"{full.get('total_chunks')} chunks. Call again with "
                f"offset={full.get('next_offset')} for the next page, or use mode='search' "
                "for specific sections.")
    if offset and not full.get("chunk_count"):
        return "No content at this offset (past the end of the file)."
    if not full.get("chunk_count"):
        return ("This file has no indexed content: it failed to index or has not been indexed "
                "yet. Its status is in the console; re-index it there.")
    return "End of file." if offset else "Full content loaded."
