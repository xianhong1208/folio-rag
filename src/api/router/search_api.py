"""Search across folders: POST /api/search.

One search over all of the caller's folders (or the ``folders`` named), optionally
limited by a document filter, answered as passages or as documents. The work is in
src/domain/rag/doc_search.py; this module only validates the body and maps errors.

Auth is the console's (as for /api/documents/search): a Bearer PAT (scripts, agents)
or the console session cookie, whose POSTs need the CSRF header. Only the folders of
the person the token belongs to (its ``sub``) are ever searched; in open mode (auth
disabled) every folder.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.auth.console_session import ConsoleIdentity, require_console_session
from src.domain.rag import doc_filter as DF
from src.domain.rag import doc_search as DS
from src.domain.rag.query_limits import MAX_QUERY_CHARS, MIN_QUERY_CHARS
from src.log import get_api_logger

logger = get_api_logger()

router = APIRouter(tags=["Search"], dependencies=[Depends(require_console_session)])

_Session = Depends(require_console_session)


class SearchBody(BaseModel):
    """Body of POST /api/search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., max_length=MAX_QUERY_CHARS, description="What to look for")
    folders: Optional[List[Union[int, str]]] = Field(
        None, min_length=1, max_length=DF.MAX_FILTER_LIST,
        description="Folder names or ids to search (your own); omit for all your folders")
    filters: Optional[DF.DocumentFilter] = Field(
        None, description="Only documents matching this filter (docs/api.md#document-filters)")
    group_by: Literal["passage", "document"] = Field(
        "passage", description="passage = ranked passages; document = ranked documents with snippets")
    top_k: Optional[int] = Field(
        None, ge=1, le=DS.MAX_TOP_K,
        description="Passages (default rag.retrieval.default_top_k) or documents (default 10)")
    snippets: int = Field(DS.DEFAULT_SNIPPETS, ge=1, le=DS.MAX_SNIPPETS,
                          description="document: passages shown per document")
    auto_merge: Optional[bool] = Field(
        None, description="passage: merge hit leaves into their section; null = as configured")

    @field_validator("query")
    @classmethod
    def _strip_min(cls, v: str) -> str:
        v = v.strip()
        if len(v) < MIN_QUERY_CHARS:
            raise ValueError(f"query too short (min {MIN_QUERY_CHARS} chars)")
        return v


@router.post("/api/search")
async def search(
    body: SearchBody = Body(..., examples=[{
        "query": "payment terms", "group_by": "document",
        "filters": {"tags_any": ["signed"]}, "top_k": 10}]),
    ident: ConsoleIdentity = _Session,
):
    """Search all your folders (or ``folders``) at once.

    ``group_by: passage`` returns ``results`` (passages, each with its ``folder``);
    ``group_by: document`` returns ``documents``: file_id, name, folder, mime, tags,
    metadata, score (its best passage's), hits (matching passages) and snippets.
    ``skipped_folders`` lists folders that were not searched and why
    (nothing_indexed, embedding_model, folder_limit, error).
    """
    from src.adapter.rag import get_rag_adapter

    req = DS.SearchRequest(query=body.query, folders=body.folders, filters=body.filters,
                           group_by=body.group_by, top_k=body.top_k, snippets=body.snippets,
                           auto_merge=body.auto_merge)
    try:
        data = await get_rag_adapter().search_across(req, owner=DF.owner_scope(ident.owner_key))
    except DS.FolderNotFound as e:
        raise HTTPException(404, str(e))
    except HTTPException:
        raise
    except Exception as e:
        from src.domain.rag.query_limits import QUERY_TOO_LONG_DETAIL, is_input_too_long_error
        if is_input_too_long_error(e):
            raise HTTPException(422, QUERY_TOO_LONG_DETAIL)
        logger.warning(f"[SEARCH] failed: {type(e).__name__}: {e}")
        raise HTTPException(500, f"Search failed: {e}")
    n = data.get("total_results", 0)
    return {"data": data, "message": f"Search returned {n} {body.group_by}(s)"}
