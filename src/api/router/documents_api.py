"""Documents across all of a person's folders: list with filters, tags, metadata keys.

- POST /api/documents/search   one page of documents matching a filter (JSON body)
- GET  /api/tags               the person's tags with document counts
- GET  /api/metadata/keys      the person's metadata keys, value types and counts

Auth is the console's: a Bearer PAT (scripts, agents) or the console session cookie
(unsafe methods then need the CSRF header, as for every console write; a search is a
POST only because the filter is a JSON object). Everything is limited to the folders
of the person the token belongs to (its ``sub``) — the filter can narrow that, never
widen it (src/domain/rag/doc_filter.py). In open mode (auth disabled) every folder.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from src.auth.console_session import ConsoleIdentity, require_console_session
from src.domain.rag import doc_filter as DF

router = APIRouter(tags=["Documents"], dependencies=[Depends(require_console_session)])

_Session = Depends(require_console_session)


class DocumentSearch(BaseModel):
    """Body of POST /api/documents/search."""

    model_config = ConfigDict(extra="forbid")

    filters: Optional[DF.DocumentFilter] = Field(
        None, description="The document filter (docs/api.md#document-filters); omit for all documents")
    sort: Literal["name", "uploaded", "size"] = "uploaded"
    order: Literal["asc", "desc"] = "desc"
    limit: int = Field(50, ge=1, le=DF.MAX_PAGE)
    offset: int = Field(0, ge=0, le=1_000_000)


def _connect():
    from db.db import get_engine
    return get_engine().connect()


def _search(body: DocumentSearch, owner) -> dict:
    with _connect() as conn:
        items, total = DF.search_documents(
            conn, body.filters, owner=owner, sort=body.sort, order=body.order,
            limit=body.limit, offset=body.offset)
    nxt = body.offset + len(items)
    return {"documents": items, "total": total, "limit": body.limit, "offset": body.offset,
            "next_offset": nxt if nxt < total else None}


@router.post("/api/documents/search")
async def search_documents(
    body: DocumentSearch = Body(..., examples=[{
        "filters": {"tags_any": ["invoice"], "metadata": {"year": {"gte": 2024}}},
        "sort": "uploaded", "order": "desc", "limit": 50}]),
    ident: ConsoleIdentity = _Session,
):
    """Your documents across all your folders that match ``filters``, one page at a time.

    Each document: id, name, folder {id, name}, size_bytes, mime, uploaded_at,
    updated_at, status (index status), tags, metadata, description. ``next_offset`` is
    the offset of the next page (null on the last one).
    """
    data = await run_in_threadpool(_search, body, DF.owner_scope(ident.owner_key))
    return {"data": data, "message": "ok"}


def _tags(owner) -> list:
    with _connect() as conn:
        return DF.tag_counts(conn, owner=owner)


def _keys(owner) -> list:
    with _connect() as conn:
        return DF.metadata_keys(conn, owner=owner)


@router.get("/api/tags")
async def list_tags(ident: ConsoleIdentity = _Session):
    """Your tags across all your folders, with how many documents carry each (tags that
    differ only in case are one tag, shown in its most common spelling)."""
    tags = await run_in_threadpool(_tags, DF.owner_scope(ident.owner_key))
    return {"data": {"tags": tags}, "message": "ok"}


@router.get("/api/metadata/keys")
async def list_metadata_keys(ident: ConsoleIdentity = _Session):
    """Your metadata keys: how many documents carry each, and the value types seen
    (string, number, boolean, date) with their counts."""
    keys = await run_in_threadpool(_keys, DF.owner_scope(ident.owner_key))
    return {"data": {"keys": keys}, "message": "ok"}
