"""Find documents across a person's folders (the MCP ``folio_find`` tool), and limit a
folder search to some documents (``file_ids`` / ``filters`` on the folder tools).

``folio_find`` answers "which documents?" before a folder tool answers "what do they
say?" (RAGFlow's ``search_metadata`` → ``search_chunks(doc_scope)``, R2R's
``search_file_descriptions`` vs ``search_file_knowledge``):

- **With a query**: the person's documents ranked by their best passage, across all
  their folders (or the ``folders`` named) — :func:`src.domain.rag.doc_search.search_across`
  with ``group_by="document"``, so the ranking, the owner scope, the filter, the skip
  reasons and the cache are those of ``POST /api/search``.
- **Without a query**: a listing of the documents matching ``filters``, newest upload
  first — :func:`src.domain.rag.doc_filter.search_documents`, the logic of
  ``POST /api/documents/search``.

Both return compact JSON trimmed to ``rag.retrieval.mcp_response_token_budget``
(:func:`fit_documents_to_budget`): lower-ranked documents are dropped first, and
snippets are short (:data:`SNIPPET_MAX_TOKENS`). Each document names the folder tool
to search it with (``tool``), so the agent can go on with ``file_ids``.

The owner scope is never taken from the arguments: the delivery layer passes the
caller's owner key (``sub:<sub>``), or ``ALL_FOLDERS`` in open mode, as for REST.
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from src.domain.rag import doc_filter as DF
from src.domain.rag import doc_search as DS
from src.domain.rag.query_limits import MAX_QUERY_CHARS, MIN_QUERY_CHARS

DEFAULT_LIMIT = 10
MAX_LIMIT = 50
MAX_OFFSET = 100_000
FIND_SNIPPETS = 2
SNIPPET_MAX_TOKENS = 220
DESCRIPTION_MAX_CHARS = 200
MAX_FILE_IDS = DF.MAX_FILTER_LIST
# Shown in folio_find's description (and in an overview listing).
VOCAB_TAGS = 25
VOCAB_KEYS = 15

_HINT_FOUND = (
    "Documents ranked by their best passage (score_type={score_type}). To answer from "
    "them, call the folder tool named in `tool` with mode='search' and file_ids=[...] of "
    "the documents you want.")
_HINT_NONE = (
    "No document matched. Rephrase the query with the documents' own terms, loosen the "
    "filters, or call folio_find without a query to list documents by tags or metadata.")
_HINT_LIST = (
    "Newest upload first. Search inside these documents with the folder tool named in "
    "`tool` (mode='search', file_ids=[...]); pass next_offset as offset for the next page.")
_HINT_LIST_EMPTY = "No document matches these filters."
_HINT_OMITTED = (
    " {n} lower-ranked document(s) were left out to keep this response under ~{budget} "
    "tokens; narrow the query or the filters, or lower limit.")


class FindError(Exception):
    """A refused request (bad arguments, a folder that is not the caller's): returned
    to the agent as ``{"error": code, "detail": ...}``."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail

    def payload(self) -> Dict[str, Any]:
        return {"error": self.code, "detail": self.detail}


# ------------------------------------------------------------- arguments


def parse_filters(raw: Any) -> Optional[DF.DocumentFilter]:
    """The filter object, validated like the REST body (FindError naming the bad key)."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise FindError("invalid_filter", "invalid filter: filters must be a JSON object")
    try:
        return DF.parse_filter(raw)
    except Exception as e:  # pydantic.ValidationError
        raise FindError("invalid_filter", DF.filter_error_detail(e)) from None


def parse_folders(raw: Any) -> Optional[List[Any]]:
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw or len(raw) > DF.MAX_FILTER_LIST:
        raise FindError("invalid_request",
                        f"folders must be a list of 1-{DF.MAX_FILTER_LIST} folder names or ids")
    out: List[Any] = []
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, (int, str)):
            raise FindError("invalid_request", "folders entries must be folder names or ids")
        out.append(x)
    return out


def parse_query(raw: Any) -> str:
    q = (raw or "").strip() if isinstance(raw, str) or raw is None else None
    if q is None:
        raise FindError("invalid_request", "query must be a string")
    if q and len(q) < MIN_QUERY_CHARS:
        raise FindError("invalid_request", f"query too short (min {MIN_QUERY_CHARS} characters)")
    if len(q) > MAX_QUERY_CHARS:
        raise FindError("invalid_request", f"query too long (max {MAX_QUERY_CHARS} characters)")
    return q


def clamp(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(value) if value is not None else default
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


# ------------------------------------------------------------- shaping


def _snippet(s: Dict[str, Any]) -> Dict[str, Any]:
    from src.domain.rag.token_budget import truncate_to_tokens

    text, cut = truncate_to_tokens(s.get("text") or "", SNIPPET_MAX_TOKENS)
    out: Dict[str, Any] = {"text": text + ("…" if cut else "")}
    if s.get("page") is not None:
        out["page"] = s["page"]
    if s.get("score") is not None:
        out["score"] = round(float(s["score"]), 4)
    return out


def _description(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return text if len(text) <= DESCRIPTION_MAX_CHARS else text[:DESCRIPTION_MAX_CHARS - 1] + "…"


def shape_found(data: Dict[str, Any], tool_of: Callable[[str], str]) -> Dict[str, Any]:
    """``search_across`` document results as the agent sees them."""
    docs = []
    for d in data.get("documents") or []:
        folder = (d.get("folder") or {}).get("name")
        docs.append({
            "file_id": d["file_id"], "name": d["name"], "folder": folder,
            "tool": tool_of(folder) if folder else None,
            "tags": d.get("tags") or [], "metadata": d.get("metadata") or {},
            "score": round(float(d.get("score") or 0.0), 4), "hits": d.get("hits", 0),
            "snippets": [_snippet(s) for s in d.get("snippets") or []],
        })
    out: Dict[str, Any] = {
        "mode": "find", "query": data.get("query"), "score_type": data.get("score_type"),
        "total_results": len(docs), "documents": docs,
    }
    skipped = [{"folder": s.get("name"), "reason": s.get("reason")}
               for s in data.get("skipped_folders") or []]
    if skipped:
        out["skipped_folders"] = skipped
    out["_hint"] = (_HINT_FOUND.format(score_type=out["score_type"]) if docs else _HINT_NONE)
    return out


def shape_listing(items: Sequence[Dict[str, Any]], total: int, offset: int,
                  tool_of: Callable[[str], str]) -> Dict[str, Any]:
    """``search_documents`` rows as the agent sees them."""
    docs = []
    for d in items:
        folder = (d.get("folder") or {}).get("name")
        row = {
            "file_id": d["id"], "name": d["name"], "folder": folder,
            "tool": tool_of(folder) if folder else None,
            "tags": d.get("tags") or [], "metadata": d.get("metadata") or {},
            "mime": d.get("mime"), "uploaded_at": d.get("uploaded_at"), "status": d.get("status"),
        }
        desc = _description(d.get("description"))
        if desc:
            row["description"] = desc
        docs.append(row)
    nxt = offset + len(docs)
    return {"mode": "list", "total": int(total), "offset": offset,
            "next_offset": nxt if nxt < total else None,
            "total_results": len(docs), "documents": docs,
            "_hint": _HINT_LIST if docs else _HINT_LIST_EMPTY}


def fit_documents_to_budget(response: Dict[str, Any], budget: int) -> Dict[str, Any]:
    """Drop the lowest-ranked documents until ``response`` fits ``budget`` estimated
    tokens; when even the first document alone is too big, its snippets are dropped
    (then cut). Records ``omitted_results`` / ``response_tokens_estimated``. A budget
    of 0 (or less) disables the check."""
    import json

    from src.domain.rag.token_budget import estimate_tokens

    def size(obj) -> int:
        return estimate_tokens(json.dumps(obj, ensure_ascii=False))

    docs = list(response.get("documents") or [])
    if budget <= 0 or not docs:
        response["response_tokens_estimated"] = size(response)
        return response
    envelope = {k: v for k, v in response.items() if k != "documents"}
    used = size(envelope) + 60          # room for the hint to grow by one sentence
    kept: List[Dict[str, Any]] = []
    for d in docs:
        cost = size(d)
        if used + cost > budget:
            if not kept:
                d = dict(d)
                while d.get("snippets") and used + size(d) > budget:
                    d["snippets"] = d["snippets"][:-1]
                d["snippets_truncated"] = True
                kept.append(d)
            break
        kept.append(d)
        used += cost
    omitted = len(docs) - len(kept)
    response["documents"] = kept
    response["total_results"] = len(kept)
    if omitted:
        response["omitted_results"] = omitted
        response["_hint"] = (response.get("_hint") or "") + _HINT_OMITTED.format(
            n=omitted, budget=budget)
    response["response_tokens_estimated"] = size(response)
    return response


# ------------------------------------------------------------- vocabulary


def vocabulary(conn, *, owner: DF.Owner, tags: int = VOCAB_TAGS, keys: int = VOCAB_KEYS
               ) -> Dict[str, Any]:
    """The caller's most used tags and metadata keys (with their value types)."""
    t = DF.tag_counts(conn, owner=owner)
    k = DF.metadata_keys(conn, owner=owner)
    return {
        "tags": [{"tag": x["tag"], "count": x["count"]} for x in t[:tags]],
        "more_tags": max(0, len(t) - tags),
        "metadata_keys": [{"key": x["key"], "types": sorted(x["types"])} for x in k[:keys]],
        "more_metadata_keys": max(0, len(k) - keys),
    }


def vocabulary_text(vocab: Dict[str, Any]) -> str:
    """The vocabulary as a few description lines (empty when there is none)."""
    lines = []
    if vocab.get("tags"):
        tags = ", ".join(f"{x['tag']} ({x['count']})" for x in vocab["tags"])
        more = f" and {vocab['more_tags']} more" if vocab.get("more_tags") else ""
        lines.append(f"Your tags (documents): {tags}{more}.")
    if vocab.get("metadata_keys"):
        keys = ", ".join(f"{x['key']} ({'/'.join(x['types'])})" for x in vocab["metadata_keys"])
        more = f" and {vocab['more_metadata_keys']} more" if vocab.get("more_metadata_keys") else ""
        lines.append(f"Your metadata keys: {keys}{more}.")
    if not lines:
        return ""
    return "\n".join(["Use these exact values in filters:"] + lines)


# ------------------------------------------------------------- find


def _resolve_listing_folders(conn, owner: DF.Owner, folders: Optional[List[Any]],
                             flt: Optional[DF.DocumentFilter]) -> Optional[DF.DocumentFilter]:
    """``folders`` (FolderNotFound for one that is not the caller's, as for a search)
    folded into the filter as folder ids; a filter's own ``folders`` still narrows."""
    if folders is None:
        return flt
    ids = [f.id for f in DS.owned_folders(conn, owner, folders)]
    if flt is not None and flt.folders is not None:
        named = DS.owned_folders(conn, owner)
        allowed = {f.id for f in named
                   if f.id in {x for x in flt.folders if isinstance(x, int)}
                   or f.name in {x for x in flt.folders if isinstance(x, str)}}
        ids = [i for i in ids if i in allowed]
    base = flt if flt is not None else DF.DocumentFilter()
    return base.model_copy(update={"folders": ids})


async def find(*, owner: DF.Owner, query: Any = None, filters: Any = None, folders: Any = None,
               limit: Any = None, offset: Any = None,
               search_across: Callable[..., Awaitable[Dict[str, Any]]],
               connect: Callable[[], Any], tool_of: Callable[[str], str],
               budget: int, run_sync: Callable[..., Awaitable[Any]]) -> Dict[str, Any]:
    """One ``folio_find`` call. Raises FindError for a refused request."""
    if owner is None or (isinstance(owner, str) and not owner):
        raise ValueError("an owner key (or ALL_FOLDERS) is required")
    q = parse_query(query)
    flt = parse_filters(filters)
    names = parse_folders(folders)
    n = clamp(limit, DEFAULT_LIMIT, 1, MAX_LIMIT)

    if q:
        req = DS.SearchRequest(query=q, folders=names, filters=flt, group_by=DS.GROUP_DOCUMENT,
                               top_k=n, snippets=FIND_SNIPPETS)
        try:
            data = await search_across(req, owner=owner)
        except DS.FolderNotFound as e:
            raise FindError("folder_not_found", str(e)) from None
        response = shape_found(data, tool_of)
    else:
        start = clamp(offset, 0, 0, MAX_OFFSET)
        overview = names is None and (flt is None or flt.is_empty()) and start == 0

        def _list():
            with connect() as conn:
                scoped = _resolve_listing_folders(conn, owner, names, flt)
                items, total = DF.search_documents(conn, scoped, owner=owner, sort="uploaded",
                                                   order="desc", limit=n, offset=start)
                vocab = vocabulary(conn, owner=owner) if overview else None
            return items, total, vocab

        try:
            items, total, vocab = await run_sync(_list)
        except DS.FolderNotFound as e:
            raise FindError("folder_not_found", str(e)) from None
        response = shape_listing(items, total, start, tool_of)
        if vocab is not None and (vocab["tags"] or vocab["metadata_keys"]):
            response["vocabulary"] = vocab
    return fit_documents_to_budget(response, budget)


# ------------------------------------------------- folder search scope


def _uuids(values: Sequence[Any]) -> Tuple[List[str], List[str]]:
    good, bad = [], []
    for v in values:
        try:
            good.append(str(uuid.UUID(str(v))))
        except (ValueError, TypeError, AttributeError):
            bad.append(str(v))
    return good, bad


def search_scope(conn, *, folder_id: int, owner: DF.Owner, file_ids: Any = None,
                 filters: Any = None) -> Optional[Dict[str, Any]]:
    """The documents a folder search is limited to, or None for the whole folder.

    ``file_ids``: ids from folio_find or mode='list'; ids that are not documents of
    this folder are ignored (and listed), and FindError when none is left — a search
    limited to other folders' documents would silently search nothing. ``filters``:
    the document filter, scoped to this folder and the caller. Both given: documents
    that satisfy both. Returns {"file_ids": tuple (may be empty when the filter matches
    nothing), "ignored_file_ids": [...]}.
    """
    from sqlalchemy import text

    flt = parse_filters(filters)
    if file_ids is None and (flt is None or flt.is_empty()):
        return None
    ids: Optional[set] = None
    ignored: List[str] = []
    if file_ids is not None:
        if isinstance(file_ids, str):
            file_ids = [file_ids]
        if not isinstance(file_ids, list) or not file_ids or len(file_ids) > MAX_FILE_IDS:
            raise FindError("invalid_request",
                            f"file_ids must be a list of 1-{MAX_FILE_IDS} file ids")
        good, bad = _uuids(file_ids)
        rows = conn.execute(text(
            'SELECT f.id FROM "Files" f WHERE f.folder_id = :fid '
            "AND f.id = ANY(CAST(:ids AS uuid[]))"),
            {"fid": int(folder_id), "ids": good}).fetchall() if good else []
        ids = {str(r[0]) for r in rows}
        ignored = bad + [g for g in good if g not in ids]
        if not ids:
            raise FindError("invalid_request",
                            "none of file_ids is a document of this folder; use the file_id "
                            "values folio_find (or mode='list') returned for this folder")
    if flt is not None and not flt.is_empty():
        matched = set(DF.matching_file_ids(conn, flt, owner=owner, folder_id=folder_id))
        ids = matched if ids is None else ids & matched
    return {"file_ids": tuple(sorted(ids or ())), "ignored_file_ids": ignored}


__all__: Sequence[str] = (
    "FindError", "find", "fit_documents_to_budget", "search_scope", "shape_found",
    "shape_listing", "vocabulary", "vocabulary_text",
)
