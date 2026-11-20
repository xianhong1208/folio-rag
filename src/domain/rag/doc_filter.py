"""The document filter: one flat, AND-only object, turned into parameterised SQL.

    {"tags_any": ["invoice", "2024"], "tags_all": ["signed"],
     "folders": ["contracts", 12], "file_type": ["pdf", "text/markdown", "image/*"],
     "uploaded_after": "2024-01-01", "uploaded_before": "2025-01-01",
     "name_contains": "q3", "file_ids": ["2f0c…"],
     "metadata": {"client": "Acme", "year": {"gte": 2024},
                  "lang": {"in": ["zh", "en"]}, "due": {"lte": "2024-12-31"},
                  "reviewed": {"exists": true}}}

Every condition must hold. ``tags_any`` matches a document with at least one of the
tags, ``tags_all`` one with all of them; tags match case-insensitively. ``folders``
takes folder names or ids, always among the caller's own folders. ``file_type`` takes
extensions (``pdf``, ``.docx``) and MIME types (``application/pdf``, ``image/*``).
``uploaded_after`` is inclusive, ``uploaded_before`` exclusive (a date or an ISO
date-time). ``file_ids`` names documents by id (a console "search inside this
document"). A metadata condition is a value (equal; strings match exactly), or an
object of operators: ``in`` (one of a list), ``gte`` / ``lte`` (a number, or a
``YYYY-MM-DD`` date compared with values of the same kind) and ``exists``. Unknown
fields and operators are refused (422) with their names.

The SQL is built here from bound parameters only: no key, value or name from the
filter is ever pasted into the statement text (llama-index's ``MetadataFilters`` for
PGVectorStore does paste them, so it is never used for user input). The caller's
owner scope is always added by :func:`build_where`, so a filter can narrow what a
person sees but never widen it.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.domain.rag.doc_meta import (
    DATE_RE,
    DocMetaError,
    check_meta_key,
    is_date_string,
    normalize_meta_value,
    normalize_tag,
    stored_metadata,
    stored_tags,
    tag_key,
)

META_OPERATORS = ("in", "gte", "lte", "exists")
MAX_FILTER_LIST = 100
MAX_NAME_CONTAINS = 200
_EXT_RE = re.compile(r"^[a-z0-9]{1,16}$")
_MIME_RE = re.compile(r"^[a-z0-9][a-z0-9.+-]*/(\*|[a-z0-9][a-z0-9.+_-]*)$")

SORTS = {"name": "lower(f.file_name)", "uploaded": "f.upload_time", "size": "f.file_size"}
ORDERS = {"asc": "ASC", "desc": "DESC"}
MAX_PAGE = 200


class _AllFolders:
    """Owner sentinel: no owner restriction (open mode only, where the console owns
    every folder)."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "ALL_FOLDERS"


ALL_FOLDERS = _AllFolders()
Owner = Union[str, _AllFolders]


@dataclass(frozen=True)
class MetaCondition:
    key: str
    op: str          # eq | in | gte | lte | exists
    value: Any


def _scalar(key: str, value: Any) -> Any:
    try:
        return normalize_meta_value(key, value)
    except DocMetaError as e:
        raise ValueError(str(e)) from None


def _parse_meta_conditions(raw: Dict[str, Any]) -> List[MetaCondition]:
    out: List[MetaCondition] = []
    for key, cond in raw.items():
        try:
            check_meta_key(key)
        except DocMetaError as e:
            raise ValueError(str(e)) from None
        if isinstance(cond, dict):
            if not cond:
                raise ValueError(f"metadata {key!r}: an empty condition; use a value or one of "
                                 + ", ".join(META_OPERATORS))
            unknown = sorted(set(cond) - set(META_OPERATORS))
            if unknown:
                raise ValueError(f"metadata {key!r}: unknown operator(s) {', '.join(unknown)}; "
                                 f"use {', '.join(META_OPERATORS)}")
            for op in META_OPERATORS:
                if op not in cond:
                    continue
                v = cond[op]
                if op == "in":
                    if not isinstance(v, list) or not v or len(v) > MAX_FILTER_LIST:
                        raise ValueError(f"metadata {key!r}: 'in' takes a list of 1-{MAX_FILTER_LIST} values")
                    out.append(MetaCondition(key, "in", tuple(_scalar(key, x) for x in v)))
                elif op == "exists":
                    if not isinstance(v, bool):
                        raise ValueError(f"metadata {key!r}: 'exists' takes true or false")
                    out.append(MetaCondition(key, "exists", v))
                else:
                    is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
                    if not is_num and not is_date_string(v):
                        raise ValueError(f"metadata {key!r}: {op!r} takes a number or a "
                                         "YYYY-MM-DD date")
                    out.append(MetaCondition(key, op, _scalar(key, v)))
        elif isinstance(cond, list):
            raise ValueError(f"metadata {key!r}: a list is not a value; use {{\"in\": [...]}}")
        else:
            out.append(MetaCondition(key, "eq", _scalar(key, cond)))
    return out


def _parse_instant(value: str, name: str) -> datetime:
    """A date or an ISO date-time; an aware one is converted to UTC (stored upload
    times are naive)."""
    try:
        if DATE_RE.match(value):
            return datetime.combine(date.fromisoformat(value), datetime.min.time())
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name}: {value!r} is not a date (YYYY-MM-DD) or an ISO date-time") from None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class DocumentFilter(BaseModel):
    """The filter object (see the module docstring). Every field is optional; an empty
    filter matches every document the caller can see."""

    model_config = ConfigDict(extra="forbid")

    tags_any: Optional[List[str]] = Field(None, max_length=MAX_FILTER_LIST,
                                          description="At least one of these tags (case-insensitive)")
    tags_all: Optional[List[str]] = Field(None, max_length=MAX_FILTER_LIST,
                                          description="All of these tags (case-insensitive)")
    folders: Optional[List[Union[int, str]]] = Field(
        None, max_length=MAX_FILTER_LIST, description="Folder names or ids (your own folders only)")
    file_type: Optional[List[str]] = Field(
        None, max_length=MAX_FILTER_LIST,
        description="Extensions (pdf, .docx) or MIME types (application/pdf, image/*)")
    uploaded_after: Optional[str] = Field(None, description="Inclusive; YYYY-MM-DD or ISO date-time")
    uploaded_before: Optional[str] = Field(None, description="Exclusive; YYYY-MM-DD or ISO date-time")
    name_contains: Optional[str] = Field(None, max_length=MAX_NAME_CONTAINS,
                                         description="Part of the file name (case-insensitive)")
    file_ids: Optional[List[str]] = Field(
        None, min_length=1, max_length=MAX_FILTER_LIST, description="These documents (file ids)")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description='{key: value | {"in": [...]} | {"gte": x} | {"lte": x} | {"exists": bool}}')

    @field_validator("tags_any", "tags_all")
    @classmethod
    def _tags(cls, v):
        if v is None:
            return v
        try:
            out = [normalize_tag(t) for t in v]
        except DocMetaError as e:
            raise ValueError(str(e)) from None
        return [t for t in out if t]

    @field_validator("file_type", mode="before")
    @classmethod
    def _file_type(cls, v):
        if isinstance(v, str):
            v = [v]
        if v is None:
            return v
        out = []
        for raw in v:
            if not isinstance(raw, str):
                raise ValueError("file_type entries must be strings")
            t = raw.strip().lower()
            if "/" in t:
                if not _MIME_RE.match(t):
                    raise ValueError(f"file_type {raw!r} is not a MIME type")
            else:
                t = t.lstrip(".")
                if not _EXT_RE.match(t):
                    raise ValueError(f"file_type {raw!r} is not a file extension")
            out.append(t)
        return out

    @field_validator("file_ids")
    @classmethod
    def _file_ids(cls, v):
        if v is None:
            return v
        out = []
        for raw in v:
            try:
                out.append(str(uuid.UUID(str(raw))))
            except ValueError:
                raise ValueError(f"file_ids {raw!r} is not a file id") from None
        return out

    @field_validator("uploaded_after", "uploaded_before")
    @classmethod
    def _instant(cls, v, info):
        if v is None:
            return v
        _parse_instant(v, info.field_name)
        return v

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, v):
        if v is None:
            return v
        if len(v) > MAX_FILTER_LIST:
            raise ValueError(f"at most {MAX_FILTER_LIST} metadata conditions")
        _parse_meta_conditions(v)
        return v

    def meta_conditions(self) -> List[MetaCondition]:
        return _parse_meta_conditions(self.metadata or {})

    def is_empty(self) -> bool:
        return not any(getattr(self, n) not in (None, [], {}, "") for n in type(self).model_fields)

    def fingerprint(self) -> str:
        """A canonical string of the filter (for cache keys)."""
        return json.dumps(self.model_dump(exclude_none=True), sort_keys=True, ensure_ascii=False,
                          default=str)


def parse_filter(raw: Any) -> Optional[DocumentFilter]:
    """A filter from untrusted JSON; None for None. Raises pydantic.ValidationError."""
    if raw is None:
        return None
    if isinstance(raw, DocumentFilter):
        return raw
    return DocumentFilter.model_validate(raw)


def filter_error_detail(exc: Exception) -> str:
    """A one-line 422 detail from a filter ValidationError, naming the field(s)."""
    errs = getattr(exc, "errors", None)
    if not callable(errs):
        return str(exc)
    parts = []
    for e in errs():
        loc = ".".join(str(x) for x in e.get("loc", ()))
        msg = e.get("msg", "")
        if e.get("type") == "extra_forbidden":
            msg = "unknown filter field"
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "invalid filter: " + "; ".join(parts)


# ------------------------------------------------------------------- SQL

_TAGS = ("jsonb_array_elements_text(CASE WHEN jsonb_typeof(f.tags) = 'array' "
         "THEN f.tags ELSE CAST('[]' AS jsonb) END)")
_META = "CASE WHEN jsonb_typeof(f.metadata) = 'object' THEN f.metadata ELSE CAST('{}' AS jsonb) END"


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass
class _Builder:
    clauses: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def param(self, prefix: str, value: Any) -> str:
        name = f"{prefix}_{len(self.params)}"
        self.params[name] = value
        return f":{name}"


def build_where(flt: Optional[DocumentFilter], *, owner: Owner,
                folder_id: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """(SQL condition over ``"Files" f``, bound parameters).

    ``owner`` is the caller's owner key (``sub:<sub>``), or :data:`ALL_FOLDERS` in open
    mode; it is always applied, whatever the filter says. ``folder_id`` limits the
    match to one folder (a search inside that folder).
    """
    b = _Builder()
    if isinstance(owner, _AllFolders):
        pass
    elif isinstance(owner, str) and owner:
        b.clauses.append('f.folder_id IN (SELECT d.id FROM "Folders" d WHERE d.user_token = '
                         + b.param("owner", owner) + ")")
    else:
        raise ValueError("an owner key (or ALL_FOLDERS) is required")
    if folder_id is not None:
        b.clauses.append("f.folder_id = " + b.param("folder", int(folder_id)))
    if flt is not None:
        _filter_clauses(flt, b)
    return (" AND ".join(b.clauses) or "TRUE"), b.params


def _filter_clauses(flt: DocumentFilter, b: _Builder) -> None:
    if flt.tags_any is not None:
        keys = sorted({tag_key(t) for t in flt.tags_any})
        b.clauses.append(f"EXISTS (SELECT 1 FROM {_TAGS} AS t(v) WHERE lower(t.v) = ANY("
                         + b.param("tags_any", keys) + "))")
    for t in sorted({tag_key(t) for t in (flt.tags_all or [])}):
        b.clauses.append(f"EXISTS (SELECT 1 FROM {_TAGS} AS t(v) WHERE lower(t.v) = "
                         + b.param("tag_all", t) + ")")
    if flt.folders is not None:
        ids = [int(x) for x in flt.folders if isinstance(x, int) and not isinstance(x, bool)]
        names = [x for x in flt.folders if isinstance(x, str)]
        b.clauses.append('f.folder_id IN (SELECT d.id FROM "Folders" d WHERE d.id = ANY('
                         + b.param("folder_ids", ids) + ") OR d.name = ANY("
                         + b.param("folder_names", names) + "))")
    if flt.file_type is not None:
        alts = []
        for t in flt.file_type:
            if t.endswith("/*"):
                alts.append("lower(f.mime_type) LIKE " + b.param("mime_prefix", _like_escape(t[:-1]) + "%")
                            + " ESCAPE '\\'")
            elif "/" in t:
                alts.append("lower(f.mime_type) = " + b.param("mime", t))
            else:
                alts.append("lower(f.file_name) LIKE " + b.param("ext", "%." + _like_escape(t))
                            + " ESCAPE '\\'")
        b.clauses.append("(" + (" OR ".join(alts) or "FALSE") + ")")
    if flt.uploaded_after is not None:
        b.clauses.append("f.upload_time >= " + b.param(
            "after", _parse_instant(flt.uploaded_after, "uploaded_after")))
    if flt.uploaded_before is not None:
        b.clauses.append("f.upload_time < " + b.param(
            "before", _parse_instant(flt.uploaded_before, "uploaded_before")))
    if flt.file_ids is not None:
        b.clauses.append("f.id = ANY(CAST(" + b.param("file_ids", list(flt.file_ids)) + " AS uuid[]))")
    if flt.name_contains:
        b.clauses.append("f.file_name ILIKE " + b.param(
            "name", "%" + _like_escape(flt.name_contains) + "%") + " ESCAPE '\\'")
    for c in flt.meta_conditions():
        b.clauses.append(_meta_clause(c, b))


def _meta_clause(c: MetaCondition, b: _Builder) -> str:
    if c.op == "eq":
        return "f.metadata @> CAST(" + b.param("meta", json.dumps({c.key: c.value})) + " AS jsonb)"
    if c.op == "in":
        alts = ["f.metadata @> CAST(" + b.param("meta", json.dumps({c.key: v})) + " AS jsonb)"
                for v in c.value]
        return "(" + " OR ".join(alts) + ")"
    if c.op == "exists":
        expr = f"jsonb_exists({_META}, " + b.param("meta_key", c.key) + ")"
        return expr if c.value else "NOT " + expr
    # gte / lte: a number against numbers, a date against YYYY-MM-DD strings.
    k = b.param("meta_key", c.key)
    cmp = ">=" if c.op == "gte" else "<="
    if isinstance(c.value, (int, float)) and not isinstance(c.value, bool):
        return (f"(CASE WHEN jsonb_typeof(f.metadata -> {k}) = 'number' "
                f"THEN CAST(f.metadata ->> {k} AS numeric) {cmp} " + b.param("meta_num", c.value)
                + " ELSE FALSE END)")
    return (f"(CASE WHEN jsonb_typeof(f.metadata -> {k}) = 'string' "
            f"AND (f.metadata ->> {k}) ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$' "
            f"THEN (f.metadata ->> {k}) COLLATE \"C\" {cmp} " + b.param("meta_date", c.value)
            + " ELSE FALSE END)")


# ------------------------------------------------------------ queries

def matching_file_ids(conn, flt: Optional[DocumentFilter], *, owner: Owner,
                      folder_id: Optional[int] = None) -> List[str]:
    """Ids (strings) of the documents matching ``flt`` within the owner scope."""
    from sqlalchemy import text

    where, params = build_where(flt, owner=owner, folder_id=folder_id)
    rows = conn.execute(text(f'SELECT f.id FROM "Files" f WHERE {where}'), params).fetchall()
    return [str(r[0]) for r in rows]


def resolve_file_ids(flt: Optional[DocumentFilter], *, owner: Owner,
                     folder_id: Optional[int] = None) -> Optional[Tuple[str, ...]]:
    """The file ids a search should be limited to: None when there is no filter (or
    an empty one), else the matching ids (possibly none). Blocking (sync engine)."""
    if flt is None or flt.is_empty():
        return None
    from db.db import get_engine

    with get_engine().connect() as conn:
        return tuple(sorted(matching_file_ids(conn, flt, owner=owner, folder_id=folder_id)))


def _iso(v) -> Optional[str]:
    return v.isoformat() if v is not None else None


def search_documents(conn, flt: Optional[DocumentFilter], *, owner: Owner, sort: str = "uploaded",
                     order: str = "desc", limit: int = 50, offset: int = 0
                     ) -> Tuple[List[Dict[str, Any]], int]:
    """One page of the caller's documents matching ``flt``, and the total count.

    Folder names are read live from ``Folders``; ``status`` is the index status
    (``unindexed`` without an index record).
    """
    from sqlalchemy import text

    if sort not in SORTS:
        raise ValueError(f"sort must be one of {', '.join(SORTS)}")
    if order not in ORDERS:
        raise ValueError("order must be asc or desc")
    limit = max(1, min(int(limit), MAX_PAGE))
    offset = max(0, int(offset))
    where, params = build_where(flt, owner=owner)
    total = conn.execute(text(f'SELECT count(*) FROM "Files" f WHERE {where}'), params).scalar() or 0
    rows = conn.execute(text(
        "SELECT f.id, f.file_name, f.file_size, f.mime_type, f.upload_time, f.updated_time, "
        "f.tags, f.metadata, f.description, d.id AS folder_id, d.name AS folder_name, "
        "fi.status AS index_status "
        'FROM "Files" f JOIN "Folders" d ON d.id = f.folder_id '
        'LEFT JOIN "FileIndices" fi ON fi.file_id = f.id '
        f"WHERE {where} ORDER BY {SORTS[sort]} {ORDERS[order]}, f.id {ORDERS[order]} "
        "LIMIT :_limit OFFSET :_offset"),
        {**params, "_limit": limit, "_offset": offset}).mappings().fetchall()
    items = [{
        "id": str(r["id"]),
        "name": r["file_name"],
        "folder": {"id": r["folder_id"], "name": r["folder_name"]},
        "size_bytes": int(r["file_size"] or 0),
        "mime": r["mime_type"],
        "uploaded_at": _iso(r["upload_time"]),
        "updated_at": _iso(r["updated_time"]),
        "status": r["index_status"] or "unindexed",
        "tags": stored_tags(r["tags"]),
        "metadata": stored_metadata(r["metadata"]),
        "description": r["description"],
    } for r in rows]
    return items, int(total)


def tag_counts(conn, *, owner: Owner) -> List[Dict[str, Any]]:
    """The caller's tags with document counts, grouped case-insensitively; each group
    is shown in its most common spelling (a tie goes to the first in byte order, the
    same on every database collation). Most used first."""
    from sqlalchemy import text

    where, params = build_where(None, owner=owner)
    rows = conn.execute(text(
        f"SELECT mode() WITHIN GROUP (ORDER BY t.v COLLATE \"C\") AS tag, count(DISTINCT f.id) AS n "
        f'FROM "Files" f, {_TAGS} AS t(v) WHERE {where} '
        "GROUP BY lower(t.v) ORDER BY n DESC, lower(t.v)"), params).fetchall()
    return [{"tag": r[0], "count": int(r[1])} for r in rows]


def metadata_keys(conn, *, owner: Owner) -> List[Dict[str, Any]]:
    """The caller's metadata keys: how many documents carry each, and the value types
    seen (string, number, boolean, date) with their counts."""
    from sqlalchemy import text

    where, params = build_where(None, owner=owner)
    rows = conn.execute(text(
        "SELECT e.key, CASE WHEN jsonb_typeof(e.value) = 'string' "
        "AND (e.value #>> '{}') ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN 'date' "
        "ELSE jsonb_typeof(e.value) END AS vtype, count(*) AS n "
        f'FROM "Files" f, jsonb_each({_META}) AS e WHERE {where} '
        "GROUP BY 1, 2 ORDER BY 1, 3 DESC"), params).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for key, vtype, n in rows:
        entry = out.setdefault(key, {"key": key, "count": 0, "types": {}})
        entry["count"] += int(n)
        entry["types"][vtype] = int(n)
    return sorted(out.values(), key=lambda e: (-e["count"], e["key"]))


def owner_scope(owner_key: Optional[str]) -> Owner:
    """An identity's owner key as a filter scope: None (open mode) → ALL_FOLDERS."""
    return ALL_FOLDERS if owner_key is None else owner_key


__all__: Sequence[str] = (
    "ALL_FOLDERS", "DocumentFilter", "MetaCondition", "build_where", "filter_error_detail",
    "matching_file_ids", "metadata_keys", "owner_scope", "parse_filter", "search_documents",
    "tag_counts",
)
