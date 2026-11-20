"""Tags and metadata on documents: the rules every write path applies.

A document (a ``Files`` row) carries

- **tags**: a list of short labels. One namespace per person, across all of their
  folders. Each tag is trimmed and its inner whitespace collapsed to one space; at
  most :data:`MAX_TAG_LEN` characters and :data:`MAX_TAGS` per document. Duplicates
  that differ only in case are dropped (the first spelling is kept), and tags are
  matched case-insensitively everywhere.
- **metadata**: a flat ``{key: value}`` object. Keys match ``^[a-z0-9_]{1,40}$``; at
  most :data:`MAX_META_KEYS` keys. A value is a string (at most
  :data:`MAX_META_STR` characters), a number, a boolean, or a date written
  ``YYYY-MM-DD`` (stored as that string, so dates compare as strings and as dates
  alike). Lists and objects are refused: no nesting.

Every write (upload, PATCH, console edit, folder import) goes through
:func:`normalize_tags` / :func:`normalize_metadata`; a value they refuse raises
:class:`DocMetaError`, whose message is meant for the caller (a 422 detail).
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

MAX_TAG_LEN = 64
MAX_TAGS = 32
MAX_META_KEYS = 32
MAX_META_STR = 500
META_KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class DocMetaError(ValueError):
    """A tag or metadata value that breaks the rules; the message names it."""


def tag_key(tag: str) -> str:
    """The form tags are compared in (PostgreSQL ``lower()`` does the same in SQL)."""
    return tag.lower()


def normalize_tag(tag: Any) -> str:
    """One tag, trimmed with inner whitespace collapsed ('' when nothing is left)."""
    if not isinstance(tag, str):
        raise DocMetaError(f"a tag must be a string, not {type(tag).__name__}")
    out = " ".join(tag.split())
    if len(out) > MAX_TAG_LEN:
        raise DocMetaError(f"tag {out[:20]!r}… is longer than {MAX_TAG_LEN} characters")
    return out


def normalize_tags(tags: Optional[Iterable[Any]]) -> List[str]:
    """A document's tags: normalised, empty ones dropped, case-insensitive duplicates
    dropped (first spelling kept), at most MAX_TAGS."""
    if tags is None:
        return []
    if isinstance(tags, (str, bytes, dict)):
        raise DocMetaError("tags must be a list of strings")
    out: List[str] = []
    seen = set()
    for raw in tags:
        t = normalize_tag(raw)
        if not t or tag_key(t) in seen:
            continue
        seen.add(tag_key(t))
        out.append(t)
    if len(out) > MAX_TAGS:
        raise DocMetaError(f"a document can have at most {MAX_TAGS} tags (got {len(out)})")
    return out


def parse_tag_string(raw: Optional[str]) -> List[str]:
    """Tags from a form field: comma-separated (``"invoice, 2024"``)."""
    if raw is None:
        return []
    return normalize_tags(raw.split(","))


def stored_tags(value: Any) -> List[str]:
    """Tags as read from the database: anything that is not a list of strings (JSON
    ``null`` from uploads before 1.6) reads as no tags."""
    if not isinstance(value, list):
        return []
    return [t for t in value if isinstance(t, str)]


def stored_metadata(value: Any) -> Dict[str, Any]:
    """Metadata as read from the database ({} for anything that is not an object)."""
    return dict(value) if isinstance(value, dict) else {}


def is_date_string(value: Any) -> bool:
    """True for a real calendar date written YYYY-MM-DD."""
    if not isinstance(value, str) or not DATE_RE.match(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def check_meta_key(key: Any) -> str:
    if not isinstance(key, str) or not META_KEY_RE.match(key):
        raise DocMetaError(
            f"metadata key {str(key)[:50]!r} is not allowed: use 1-40 lowercase letters, "
            "digits or underscores")
    return key


def normalize_meta_value(key: str, value: Any) -> Any:
    """One metadata value: a string (trimmed, at most MAX_META_STR), a finite number,
    a boolean or a YYYY-MM-DD date. Lists, objects and null are refused."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise DocMetaError(f"metadata {key!r}: numbers must be finite")
        return value
    if isinstance(value, str):
        v = value.strip()
        if DATE_RE.match(v) and not is_date_string(v):
            raise DocMetaError(f"metadata {key!r}: {v!r} is not a valid date (YYYY-MM-DD)")
        if len(v) > MAX_META_STR:
            raise DocMetaError(f"metadata {key!r}: longer than {MAX_META_STR} characters")
        return v
    if isinstance(value, (dict, list, tuple)):
        raise DocMetaError(
            f"metadata {key!r}: nested values are not allowed; use a string, number, "
            "boolean or YYYY-MM-DD date")
    if value is None:
        raise DocMetaError(f"metadata {key!r}: null is not a value; leave the key out to remove it")
    raise DocMetaError(f"metadata {key!r}: unsupported value type {type(value).__name__}")


def normalize_metadata(meta: Any) -> Dict[str, Any]:
    """A document's metadata object, validated (see the module docstring)."""
    if meta is None:
        return {}
    if not isinstance(meta, dict):
        raise DocMetaError("metadata must be an object of key: value pairs")
    if len(meta) > MAX_META_KEYS:
        raise DocMetaError(f"a document can have at most {MAX_META_KEYS} metadata keys (got {len(meta)})")
    return {check_meta_key(k): normalize_meta_value(k, v) for k, v in meta.items()}


def parse_metadata_json(raw: Optional[str]) -> Dict[str, Any]:
    """Metadata from a form field holding a JSON object ('' or None = none)."""
    if raw is None or not raw.strip():
        return {}
    import json

    try:
        data = json.loads(raw)
    except ValueError as e:
        raise DocMetaError(f"metadata is not valid JSON: {e.msg}") from None
    return normalize_metadata(data)


def value_type(value: Any) -> str:
    """The type a stored metadata value is listed as: string, number, boolean or date."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if is_date_string(value):
        return "date"
    return "string"
