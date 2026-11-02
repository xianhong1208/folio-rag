"""Normalisation of user-supplied folder names and upload file names.

Folder and file names land in ``String(255)`` columns, in MCP tool names and in
HTTP headers. Before this module an over-long name reached the database and came
back as a 500 carrying SQL details, a macOS (NFD) name such as "café" failed the
safe-name check because its combining accent is not a word character, and an
upload's raw ``filename`` (which may carry a client path or control characters)
was stored as-is.
"""

from __future__ import annotations

import os
import unicodedata
from urllib.parse import quote

# Matches the String(255) columns (Folders.name, Files.file_name).
MAX_NAME_LEN = 255


def normalize_name(name: str) -> str:
    """NFC-normalise and trim a folder name (no validation beyond that)."""
    return unicodedata.normalize("NFC", name or "").strip()


def _strip_controls(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch) not in ("Cc", "Cf"))


def sanitize_filename(raw: str | None, fallback: str = "file") -> str:
    """A storable display name for an uploaded file.

    Drops any client-side directory part (both separators), control and format
    characters, NFC-normalises, and truncates to MAX_NAME_LEN characters while
    keeping the extension (it decides the parser). Never returns an empty string.
    """
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _strip_controls(unicodedata.normalize("NFC", name)).strip()
    if not name or name in (".", ".."):
        name = fallback
    if len(name) > MAX_NAME_LEN:
        stem, ext = os.path.splitext(name)
        if len(ext) > 16:  # not a real extension
            stem, ext = name, ""
        name = stem[: MAX_NAME_LEN - len(ext)] + ext
    return name


def content_disposition(filename: str, disposition: str = "attachment") -> str:
    """Content-Disposition value safe for any file name (RFC 6266 / RFC 5987).

    The quoted ASCII fallback has quotes, backslashes and control characters
    replaced, so a name like ``a"b.pdf`` cannot break out of the quoted string;
    the exact name travels in ``filename*``.
    """
    clean = _strip_controls(filename or "") or "download"
    ascii_name = clean.encode("ascii", "ignore").decode("ascii")
    ascii_name = "".join("_" if ch in '"\\' else ch for ch in ascii_name).strip() or "download"
    return (f'{disposition}; filename="{ascii_name}"; '
            f"filename*=UTF-8''{quote(clean, safe='')}")
