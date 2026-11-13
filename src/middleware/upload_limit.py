"""Refuse an oversized single-file upload while it streams in, not after.

The multipart parser spools every file part to a temporary file before the endpoint
runs, so without this guard a 5 GB request is read to the end before the 100 MB check
can refuse it. For the single-file upload routes this guard:

- answers 413 at once when ``Content-Length`` already exceeds the limit (nothing of the
  body is read), and
- counts the body as it arrives (chunked requests have no ``Content-Length``) and
  aborts with 413 as soon as it crosses the limit; the partial spooled part is a
  temporary file that is discarded with the request.

The limit is the file limit plus a small allowance for the multipart framing and the
form's text fields (each at most 1 MB in the parser). A file that slips under the
allowance is still refused by the per-file check, with the same status and message.
The batch route is not capped here (its body legitimately holds several files); each of
its files is checked on its own.
"""

from __future__ import annotations

import json
import re

from fastapi import HTTPException

from db.filedb import FileDB, file_too_large_message

# POST routes that carry exactly one file: REST `/folders/{id}/file` and the console's
# `/api/admin/manage/folders/{id}/files`. (REST `/folders/{id}/files` is the batch route.)
_SINGLE_UPLOAD = re.compile(r"(?:/admin/manage/folders/\d+/files|/folders/\d+/file)/?$")

# Multipart boundaries, part headers and the description/tags/auto_index fields.
FORM_ALLOWANCE = 4 * 1024 * 1024


def body_limit() -> int:
    return FileDB.MAX_FILE_SIZE + FORM_ALLOWANCE


class UploadSizeLimit:
    """Pure-ASGI guard for the single-file upload routes (assembled in app.py)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http" or scope.get("method") != "POST"
                or not _SINGLE_UPLOAD.search(scope.get("path", ""))):
            await self.app(scope, receive, send)
            return

        limit = body_limit()
        message = file_too_large_message(FileDB.MAX_FILE_SIZE)
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            await _send_413(send, message)
            return

        received = 0

        async def counting_receive():
            nonlocal received
            event = await receive()
            if event.get("type") == "http.request":
                received += len(event.get("body", b""))
                if received > limit:
                    # FastAPI re-raises an HTTPException from body parsing as-is, so the
                    # client gets the same 413 the per-file check gives.
                    raise HTTPException(status_code=413, detail=message)
            return event

        await self.app(scope, counting_receive, send)


def _content_length(scope):
    for name, value in scope.get("headers") or ():
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_413(send, message: str) -> None:
    body = json.dumps({"detail": message}, separators=(",", ":")).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"connection", b"close"),
        ],
    })
    await send({"type": "http.response.body", "body": body})
