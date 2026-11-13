"""Advance notice: the built-in OCR, speech-to-text and the ``/v1`` media endpoints go in 1.5.

1.4.2 removes nothing. It says so in four places:

- a startup log line when the configuration uses any of them;
- a warning logged once per process on the first in-process OCR or speech-to-text use
  and on the first ``/v1`` media call;
- ``Deprecation`` (RFC 9745) and ``Link: rel="deprecation"`` headers on every ``/v1``
  media response, errors included (no ``Sunset`` header: 1.5 has no release date yet);
- the README, the docs and the CHANGELOG.

What replaces them: an OpenAI-compatible vision model for scanned PDFs and images
(``rag.parsing.pdf/images: cloud`` with ``rag.parsing.vision``), and an OpenAI-compatible
transcription endpoint for audio (``rag.asr.provider: openai-compatible``), for example
both served by Forgeyard.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import List

from src.log import get_api_logger

logger = get_api_logger()

REMOVED_IN = "1.5"
# When the deprecation was announced (Folio 1.4.2): RFC 9745 wants a date, as "@<epoch>".
DEPRECATED_SINCE = datetime(2026, 11, 13, tzinfo=timezone.utc)
DEPRECATION_HEADER = f"@{int(DEPRECATED_SINCE.timestamp())}"
NOTICE_URL = ("https://github.com/xianhong1208/folio-rag/blob/main/docs/api.md"
              "#removed-in-15-built-in-ocr-speech-to-text-and-the-v1-media-endpoints")
LINK_HEADER = f'<{NOTICE_URL}>; rel="deprecation"; type="text/html"'

REPLACE_OCR = ("read scanned PDFs and images with an OpenAI-compatible vision model instead "
               "(rag.parsing.pdf / rag.parsing.images: cloud, with rag.parsing.vision; "
               "e.g. Qwen3-VL served by Forgeyard)")
REPLACE_ASR = ("send audio to an OpenAI-compatible transcription endpoint instead "
               "(rag.asr.provider: openai-compatible with base_url; e.g. Forgeyard's)")
REPLACE_MEDIA = ("call the vision model or transcription endpoint directly instead "
                 "(any OpenAI-compatible server, e.g. Forgeyard)")

MESSAGES = {
    "ocr": f"The built-in OCR is deprecated and will be removed in {REMOVED_IN}: {REPLACE_OCR}.",
    "asr": f"The built-in speech-to-text is deprecated and will be removed in {REMOVED_IN}: "
           f"{REPLACE_ASR}.",
    "media": f"The /v1 media endpoints (/v1/ocr/general, /v1/transcriptions/general, /v1/models) "
             f"are deprecated and will be removed in {REMOVED_IN}: {REPLACE_MEDIA}.",
}

_warned: set = set()
_lock = threading.Lock()


def warn_once(key: str) -> bool:
    """Log the ``key`` notice once per process. Returns True when it was logged now."""
    with _lock:
        if key in _warned:
            return False
        _warned.add(key)
    logger.warning(f"[DEPRECATION] {MESSAGES[key]} See {NOTICE_URL}")
    return True


def _reset_for_tests() -> None:
    with _lock:
        _warned.clear()


def deprecated_in_use(config) -> List[str]:
    """Which of the deprecated features this configuration uses (``ocr``, ``asr``, ``media``)."""
    from src.config.roles import asr_runs_in_process, needs_local_docling_models
    used: List[str] = []
    rag = getattr(config, "rag", None)
    if rag is not None:
        docling = getattr(rag, "docling", None)
        if needs_local_docling_models(rag) and bool(getattr(docling, "ocr_enabled", True)):
            used.append("ocr")
        if asr_runs_in_process(rag):
            used.append("asr")
    modules = getattr(getattr(config, "modules", None), "enabled", None) or []
    if "media" in modules:
        used.append("media")
    return used


def log_startup_notice(config) -> List[str]:
    """One startup line per deprecated feature the configuration uses; returns their keys."""
    try:
        used = deprecated_in_use(config)
    except Exception as e:  # noqa: BLE001 - a notice must never block startup
        logger.debug(f"[DEPRECATION] startup check skipped: {e}")
        return []
    for key in used:
        logger.warning(f"[DEPRECATION] This configuration uses it: {MESSAGES[key]}")
    return used


class MediaDeprecationMiddleware:
    """Pure-ASGI: marks every ``/v1`` media response as deprecated and logs the first call."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "") if scope.get("type") == "http" else ""
        if not (path == "/v1" or path.startswith("/v1/")):
            await self.app(scope, receive, send)
            return
        warn_once("media")

        async def send_with_headers(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"deprecation", DEPRECATION_HEADER.encode()))
                headers.append((b"link", LINK_HEADER.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)
