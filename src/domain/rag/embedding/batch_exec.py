
"""Embedding batch execution — retry/backoff.

Transient errors (connection/5xx/rate-limit/timeout) retry with 1→2→4s backoff; permanent errors raise immediately.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from src.log import get_api_logger

logger = get_api_logger()

_EMBEDDING_BATCH_SIZE = 50

_EMBED_RETRY_ATTEMPTS = 3
_EMBED_RETRY_BASE_DELAY = 1.0  # seconds; doubles per attempt
_EMBED_RETRYABLE_PATTERNS = (
    "timeout", "timed out",
    "connection", "connectionerror", "remotedisconnected",
    "503", "502", "504", "429", "service unavailable", "bad gateway",
    "rate limit", "too many requests", "server error",
    "network", "broken pipe", "reset by peer",
)


def _is_retryable_embedding_error(exc: BaseException) -> bool:
    """Return True if the exception looks transient (retryable), matching class name
    and message against known patterns. The embedding client may be httpx / requests /
    urllib3 / a llama_index wrapper, so isinstance() is unreliable.
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(p in name or p in msg for p in _EMBED_RETRYABLE_PATTERNS)


async def _embed_batch_with_retry(
    embed_model, batch_texts, file_name: str,
    batch_start: int, batch_end: int, n_leaves: int,
):
    """Run get_text_embedding_batch, retrying transient errors with exponential backoff.

    Permanent errors re-raise immediately. Retries use async sleep so the event loop
    isn't blocked. Raises the final attempt's exception when exhausted or non-retryable.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(_EMBED_RETRY_ATTEMPTS + 1):
        try:
            # Offload the sync HTTP call to a thread so it doesn't block the event loop:
            # a 50-chunk batch can take seconds against vLLM, and a stalled heartbeat
            # would make the watchdog wrongly declare the job dead.
            result = await asyncio.to_thread(
                embed_model.get_text_embedding_batch, batch_texts
            )
            # A short result (partial failure behind HTTP 200) would let the caller's zip
            # silently drop the tail → NULL embeddings stored, those chunks unretrievable.
            # Require an exact length match; a mismatch raises (and is retryable).
            if len(result) != len(batch_texts):
                raise ValueError(
                    f"Embedding batch size mismatch for {file_name}: sent "
                    f"{len(batch_texts)} texts, got {len(result)} embeddings "
                    f"(batch [{batch_start}-{batch_end}))"
                )
            return result
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_embedding_error(exc):
                raise
            if attempt >= _EMBED_RETRY_ATTEMPTS:
                raise
            delay = _EMBED_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                f"Embedding batch [{batch_start}-{batch_end}) of {n_leaves} "
                f"({file_name}) transient failure ({type(exc).__name__}: {exc}); "
                f"retry {attempt + 1}/{_EMBED_RETRY_ATTEMPTS} after {delay:.1f}s"
            )
            await asyncio.sleep(delay)
    # Defensive — should be unreachable because the loop above re-raises.
    if last_exc:
        raise last_exc
    raise RuntimeError("embedding retry loop exited without result")
