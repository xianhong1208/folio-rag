"""One retry policy for every model HTTP call that does not go through the OpenAI SDK.

Retries 408 / 409 / 425 / 429 / 5xx responses and transport errors (timeouts, refused or
reset connections) with exponential backoff and full jitter, honouring ``Retry-After``
(seconds or HTTP-date, capped). Anything else — 4xx auth/validation errors — returns
immediately: retrying a bad key or a wrong model name only multiplies the latency.

Request bodies must be re-sendable (bytes / dict / tuples of bytes), never open file
handles, because a retry sends them again.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional

import httpx

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    backoff_base: float = 0.5   # first backoff window (seconds); doubles per attempt
    backoff_max: float = 20.0   # cap of the exponential window
    retry_after_max: float = 60.0  # never sleep longer than this on a server's say-so


def retry_after_seconds(response: Optional[httpx.Response]) -> Optional[float]:
    """Parse ``Retry-After`` (delta-seconds or HTTP-date); None when absent/invalid."""
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def compute_delay(policy: RetryPolicy, attempt: int,
                  response: Optional[httpx.Response] = None,
                  rng: Callable[[], float] = random.random) -> float:
    """Delay before retry number ``attempt`` (0-based)."""
    hinted = retry_after_seconds(response)
    if hinted is not None:
        return min(hinted, policy.retry_after_max)
    window = min(policy.backoff_max, policy.backoff_base * (2 ** attempt))
    return rng() * window  # full jitter


def is_retryable_exception(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def request_with_retry(client: httpx.Client, method: str, url: str, *,
                       policy: RetryPolicy, sleep: Callable[[float], Any] = time.sleep,
                       on_retry: Optional[Callable[[int, str], None]] = None,
                       **kwargs) -> httpx.Response:
    """Send with retries; returns the last response (callers decide on raise_for_status)."""
    for attempt in range(policy.max_retries + 1):
        try:
            response = client.request(method, url, **kwargs)
        except Exception as exc:
            if not is_retryable_exception(exc) or attempt >= policy.max_retries:
                raise
            if on_retry:
                on_retry(attempt + 1, type(exc).__name__)
            sleep(compute_delay(policy, attempt))
            continue
        if response.status_code in RETRYABLE_STATUS and attempt < policy.max_retries:
            if on_retry:
                on_retry(attempt + 1, f"HTTP {response.status_code}")
            sleep(compute_delay(policy, attempt, response))
            continue
        return response
    raise AssertionError("unreachable")  # pragma: no cover


async def arequest_with_retry(client: httpx.AsyncClient, method: str, url: str, *,
                              policy: RetryPolicy, sleep: Callable[[float], Any] = asyncio.sleep,
                              on_retry: Optional[Callable[[int, str], None]] = None,
                              **kwargs) -> httpx.Response:
    """Async twin of request_with_retry."""
    for attempt in range(policy.max_retries + 1):
        try:
            response = await client.request(method, url, **kwargs)
        except Exception as exc:
            if not is_retryable_exception(exc) or attempt >= policy.max_retries:
                raise
            if on_retry:
                on_retry(attempt + 1, type(exc).__name__)
            await sleep(compute_delay(policy, attempt))
            continue
        if response.status_code in RETRYABLE_STATUS and attempt < policy.max_retries:
            if on_retry:
                on_retry(attempt + 1, f"HTTP {response.status_code}")
            await sleep(compute_delay(policy, attempt, response))
            continue
        return response
    raise AssertionError("unreachable")  # pragma: no cover


def join_endpoint(base_url: str, path: str) -> str:
    """``base_url`` + ``path`` without doubling the API version segment.

    Model base_urls are inconsistent about ``/v1`` (embedding/LLM ones usually include it,
    reranker ones usually don't), so ``join_endpoint("http://h/v1", "/v1/score")`` and
    ``join_endpoint("http://h", "/v1/score")`` both give ``http://h/v1/score``.
    """
    base = (base_url or "").rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    tail = base.rsplit("/", 1)[-1]
    head = path.split("/", 2)[1] if path.count("/") >= 2 else ""
    if tail and head and tail == head and tail[:1] == "v" and tail[1:].isdigit():
        path = path[len(head) + 1:]
    return base + path
