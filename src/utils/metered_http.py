"""httpx clients that count and time every request to a model endpoint.

``metered_client(role)`` returns an ordinary ``httpx.Client`` (``metered_async_client``:
``httpx.AsyncClient``) whose ``send`` records ``folio_model_requests_total{role, outcome}``
and ``folio_model_request_duration_seconds{role}``. Every request method of httpx, and the
OpenAI SDK (which takes it as ``http_client``), goes through ``send``, so retries are
counted one by one. Overriding ``send`` rather than passing a custom transport keeps
httpx's proxy settings from the environment (a ``transport=`` argument turns them off).

``openai=True`` builds on the OpenAI SDK's ``DefaultHttpxClient`` so the SDK keeps its own
connection limits, timeout and redirect defaults.
"""

from __future__ import annotations

import functools
import time

import httpx

from src import metrics


class _SyncMeter:
    _folio_role = metrics.OTHER

    def send(self, request, **kwargs):
        started = time.perf_counter()
        try:
            response = super().send(request, **kwargs)
        except Exception:
            metrics.observe_model_call(self._folio_role, None, time.perf_counter() - started)
            raise
        metrics.observe_model_call(self._folio_role, response.status_code, time.perf_counter() - started)
        return response


class _AsyncMeter:
    _folio_role = metrics.OTHER

    async def send(self, request, **kwargs):
        started = time.perf_counter()
        try:
            response = await super().send(request, **kwargs)
        except Exception:
            metrics.observe_model_call(self._folio_role, None, time.perf_counter() - started)
            raise
        metrics.observe_model_call(self._folio_role, response.status_code, time.perf_counter() - started)
        return response


@functools.lru_cache(maxsize=None)
def _metered_class(base: type) -> type:
    meter = _AsyncMeter if issubclass(base, httpx.AsyncClient) else _SyncMeter
    return type(f"Metered{base.__name__}", (meter, base), {})


def _base(async_: bool, openai: bool) -> type:
    if openai:
        from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

        return DefaultAsyncHttpxClient if async_ else DefaultHttpxClient
    return httpx.AsyncClient if async_ else httpx.Client


def metered_client(role: str, *, openai: bool = False, **kwargs) -> httpx.Client:
    """A sync httpx client recording model-call metrics under ``role``."""
    client = _metered_class(_base(False, openai))(**kwargs)
    client._folio_role = role
    return client


def metered_async_client(role: str, *, openai: bool = False, **kwargs) -> httpx.AsyncClient:
    """An async httpx client recording model-call metrics under ``role``."""
    client = _metered_class(_base(True, openai))(**kwargs)
    client._folio_role = role
    return client


def openai_sdk_clients(role: str) -> dict:
    """``http_client`` / ``async_http_client`` keyword arguments for a llama-index OpenAI
    model (e.g. ``OpenAILikeEmbedding``), both metered under ``role``."""
    return {"http_client": metered_client(role, openai=True),
            "async_http_client": metered_async_client(role, openai=True)}
