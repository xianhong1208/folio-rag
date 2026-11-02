"""OpenAI-compatible chat client factory shared by every LLM role.

One factory instead of a per-call-site switch, so every role gets the same rules:

* ``openai`` honours ``base_url`` (it used to be dropped, which forced users to
  label an OpenAI-compatible cloud "vllm" just to set the endpoint). Empty base_url means
  the official OpenAI endpoint.
* ``openai-compatible`` / ``vllm`` / ``ollama`` require ``base_url`` and accept a missing
  key (self-hosted servers usually ignore it).
* ``azure`` uses the Azure client (``base_url`` is the resource endpoint).
* ``timeout`` and ``max_retries`` come from the role config. The OpenAI SDK retries
  429 / 408 / 409 / 5xx and connection errors with exponential backoff and honours
  ``Retry-After``, so no second retry layer is stacked on top.

Configs may be pydantic models or SimpleNamespace stand-ins, hence the getattr defaults.
"""

from __future__ import annotations

import os
from typing import Any, Optional

OPENAI_COMPATIBLE_PROVIDERS = ("openai", "openai-compatible", "vllm", "ollama")
SUPPORTED_PROVIDERS = (*OPENAI_COMPATIBLE_PROVIDERS, "azure")

DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 2


def _tuning(cfg) -> dict:
    timeout = getattr(cfg, "timeout", None) or DEFAULT_TIMEOUT
    retries = getattr(cfg, "max_retries", None)
    return {"timeout": float(timeout),
            "max_retries": DEFAULT_MAX_RETRIES if retries is None else int(retries)}


def create_openai_client(cfg: Any):
    """Build a sync OpenAI (or AzureOpenAI) client for an LLM / vision role config."""
    from openai import OpenAI

    provider = (getattr(cfg, "provider", None) or "").lower().strip()
    base_url = getattr(cfg, "base_url", None) or None
    api_key = getattr(cfg, "api_key", None) or None

    if provider == "azure":
        from openai import AzureOpenAI

        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Azure LLM api_key is required")
        endpoint = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("Azure endpoint (base_url) is required")
        return AzureOpenAI(
            api_key=api_key,
            api_version=getattr(cfg, "api_version", None) or "2024-12-01-preview",
            azure_endpoint=endpoint,
            **_tuning(cfg),
        )

    if provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI api_key is required (set api_key: \"${OPENAI_API_KEY}\")")
        return OpenAI(api_key=api_key, base_url=base_url, **_tuning(cfg))

    if provider in ("openai-compatible", "vllm", "ollama"):
        if not base_url:
            raise ValueError(f"{provider} requires base_url (e.g., http://localhost:8000/v1)")
        return OpenAI(api_key=api_key or "not-needed", base_url=base_url, **_tuning(cfg))

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def model_name(cfg: Any) -> Optional[str]:
    """The model id to send: the Azure deployment name when set, else ``model``."""
    return getattr(cfg, "azure_deployment", None) or getattr(cfg, "model", None)


def role_llm(role: str):
    """The effective config for an LLM role from the live Config, or raise if unset."""
    from src.config.config_manager import Config
    from src.config.roles import resolve_llm

    rag = getattr(Config.get_config_model(), "rag", None)
    cfg = resolve_llm(rag, role) if rag is not None else None
    if cfg is None:
        raise ValueError(f"No LLM configured for {role} (set rag.llm or rag.{role})")
    return cfg
