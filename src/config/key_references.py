"""Which ``${VAR}`` key references a runtime settings change may use, and for which server.

A model key written as ``${VAR}`` is read from the server's environment. config.yaml
(the startup config) is the operator's statement of which variables hold model keys
and which servers they belong to. A runtime change (the console's Settings page,
``PUT /api/admin/settings``, a stored override) may therefore only

- reference a variable that config.yaml itself uses as some model role's ``api_key``,
- and only while that role's server is one config.yaml pairs the variable with.

The server is the *origin* of the role's effective ``base_url``: scheme, host and port
(a role without a ``base_url`` uses its provider's default endpoint, recorded as
``default:<provider>``). LLM roles inherit ``base_url`` and ``api_key`` from ``rag.llm``.

Pure config logic: no I/O, nothing here reads the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Set
from urllib.parse import urlsplit

from src.config.roles import LLM_ROLES

# One generic answer for any reference that is not allowed: unknown, unset, or not paired
# with this server. It must not reveal whether a variable exists in the environment.
NOT_ALLOWED = ("this ${...} reference is not allowed here: use a variable that config.yaml "
               "uses as this server's api_key, or type the key itself")

_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True)
class RoleKey:
    """Where one role sends its key: the server origin and the key field it comes from."""
    role: str
    origin: Optional[str]          # None when the role is not configured
    key_path: Optional[str]        # api_key field the effective key is read from (None = no key)
    key: Optional[str]             # the effective key value


def origin_of(base_url: Optional[str], provider: Optional[str] = None) -> str:
    """``scheme://host:port`` of a base_url (default ports made explicit, case folded);
    ``default:<provider>`` when no base_url is set (the provider's own endpoint)."""
    if not base_url or not str(base_url).strip():
        return f"default:{(provider or '').strip().lower()}"
    parts = urlsplit(str(base_url).strip())
    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()
    try:
        port = parts.port or _DEFAULT_PORTS.get(scheme)
    except ValueError:
        port = None
    return f"{scheme}://{host}:{port}"


def _sec(rag, name):
    return getattr(rag, name, None) if rag is not None else None


def role_keys(rag) -> Dict[str, RoleKey]:
    """The effective (origin, key source) of every model role that can carry a key."""
    out: Dict[str, RoleKey] = {}

    def plain(role, sec, path_prefix):
        if sec is None:
            out[role] = RoleKey(role, None, None, None)
            return
        key = getattr(sec, "api_key", None)
        out[role] = RoleKey(role, origin_of(getattr(sec, "base_url", None), getattr(sec, "provider", None)),
                            f"{path_prefix}.api_key" if key else None, key or None)

    plain("embedding", _sec(rag, "embedding"), "rag.embedding")
    llm = _sec(rag, "llm")
    plain("llm", llm, "rag.llm")
    for role in LLM_ROLES:
        own = _sec(rag, role)
        if own is None and llm is None:
            out[role] = RoleKey(role, None, None, None)
            continue
        base_url = (getattr(own, "base_url", None) if own is not None else None) or \
            (getattr(llm, "base_url", None) if llm is not None else None)
        provider = (getattr(own, "provider", None) if own is not None else None) or \
            (getattr(llm, "provider", None) if llm is not None else None)
        own_key = getattr(own, "api_key", None) if own is not None else None
        if own_key:
            key, path = own_key, f"rag.{role}.api_key"
        else:
            key = getattr(llm, "api_key", None) if llm is not None else None
            path = "rag.llm.api_key" if key else None
        out[role] = RoleKey(role, origin_of(base_url, provider), path, key or None)
    plain("rerank", _sec(rag, "rerank"), "rag.rerank")
    plain("asr", _sec(rag, "asr"), "rag.asr")
    plain("vision", _sec(_sec(rag, "parsing"), "vision"), "rag.parsing.vision")
    return out


def paired_origins(startup_rag, refs: Mapping[str, str]) -> Dict[str, Set[str]]:
    """Variable -> the origins config.yaml sends it to.

    ``refs`` maps a secret path to the variable config.yaml wrote there (Config's
    pre-expansion references). Only api_key references count.
    """
    pairs: Dict[str, Set[str]] = {}
    for rk in role_keys(startup_rag).values():
        if rk.key_path and rk.origin is not None:
            var = refs.get(rk.key_path)
            if var:
                pairs.setdefault(var, set()).add(rk.origin)
    return pairs


def reference_allowed(var: Optional[str], origin: Optional[str],
                      pairs: Mapping[str, Iterable[str]]) -> bool:
    """True when ``var`` is a config.yaml key variable paired with ``origin``."""
    return bool(var) and origin is not None and origin in set(pairs.get(var, ()))


def key_variables(pairs: Mapping[str, Iterable[str]]) -> Set[str]:
    return set(pairs)
