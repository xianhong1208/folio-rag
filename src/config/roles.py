"""Model roles — one place that answers "which model serves this job, and where does it run".

A *role* is a job the server needs a model for. Every role is configured with the same
shape (provider / base_url / model / api_key / timeout / max_retries), so a deployment can
point each one at a different OpenAI-compatible endpoint:

    embedding       rag.embedding
    llm             rag.llm                                (answers, evaluation, and the default)
    contextual_llm  rag.contextual_llm  → inherits rag.llm   (contextual-retrieval prefixes:
                                                             one short sentence per chunk)
    rerank          rag.rerank
    vision          rag.parsing.vision                     (cloud PDF / image parsing)
    asr             rag.asr                                (speech-to-text)

Pure config logic: nothing here imports a model library or makes a request.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config.model import LLMRoleConfig, ParsingConfig, RAGLLMConfig

# Optional per-role overrides of rag.llm (answer_llm / eval_llm were removed in 1.5).
LLM_ROLES = ("contextual_llm",)

# Providers that load weights into this process (everything else is an HTTP endpoint).
IN_PROCESS_EMBEDDING = frozenset({"huggingface", "local"})
IN_PROCESS_ASR = frozenset({"docling-whisper", "fireredasr"})

ROLE_LABELS = {
    "embedding": "Embedding",
    "contextual_llm": "Contextual-prefix LLM",
    "llm": "LLM (answers, evaluation)",
    "rerank": "Reranker",
    "vision": "Document parsing (vision)",
    "asr": "Speech-to-text",
}


def resolve_llm(rag_config, role: str) -> Optional[RAGLLMConfig]:
    """The effective LLM config for ``role``: ``rag.<role>`` fields over ``rag.llm``
    (``role="llm"``: ``rag.llm`` itself, used for answers and evaluation).

    Returns None when neither is configured. A role override without a base ``rag.llm``
    is still usable on its own (it is merged over the RAGLLMConfig defaults).
    """
    if role == "llm":
        return getattr(rag_config, "llm", None)
    if role not in LLM_ROLES:
        raise ValueError(f"Unknown LLM role: {role!r} (expected one of {LLM_ROLES})")
    base: Optional[RAGLLMConfig] = getattr(rag_config, "llm", None)
    override: Optional[LLMRoleConfig] = getattr(rag_config, role, None)
    if override is None:
        return base
    data = base.model_dump() if base is not None else RAGLLMConfig().model_dump()
    data.update({k: v for k, v in override.model_dump().items() if v is not None})
    return RAGLLMConfig.model_validate(data)


# rag.retrieval.segmenter when unset: CKIP is an in-process torch model, which the cloud
# profile never loads, so cloud deployments default to the model-free bigram segmenter.
DEFAULT_SEGMENTER = "ckip"
CLOUD_DEFAULT_SEGMENTER = "bigram"


def default_segmenter(profile: Optional[str]) -> str:
    return CLOUD_DEFAULT_SEGMENTER if profile == "cloud" else DEFAULT_SEGMENTER


def effective_segmenter(rag_config) -> str:
    """The FTS segmenter in force: ``rag.retrieval.segmenter`` or the profile default."""
    retrieval = getattr(rag_config, "retrieval", None)
    explicit = getattr(retrieval, "segmenter", None) if retrieval is not None else None
    if explicit:
        return str(explicit).strip().lower()
    return default_segmenter(getattr(rag_config, "profile", "local"))


def parsing_mode(rag_config, kind: str) -> str:
    """``"local"`` or ``"cloud"`` for ``kind`` in {"pdf", "images"}; unset follows the profile."""
    if kind not in ("pdf", "images"):
        raise ValueError(f"Unknown parsing kind: {kind!r}")
    parsing: Optional[ParsingConfig] = getattr(rag_config, "parsing", None)
    explicit = getattr(parsing, kind, None) if parsing is not None else None
    if explicit:
        return explicit
    return "cloud" if getattr(rag_config, "profile", "local") == "cloud" else "local"


def uses_cloud_parsing(rag_config) -> bool:
    return any(parsing_mode(rag_config, k) == "cloud" for k in ("pdf", "images"))


def needs_local_docling_models(rag_config) -> bool:
    """Whether the in-process docling PDF/image models (layout, TableFormer, OCR) are used."""
    return any(parsing_mode(rag_config, k) == "local" for k in ("pdf", "images"))


def asr_runs_in_process(rag_config) -> bool:
    asr = getattr(rag_config, "asr", None)
    if asr is None:  # legacy default: local docling-whisper when weights are present
        return getattr(rag_config, "profile", "local") != "cloud"
    return bool(asr.enabled and asr.provider in IN_PROCESS_ASR)


def docling_warmup_needed(rag_config) -> bool:
    """Whether startup should pre-load docling's models: only when PDFs/images are parsed
    locally or local docling-whisper transcribes audio. Office/Markdown/HTML need no models."""
    if rag_config is None:
        return True  # unknown config: keep the historical behaviour
    asr = getattr(rag_config, "asr", None)
    local_whisper = asr_runs_in_process(rag_config) and (asr is None or asr.provider == "docling-whisper")
    return needs_local_docling_models(rag_config) or local_whisper


def _endpoint(provider, model, base_url, enabled=True, in_process=False) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "runs": "off" if not enabled else ("in-process" if in_process else "endpoint"),
        "provider": provider,
        "model": model,
        "base_url": base_url,
    }


def role_summary(rag_config) -> List[Dict[str, Any]]:
    """One entry per role: where it runs and which model serves it (never secrets).

    ``runs`` is ``endpoint`` (HTTP API), ``in-process`` (weights loaded here) or ``off``.
    ``secret_path`` names the api_key field that the console's key indicator reads.
    """
    out: List[Dict[str, Any]] = []

    emb = getattr(rag_config, "embedding", None)
    if emb is not None:
        entry = _endpoint(emb.provider, emb.model, emb.base_url,
                          in_process=emb.provider in IN_PROCESS_EMBEDDING)
        out.append({"role": "embedding", "secret_path": "rag.embedding.api_key", **entry})

    cr = getattr(rag_config, "contextual_retrieval", None)
    llm = getattr(rag_config, "llm", None)
    if llm is None:
        entry = _endpoint(None, None, None, enabled=False)
    else:
        entry = _endpoint(llm.provider, llm.azure_deployment or llm.model, llm.base_url)
    out.append({"role": "llm", "secret_path": "rag.llm.api_key", **entry})

    role_enabled = {"contextual_llm": bool(cr and cr.enabled)}
    for role in LLM_ROLES:
        llm = resolve_llm(rag_config, role)
        own = getattr(rag_config, role, None) is not None
        if llm is None:
            entry = _endpoint(None, None, None, enabled=False)
        else:
            entry = _endpoint(llm.provider, llm.azure_deployment or llm.model, llm.base_url,
                              enabled=role_enabled[role])
        secret = f"rag.{role}.api_key" if own and getattr(rag_config, role).api_key else "rag.llm.api_key"
        out.append({"role": role, "secret_path": secret, "inherits": None if own else "llm", **entry})

    rr = getattr(rag_config, "rerank", None)
    entry = _endpoint(getattr(rr, "api", None), getattr(rr, "model", None),
                      getattr(rr, "base_url", None), enabled=bool(rr and rr.enabled))
    out.append({"role": "rerank", "secret_path": "rag.rerank.api_key", **entry})

    parsing = getattr(rag_config, "parsing", None)
    vision = getattr(parsing, "vision", None) if parsing is not None else None
    modes = {k: parsing_mode(rag_config, k) for k in ("pdf", "images")}
    if uses_cloud_parsing(rag_config):
        entry = _endpoint(getattr(vision, "provider", None), getattr(vision, "model", None),
                          getattr(vision, "base_url", None))
    else:
        entry = _endpoint("docling", "layout + TableFormer + OCR", None, in_process=True)
    out.append({"role": "vision", "secret_path": "rag.parsing.vision.api_key",
                "modes": modes, **entry})

    asr = getattr(rag_config, "asr", None)
    if asr is None:
        entry = _endpoint("docling-whisper", None, None,
                          enabled=getattr(rag_config, "profile", "local") != "cloud", in_process=True)
    else:
        entry = _endpoint(asr.provider, asr.model, asr.base_url, enabled=asr.enabled,
                          in_process=asr.provider in IN_PROCESS_ASR)
    out.append({"role": "asr", "secret_path": "rag.asr.api_key", **entry})

    for e in out:
        e["label"] = ROLE_LABELS[e["role"]]
    return out


def startup_lines(rag_config) -> List[str]:
    """Human-readable "role → where it runs" lines for the startup log."""
    lines = [f"Deployment profile: {getattr(rag_config, 'profile', 'local')}"]
    for e in role_summary(rag_config):
        if e["runs"] == "off":
            lines.append(f"  {e['label']:<28} off")
            continue
        where = "in-process" if e["runs"] == "in-process" else f"endpoint {e['base_url'] or '(provider default)'}"
        extra = ""
        if e["role"] == "vision":
            extra = f" [pdf={e['modes']['pdf']}, images={e['modes']['images']}]"
        lines.append(f"  {e['label']:<28} {where} — {e['provider']} / {e['model']}{extra}")
    retrieval = getattr(rag_config, "retrieval", None)
    if (retrieval is not None and getattr(retrieval, "hybrid_search", False)
            and getattr(retrieval, "text_search_config", "simple") == "simple"):
        seg = effective_segmenter(rag_config)
        where = "in-process model" if seg == "ckip" else "in-process, no model"
        lines.append(f"  {'FTS segmenter':<28} {where} — {seg}")
    return lines
