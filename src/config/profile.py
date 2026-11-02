"""Deployment-profile validation (``rag.profile: local | hybrid | cloud``).

The profile sets defaults (see RAGConfig._apply_profile_defaults and roles.parsing_mode)
and validates the per-role choices against them:

* ``local``  — everything is allowed (the pre-profile behaviour; existing configs are
  implicitly ``local`` and validate exactly as before).
* ``hybrid`` — everything is allowed; API keys written literally in config.yaml are
  reported as warnings, because a hybrid deployment usually holds at least one billable
  cloud key.
* ``cloud``  — every role must be an HTTP endpoint. In-process providers (HuggingFace
  embedding, docling-whisper / FireRedASR, local docling PDF/image parsing) are errors,
  and so is any API key that is not an ``${ENV_VAR}`` reference.

Why reject (not warn) literal keys in ``cloud``: a cloud key is a billable credential
and config.yaml is the file people copy, paste into issues and commit. ``local`` keeps
accepting literal values because self-hosted vLLM/Ollama endpoints take placeholder keys
("not-needed", "not-needed") that are not secrets. Errors fail startup (main.py exits) and
reject the offending console change (HTTP 422); they never degrade silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from src.config.roles import (
    IN_PROCESS_ASR,
    IN_PROCESS_EMBEDDING,
    LLM_ROLES,
    parsing_mode,
)

# ${VAR} or ${VAR:-default} anywhere in a string (same syntax as config_manager).
ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}")
# A value that is *only* a reference — what the console accepts for secrets.
WHOLE_ENV_REF_RE = re.compile(r"^\s*\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}\s*$")


@dataclass
class ProfileReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def env_refs(raw: Any, prefix: str = "") -> Dict[str, str]:
    """Map dotted path → first referenced env-var name, for every raw (pre-expansion) string
    value that contains ``${VAR}``. Lets the server tell a key that came from the environment
    from one written into config.yaml after expansion has erased the difference."""
    out: Dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out.update(env_refs(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(raw, str):
        m = ENV_REF_RE.search(raw)
        if m:
            out[prefix] = m.group(1)
    return out


def iter_secret_fields(rag_config) -> Iterator[Tuple[str, Optional[str]]]:
    """Yield ``(dotted path, value)`` for every api_key field present in the rag config."""
    sections = ["embedding", "llm", *LLM_ROLES, "rerank", "asr"]
    for name in sections:
        sec = getattr(rag_config, name, None)
        if sec is not None and hasattr(sec, "api_key"):
            yield f"rag.{name}.api_key", sec.api_key
    parsing = getattr(rag_config, "parsing", None)
    vision = getattr(parsing, "vision", None) if parsing is not None else None
    if vision is not None:
        yield "rag.parsing.vision.api_key", vision.api_key


def literal_secret_paths(rag_config, refs: Mapping[str, str]) -> List[str]:
    """Secret paths whose value is set but did not come from an ``${ENV}`` reference."""
    return [p for p, v in iter_secret_fields(rag_config) if v and p not in refs]


def check_profile(rag_config, refs: Optional[Mapping[str, str]] = None) -> ProfileReport:
    """Validate ``rag_config`` against its profile.

    Args:
        rag_config: a RAGConfig (or a deep copy with pending runtime changes applied).
        refs: secret path → env-var name for keys that came from ``${ENV}`` references
            (from config.yaml or a console-stored reference). None = treat as unknown, so
            the literal-key rule is skipped (used where the source cannot be known).
    """
    report = ProfileReport()
    if rag_config is None:
        return report
    profile = getattr(rag_config, "profile", "local")

    # Cloud parsing needs a vision model in every profile (it was asked for explicitly or
    # by the cloud profile's default).
    parsing = getattr(rag_config, "parsing", None)
    vision = getattr(parsing, "vision", None) if parsing is not None else None
    cloud_kinds = [k for k in ("pdf", "images") if parsing_mode(rag_config, k) == "cloud"]
    if cloud_kinds and not (vision and vision.model):
        report.errors.append(
            f"rag.parsing: {'/'.join(cloud_kinds)} parsing is 'cloud' but rag.parsing.vision.model "
            "is not set — configure an OpenAI-compatible vision model (provider, base_url, model, "
            "api_key), or set rag.parsing.pdf/images: local"
            + (" (not allowed in profile: cloud)" if profile == "cloud" else ""))

    if profile == "cloud":
        emb = getattr(rag_config, "embedding", None)
        if emb is not None and emb.provider in IN_PROCESS_EMBEDDING:
            report.errors.append(
                f"rag.embedding.provider '{emb.provider}' loads the model in-process, which "
                "profile: cloud forbids — use openai / openai-compatible / azure / vllm / ollama")
        asr = getattr(rag_config, "asr", None)
        if asr is not None and asr.enabled and asr.provider in IN_PROCESS_ASR:
            report.errors.append(
                f"rag.asr.provider '{asr.provider}' runs locally, which profile: cloud forbids — "
                "use openai-compatible, or set rag.asr.enabled: false")
        for kind in ("pdf", "images"):
            if parsing_mode(rag_config, kind) == "local":
                report.errors.append(
                    f"rag.parsing.{kind}: local uses in-process docling models, which "
                    "profile: cloud forbids — remove it (cloud is the default) or set it to cloud")
        retrieval = getattr(rag_config, "retrieval", None)
        seg = getattr(retrieval, "segmenter", None) if retrieval is not None else None
        if (retrieval is not None and getattr(retrieval, "hybrid_search", False)
                and getattr(retrieval, "text_search_config", "simple") == "simple"
                and (seg is None or seg == "ckip")):
            report.warnings.append(
                "rag.retrieval: Chinese BM25 segmentation defaults to the CKIP model, an in-process "
                "torch model that profile: cloud does not load, so Chinese text is indexed "
                "unsegmented — set rag.retrieval.segmenter: bigram for model-free segmentation")

    if refs is not None and profile in ("cloud", "hybrid"):
        for path in literal_secret_paths(rag_config, refs):
            msg = (f"{path} is a literal value — reference an environment variable instead, "
                   f"e.g. api_key: \"${{OPENAI_API_KEY}}\"")
            (report.errors if profile == "cloud" else report.warnings).append(msg)

    return report
