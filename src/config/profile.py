"""Deployment-profile validation (``rag.profile: local | hybrid | cloud``).

The profile sets defaults (see RAGConfig._apply_profile_defaults and roles.default_segmenter)
and validates the per-role choices against them:

* ``local``  — everything is allowed (the pre-profile behaviour; existing configs are
  implicitly ``local`` and validate exactly as before).
* ``hybrid`` — everything is allowed; API keys written literally in config/models.yaml are
  reported as warnings, because a hybrid deployment usually holds at least one billable
  cloud key.
* ``cloud``  — every role must be an HTTP endpoint. In-process providers (HuggingFace
  embedding, the CKIP segmenter) are errors, and so is any API key that is not an
  ``${ENV_VAR}`` reference.
  The FTS segmenter defaults to the model-free ``bigram`` there.

Why reject (not warn) literal keys in ``cloud``: a cloud key is a billable credential
and config/models.yaml is the file people copy, paste into issues and commit. ``local`` keeps
accepting literal values because self-hosted vLLM/Ollama endpoints take placeholder keys
(such as "not-needed") that are not secrets. Errors fail startup (main.py exits) and
reject the offending console change (HTTP 422); they never degrade silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from src.config.roles import (
    IN_PROCESS_EMBEDDING,
    LLM_ROLES,
    pdf_parsing,
    vision_model,
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
    from one written into the config files after expansion has erased the difference."""
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
            (from config/models.yaml or a console-stored reference). None = treat as unknown, so
            the literal-key rule is skipped (used where the source cannot be known).
    """
    report = ProfileReport()
    if rag_config is None:
        return report
    profile = getattr(rag_config, "profile", "local")

    # PDFs and images need no model in any profile (the text layer, or a refusal for images).
    # Forcing the vision model without one cannot work, so it is refused.
    if pdf_parsing(rag_config) == "vision" and vision_model(rag_config) is None:
        report.errors.append(
            "rag.parsing.pdf is 'vision' but rag.parsing.vision is not configured — set its "
            "model and base_url (env VISION_MODEL, VISION_BASE_URL) to an OpenAI-compatible vision "
            "model, or set rag.parsing.pdf to auto (text layer until a vision model is set)")

    if profile == "cloud":
        emb = getattr(rag_config, "embedding", None)
        if emb is not None and emb.provider in IN_PROCESS_EMBEDDING:
            report.errors.append(
                f"rag.embedding.provider '{emb.provider}' loads the model in-process, which "
                "profile: cloud forbids — use openai / openai-compatible / azure / vllm / ollama")
        retrieval = getattr(rag_config, "retrieval", None)
        seg = getattr(retrieval, "segmenter", None) if retrieval is not None else None
        if seg == "ckip":
            report.errors.append(
                "rag.retrieval.segmenter: ckip is an in-process torch model, which profile: "
                "cloud never loads — use bigram (the cloud default, no dependency) or jieba "
                "(`uv sync --extra jieba`), or keep CKIP with profile: hybrid and "
                "`uv sync --extra local`")

    if refs is not None and profile in ("cloud", "hybrid"):
        for path in literal_secret_paths(rag_config, refs):
            msg = (f"{path} is a literal value — reference an environment variable instead, "
                   f"e.g. api_key: \"${{OPENAI_API_KEY}}\"")
            (report.errors if profile == "cloud" else report.warnings).append(msg)

    return report
