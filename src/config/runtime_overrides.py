
"""Runtime settings overrides — whitelist, validation, in-place mutation (the core of admin hot-reconfig).

Layering: config.yaml = factory defaults (read-only) -> the RuntimeSettings table = live overrides
-> the merged ConfigModel = the single source of truth every service reads.

Why in-place setattr rather than swapping in a fresh ConfigModel: consumers (context_generator
holds llm_config, sub-services hold ctx) hold object references — replacing the model would orphan
all old references, so mutating in place is what makes changes take effect everywhere.

Three whitelist levels:
- live: takes effect immediately (HTTP-endpoint services, request-time-read parameters)
- warn: takes effect but with an impact warning (embedding-family — vector space incompatible,
  existing folders must be reindexed)
- not listed: always rejected (infrastructure such as DB/port/auth; requires a restart)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Dotted path -> level. Any path not listed is rejected.
# Note: values like score_threshold are frozen into the Reranker instance, but rebind (rag_context)
# rebuilds that instance — here we only handle value validation and writing; rebind is a separate layer.
EDITABLE: Dict[str, str] = {
    # Model services — embedding (warn: changing the model = vector space incompatible, old folders need rebuilding)
    "rag.embedding.provider": "warn",
    "rag.embedding.model": "warn",
    "rag.embedding.dimension": "warn",
    "rag.embedding.base_url": "warn",
    "rag.embedding.api_key": "warn",
    "rag.embedding.query_prefix": "warn",
    "rag.embedding.passage_prefix": "warn",
    # Model services — LLM (contextual retrieval)
    "rag.llm.provider": "live",
    "rag.llm.model": "live",
    "rag.llm.base_url": "live",
    "rag.llm.api_key": "live",
    # Contextual retrieval behavior
    "rag.contextual_retrieval.enabled": "live",
    "rag.contextual_retrieval.max_context_length": "live",
    "rag.contextual_retrieval.max_concurrent": "live",
    "rag.contextual_retrieval.max_doc_chars": "live",
    "rag.contextual_retrieval.max_tokens": "live",
    "rag.contextual_retrieval.reasoning_effort": "live",
    # Model services — rerank
    "rag.rerank.enabled": "live",
    "rag.rerank.model": "live",
    "rag.rerank.base_url": "live",
    "rag.rerank.api_key": "live",
    "rag.rerank.top_n": "live",
    "rag.rerank.score_threshold": "live",
    "rag.rerank.query_template": "live",
    "rag.rerank.document_template": "live",
    # Model services — ASR
    "rag.asr.enabled": "live",
    "rag.asr.provider": "live",
    "rag.asr.base_url": "live",
    "rag.asr.api_key": "live",
    "rag.asr.model": "live",
    # Retrieval parameters (both REST/agentic read config at request time; ctx scalars synced via rebind)
    "rag.retrieval.default_top_k": "live",
    "rag.retrieval.default_similarity_cutoff": "live",
    "rag.retrieval.default_sparse_top_k": "live",
    "rag.retrieval.default_hybrid_alpha": "live",
    "rag.retrieval.hybrid_fusion": "live",
    "rag.retrieval.expand_context_default": "live",
    "rag.retrieval.expand_context_neighbors": "live",
    "rag.retrieval.auto_merging.enabled": "live",
    "rag.retrieval.auto_merging.merge_threshold": "live",
}

# Fields to mask on display (GET returns "•••" as a placeholder; PUT writes are unaffected)
SECRET_PATHS = {p for p in EDITABLE if p.endswith("api_key")}

# Path prefix -> rebind group (dispatch key for rag_context.apply_runtime_changes)
REBIND_GROUPS = ("rag.embedding", "rag.llm", "rag.contextual_retrieval",
                 "rag.rerank", "rag.asr", "rag.retrieval")


def _resolve(config_model, path: str, create: bool = True):
    """Dotted path -> (parent object, field name).

    create=True: intermediate None optional segments are built up with default values (for the
    write path); create=False: the read path — must have no side effects, so a None segment returns
    (None, field).
    """
    parts = path.split(".")
    obj = config_model
    for i, part in enumerate(parts[:-1]):
        nxt = getattr(obj, part)
        if nxt is None:
            if not create:
                return None, parts[-1]
            # optional segment (e.g. rag.asr unset) -> build it from the field type's default
            field = type(obj).model_fields[part]
            ann = field.annotation
            # Optional[X] → X
            import typing
            args = typing.get_args(ann)
            target = next((a for a in args if a is not type(None)), ann) if args else ann
            nxt = target()
            setattr(obj, part, nxt)
        obj = nxt
    return obj, parts[-1]


def get_effective(config_model, mask_secrets: bool = True) -> Dict[str, Any]:
    """Current effective values for all whitelisted paths (secrets masked)."""
    out: Dict[str, Any] = {}
    for path in EDITABLE:
        try:
            parent, field = _resolve(config_model, path, create=False)
            v = getattr(parent, field) if parent is not None else None
        except Exception:
            v = None
        if mask_secrets and path in SECRET_PATHS and v:
            v = "•••"
        out[path] = v
    return out


def validate_and_apply(config_model, patch: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate the patch, then write it into the ConfigModel in place.

    All-or-nothing: any single validation failure -> ValueError, and nothing is written.

    Returns:
        (applied, warnings) — the actually-written {path: value} and warnings for warn-level paths.

    Raises:
        ValueError: Path not whitelisted / type validation failed (message includes details).
    """
    unknown = [p for p in patch if p not in EDITABLE]
    if unknown:
        raise ValueError(f"Setting path cannot be changed at runtime: {unknown} (see GET /api/admin/settings for the allow-list)")

    # Validate everything first (pydantic: apply into a section copy to check types); write only if all pass
    staged = []  # [(parent, field, validated_value, path)]
    errors = []
    for path, value in patch.items():
        try:
            parent, field = _resolve(config_model, path)
            section_cls = type(parent)
            data = parent.model_dump()
            data[field] = value
            validated = section_cls.model_validate(data)  # type/constraint validation
            staged.append((parent, field, getattr(validated, field), path))
        except ValueError as e:
            errors.append(f"{path}: {e}")
    if errors:
        raise ValueError("Settings validation failed: " + "; ".join(errors))

    applied: Dict[str, Any] = {}
    warnings: List[str] = []
    for parent, field, value, path in staged:
        setattr(parent, field, value)  # in place — consumers holding references see it immediately
        applied[path] = value
        if EDITABLE[path] == "warn":
            warnings.append(
                f"{path}: embedding change — vector space/dimension incompatible; "
                "existing folders must be reindexed before retrieval works. New indexes use the new setting immediately."
            )
    return applied, warnings


def rebind_groups_for(paths) -> List[str]:
    """A batch of applied paths -> the rebind groups to trigger (deduplicated, in a fixed order)."""
    hit = set()
    for p in paths:
        for g in REBIND_GROUPS:
            if p.startswith(g + "."):
                hit.add(g)
    return [g for g in REBIND_GROUPS if g in hit]
