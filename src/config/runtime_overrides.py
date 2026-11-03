
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

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

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
    "rag.embedding.request_dimensions": "warn",
    "rag.embedding.timeout": "live",
    "rag.embedding.max_retries": "live",
    # Model services — default LLM (every LLM role inherits unset fields from it)
    "rag.llm.provider": "live",
    "rag.llm.model": "live",
    "rag.llm.base_url": "live",
    "rag.llm.api_key": "live",
    "rag.llm.timeout": "live",
    "rag.llm.max_retries": "live",
    # Per-role LLM overrides (empty = inherit rag.llm)
    **{f"rag.{role}.{f}": "live"
       for role in ("contextual_llm", "answer_llm", "eval_llm")
       for f in ("provider", "model", "base_url", "api_key")},
    # Contextual retrieval behavior
    "rag.contextual_retrieval.enabled": "live",
    "rag.contextual_retrieval.max_context_length": "live",
    "rag.contextual_retrieval.max_concurrent": "live",
    "rag.contextual_retrieval.max_doc_chars": "live",
    "rag.contextual_retrieval.max_tokens": "live",
    "rag.contextual_retrieval.reasoning_effort": "live",
    "rag.contextual_retrieval.max_calls_per_file": "live",
    "rag.contextual_retrieval.max_tokens_per_day": "live",
    # Model services — rerank
    "rag.rerank.enabled": "live",
    "rag.rerank.model": "live",
    "rag.rerank.base_url": "live",
    "rag.rerank.api_key": "live",
    "rag.rerank.top_n": "live",
    "rag.rerank.score_threshold": "live",
    "rag.rerank.query_template": "live",
    "rag.rerank.document_template": "live",
    "rag.rerank.auto_template": "live",
    "rag.rerank.gap_threshold": "live",
    "rag.rerank.api": "live",
    "rag.rerank.endpoint_path": "live",
    "rag.rerank.timeout": "live",
    "rag.rerank.max_retries": "live",
    # Model services — ASR
    "rag.asr.enabled": "live",
    "rag.asr.provider": "live",
    "rag.asr.base_url": "live",
    "rag.asr.api_key": "live",
    "rag.asr.model": "live",
    "rag.asr.language": "live",
    "rag.asr.segment_seconds": "live",
    "rag.asr.timeout": "live",
    "rag.asr.max_retries": "live",
    # Document parsing (read per file, so changes apply to the next indexed file)
    "rag.parsing.pdf": "live",
    "rag.parsing.images": "live",
    "rag.parsing.vision.provider": "live",
    "rag.parsing.vision.model": "live",
    "rag.parsing.vision.base_url": "live",
    "rag.parsing.vision.api_key": "live",
    "rag.parsing.max_concurrency": "live",
    "rag.parsing.max_pages_per_file": "live",
    "rag.parsing.max_pages_per_day": "live",
    "rag.parsing.on_page_failure": "live",
    # Retrieval parameters (both REST and MCP queries read config at request time; ctx scalars synced via rebind)
    "rag.retrieval.default_top_k": "live",
    "rag.retrieval.default_similarity_cutoff": "live",
    "rag.retrieval.default_sparse_top_k": "live",
    "rag.retrieval.default_hybrid_alpha": "live",
    "rag.retrieval.hybrid_fusion": "live",
    "rag.retrieval.segmenter": "live",
    "rag.retrieval.expand_context_default": "live",
    "rag.retrieval.expand_context_neighbors": "live",
    "rag.retrieval.auto_merging.enabled": "live",
    "rag.retrieval.auto_merging.merge_threshold": "live",
}

# Value constraints for the editable paths, checked on every write through the settings
# API (the config models themselves only check types). Without them max_concurrent=0
# built Semaphore(0) and hung indexing, score_threshold=5 silently emptied every query,
# and an unknown provider only failed at the next model rebuild.
# Numeric: (min, max) inclusive; None is allowed only where listed in _NULLABLE.
_RANGES: Dict[str, Tuple[float, float]] = {
    "rag.embedding.dimension": (1, 16384),
    "rag.contextual_retrieval.max_context_length": (1, 10_000),
    "rag.contextual_retrieval.max_concurrent": (1, 64),
    "rag.contextual_retrieval.max_doc_chars": (1_000, 10_000_000),
    "rag.contextual_retrieval.max_tokens": (1, 131_072),
    "rag.rerank.top_n": (1, 1_000),
    # Rerank scores from the /v1/score API are probabilities; the console slider is 0–1 too.
    "rag.rerank.score_threshold": (0.0, 1.0),
    "rag.rerank.gap_threshold": (0.0, 1.0),
    # 20 matches the REST QueryRequest.top_k ceiling.
    "rag.retrieval.default_top_k": (1, 20),
    "rag.retrieval.default_similarity_cutoff": (0.0, 1.0),
    "rag.retrieval.default_sparse_top_k": (1, 200),
    "rag.retrieval.default_hybrid_alpha": (0.0, 1.0),
    "rag.retrieval.expand_context_neighbors": (0, 10),
    "rag.retrieval.auto_merging.merge_threshold": (0.0, 1.0),
    # Per-role HTTP behaviour and cost guards (bounds match the config models).
    "rag.embedding.timeout": (0.1, 3600),
    "rag.embedding.max_retries": (0, 10),
    "rag.llm.timeout": (0.1, 3600),
    "rag.llm.max_retries": (0, 10),
    "rag.contextual_retrieval.max_calls_per_file": (0, 10_000_000),
    "rag.contextual_retrieval.max_tokens_per_day": (0, 10_000_000_000),
    "rag.rerank.timeout": (0.1, 600),
    "rag.rerank.max_retries": (0, 5),
    "rag.asr.segment_seconds": (30, 780),
    "rag.asr.timeout": (0.1, 3600),
    "rag.asr.max_retries": (0, 10),
    "rag.parsing.max_concurrency": (1, 32),
    "rag.parsing.max_pages_per_file": (0, 1_000_000),
    "rag.parsing.max_pages_per_day": (0, 100_000_000),
}
_LLM_ROLES = ("contextual_llm", "answer_llm", "eval_llm")
_NULLABLE = {"rag.rerank.top_n", "rag.contextual_retrieval.reasoning_effort",
             # per-role LLM overrides: empty = inherit rag.llm
             *(f"rag.{r}.{f}" for r in _LLM_ROLES for f in ("provider", "model", "base_url")),
             "rag.rerank.endpoint_path", "rag.asr.language",
             # empty = follow the profile (cloud profile -> cloud parsing)
             "rag.parsing.pdf", "rag.parsing.images",
             # unset = the profile's default (ckip; bigram under profile: cloud)
             "rag.retrieval.segmenter"}

# Choice lists. Providers are lower-cased and trimmed before the check (the factories
# compare lower-case names). Embedding providers come from the factory itself.
LLM_PROVIDERS = ("azure", "openai", "openai-compatible", "vllm", "ollama")
ASR_PROVIDERS = ("docling-whisper", "openai-compatible", "fireredasr")
# Vision models are called through the OpenAI-compatible chat API (azure: its own client).
VISION_PROVIDERS = ("openai-compatible", "openai", "vllm", "ollama", "azure")
RERANK_APIS = ("score", "cohere", "jina", "generic")
PARSING_MODES = ("local", "cloud")
_LOWERCASE_PATHS = ("rag.embedding.provider", "rag.llm.provider", "rag.asr.provider",
                    *(f"rag.{r}.provider" for r in _LLM_ROLES),
                    "rag.parsing.vision.provider", "rag.rerank.api",
                    "rag.parsing.pdf", "rag.parsing.images", "rag.parsing.on_page_failure",
                    "rag.retrieval.segmenter")


def _segmenters():
    from src.domain.rag.text_segmenter import SEGMENTERS
    return SEGMENTERS


def _allowed_values(path: str):
    if path == "rag.embedding.provider":
        from src.domain.rag.embedding.factory import EmbeddingProviderFactory
        return tuple(EmbeddingProviderFactory.list_supported_providers())
    return {
        "rag.llm.provider": LLM_PROVIDERS,
        **{f"rag.{r}.provider": LLM_PROVIDERS for r in _LLM_ROLES},
        "rag.asr.provider": ASR_PROVIDERS,
        "rag.parsing.vision.provider": VISION_PROVIDERS,
        "rag.rerank.api": RERANK_APIS,
        "rag.parsing.pdf": PARSING_MODES,
        "rag.parsing.images": PARSING_MODES,
        "rag.parsing.on_page_failure": ("mark", "fail"),
        "rag.retrieval.hybrid_fusion": ("rrf", "concat"),
        "rag.retrieval.segmenter": _segmenters(),
        "rag.contextual_retrieval.reasoning_effort": ("low", "medium", "high"),
    }.get(path)


def _check_constraints(path: str, value: Any) -> Any:
    """Normalise and range/choice-check one value; returns the value to write.

    Raises ValueError with a readable reason. Type coercion is left to pydantic.
    """
    if path in _LOWERCASE_PATHS and isinstance(value, str):
        value = value.strip().lower()
    if value is None or (value == "" and path in _NULLABLE):
        if path in _NULLABLE:
            return None
        if path in _RANGES or _allowed_values(path):
            raise ValueError("a value is required")
        return value
    choices = _allowed_values(path)
    if choices and value not in choices:
        raise ValueError(f"must be one of {', '.join(choices)} (got {value!r})")
    if path in _RANGES:
        lo, hi = _RANGES[path]
        try:
            num = float(value)
        except (TypeError, ValueError):
            return value  # not a number: let pydantic report the type error
        if not lo <= num <= hi:
            raise ValueError(f"must be between {lo:g} and {hi:g} (got {value!r})")
    if path.endswith(".base_url") and isinstance(value, str) and value.strip():
        if not value.strip().lower().startswith(("http://", "https://")):
            raise ValueError(f"must be an http(s):// URL (got {value!r})")
    if path == "rag.rerank.endpoint_path" and isinstance(value, str) and not value.startswith("/"):
        raise ValueError(f"must start with / (got {value!r})")
    if path.endswith(("_template",)) and isinstance(value, str) and "{text}" not in value:
        raise ValueError("must contain the {text} placeholder")
    if path.endswith((".model", ".provider")) and isinstance(value, str) and not value.strip():
        raise ValueError("must not be empty")
    return value


# Fields to mask on display (GET returns "•••" as a placeholder; PUT writes are unaffected)
SECRET_PATHS = {p for p in EDITABLE if p.endswith("api_key")}
MASK = "•••"

# Path prefix -> rebind group (dispatch key for rag_context.apply_runtime_changes).
# answer_llm / eval_llm / parsing are read per request or per file, so they need no rebind.
REBIND_GROUPS = ("rag.embedding", "rag.llm", "rag.contextual_llm", "rag.contextual_retrieval",
                 "rag.rerank", "rag.asr", "rag.retrieval")

# Secret path -> env-var name (or None for a literal) for keys written through the console.
# Console writes store the ${VAR} reference in the DB, never the resolved key.
_secret_source: Dict[str, Optional[str]] = {}


def secret_ref(path: str) -> Optional[str]:
    """Env-var name a secret currently comes from (console write first, then config.yaml)."""
    if path in _secret_source:
        return _secret_source[path]
    from src.config.config_manager import Config
    return Config.get_env_refs().get(path)


def _effective_refs(pending: Dict[str, Optional[str]]) -> Dict[str, str]:
    from src.config.config_manager import Config
    refs = dict(Config.get_env_refs())
    for path, ref in {**_secret_source, **pending}.items():
        if ref:
            refs[path] = ref
        else:
            refs.pop(path, None)
    return refs


def _resolve_secret(path: str, value: Any) -> Tuple[Any, Optional[str]]:
    """``${VAR}`` → (resolved key, "VAR"); literal → (value, None). Rejects the mask."""
    if not isinstance(value, str) or not value:
        return value, None
    if value.strip() == MASK:
        raise ValueError("send a new key or an ${ENV_VAR} reference, not the masked placeholder")
    from src.config.profile import WHOLE_ENV_REF_RE
    m = WHOLE_ENV_REF_RE.match(value)
    if not m:
        return value, None
    from src.config.config_manager import expand_env_vars
    return expand_env_vars(value.strip()), m.group(1)


def _resolve_parent(config_model, path: str, create: bool = True):
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
            parent, field = _resolve_parent(config_model, path, create=False)
            v = getattr(parent, field) if parent is not None else None
        except Exception:
            v = None
        if mask_secrets and path in SECRET_PATHS and v:
            v = MASK
        out[path] = v
    return out


def validate_and_apply(config_model, patch: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate the patch, then write it into the ConfigModel in place.

    All-or-nothing: any single validation failure -> ValueError, and nothing is written.

    Returns:
        (applied, warnings) — the actually-written {path: value} and warnings for warn-level paths.

    Raises:
        ValueError: Path not whitelisted / type, range or choice validation failed
            (message includes details).
    """
    unknown = [p for p in patch if p not in EDITABLE]
    if unknown:
        raise ValueError(
            f"Setting path(s) cannot be changed at runtime: {unknown} "
            "(see GET /api/admin/settings for the editable list)")

    # Validate everything first (pydantic: apply into a section copy to check types); write only if all pass
    staged = []  # [(parent, field, validated_value, path, as_written)]
    pending_refs: Dict[str, Optional[str]] = {}
    errors = []
    for path, value in patch.items():
        try:
            value = _check_constraints(path, value)
            written = value
            if path in SECRET_PATHS:
                value, pending_refs[path] = _resolve_secret(path, value)
            parent, field = _resolve_parent(config_model, path)
            section_cls = type(parent)
            data = parent.model_dump()
            data[field] = value
            validated = section_cls.model_validate(data)  # type/constraint validation
            staged.append((parent, field, getattr(validated, field), path, written))
        except ValueError as e:
            errors.append(f"{path}: {e}")
    if errors:
        raise ValueError("Settings validation failed: " + "; ".join(errors))

    # The result must still satisfy rag.profile (e.g. no in-process provider and no literal
    # key under profile: cloud). Checked on a copy so a rejected batch changes nothing.
    rag = getattr(config_model, "rag", None)
    if rag is not None and hasattr(rag, "model_copy"):
        from src.config.profile import check_profile
        trial = SimpleNamespace(rag=rag.model_copy(deep=True))
        for _parent, field, value, path, _w in staged:
            t_parent, t_field = _resolve_parent(trial, path)
            setattr(t_parent, t_field, value)
        report = check_profile(trial.rag, refs=_effective_refs(pending_refs))
        if report.errors:
            raise ValueError(f"rag.profile '{rag.profile}' rejects this change: "
                             + "; ".join(report.errors))

    applied: Dict[str, Any] = {}
    warnings: List[str] = []
    for parent, field, value, path, written in staged:
        setattr(parent, field, value)  # in place — consumers holding references see it immediately
        if path in pending_refs:
            _secret_source[path] = pending_refs[path]
            value = written if pending_refs[path] else value  # persist the reference, not the key
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
