
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
    "rag.retrieval.rerank_candidates": "live",
    "rag.retrieval.rerank_pool": "live",
    "rag.retrieval.rerank_dense_candidates": "live",
    "rag.retrieval.rerank_sparse_candidates": "live",
    "rag.retrieval.min_relevance": "live",
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
    "rag.retrieval.rerank_candidates": (1, 200),
    "rag.retrieval.rerank_dense_candidates": (0, 200),
    "rag.retrieval.rerank_sparse_candidates": (0, 200),
    # A rerank score (probability), like rag.rerank.score_threshold.
    "rag.retrieval.min_relevance": (0.0, 1.0),
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
        "rag.retrieval.rerank_pool": ("per_list", "fused"),
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

# Variable -> origins config.yaml sends it to (see src/config/key_references.py), taken
# from the startup config before any runtime override is applied.
_startup_pairs: Optional[Dict[str, set]] = None


def capture_startup(config_model, refs: Optional[Dict[str, str]] = None) -> Dict[str, set]:
    """Record which key variables config.yaml pairs with which servers. Called at startup
    before the stored overrides are applied; ``refs`` defaults to Config's references."""
    global _startup_pairs
    from src.config.key_references import paired_origins
    if refs is None:
        from src.config.config_manager import Config
        refs = Config.get_env_refs()
    _startup_pairs = paired_origins(getattr(config_model, "rag", None), refs)
    return _startup_pairs


def _pairs(config_model) -> Dict[str, set]:
    # First use without an explicit capture: the process has applied no runtime change
    # yet, so the model in hand is still the startup config.
    return _startup_pairs if _startup_pairs is not None else capture_startup(config_model)


def _reset_startup_for_tests() -> None:
    global _startup_pairs
    _startup_pairs = None


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


def _resolve_secret(path: str, value: Any, key_vars) -> Tuple[Any, Optional[str]]:
    """``${VAR}`` → (resolved key, "VAR"); literal → (value, None). Rejects the mask.

    Only a variable in ``key_vars`` (those config.yaml uses as a model key) is ever read;
    any other reference, set or not, gets the same generic error, so the answer never
    tells whether a variable exists. Which server it may go to is checked by the caller.
    """
    if not isinstance(value, str) or not value:
        return value, None
    if value.strip() == MASK:
        raise ValueError("send a new key or an ${ENV_VAR} reference, not the masked placeholder")
    from src.config.profile import ENV_REF_RE, WHOLE_ENV_REF_RE
    from src.config.key_references import NOT_ALLOWED
    m = WHOLE_ENV_REF_RE.match(value)
    if not m:
        if ENV_REF_RE.search(value):
            raise ValueError(NOT_ALLOWED)  # a reference inside a longer value is never expanded
        return value, None
    if m.group(1) not in key_vars:
        raise ValueError(NOT_ALLOWED)
    from src.config.config_manager import expand_env_vars
    try:
        return expand_env_vars(value.strip()), m.group(1)
    except ValueError:
        raise ValueError(NOT_ALLOWED) from None


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


# Role -> its own api_key field (LLM roles may instead inherit rag.llm.api_key).
ROLE_KEY_PATHS = {
    "embedding": "rag.embedding.api_key", "llm": "rag.llm.api_key",
    **{r: f"rag.{r}.api_key" for r in _LLM_ROLES},
    "rerank": "rag.rerank.api_key", "asr": "rag.asr.api_key",
    "vision": "rag.parsing.vision.api_key",
}

KEY_CLEARED = ("{path} was cleared: {role}'s base_url now points to another server, so the "
               "key it had is not sent there. Enter the key for the new server.")


def validate_and_apply(config_model, patch: Dict[str, Any], *, carry_check: bool = True
                       ) -> Tuple[Dict[str, Any], List[str]]:
    """Validate the patch, then write it into the ConfigModel in place.

    All-or-nothing: any single validation failure -> ValueError, and nothing is written.

    Model keys (see src/config/key_references.py):
    - a ``${VAR}`` key is accepted only for a variable config.yaml uses as a model key,
      and only while the role's server (base_url origin) is one config.yaml pairs it with;
      any other reference gets one generic error;
    - when a role's base_url moves to another origin and the patch gives that role no new
      key, its own key is cleared (``applied`` holds ``None`` for it, and a warning says
      so); a key it inherits from ``rag.llm`` makes the change fail instead.
    ``carry_check=False`` (startup replay of stored overrides) skips the clearing: the
    stored rows already are the result of one.

    Returns:
        (applied, warnings) — the actually-written {path: value} and warnings for warn-level
        paths and cleared keys.

    Raises:
        ValueError: Path not whitelisted / type, range or choice validation failed
            (message includes details).
    """
    unknown = [p for p in patch if p not in EDITABLE]
    if unknown:
        raise ValueError(
            f"Setting path(s) cannot be changed at runtime: {unknown} "
            "(see GET /api/admin/settings for the editable list)")

    pairs = _pairs(config_model)
    # Validate everything first (pydantic: apply into a section copy to check types); write only if all pass
    staged = []  # [(parent, field, validated_value, path, as_written)]
    pending_refs: Dict[str, Optional[str]] = {}
    errors = []
    for path, value in patch.items():
        try:
            value = _check_constraints(path, value)
            written = value
            if path in SECRET_PATHS:
                value, pending_refs[path] = _resolve_secret(path, value, pairs)
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

    # Checked on a copy so a rejected batch changes nothing.
    dropped: List[Tuple[str, str]] = []  # (api_key path, role)
    rag = getattr(config_model, "rag", None)
    if rag is not None and hasattr(rag, "model_copy"):
        from src.config.key_references import NOT_ALLOWED, reference_allowed, role_keys
        trial = SimpleNamespace(rag=rag.model_copy(deep=True))
        for _parent, field, value, path, _w in staged:
            t_parent, t_field = _resolve_parent(trial, path)
            setattr(t_parent, t_field, value)

        before = role_keys(rag)
        if carry_check:
            # A key never follows its role to another server.
            after = role_keys(trial.rag)
            for role, a in after.items():
                b = before.get(role)
                moved = b is not None and b.origin is not None and a.origin != b.origin
                if moved and a.key and a.key_path not in patch and a.key_path == ROLE_KEY_PATHS[role]:
                    dropped.append((a.key_path, role))
            for path, _role in dropped:
                t_parent, t_field = _resolve_parent(trial, path)
                setattr(t_parent, t_field, None)
            gone = {p for p, _ in dropped}
            for role, a in role_keys(trial.rag).items():
                b = before.get(role)
                moved = b is not None and b.origin is not None and a.origin != b.origin
                if moved and a.key and a.key_path not in patch and a.key_path not in gone:
                    errors.append(
                        f"{ROLE_KEY_PATHS[role]}: {role}'s base_url now points to another server "
                        f"and it would send the key it inherits from {a.key_path} there; give it "
                        "its own api_key in the same change")

        # A reference only for the server config.yaml pairs it with.
        refs = _effective_refs(pending_refs)
        for path, _role in dropped:
            refs.pop(path, None)
        # Only the keys this change sets: a key carried to another server was cleared or
        # refused above, and at startup a carried one is settled by enforce_reference_pairing.
        for role, a in role_keys(trial.rag).items():
            var = refs.get(a.key_path) if a.key_path else None
            if var and a.key_path in patch and not reference_allowed(var, a.origin, pairs):
                errors.append(f"{a.key_path}: {NOT_ALLOWED}")
        if errors:
            raise ValueError("Settings validation failed: " + "; ".join(dict.fromkeys(errors)))

        # The result must still satisfy rag.profile (e.g. no in-process provider and no
        # literal key under profile: cloud).
        from src.config.profile import check_profile
        report = check_profile(trial.rag, refs=refs)
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
    for path, role in dropped:
        parent, field = _resolve_parent(config_model, path)
        setattr(parent, field, None)
        _secret_source[path] = None
        applied[path] = None
        warnings.append(KEY_CLEARED.format(path=path, role=role))
    return applied, warnings


def enforce_reference_pairing(config_model) -> List[str]:
    """Startup, after the stored overrides: clear any key that comes from a ``${VAR}``
    reference while its role points at a server config.yaml does not pair with that
    variable (e.g. a base_url override stored before 1.4.2). Returns the cleared paths."""
    from src.config.key_references import reference_allowed, role_keys
    rag = getattr(config_model, "rag", None)
    if rag is None:
        return []
    pairs = _pairs(config_model)
    refs = _effective_refs({})
    cleared: List[str] = []
    for role, a in role_keys(rag).items():
        var = refs.get(a.key_path) if a.key_path else None
        if var and not reference_allowed(var, a.origin, pairs):
            parent, field = _resolve_parent(config_model, a.key_path, create=False)
            if parent is not None:
                setattr(parent, field, None)
                _secret_source[a.key_path] = None
                cleared.append(a.key_path)
    return cleared


def rebind_groups_for(paths) -> List[str]:
    """A batch of applied paths -> the rebind groups to trigger (deduplicated, in a fixed order)."""
    hit = set()
    for p in paths:
        for g in REBIND_GROUPS:
            if p.startswith(g + "."):
                hit.add(g)
    return [g for g in REBIND_GROUPS if g in hit]
