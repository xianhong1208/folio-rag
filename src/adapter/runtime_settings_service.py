
"""Runtime settings orchestration — the end-to-end flow behind the admin API.

apply:   validate -> mutate ConfigModel in place -> persist to DB (survives
         restart) -> rebind live objects.
reset:   delete the DB override -> restore the field from its config.yaml
         value -> rebind.
startup: at main.py startup, overlay the DB overrides onto the freshly loaded
         ConfigModel. The adapter is not built yet, so there is nothing to
         rebind — the first construction picks up the overlaid values.

Concurrency: a single lock serializes admin writes (the settings panel is not a
high-frequency path); reads take no lock — an in-place setattr is an atomic
attribute swap, so a reader at worst sees the previous value for one beat.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from src.config.config_manager import Config
from src.config import runtime_overrides as ro
from db.runtime_settings_db import RuntimeSettingsDB
from src.log import get_adapter_logger

logger = get_adapter_logger()

_apply_lock = threading.Lock()


def load_overrides_on_startup() -> int:
    """Overlay the DB overrides onto the ConfigModel at startup (called by main.py once the DB is ready).

    Invalid overrides (a changed whitelist or a type mismatch) are skipped one
    by one and logged — a single bad row must not block service startup.
    Returns the number successfully applied.
    """
    overrides = RuntimeSettingsDB.load_all()
    if not overrides:
        return 0
    config_model = Config.get_config_model()
    applied = 0
    for path, value in overrides.items():
        try:
            ro.validate_and_apply(config_model, {path: value})
            applied += 1
        except ValueError as e:
            logger.warning(f"[RUNTIME] Skipping invalid override {path}={value!r}: {e}")
    logger.info(f"[RUNTIME] Applied {applied}/{len(overrides)} runtime override(s) on startup")
    return applied


def get_settings_view() -> Dict[str, Any]:
    """Full view for the settings panel: effective value / level / whether overridden, for each whitelisted path."""
    config_model = Config.get_config_model()
    effective = ro.get_effective(config_model)
    overridden = set(RuntimeSettingsDB.load_all().keys())
    settings = []
    for path, level in ro.EDITABLE.items():
        entry = {
            "path": path,
            "value": effective.get(path),
            "level": level,
            "overridden": path in overridden,
            "secret": path in ro.SECRET_PATHS,
        }
        if entry["secret"]:
            # Key indicator only: whether a key is set and which env var it comes from.
            entry["is_set"] = bool(effective.get(path))
            entry["env"] = ro.secret_ref(path)
        settings.append(entry)
    rag = getattr(config_model, "rag", None)
    return {"settings": settings, **_profile_view(rag, effective)}


def _profile_view(rag, effective: Dict[str, Any]) -> Dict[str, Any]:
    """Profile + one summary per model role (where it runs, key indicator; never key values)."""
    if rag is None:
        return {"profile": None, "roles": [], "usage": {}}
    from src.config.roles import role_summary
    from src.domain.rag.usage_budget import BUDGET
    roles = []
    for r in role_summary(rag):
        secret_path = r.pop("secret_path")
        key_set = bool(_secret_value(rag, secret_path))
        roles.append({**r, "key_set": key_set,
                      "key_env": ro.secret_ref(secret_path) if key_set else None})
    return {"profile": rag.profile, "roles": roles, "usage": BUDGET.snapshot()}


def _secret_value(rag, path: str):
    from types import SimpleNamespace
    parent, field = ro._resolve(SimpleNamespace(rag=rag), path, create=False)
    return getattr(parent, field, None) if parent is not None else None


def apply_settings(patch: Dict[str, Any], updated_by: Optional[str] = None
                   ) -> Tuple[Dict[str, Any], List[str]]:
    """Validate -> mutate config in place -> persist -> rebind.

    Validation is all-or-nothing. Persistence and rebind then run in order as
    best-effort; a failure is re-raised as-is so the API returns 500.
    """
    with _apply_lock:
        config_model = Config.get_config_model()
        before = ro.get_effective(config_model, mask_secrets=True)  # previous values (secrets masked)
        applied, warnings = ro.validate_and_apply(config_model, patch)
        for path, value in applied.items():
            RuntimeSettingsDB.upsert(path, value, updated_by=updated_by)
            _audit(path, "set", before.get(path), _audit_value(path, value), updated_by)
        _rebind(list(applied.keys()))
        logger.info(f"[RUNTIME] applied {sorted(applied)} by={updated_by}")
        return applied, warnings


def reset_setting(path: str, updated_by: Optional[str] = None) -> bool:
    """Remove a single override, restore the config.yaml value, and rebind. Returns whether an override existed to delete.

    The factory value is resolved and validated *before* the override row is deleted:
    deleting first meant a failure here (e.g. a whole optional section missing from
    config.yaml) left the DB reset but the running value unchanged, behind a 500.
    """
    if path not in ro.EDITABLE:
        raise ValueError(f"Setting path is not editable: {path}")
    with _apply_lock:
        yaml_value = _yaml_value(path)
        config_model = Config.get_config_model()
        # Dry run on a copy: raises without touching the live model or the DB.
        ro.validate_and_apply(config_model.model_copy(deep=True), {path: yaml_value})
        removed = RuntimeSettingsDB.delete(path)
        if not removed:
            return False
        ro.validate_and_apply(config_model, {path: yaml_value})
        _rebind([path])
        _audit(path, "reset", None, _audit_value(path, yaml_value), updated_by)
        logger.info(f"[RUNTIME] reset {path} -> yaml default {yaml_value!r}")
        return True


def _audit_value(path: str, value: Any) -> Any:
    """Secrets are audited as the env reference they point at, or masked."""
    if path not in ro.SECRET_PATHS:
        return value
    ref = ro.secret_ref(path)
    return f"${{{ref}}}" if ref else ro.MASK


def _audit(key, action, old, new, by):
    from db.settings_audit_db import SettingsAuditDB
    SettingsAuditDB.record(key, action, old, new, by)


def audit_log(limit: int = 100):
    from db.settings_audit_db import SettingsAuditDB
    return SettingsAuditDB.recent(limit)


def _yaml_value(path: str) -> Any:
    """The factory value at a dotted path: config.yaml on disk (``${ENV}`` placeholders
    expanded, as at startup), else the field's pydantic default.

    The default is found by walking the schema *types*, not the live objects: an
    optional section absent from config.yaml (e.g. ``rag.asr``) is None at runtime,
    and walking instances raised AttributeError (a 500 on reset).
    """
    import typing
    import yaml
    from src.config.config_manager import expand_env_vars
    raw_path = Config._config_path
    try:
        if raw_path:
            with open(raw_path, "r", encoding="utf-8") as f:
                data = expand_env_vars(yaml.safe_load(f) or {})
            node: Any = data
            for part in path.split("."):
                node = node[part]
            return node
    except Exception:
        pass
    # Fallback: the field's declared default, found through the model classes
    parent_path, _, field = path.rpartition(".")
    cls = type(Config.get_config_model())
    for part in parent_path.split("."):
        ann = cls.model_fields[part].annotation
        args = typing.get_args(ann)
        cls = next((a for a in args if a is not type(None)), ann) if args else ann
    return cls.model_fields[field].get_default(call_default_factory=True)


def _rebind(paths: List[str]) -> List[str]:
    """Rebind only if the adapter is already built (if not, there is no live object yet — the first construction picks up the new config)."""
    groups = ro.rebind_groups_for(paths)
    if not groups:
        return []
    from src.adapter.rag import peek_rag_adapter
    adapter = peek_rag_adapter()
    if adapter is None:
        logger.info(f"[RUNTIME] adapter not built yet; {groups} will take effect at first construction")
        return []
    adapter._ctx.apply_runtime_changes(groups)
    return groups
