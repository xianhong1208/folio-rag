"""Convert a Folio 1.4 ``config/config.yaml`` into the six files of 1.5 plus ``.env`` lines.

Used by ``scripts/split_config.py``. The conversion keeps every value: the merged config
the six files and the generated ``.env`` lines produce equals what the old file produced,
except for the removed LLM roles (``rag.answer_llm``, ``rag.eval_llm``) and the module
wiring (now in code). ``check`` proves it field by field and reports field names only.

Per-machine and secret values (``layout.ENV_SETTINGS``) are written as ``${VAR:-default}``
references with safe defaults; the actual value goes to the ``.env`` lines when the
reference would not reproduce it on its own. Values are never printed or logged.
"""

from __future__ import annotations

import contextlib
import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import yaml

from src.config.layout import (CONFIG_FILES, ENV_SETTINGS, REMOVED_SECTIONS, EnvSetting,
                               env_reference, file_for)
# A value that is only a reference: group 1 = variable, group 2 = default (None = none).
_REF_PARTS_RE = re.compile(r"^\s*\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}\s*$")


@dataclass
class SplitResult:
    files: Dict[str, Dict[str, Any]]           # file name -> settings tree
    env_lines: Dict[str, str]                  # VAR -> value for .env
    notes: List[str] = field(default_factory=list)


def _as_env_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _blank(value: Optional[str]) -> bool:
    return value is None or not str(value).strip()


def _convert_value(raw: Any, setting: EnvSetting, env: Mapping[str, str], path: str,
                   notes: List[str]) -> Tuple[Any, Optional[Tuple[str, str]]]:
    """(new raw value, (VAR, value) for .env or None) for one per-machine setting."""
    if raw is None:
        if not setting.blank_is_none:
            return None, None  # stays an explicit null
        if _blank(env.get(setting.var)):
            return f"${{{setting.var}:-}}", None
        notes.append(f"{path} was null; {setting.var} is set in your environment, so the "
                     f".env lines clear it ({setting.var}=) to keep the value unset")
        return f"${{{setting.var}:-}}", (setting.var, "")
    if isinstance(raw, str):
        m = _REF_PARTS_RE.match(raw)
        if m:
            var, old_default = m.group(1), m.group(2)
            old = env.get(var, old_default)
            if old is None:
                return raw, None  # unset and no default: the old file failed here as well
            new_raw = env_reference(setting, var)
            new = env.get(var, setting.default)
            if setting.blank_is_none and _blank(old) and _blank(new):
                return new_raw, None
            return new_raw, (None if new == old else (var, old))
    # A literal (or a reference inside a longer string): expand it and move it to .env.
    from src.config.config_manager import expand_env_vars
    with patched_environ(env):
        old_text = _as_env_text(expand_env_vars(raw))
    new_raw = env_reference(setting)
    current = env.get(setting.var)
    new = current if current is not None else setting.default
    if new == old_text or (setting.blank_is_none and _blank(new) and _blank(old_text)):
        return new_raw, None
    if current is not None:
        notes.append(f"{setting.var} is already set in your environment or .env with another "
                     f"value; the generated line for {path} must replace it")
    return new_raw, (setting.var, old_text)


def _locate(tree: Dict[str, Any], path: str) -> Optional[Tuple[Dict[str, Any], str]]:
    parts = path.split(".")
    node: Any = tree
    for part in parts[:-1]:
        if not isinstance(node, dict) or not isinstance(node.get(part), dict):
            return None
        node = node[part]
    return (node, parts[-1]) if isinstance(node, dict) and parts[-1] in node else None


def _put(tree: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _modules(raw_modules: Any, notes: List[str]) -> Dict[str, Any]:
    from src.config.modules import MODULES
    raw_modules = raw_modules if isinstance(raw_modules, dict) else {}
    enabled = list(raw_modules.get("enabled") or [])
    kept = []
    for name in enabled:
        if name not in MODULES:
            notes.append(f"modules.enabled: '{name}' is not a built-in module and was dropped")
            continue
        kept.append(name)
    for name, wiring in raw_modules.items():
        if name == "enabled" or not isinstance(wiring, dict) or name not in MODULES:
            continue
        built_in = MODULES[name].model_dump()
        for part in ("api_router", "mcp_tools"):
            old = wiring.get(part) or {}
            new = built_in.get(part) or {}
            old_set = {k: v for k, v in old.items() if v}
            new_set = {k: v for k, v in new.items() if v}
            if old_set != new_set:
                notes.append(f"modules.{name}.{part} differs from the built-in wiring "
                             "(src/config/modules.py); wiring is no longer configurable")
    return {"enabled": kept}


def place_sections(tree: Dict[str, Any], notes: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Distribute a merged settings tree over the six files (layout.FILE_SECTIONS).
    Removed LLM roles are dropped and module wiring is reduced to ``enabled``."""
    notes = notes if notes is not None else []
    raw = tree
    files: Dict[str, Dict[str, Any]] = {name: {} for name in CONFIG_FILES}
    for top, value in raw.items():
        if top == "modules":
            files["mcp.yaml"]["modules"] = _modules(value, notes)
            continue
        if top in ("rag", "auth", "app") and isinstance(value, dict):
            for sub, sub_value in value.items():
                path = f"{top}.{sub}"
                if path in REMOVED_SECTIONS:
                    notes.append(f"{path} was removed in Folio 1.5 (answers and evaluation use "
                                 "rag.llm); its settings were not carried over")
                    continue
                target = file_for(path)
                if target is None:
                    target = "retrieval.yaml" if top == "rag" else "system.yaml"
                    notes.append(f"{path} is not a known setting; written to {target}")
                _put(files[target], path, sub_value)
            continue
        target = file_for(top)
        if target is None:
            target = "system.yaml"
            notes.append(f"{top} is not a known section; written to {target}")
        files[target][top] = value
    return files


def split_legacy(old_raw: Dict[str, Any], env: Mapping[str, str]) -> SplitResult:
    """Split an old config tree (as read from YAML, unexpanded) into the six files."""
    raw = copy.deepcopy(old_raw or {})
    notes: List[str] = []
    env_lines: Dict[str, str] = {}
    for path, setting in ENV_SETTINGS.items():
        loc = _locate(raw, path)
        if loc is None:
            continue
        parent, key = loc
        new_raw, line = _convert_value(parent[key], setting, env, path, notes)
        parent[key] = new_raw
        if line is not None:
            env_lines[line[0]] = line[1]

    files = place_sections(raw, notes)
    return SplitResult(files=files, env_lines=env_lines, notes=notes)


@contextlib.contextmanager
def patched_environ(env: Mapping[str, str]) -> Iterator[None]:
    """Run with os.environ replaced by ``env`` (restored afterwards)."""
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def legacy_config_dict(old_raw: Dict[str, Any], env: Mapping[str, str]) -> Dict[str, Any]:
    """What Folio 1.4 loaded from the old file (with today's models): expanded, modules
    reduced to ``enabled``, removed roles left out."""
    from src.config.config_manager import expand_env_vars
    from src.config.model import ConfigModel
    with patched_environ(env):
        data = expand_env_vars(copy.deepcopy(old_raw or {}))
    from src.config.modules import MODULES
    modules = data.get("modules") or {}
    # 1.4 skipped an enabled module it had no wiring for; 1.5 knows only the built-in ones.
    data["modules"] = {"enabled": [m for m in (modules.get("enabled") or []) if m in MODULES]}
    rag = data.get("rag")
    if isinstance(rag, dict):
        for sec in REMOVED_SECTIONS:
            rag.pop(sec.split(".", 1)[1], None)
    return ConfigModel(**data).model_dump()


def split_config_dict(config_dir: Path, env: Mapping[str, str]) -> Dict[str, Any]:
    """What 1.5 loads from ``config_dir`` under ``env`` (validated, as a dict)."""
    from src.config.config_manager import expand_env_vars, load_config_dir
    from src.config.model import ConfigModel
    raw, _ = load_config_dir(config_dir)
    with patched_environ(env):
        data = expand_env_vars(raw)
    return ConfigModel(**data).model_dump()


def diff_paths(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Dotted paths where two plain trees differ (names only, never values)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(path)
            else:
                out.extend(diff_paths(a[key], b[key], path))
        return out
    return [] if a == b else [prefix or "<root>"]


def _dump(tree: Dict[str, Any]) -> str:
    return yaml.safe_dump(tree, sort_keys=False, allow_unicode=True, default_flow_style=False,
                          width=1000)


def _env_line(var: str, value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError(f"{var}: a value with a line break cannot go into .env")
    needs_quotes = value != value.strip() or value[:1] in ("'", '"', "#") or value[-1:] in ("'", '"')
    return f'{var}="{value}"' if needs_quotes else f"{var}={value}"


HEADER = ("# Generated by scripts/split_config.py from a Folio 1.4 config.yaml.\n"
          "# The commented templates are in the repository (git show HEAD:config/{name}).\n")


def write_outputs(result: SplitResult, out_dir: Path, env_out: Path, force: bool = False) -> None:
    """Write the six files and the .env lines (mode 600). Refuses to overwrite unless ``force``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [n for n in CONFIG_FILES if (out_dir / n).exists()]
    if existing and not force:
        raise FileExistsError(
            f"{out_dir} already has {', '.join(existing)}; pass --force to replace them "
            "(in a git checkout, `git checkout -- config/` restores the templates)")
    if env_out.exists() and not force:
        raise FileExistsError(f"{env_out} exists; pass --force to replace it")
    for name in CONFIG_FILES:
        (out_dir / name).write_text(HEADER.format(name=name) + _dump(result.files[name]),
                                    encoding="utf-8")
    lines = ["# .env lines generated by scripts/split_config.py: merge them into .env.",
             "# They hold secrets: keep this file private (mode 600) and delete it afterwards."]
    lines += [_env_line(var, value) for var, value in sorted(result.env_lines.items())]
    fd = os.open(env_out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, ("\n".join(lines) + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def check(old_raw: Dict[str, Any], out_dir: Path, env: Mapping[str, str],
          env_lines: Mapping[str, str]) -> List[str]:
    """Field names where the converted config differs from the old one (empty = equal)."""
    before = legacy_config_dict(old_raw, env)
    after = split_config_dict(out_dir, {**env, **env_lines})
    return diff_paths(before, after)


__all__ = ["SplitResult", "place_sections", "split_legacy", "write_outputs", "check", "diff_paths",
           "legacy_config_dict", "split_config_dict", "patched_environ"]
