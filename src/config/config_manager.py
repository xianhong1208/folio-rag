
"""Configuration management module.

Reads the six files under config/ (see src/config/layout.py), merges them into one
tree, expands ${VAR} references and validates the result with Pydantic.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from loguru import logger  # route errors to the log sink instead of only print_exc to stderr

from .model import (
    ServerConfig,
    DatabaseConfig,
    AuthConfig,
    LoggingConfig,
    AppConfig,
    ModuleConfig,
    ModulesConfig,
    ConfigModel,
)
from .layout import CONFIG_FILES, LEGACY_CONFIG_FILE, REMOVED_SECTIONS

DEFAULT_CONFIG_DIR = "config"


# Environment-variable expansion for config values, matching MCP Gate's convention:
# `${VAR}` / `${VAR:-default}`. Lets the DB URL (and any other value) come from the
# environment while the config files keep a safe default.
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in strings, dicts and lists."""
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default_value is not None:
                return default_value
            raise ValueError(
                f"Environment variable '{var_name}' is not set and has no default. "
                f"Set it or provide a default in config: ${{{var_name}:-default}}"
            )
        return ENV_VAR_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value

class ConfigFileError(ValueError):
    """The config/ directory cannot be read: a file is missing or unreadable, or one
    setting is written in two files."""


def _merge_into(merged: Dict[str, Any], data: Dict[str, Any], name: str,
                origins: Dict[str, str], prefix: str = "") -> None:
    """Deep-merge ``data`` (from file ``name``) into ``merged``.

    Mappings merge key by key, so ``rag:`` may appear in several files; any other value
    may be written once only. A setting in two files is an error naming both files."""
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(value, dict):
                _merge_into(merged[key], value, name, origins, path)
                continue
            raise ConfigFileError(
                f"'{path}' is set in both config/{origins.get(path, '?')} and config/{name}; "
                "a setting may be written in one file only")
        merged[key] = value
        origins[path] = name
        if isinstance(value, dict):
            for sub in _sub_paths(value, path):
                origins[sub] = name


def _sub_paths(value: Dict[str, Any], prefix: str):
    """Every dotted path under a mapping (for duplicate reports)."""
    for k, v in value.items():
        p = f"{prefix}.{k}"
        yield p
        if isinstance(v, dict):
            yield from _sub_paths(v, p)


def load_config_dir(config_dir: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Read and merge the six files of ``config_dir`` (no ${VAR} expansion).

    Returns (merged raw tree, dotted path -> file it came from).
    Raises ConfigFileError for a missing file, invalid YAML, a file that is not a
    mapping, or a setting written in two files."""
    config_dir = Path(config_dir)
    missing = [n for n in CONFIG_FILES if not (config_dir / n).is_file()]
    if missing:
        hint = ""
        if (config_dir / LEGACY_CONFIG_FILE).is_file():
            hint = (f" config/{LEGACY_CONFIG_FILE} is no longer read: convert it with "
                    "`uv run python scripts/split_config.py` (docs/configuration.md, Upgrading).")
        raise ConfigFileError(
            f"Missing config file(s) in {config_dir}: {', '.join(missing)}.{hint}")
    merged: Dict[str, Any] = {}
    origins: Dict[str, str] = {}
    for name in CONFIG_FILES:
        try:
            data = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigFileError(f"config/{name} is not valid YAML: {e}") from None
        if data is None:
            continue
        if not isinstance(data, dict):
            raise ConfigFileError(f"config/{name} must be a mapping of settings")
        _merge_into(merged, data, name, origins)
    return merged, origins


def legacy_config_warning(config_dir: Union[str, Path]) -> Optional[str]:
    """The warning for an old single config.yaml left in ``config_dir``, if there is one."""
    if not (Path(config_dir) / LEGACY_CONFIG_FILE).is_file():
        return None
    return (f"config/{LEGACY_CONFIG_FILE} is no longer read: since Folio 1.5 the configuration "
            f"is {', '.join(CONFIG_FILES)} plus .env. Its values are IGNORED. Convert it with "
            "`uv run python scripts/split_config.py`, then move it away "
            "(docs/configuration.md, Upgrading from 1.4).")


def removed_section_warnings(raw: Dict[str, Any]) -> List[str]:
    out = []
    rag = raw.get("rag") if isinstance(raw.get("rag"), dict) else {}
    for sec in REMOVED_SECTIONS:
        if sec.split(".", 1)[1] in rag:
            out.append(f"{sec} was removed in Folio 1.5 and is ignored: answers and "
                       "evaluation use rag.llm")
    return out


class Config:
    _config: Dict[str, Any] = {}
    _config_model: Optional[ConfigModel] = None
    _config_dir: Optional[str] = None  # config/ directory (runtime settings reset re-reads factory values)
    # Dotted path -> env-var name for values written as ${VAR} in the files (pre-expansion).
    # Lets the profile check tell an env-sourced API key from a literal one.
    _env_refs: Dict[str, str] = {}
    # (level, message) of the notices set_config logged while loading. They are logged
    # before main.py sets up the log files from this very config, so they reach only the
    # terminal; main.py logs them again once the files exist (log_startup_notices).
    _startup_notices: List[Tuple[str, str]] = []

    @classmethod
    def _validate_config_dir(cls, config_dir: str) -> Path:
        """The config directory, which must be inside the working directory
        (guards against path traversal)."""
        base_dir = os.path.realpath(os.getcwd())
        full_path = os.path.realpath(os.path.join(base_dir, config_dir))
        if not full_path.startswith(base_dir + os.sep) and full_path != base_dir:
            raise ValueError(
                f"Security Error: config dir '{config_dir}' resolves to '{full_path}' "
                f"which is outside the allowed base directory '{base_dir}'")
        return Path(full_path)

    @classmethod
    def set_config(cls, config_dir: str = DEFAULT_CONFIG_DIR):
        """Load and set the global config from the six files in ``config_dir``.

        Read and merge the files -> expand ${VAR} -> validate with Pydantic. File-level
        problems (missing file, duplicate setting) raise ConfigFileError; a validation
        error leaves the model unset (logged), and main.py refuses to start.
        """
        path = cls._validate_config_dir(config_dir)
        cls._startup_notices = []
        warning = legacy_config_warning(path)
        if warning:
            cls._notice("error", warning)
        raw, _origins = load_config_dir(path)
        for w in removed_section_warnings(raw):
            cls._notice("warning", w)
        from src.config.profile import env_refs
        cls._env_refs = env_refs(raw)
        cls._config = expand_env_vars(raw)
        cls._config_dir = str(path)
        try:
            cls._config_model = ConfigModel(**cls._config)
            cls._check_profile(cls._config_model)
        except Exception:
            logger.exception("Config model validation failed — no config model loaded")
            cls._config_model = None

    @classmethod
    def _notice(cls, level: str, message: str) -> None:
        """Log a load-time notice now (the terminal) and keep it for log_startup_notices."""
        getattr(logger, level)(message)
        cls._startup_notices.append((level, message))

    @classmethod
    def log_startup_notices(cls, log) -> List[str]:
        """Log the notices of the last set_config again through ``log``, once.

        Call it right after the log files are set up: set_config ran before, so its own
        copy of each notice reached only the terminal. Returns the messages logged.
        """
        notices, cls._startup_notices = cls._startup_notices, []
        for level, message in notices:
            getattr(log, level)(message)
        return [message for _level, message in notices]

    @classmethod
    def factory_raw(cls) -> Dict[str, Any]:
        """The config files as they are on disk now (merged, unexpanded): the factory
        values the console's "reset" restores."""
        if not cls._config_dir:
            return {}
        return load_config_dir(cls._config_dir)[0]

    @classmethod
    def _check_profile(cls, config_model: ConfigModel) -> None:
        """Validate rag.profile (src/config/profile.py). Errors raise (-> the model is not
        loaded and main.py refuses to start); warnings are logged once."""
        rag = getattr(config_model, "rag", None)
        if rag is None:
            return
        from src.config.profile import check_profile
        report = check_profile(rag, refs=cls._env_refs)
        for w in report.warnings:
            cls._notice("warning", f"[PROFILE] {w}")
        if report.errors:
            raise ValueError(
                f"rag.profile '{rag.profile}' validation failed:\n  - "
                + "\n  - ".join(report.errors))

    @classmethod
    def get_env_refs(cls) -> Dict[str, str]:
        """Dotted path -> env-var name for config values written as ${VAR} references."""
        return dict(cls._env_refs)

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return the full config dict."""
        return cls._config


    @classmethod
    def get_config_model(cls) -> Optional[ConfigModel]:
        """Return the parsed Pydantic ConfigModel (if any)."""
        return cls._config_model

    @classmethod
    def get_server_config(cls) -> Optional[ServerConfig]:
        if cls._config_model is None:
            return None
        return cls._config_model.server

    @classmethod
    def get_database_config(cls) -> Optional[DatabaseConfig]:
        if cls._config_model is None:
            return None
        return cls._config_model.database

    @classmethod
    def get_auth_config(cls) -> Optional[AuthConfig]:
        if cls._config_model is None:
            return None
        return cls._config_model.auth

    @classmethod
    def get_logging_config(cls) -> Optional[LoggingConfig]:
        if cls._config_model is None:
            return None
        return cls._config_model.logging

    @classmethod
    def get_app_config_model(cls) -> Optional[AppConfig]:
        if cls._config_model is None:
            return None
        return cls._config_model.app

    @classmethod
    def get_modules_config(cls) -> Optional[ModulesConfig]:
        if cls._config_model is None:
            return None
        return cls._config_model.modules

    @classmethod
    def get_module_model(cls, module_name: str) -> Optional[ModuleConfig]:
        """An enabled module's wiring from the registry (src/config/modules.py), or None
        when the module is not enabled or no config is loaded."""
        if cls._config_model is None or module_name not in cls._config_model.modules.enabled:
            return None
        from src.config.modules import MODULES
        return MODULES.get(module_name)


def get_config(config_dir: str = DEFAULT_CONFIG_DIR) -> Optional[ConfigModel]:
    """Convenience function: load the config and return the parsed ConfigModel (if parsing succeeded)."""
    Config.set_config(config_dir)
    return Config.get_config_model()