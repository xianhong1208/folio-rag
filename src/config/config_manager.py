
"""Configuration management module.

Reads and parses the YAML config file and provides config-driven application startup.
Uses Pydantic models for type-safe configuration access.
"""

import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List
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


# Environment-variable expansion for config values, matching MCP Center's convention:
# `${VAR}` / `${VAR:-default}`. Lets the DB URL (and any other value) come from the
# environment while config.yaml keeps a safe default.
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

# Allowed config-file directories (whitelist)
ALLOWED_CONFIG_DIRS: List[str] = [
    "config",
    ".",
]


class Config:
    _config: Dict[str, Any] = {}
    _config_model: Optional[ConfigModel] = None
    _module_configs: Dict[str, ModuleConfig] = {}  # Safe module-config storage (dict avoids setattr injection)
    _config_path: Optional[str] = None  # Original yaml path (used by runtime settings reset to read factory values)
    
    @classmethod
    def _validate_config_path(cls, config_path: str) -> Path:
        """Validate the config-file path to prevent path-traversal attacks.

        Args:
            config_path: The user-specified config-file path.

        Returns:
            A validated, safe Path object.

        Raises:
            ValueError: If the path is unsafe or not within an allowed directory.
        """
        base_dir = os.path.realpath(os.getcwd())
        if not os.path.isabs(config_path):
            full_path = os.path.realpath(os.path.join(base_dir, config_path))
        else:
            full_path = os.path.realpath(config_path)

        if not full_path.startswith(base_dir + os.sep) and full_path != base_dir:
            raise ValueError(
                f"Security Error: config_path '{config_path}' resolves to '{full_path}' "
                f"which is outside the allowed base directory '{base_dir}'"
            )
        
        rel_path = os.path.relpath(full_path, base_dir)
        path_parts = Path(rel_path).parts
        if path_parts:
            first_dir = path_parts[0]
            is_allowed = (
                first_dir in ALLOWED_CONFIG_DIRS or
                len(path_parts) == 1 or
                "." in ALLOWED_CONFIG_DIRS
            )
            if not is_allowed:
                raise ValueError(
                    f"Security Error: config file must be in allowed directories: {ALLOWED_CONFIG_DIRS}"
                )
        
        config_file = Path(full_path)
        if config_file.suffix.lower() not in ['.yaml', '.yml']:
            raise ValueError(
                f"Security Error: config file must have .yaml or .yml extension, got '{config_file.suffix}'"
            )
        
        return config_file
    
    @classmethod
    def set_config(cls, config_path: str):
        """Load and set the global config.

        Three steps: read YAML -> parse and validate with Pydantic -> store the detailed config of
        each enabled module into the safe dict.

        Args:
            config_path: Absolute path to config.yaml.
        """
        cls._load_config(config_path)
        cls._config_path = config_path
        try:
            config_data = dict(cls._config)
            modules_data = config_data.get('modules', {}) or {}
            enabled_modules = modules_data.get('enabled', [])

            # Simplify modules for Pydantic validation
            config_data['modules'] = {'enabled': enabled_modules}

            cls._config_model = ConfigModel(**config_data)

            # Store module configs in a safe dict instead of using setattr;
            # dict access doesn't touch object internals and is therefore safe.
            cls._module_configs: Dict[str, ModuleConfig] = {}

            for module_name in enabled_modules:
                if module_name in modules_data and module_name != 'enabled':
                    try:
                        module_config = ModuleConfig(**modules_data[module_name])
                        # Store in a dict rather than via setattr to block unsafe data flow
                        cls._module_configs[module_name] = module_config
                    except Exception:
                        # Don't silently drop the whole module (e.g. rag_indexing disappearing ->
                        # that endpoint silently 404s). Route to the log sink so ops sees it at a glance.
                        logger.exception(
                            f"Module config '{module_name}' failed to parse — "
                            f"this module will be UNAVAILABLE"
                        )
        except Exception:
            # On config-model validation failure, log the full traceback (including field-level
            # detail) to the log sink. The service still starts in degraded mode (_config_model=None,
            # so all APIs 404); switching to fail-fast would be a separate decision.
            logger.exception("Config model validation failed — starting in DEGRADED mode (no config model)")
            cls._config_model = None
    
    @classmethod
    def _load_config(cls, config_path: str):
        """Load the config file.

        Args:
            config_path: The config-file path.

        Raises:
            ValueError: If the path is unsafe.
            FileNotFoundError: If the config file does not exist.
        """
        config_file_path = cls._validate_config_path(config_path)  # guards against path traversal
        
        try:
            if config_file_path.exists():
                with open(config_file_path, 'r', encoding='utf-8') as file:
                    cls._config = expand_env_vars(yaml.safe_load(file) or {})
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
        except ValueError:
            raise
        except Exception as e:
            print(f"Failed to load config file: {e}")
            traceback.print_exc()
            cls._config = {}
    
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
        """Get a single module's ModuleConfig (via safe dict access, avoiding attribute injection).

        Args:
            module_name: A key in config.modules.enabled.

        Returns:
            The ModuleConfig, or None (module not enabled / config not yet loaded).
        """
        if cls._config_model is None:
            return None
        return cls._module_configs.get(module_name, None)


def get_config(config_path: str) -> Optional[ConfigModel]:
    """Convenience function: load the config and return the parsed ConfigModel (if parsing succeeded)."""
    Config.set_config(config_path)
    return Config.get_config_model()