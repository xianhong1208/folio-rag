"""Minimal ``.env`` reader (no dependency).

``.env`` holds the secrets and per-machine values that config/*.yaml reads through
``${VAR:-default}`` references. ``KEY=value`` per line; ``#`` comments and blank lines
are skipped; one pair of surrounding quotes is removed from the value.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse_dotenv(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        out[key.strip()] = _unquote(value)
    return out


def read_dotenv(path: Union[str, Path, None] = None) -> Dict[str, str]:
    """The pairs in ``path`` (default: the project's ``.env``); {} when it does not exist."""
    env_file = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return {}
    return parse_dotenv(env_file.read_text(encoding="utf-8"))


def load_dotenv(path: Union[str, Path, None] = None, override: bool = True) -> Optional[Path]:
    """Load ``.env`` into os.environ. ``override=True`` lets this project's ``.env`` win
    over a stale global export lingering in the shell. Returns the file read, or None."""
    env_file = Path(path) if path else PROJECT_ROOT / ".env"
    pairs = read_dotenv(env_file)
    for key, value in pairs.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return env_file if env_file.is_file() else None
