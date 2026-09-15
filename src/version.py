
"""Single source of truth for the version: [project].version in pyproject.toml.

Resolution order: importlib.metadata (primary, once installed or packaged),
then pyproject.toml at the project root (dev checkout not yet uv sync'd),
then "0.0.0" plus a warning. Note: an editable install's metadata is a
snapshot from uv sync time, so re-sync after bumping the version.
"""
from __future__ import annotations

import logging
import tomllib
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

logger = logging.getLogger(__name__)

_DISTRIBUTION_NAME = "agentic-rag"
# src/version.py -> parents[1] = project root (where a dev checkout's pyproject.toml lives)
_PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the application version string (see module docstring for sources)."""
    try:
        return _pkg_version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        pass  # Not yet installed as a package (dev checkout without uv sync) -- fall back to pyproject.toml

    try:
        with _PYPROJECT_PATH.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
        logger.warning(
            f"Could not resolve version from metadata or {_PYPROJECT_PATH} "
            f"({e}); returning 0.0.0. For packaged deployments, ensure libs/ "
            f"contains agentic_rag's dist-info."
        )
        return "0.0.0"
