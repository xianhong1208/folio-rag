"""Test helper: write a whole settings tree as the six config files.

Tests describe a config as one tree (as YAML text or a dict); the server reads it from
six files (src/config/layout.py). This splits the tree the same way the converter does,
without touching values, so ${VAR} references stay as written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from src.config.split import place_sections


def write_config_dir(directory: Union[str, Path], tree: Union[str, Dict[str, Any]]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(tree, str):
        tree = yaml.safe_load(tree) or {}
    for name, part in place_sections(tree).items():
        (directory / name).write_text(
            yaml.safe_dump(part, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return directory


def config_dir_from_file(yaml_file: Union[str, Path]) -> str:
    """Split a one-file tree (as tests write it) into six files next to it; returns the dir."""
    yaml_file = Path(yaml_file)
    return str(write_config_dir(yaml_file.parent / (yaml_file.stem + "_cfg"),
                                yaml_file.read_text(encoding="utf-8")))
