"""Where a folder's retrieval-evaluation artifacts live (reports/).

Artifacts are keyed by folder **id** plus name (``<id>_<name>``). They used to be
keyed by name alone, but folder names are only unique per owner token, so two
owners with a folder called "docs" shared — and overwrote — one report, question
set and history, and each could read the other's (questions are generated from
chunk text).

Files written under the old name-only key are adopted on first use, but only when
exactly one folder carries that name: then they can only have been that folder's.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

REPORTS_DIR = "reports"


def _legacy_slug(folder) -> str:
    return str(folder.name).replace("/", "_")


def report_slug(folder, name_is_unique: Optional[Callable[[str], bool]] = None) -> str:
    """File key for ``folder``'s eval artifacts, adopting legacy name-keyed files."""
    slug = f"{folder.id}_{_legacy_slug(folder)}"
    _adopt_legacy(folder, slug, name_is_unique or _name_is_unique)
    return slug


def report_path(slug: str) -> str:
    return os.path.join(REPORTS_DIR, f"rag_eval_{slug}.json")


def history_path(slug: str) -> str:
    return os.path.join(REPORTS_DIR, f"rag_eval_history_{slug}.jsonl")


def eval_set_path(slug: str) -> str:
    return os.path.join(REPORTS_DIR, f"eval_set_{slug}.jsonl")


_PATHS = (report_path, history_path, eval_set_path)


def _adopt_legacy(folder, slug: str, name_is_unique: Callable[[str], bool]) -> None:
    if any(os.path.exists(p(slug)) for p in _PATHS):
        return
    legacy = _legacy_slug(folder)
    old = [(p(legacy), p(slug)) for p in _PATHS if os.path.exists(p(legacy))]
    if not old or not name_is_unique(str(folder.name)):
        return
    for src, dst in old:
        os.replace(src, dst)


def _name_is_unique(name: str) -> bool:
    from db.folderdb import FolderDB
    return len(FolderDB.get(name=name) or []) == 1
