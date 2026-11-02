"""Environment variables renamed when the project became Folio.

Deployments may still export the old names, so each renamed variable is read
under its new name first and then under its old one. Using an old name works
but logs one deprecation warning per variable and process.
"""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger

# new name -> old name
RENAMED_ENV_VARS = {
    "FOLIO_CORE_DUMPS": "AGENTIC_RAG_CORE_DUMPS",
    "FOLIO_INDEXING_CONCURRENCY": "RAG_INDEXING_CONCURRENCY",
    "FOLIO_PER_FILE_TIMEOUT_SECONDS": "RAG_PER_FILE_TIMEOUT_SECONDS",
    "FOLIO_JOB_TIMEOUT_SECONDS": "RAG_JOB_TIMEOUT_SECONDS",
}

_warned: set = set()


def getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    """``os.getenv`` for a renamed variable: the Folio name wins, the old one is a fallback."""
    value = os.environ.get(name)
    if value is not None:
        return value
    old = RENAMED_ENV_VARS.get(name)
    if old is not None:
        value = os.environ.get(old)
        if value is not None:
            if old not in _warned:
                _warned.add(old)
                logger.warning(f"Environment variable {old} is deprecated; set {name} instead.")
            return value
    return default
