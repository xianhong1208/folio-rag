"""Where each setting lives: the six files under ``config/`` and the environment variables.

Folio's configuration is six YAML files, merged into one tree at startup (see
``config_manager.load_config_dir``). Each file is written in the shape of the merged
tree, so ``models.yaml`` starts with ``rag:`` and ``mcp.yaml`` holds ``app.tool_prefix``.
A setting may be written in one file only; the same key in two files stops startup.

Secrets and per-machine values (database URL, model endpoints and keys, the MCP Gate
issuer and audience, the revocation client, GPU devices) are ``${VAR:-default}``
references, read from the environment or from ``.env``. That keeps every file under
``config/`` free of secrets and safe to commit.

This module is pure data, shared by the loader, ``scripts/split_config.py`` and the tests.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

# File -> the dotted sections it holds. The longest matching prefix wins, so
# "auth.dynamic_tools" goes to mcp.yaml while the rest of "auth" goes to auth.yaml.
FILE_SECTIONS: Dict[str, Tuple[str, ...]] = {
    "database.yaml": ("database",),
    "auth.yaml": ("auth",),
    "models.yaml": ("rag.profile", "rag.embedding", "rag.llm", "rag.contextual_llm",
                    "rag.rerank", "rag.asr", "rag.docling", "rag.parsing"),
    "mcp.yaml": ("app.tool_prefix", "auth.dynamic_tools", "modules"),
    "retrieval.yaml": ("rag.enabled", "rag.chunking", "rag.indexing",
                       "rag.contextual_retrieval", "rag.vector_store", "rag.retrieval"),
    "system.yaml": ("server", "logging", "app"),
}

CONFIG_FILES: Tuple[str, ...] = tuple(FILE_SECTIONS)

# The single file used up to Folio 1.4. It is never read; its presence is reported.
LEGACY_CONFIG_FILE = "config.yaml"

# LLM roles removed in 1.5: answers and evaluation use rag.llm.
REMOVED_SECTIONS: Tuple[str, ...] = ("rag.answer_llm", "rag.eval_llm")


def file_for(path: str) -> Optional[str]:
    """The file a dotted setting path belongs to (longest section prefix), or None."""
    best, best_len = None, -1
    for name, sections in FILE_SECTIONS.items():
        for sec in sections:
            if (path == sec or path.startswith(sec + ".")) and len(sec) > best_len:
                best, best_len = name, len(sec)
    return best


class EnvSetting(NamedTuple):
    """A per-machine or secret value: the variable it is read from and the default the
    committed file gives it. ``blank_is_none``: an empty value means "not set" (None)."""
    var: str
    default: str
    blank_is_none: bool = False


# Dotted path -> the variable the committed config/*.yaml reads it from.
ENV_SETTINGS: Dict[str, EnvSetting] = {
    "database.url": EnvSetting("DATABASE_URL", "postgresql://postgres:CHANGE_ME@localhost:5432/folio"),
    "database.port": EnvSetting("DATABASE_PORT", "", True),
    "auth.issuer": EnvSetting("MCP_GATE_URL", "http://localhost:4568"),
    "auth.audience": EnvSetting("FOLIO_AUDIENCE", "http://127.0.0.1:5039/mcp"),
    "auth.revocation.enabled": EnvSetting("FOLIO_REVOCATION_ENABLED", "false"),
    "auth.revocation.client_id": EnvSetting("FOLIO_REVOCATION_CLIENT_ID", ""),
    "auth.revocation.client_secret": EnvSetting("FOLIO_REVOCATION_CLIENT_SECRET", ""),
    "rag.embedding.base_url": EnvSetting("EMBEDDING_BASE_URL", "http://localhost:7075/v1", True),
    "rag.embedding.api_key": EnvSetting("EMBEDDING_API_KEY", "not-needed", True),
    "rag.llm.base_url": EnvSetting("LLM_BASE_URL", "http://localhost:5052/v1", True),
    "rag.llm.api_key": EnvSetting("LLM_API_KEY", "not-needed", True),
    "rag.contextual_llm.base_url": EnvSetting("CONTEXTUAL_LLM_BASE_URL", "", True),
    "rag.contextual_llm.api_key": EnvSetting("CONTEXTUAL_LLM_API_KEY", "", True),
    "rag.rerank.base_url": EnvSetting("RERANK_BASE_URL", "http://localhost:8787"),
    "rag.rerank.api_key": EnvSetting("RERANK_API_KEY", "not-needed", True),
    "rag.asr.base_url": EnvSetting("ASR_BASE_URL", "", True),
    "rag.asr.api_key": EnvSetting("ASR_API_KEY", "", True),
    "rag.parsing.vision.base_url": EnvSetting("VISION_BASE_URL", "", True),
    "rag.parsing.vision.api_key": EnvSetting("VISION_API_KEY", "", True),
    "rag.docling.device": EnvSetting("DOCLING_DEVICE", "cuda"),
    "rag.retrieval.ckip_device": EnvSetting("CKIP_DEVICE", "cpu"),
}


def env_reference(setting: EnvSetting, var: Optional[str] = None) -> str:
    """``${VAR:-default}`` for a setting (``var`` overrides the variable name)."""
    return f"${{{var or setting.var}:-{setting.default}}}"


__all__ = [
    "FILE_SECTIONS", "CONFIG_FILES", "LEGACY_CONFIG_FILE", "REMOVED_SECTIONS",
    "EnvSetting", "ENV_SETTINGS", "file_for", "env_reference",
]
