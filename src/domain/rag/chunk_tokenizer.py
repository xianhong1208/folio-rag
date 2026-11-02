"""Tokenizer used to count chunk tokens: resolved locally, never downloaded.

Chunking measures every chunk against a token budget so that the text stays
inside the embedding model's context window. The tokenizer only *counts*; it
does not have to be the embedding model's own tokenizer, but the closer it is
the more accurate the budget.

Resolution order (first hit wins, nothing is ever fetched from the network):

1. ``assets/hf_tokenizers/<org>--<name>`` that exactly matches the embedding
   model name (e.g. ``intfloat/multilingual-e5-large``).
2. Any other tokenizer directory under ``assets/hf_tokenizers/`` (kept from the
   previous behaviour: the budget is approximate anyway).
3. The bundled XLM-RoBERTa tokenizer in ``resources/tokenizers/xlm-roberta/``.
   It is the tokenizer of multilingual-e5-large and bge-m3, so for those models
   it is exact; for cloud embeddings (OpenAI, Azure) whose tokenizers are not
   available offline it is a good multilingual approximation.
4. A character-based estimate, logged as a warning. Only reached when the
   bundled file is missing from a broken install; chunking still works.

Loading uses the ``tokenizers`` package (Rust, no torch, no transformers).
``transformers`` is imported lazily, and only for an ``assets`` directory that
carries no ``tokenizer.json`` — never under ``rag.profile: cloud`` (where
importing it would pull in torch) or when it is not installed; such a
directory is then skipped for the bundled tokenizer.
"""

from __future__ import annotations

import importlib.util
import lzma
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.log import get_api_logger

logger = get_api_logger()

_REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLED_TOKENIZER_NAME = "xlm-roberta"

_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


class FastTokenizerCounter:
    """Adapter giving a ``tokenizers.Tokenizer`` the ``encode(text, add_special_tokens=...)``
    call shape the chunker uses (it returns a list of ids, like transformers)."""

    def __init__(self, tokenizer: Any, name: str):
        self._tok = tokenizer
        self.name = name

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self._tok.encode(text or "", add_special_tokens=add_special_tokens).ids


class CharEstimateCounter:
    """Last-resort estimate: one token per CJK/kana/hangul character, one per
    four other characters. Deliberately generous so chunks stay under budget."""

    name = "char-estimate"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        text = text or ""
        wide = sum(1 for c in text if _is_wide(c))
        narrow = len(text) - wide
        return [0] * (wide + (narrow + 3) // 4)


def _is_wide(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3040 <= cp <= 0x30FF      # kana
        or 0x3400 <= cp <= 0x9FFF   # CJK unified (+ ext A)
        or 0xAC00 <= cp <= 0xD7AF   # hangul
        or 0xF900 <= cp <= 0xFAFF   # CJK compatibility
    )


def _assets_tokenizers_dir() -> Optional[Path]:
    from src.utils.runtime_paths import resolve_external_dir

    assets = resolve_external_dir("assets", dev_root=_REPO_ROOT)
    if not assets:
        return None
    d = assets / "hf_tokenizers"
    return d if d.is_dir() else None


def bundled_tokenizer_path() -> Optional[Path]:
    """Location of the bundled tokenizer (``resources/`` next to the app, or in the repo)."""
    from src.utils.runtime_paths import resolve_external_dir

    roots = []
    resolved = resolve_external_dir("resources", dev_root=_REPO_ROOT)
    if resolved is not None:
        roots.append(resolved)
    roots.append(_REPO_ROOT / "resources")
    for root in roots:
        p = root / "tokenizers" / BUNDLED_TOKENIZER_NAME / "tokenizer.json.xz"
        if p.is_file():
            return p
    return None


def _pick_assets_dir(model_name: Optional[str]) -> Optional[Path]:
    base = _assets_tokenizers_dir()
    if base is None:
        return None
    if model_name:
        exact = base / model_name.replace("/", "--")
        if exact.is_dir():
            return exact
    fallback = next((p for p in sorted(base.iterdir()) if p.is_dir()), None)
    if fallback is not None:
        logger.info(
            f"[TOKENIZER] no exact local tokenizer for '{model_name}'; "
            f"counting chunk tokens with '{fallback.name}'"
        )
    return fallback


def transformers_allowed() -> bool:
    """Whether a slow-tokenizer directory may be loaded through ``transformers``.

    Not under ``rag.profile: cloud`` (transformers imports torch when it is installed,
    and the cloud profile must not load the local model stack), nor when
    transformers is not installed (``uv sync`` without ``--extra local``).
    """
    try:
        from src.config.config_manager import Config

        rag = getattr(Config.get_config_model(), "rag", None)
        if getattr(rag, "profile", "local") == "cloud":
            return False
    except Exception:
        pass
    return importlib.util.find_spec("transformers") is not None


def _load_dir(path: Path) -> Any:
    tj = path / "tokenizer.json"
    if tj.is_file():
        from tokenizers import Tokenizer

        return FastTokenizerCounter(Tokenizer.from_file(str(tj)), path.name)
    # Slow-tokenizer-only directory (sentencepiece files without tokenizer.json).
    if not transformers_allowed():
        raise RuntimeError("no tokenizer.json, and transformers is unavailable here")
    from transformers import AutoTokenizer  # lazy: heavy import

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def _load_bundled(path: Path) -> Any:
    from tokenizers import Tokenizer

    data = lzma.decompress(path.read_bytes()).decode("utf-8")
    return FastTokenizerCounter(Tokenizer.from_str(data), BUNDLED_TOKENIZER_NAME)


def get_chunk_tokenizer(model_name: Optional[str] = None) -> Any:
    """Return a token counter for chunk sizing. Never downloads, never raises.

    ``model_name`` is the embedding model name, or None for providers whose
    names are not HuggingFace repos (OpenAI, Azure).
    """
    asset_dir = _pick_assets_dir(model_name)
    key = f"assets:{asset_dir}" if asset_dir is not None else "bundled"
    with _cache_lock:
        if key in _cache:
            return _cache[key]
        tok = None
        if asset_dir is not None:
            try:
                tok = _load_dir(asset_dir)
                logger.info(f"[TOKENIZER] chunk tokens counted with {asset_dir}")
            except Exception as e:
                logger.warning(
                    f"[TOKENIZER] could not load {asset_dir} ({e}); using the bundled tokenizer"
                )
        if tok is None:
            bundled = bundled_tokenizer_path()
            if bundled is not None:
                try:
                    tok = _load_bundled(bundled)
                    logger.info(f"[TOKENIZER] chunk tokens counted with the bundled {bundled.parent.name} tokenizer")
                except Exception as e:
                    logger.warning(f"[TOKENIZER] bundled tokenizer unreadable ({e})")
        if tok is None:
            logger.warning(
                "[TOKENIZER] no tokenizer available (assets/hf_tokenizers empty and "
                "resources/tokenizers missing); estimating chunk tokens from characters"
            )
            tok = CharEstimateCounter()
        _cache[key] = tok
        return tok


def clear_cache() -> None:
    """Forget loaded tokenizers (tests)."""
    with _cache_lock:
        _cache.clear()
