"""Torch-free token counting for the chunker's token budget.

The chunker only needs ``tokenizer.encode(text, add_special_tokens=False)`` to measure
chunk sizes. transformers' AutoTokenizer provides that but imports torch when it is
installed, so the cloud profile (and installs without transformers) use the Rust
``tokenizers`` package instead, which reads the same ``tokenizer.json``.

Resolution order (never raises):
1. ``assets/hf_tokenizers/<org--name>/tokenizer.json`` (exact match, else any bundled one)
2. ``Tokenizer.from_pretrained(name)`` — downloads tokenizer.json only, unless
   HF_HUB_OFFLINE=1 or no name is known
3. a character estimate (CJK char = 1 token, other text ≈ 4 chars per token) with a
   WARNING — chunk sizes are then approximate, but indexing keeps working
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.log import get_api_logger

logger = get_api_logger()

_cache: Dict[str, Any] = {}
_lock = threading.Lock()
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")


class FastTokenizer:
    """``tokenizers.Tokenizer`` behind the ``encode(text, add_special_tokens=False)`` API."""

    def __init__(self, tok, source: str):
        self._tok = tok
        self.source = source

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids


class CharEstimateTokenizer:
    """Last resort: estimate tokens from characters (no vocabulary available)."""

    source = "char-estimate"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        cjk = len(_CJK.findall(text))
        rest = len(text) - cjk
        return list(range(cjk + math.ceil(rest / 4)))


def prefer_light_tokenizer() -> bool:
    """Use the torch-free path under profile: cloud, or when transformers is not installed."""
    try:
        from src.config.config_manager import Config
        rag = getattr(Config.get_config_model(), "rag", None)
        if getattr(rag, "profile", "local") == "cloud":
            return True
    except Exception:
        pass
    return importlib.util.find_spec("transformers") is None


def _bundled(name: Optional[str], assets_dir: Optional[Path]) -> Optional[Path]:
    if not assets_dir:
        return None
    root = Path(assets_dir) / "hf_tokenizers"
    if not root.is_dir():
        return None
    if name:
        exact = root / name.replace("/", "--") / "tokenizer.json"
        if exact.is_file():
            return exact
    return next((p / "tokenizer.json" for p in sorted(root.iterdir())
                 if (p / "tokenizer.json").is_file()), None)


def load_light_tokenizer(name: Optional[str], assets_dir: Optional[Path] = None):
    """A torch-free tokenizer for ``name`` (see module docstring for the order)."""
    key = f"{name}|{assets_dir}"
    with _lock:
        if key in _cache:
            return _cache[key]
        tok = _resolve(name, assets_dir)
        _cache[key] = tok
        return tok


def _resolve(name: Optional[str], assets_dir: Optional[Path]):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        Tokenizer = None

    if Tokenizer is not None:
        path = _bundled(name, assets_dir)
        if path is not None:
            logger.info(f"[TOKENIZER] {name or '(default)'}: using bundled {path.parent.name}/tokenizer.json")
            return FastTokenizer(Tokenizer.from_file(str(path)), f"bundled:{path.parent.name}")
        if name and os.environ.get("HF_HUB_OFFLINE") != "1":
            try:
                tok = Tokenizer.from_pretrained(name)
                logger.info(f"[TOKENIZER] {name}: downloaded tokenizer.json")
                return FastTokenizer(tok, f"hub:{name}")
            except Exception as e:  # not on the Hub, gated, offline network…
                logger.debug(f"[TOKENIZER] download of {name} failed: {type(e).__name__}")
    logger.warning(
        f"[TOKENIZER] no tokenizer available for {name or 'the embedding model'} — chunk sizes "
        "use a character estimate. Put a tokenizer.json under assets/hf_tokenizers/<org--name>/ "
        "for exact budgets.")
    return CharEstimateTokenizer()
