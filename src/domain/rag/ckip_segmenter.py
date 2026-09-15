
"""CKIP Word Segmenter — Python-side Chinese word segmentation for BM25/FTS.

Chinese text has no spaces between words, so Postgres' `simple` full-text
config would treat a whole sentence as a single token and BM25 recall would
collapse. Historically this project delegated segmentation to the Postgres
`pg_jieba` (`jiebacfg`) FTS dictionary; we now segment in Python with the
Academia Sinica CKIP word segmenter (`ckip-transformers`) and feed the
space-joined token string to Postgres' `simple` config instead.

Design:
- Thread-safe lazy singleton: the (heavy) transformer model is loaded once on
  first use, guarded by a lock, and reused for every subsequent call.
- Graceful degradation: if `ckip-transformers` is not importable or the model
  fails to load, we log once and fall back to returning the input unchanged.
  Retrieval then behaves as `simple` on the raw text (degraded Chinese recall)
  rather than crashing the indexing/query path.
- The public helpers return SPACE-JOINED token strings; callers store these in
  the `text_search_tsv` vector (via `to_tsvector('simple', ...)`), never in the
  displayed `text`/`content` column.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from src.log import get_api_logger

logger = get_api_logger()

# Default CKIP word-segmentation model. "bert-base" is the standard accuracy
# model; callers may override via the constructor / get_segmenter(model=...).
_DEFAULT_MODEL = "bert-base"


class CkipSegmenter:
    """Lazy, thread-safe wrapper around ckip_transformers CkipWordSegmenter.

    A single instance is shared process-wide via `get_segmenter()`. The model
    is materialized on first `segment()`/`segment_batch()` call so importing
    this module (and constructing the object) stays cheap and never fails.
    """

    def __init__(self, model: str = _DEFAULT_MODEL):
        self._model_name = model
        self._ws = None  # the underlying CkipWordSegmenter (lazy)
        self._load_lock = threading.Lock()
        self._load_failed = False  # once True, we permanently fall back

    def _ensure_model(self) -> bool:
        """Load the CKIP model once. Returns True if the model is usable.

        Thread-safe (double-checked locking). On any import/load failure we set
        a sticky flag so we don't retry the expensive load on every call, and
        return False so callers degrade gracefully.
        """
        if self._ws is not None:
            return True
        if self._load_failed:
            return False
        with self._load_lock:
            # Re-check inside the lock (another thread may have loaded it).
            if self._ws is not None:
                return True
            if self._load_failed:
                return False
            try:
                # Imported lazily so a missing optional dependency doesn't break
                # module import or the non-Chinese retrieval path.
                from ckip_transformers.nlp import CkipWordSegmenter

                logger.info(
                    f"Loading CKIP word segmenter (model={self._model_name})..."
                )
                self._ws = CkipWordSegmenter(model=self._model_name)
                logger.info("✅ CKIP word segmenter ready")
                return True
            except Exception as e:  # noqa: BLE001 — degrade on ANY failure
                self._load_failed = True
                logger.warning(
                    f"⚠️ CKIP word segmenter unavailable ({e!r}); "
                    f"falling back to raw text for FTS tokenization. "
                    f"Chinese BM25 recall will be degraded until the "
                    f"`ckip-transformers` model can be loaded."
                )
                return False

    @staticmethod
    def _join(words: List[str]) -> str:
        """Join a model word list into a single normalized space-joined string.

        Individual tokens can themselves contain whitespace (rare) or be empty;
        we split/rejoin on arbitrary whitespace to guarantee single-space
        separators and no leading/trailing/emtpy tokens.
        """
        return " ".join(" ".join(words).split())

    def segment(self, text: str) -> str:
        """Segment one string into a space-joined token string.

        Returns "" for empty/whitespace-only input. On model unavailability or
        any runtime error, returns the input unchanged (graceful fallback).
        """
        if not text or not text.strip():
            return ""
        if not self._ensure_model():
            return text
        try:
            # CkipWordSegmenter is a batch API; wrap the single input in a list.
            result = self._ws([text])
            words = result[0] if result else []
            return self._join(words) or text
        except Exception as e:  # noqa: BLE001
            logger.warning(f"CKIP segment failed, using raw text: {e!r}")
            return text

    def segment_batch(self, texts: List[str]) -> List[str]:
        """Segment many strings using the model's batch API (indexing throughput).

        Preserves input order and length. Empty inputs map to "". On model
        unavailability or any runtime error, returns the inputs unchanged.
        """
        if not texts:
            return []
        if not self._ensure_model():
            return list(texts)
        try:
            results = self._ws(texts)
            out: List[str] = []
            for original, words in zip(texts, results):
                if not original or not original.strip():
                    out.append("")
                else:
                    out.append(self._join(words) or original)
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning(f"CKIP batch segment failed, using raw texts: {e!r}")
            return list(texts)


# ----------------------------------------------------------------------------
# Process-wide singleton accessor
# ----------------------------------------------------------------------------
_singleton: Optional[CkipSegmenter] = None
_singleton_lock = threading.Lock()


def get_segmenter(model: Optional[str] = None) -> CkipSegmenter:
    """Return the shared CkipSegmenter singleton (thread-safe, lazy).

    The model name is fixed on first construction; a later differing `model`
    argument is ignored (a warning is logged) to keep a single loaded model
    process-wide.
    """
    global _singleton
    if _singleton is not None:
        if model is not None and model != _singleton._model_name:
            logger.warning(
                f"CKIP segmenter already initialized with "
                f"model={_singleton._model_name!r}; ignoring requested "
                f"model={model!r}"
            )
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = CkipSegmenter(model=model or _DEFAULT_MODEL)
        return _singleton
