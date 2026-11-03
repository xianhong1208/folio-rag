"""Chinese word segmentation for full-text search — one choice, used everywhere.

Postgres's ``simple`` text-search config splits on whitespace and punctuation,
so space-less Chinese must be segmented in Python before it reaches
``to_tsvector``. The segmenter is chosen by ``rag.retrieval.segmenter``:

- ``ckip``   (default) — the Academia Sinica CKIP BERT segmenter. Most accurate
  on Traditional Chinese; needs torch (``uv sync --extra local``) and a model
  download on first use. Not allowed under ``rag.profile: cloud``.
- ``jieba``  — dictionary-based, pure Python, fast, no GPU. Install the optional
  extra: ``uv sync --extra jieba``.
- ``bigram`` — overlapping CJK character pairs plus Latin / digit words. No
  model and no dependency at all; high recall, larger index. The default under
  ``rag.profile: cloud``.

Unset, the segmenter follows the profile (``src/config/roles.effective_segmenter``).

:func:`get_segmenter` is the only entry point: the tsvector writer (indexing and
Rebuild FTS) and the query side both call it, so a query is always tokenised
like the index it searches. Heavy libraries are imported lazily, on first use.

Each folder's vector table records which segmenter built its tsvector (a column
comment, see ``vector_store_manager``); after switching segmenters the console
flags folders whose stamp no longer matches, and *Rebuild FTS* re-segments them.
A model-backed segmenter also offers ``segment_batch_report(texts) -> (out,
degraded)``: when indexing had to fall back to raw text, the indexer re-stamps the
table ``<name>-degraded`` so the console flags it for Rebuild FTS.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, List, Optional, Tuple

from src.log import get_api_logger

logger = get_api_logger()

SEGMENTERS = ("ckip", "jieba", "bigram")
DEFAULT_SEGMENTER = "ckip"


def _normalise(tokens) -> str:
    """Tokens → one space-joined string with single separators and no empties."""
    return " ".join(" ".join(tokens).split())


# CJK ideographs (incl. extension A and compatibility ideographs). Kana / Hangul
# are segmented by the same bigram rule.
_CJK = r"㐀-䶿一-鿿豈-﫿぀-ヿ가-힯"
_RUN = re.compile(rf"([{_CJK}]+)|([^\W_{_CJK}]+)")


def bigram_tokens(text: str) -> List[str]:
    """CJK runs → overlapping character pairs (a lone character stays itself);
    other words (Latin letters, digits) are kept whole."""
    out: List[str] = []
    for cjk, word in _RUN.findall(text or ""):
        if cjk:
            if len(cjk) == 1:
                out.append(cjk)
            else:
                out.extend(cjk[i:i + 2] for i in range(len(cjk) - 1))
        elif word:
            out.append(word)
    return out


class BigramSegmenter:
    """Dependency-free CJK bigram tokeniser."""

    name = "bigram"

    def segment(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        return _normalise(bigram_tokens(text)) or text

    def segment_batch(self, texts: List[str]) -> List[str]:
        return [self.segment(t) for t in (texts or [])]


class JiebaSegmenter:
    """jieba precise-mode segmentation; the library is imported on first use.

    Degrades like the CKIP segmenter: if jieba cannot be imported the input is
    returned unchanged (logged once), so retrieval keeps working with poorer
    Chinese keyword recall rather than failing.
    """

    name = "jieba"

    def __init__(self):
        self._jieba = None
        self._failed = False
        self._lock = threading.Lock()

    def _ensure(self) -> bool:
        if self._jieba is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._jieba is None and not self._failed:
                try:
                    import jieba  # optional dependency

                    jieba.setLogLevel(30)  # WARNING: keep the dictionary-load chatter out
                    self._jieba = jieba
                except Exception as e:  # noqa: BLE001
                    self._failed = True
                    logger.warning(
                        f"jieba segmenter unavailable ({e!r}); Chinese keyword search will "
                        "use unsegmented text. Install it with `pip install jieba`, or set "
                        "rag.retrieval.segmenter to 'bigram'.")
        return self._jieba is not None

    def _segment(self, text: str) -> Tuple[str, bool]:
        if not text or not text.strip():
            return "", False
        if not self._ensure():
            return text, True
        try:
            return _normalise(self._jieba.lcut(text)) or text, False
        except Exception as e:  # noqa: BLE001
            logger.warning(f"jieba segment failed, using raw text: {e!r}")
            return text, True

    def segment(self, text: str) -> str:
        return self._segment(text)[0]

    def segment_batch(self, texts: List[str]) -> List[str]:
        return [self.segment(t) for t in (texts or [])]

    def segment_batch_report(self, texts: List[str]) -> Tuple[List[str], bool]:
        """``segment_batch`` plus whether any text fell back to raw (see CKIP's)."""
        pairs = [self._segment(t) for t in (texts or [])]
        return [p[0] for p in pairs], any(p[1] for p in pairs)


def configured_segmenter_name() -> str:
    """``rag.retrieval.segmenter`` (live), else the profile's default (ckip; bigram under
    profile: cloud), else ckip when the config is unreadable or the value invalid."""
    try:
        from src.config.config_manager import Config
        from src.config.roles import effective_segmenter

        rag = getattr(Config.get_config_model(), "rag", None)
        if rag is None:
            return DEFAULT_SEGMENTER
        name = effective_segmenter(rag)
    except Exception:  # noqa: BLE001
        return DEFAULT_SEGMENTER
    return name if name in SEGMENTERS else DEFAULT_SEGMENTER


_instances: Dict[str, object] = {}
_instances_lock = threading.Lock()


def get_segmenter(name: Optional[str] = None):
    """The segmenter for FTS (``name`` or the configured one), created once per kind.

    Every segmenter exposes ``name``, ``segment(text) -> str`` and
    ``segment_batch(texts) -> List[str]`` (space-joined tokens).
    """
    name = (name or configured_segmenter_name()).lower()
    if name not in SEGMENTERS:
        name = DEFAULT_SEGMENTER
    inst = _instances.get(name)
    if inst is not None:
        return inst
    with _instances_lock:
        inst = _instances.get(name)
        if inst is None:
            if name == "bigram":
                inst = BigramSegmenter()
            elif name == "jieba":
                inst = JiebaSegmenter()
            else:
                from src.domain.rag.ckip_segmenter import get_segmenter as _ckip

                inst = _ckip()
            _instances[name] = inst
        return inst
