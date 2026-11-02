
"""Local FireRedASR-AED-L provider.

Engineering constraints (from the model card and upstream issues):
- **60s hard input limit**: >60s starts hallucinating, >200s overflows the
  positional encoding. Use ffmpeg silencedetect to find silence points and
  greedily split into <=55s segments (5s safety margin); hard-cut at 55s when
  a stretch has no silence.
- **16kHz mono**: ffmpeg resamples every source format to pcm_s16le wav.
- **Simplified-Chinese output**: the AED training corpus is Simplified Chinese,
  so post-process with OpenCC `s2twp` (Simplified -> Traditional + Taiwan
  terminology); Traditional-Chinese output is gated by the Simplified-character
  ratio check in evals.
- **No punctuation**: AED output has no punctuation, so it goes down the
  leaf_splitter plain-text path (chunks=None).

Weights: assets/fireredasr/FireRedASR-AED-L/ (~4.7GB, shipped separately).

The model loads as a lazy singleton (first transcribe loads it, same pattern as
the docling converter); the inference lock serializes calls because concurrent
forward passes on the 1.1B model cause OOM.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from typing import Callable, List, Optional

from src.log import get_api_logger
from src.domain.rag.asr_base import AsrProvider, AsrResult
# 55s = 60s hard limit minus a 5s safety margin (model card: >60s hallucinates,
# >200s overflows the positional encoding).
from src.domain.rag.audio_segments import MAX_SEGMENT_SECONDS  # noqa: F401

logger = get_api_logger()

_model_lock = threading.Lock()
_model = None  # lazy singleton: (FireRedAsr, use_gpu)
# Inference lock: the indexing path (document_loader) and media's
# /v1/transcriptions share the same single 1.1B model instance; concurrent
# forward passes cause KV-cache races / OOM (same rationale as docling's
# DOCLING_INFER_LOCK). Every model.transcribe call must hold this lock.
FIRERED_INFER_LOCK = threading.Lock()


def _resolve_model_dir() -> Optional[Path]:
    """Weights directory assets/fireredasr/FireRedASR-AED-L (ready only if model.pth.tar exists)."""
    from src.domain.rag.docling_loader import resolve_assets_dir
    assets = resolve_assets_dir()
    if assets is None:
        return None
    d = assets / "fireredasr" / "FireRedASR-AED-L"
    return d if (d / "model.pth.tar").is_file() else None


def _load_model(model_dir: Path):
    """Load the AED model (singleton; fireredasr is an installed package, imported normally)."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        import torch
        # fireredasr is a uv path dependency (vendor/fireredasr_src installed
        # into site-packages), so it imports normally and Nuitka compiles it
        # into the binary like any dependency; deployment does not need
        # vendor/. Raise an actionable error (rather than a bare ImportError)
        # when the package is missing.
        try:
            from fireredasr.models.fireredasr import FireRedAsr
        except ImportError as e:
            raise RuntimeError(
                "fireredasr package cannot be imported — in dev run "
                "`uv sync --group <gpu>` (path dep points at vendor/fireredasr_src); "
                "for packaged builds ensure Nuitka bundles fireredasr "
                "(add --include-package=fireredasr if needed)"
            ) from e

        use_gpu = torch.cuda.is_available()
        logger.info(f"[FIREREDASR] loading AED-L from {model_dir} (gpu={use_gpu})")
        # torch 2.6+ defaults to weights_only=True, but the official
        # checkpoint's "args" is an argparse.Namespace, which must be
        # allowlisted (only this type; all other deserialization protections
        # stay unchanged, and vendor code is not modified).
        import argparse
        with torch.serialization.safe_globals([argparse.Namespace]):
            model = FireRedAsr.from_pretrained("aed", str(model_dir))
        _model = (model, use_gpu)
        logger.info("[FIREREDASR] model ready")
        return _model


# Segment planning and ffmpeg helpers are shared with the cloud provider (audio_segments.py);
# re-exported here under their historical names.
from src.domain.rag.audio_segments import (  # noqa: E402,F401
    _detect_silences,
    _resample_to_wav16k,
    _slice_wav,
    parse_silences,
    plan_segments,
)


class FireRedAsrProvider(AsrProvider):
    """Local FireRedASR-AED-L (segment -> per-segment inference -> concatenate -> s2twp to Traditional Chinese)."""

    def __init__(self, *, beam_size: int = 3):
        self._beam_size = beam_size
        self._s2twp = None  # lazy (opencc loads its dictionaries)

    def available(self) -> bool:
        return _resolve_model_dir() is not None

    def _to_traditional(self, text: str) -> str:
        if self._s2twp is None:
            import opencc
            self._s2twp = opencc.OpenCC("s2twp")
        return self._s2twp.convert(text)

    def _transcribe_segments(self, seg_paths: List[Path]) -> List[str]:
        """Per-segment inference (no batching: segments are already near the length
        limit, so batch padding only wastes work and OOM risk is low)."""
        model_dir = _resolve_model_dir()
        if model_dir is None:  # available() already guards this; second line of defense
            raise RuntimeError("FireRedASR weights not found in assets/fireredasr/FireRedASR-AED-L")
        model, use_gpu = _load_model(model_dir)
        texts: List[str] = []
        for p in seg_paths:
            with FIRERED_INFER_LOCK:  # indexing and media STT share the model; serialize inference
                results = model.transcribe(
                    [p.stem], [str(p)],
                    {"use_gpu": use_gpu, "beam_size": self._beam_size, "nbest": 1},
                )
            texts.append(results[0]["text"].strip() if results else "")
        return texts

    def transcribe(
        self, *, audio_path: str, file_name: str,
        max_tokens: Optional[int] = None, tokenizer: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
    ) -> AsrResult:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="fireredasr_") as td:
            tdir = Path(td)
            wav = tdir / "input16k.wav"
            _resample_to_wav16k(audio_path, wav)
            with wave.open(str(wav), "rb") as w:
                duration = w.getnframes() / w.getframerate()

            segments = plan_segments(duration, _detect_silences(wav))
            seg_paths = _slice_wav(wav, segments, tdir)
            logger.info(
                f"[FIREREDASR] {file_name}: {duration:.1f}s → {len(segments)} segment(s)"
            )

            texts = []
            for i, sp in enumerate(self._transcribe_segments(seg_paths)):
                texts.append(sp)
                if progress_cb is not None:
                    try:
                        progress_cb("loading", i + 1, len(seg_paths))
                    except Exception:
                        pass

        full = self._to_traditional("\n".join(t for t in texts if t))
        return AsrResult(text=full, chunks=None)
