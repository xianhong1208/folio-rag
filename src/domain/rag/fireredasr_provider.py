
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

import re
import subprocess
import threading
import wave
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.log import get_api_logger
from src.domain.rag.asr_base import AsrProvider, AsrResult

logger = get_api_logger()

# 55s = 60s hard limit minus a 5s safety margin (model card: >60s hallucinates,
# >200s overflows the positional encoding).
MAX_SEGMENT_SECONDS = 55.0
# silencedetect parameters: below -35dB for 0.3s counts as silence (a natural
# pause in meeting speech).
_SILENCE_FILTER = "silencedetect=noise=-35dB:d=0.3"

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


# Segment planning (pure functions, unit-testable)

def parse_silences(ffmpeg_stderr: str) -> List[Tuple[float, float]]:
    """Extract (silence_start, silence_end) pairs from ffmpeg silencedetect stderr.

    The final silence may have only a start (the file ends in silence); drop the
    incomplete pair, since segment planning only needs silences within the file.
    """
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", ffmpeg_stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", ffmpeg_stderr)]
    return list(zip(starts, ends))


def plan_segments(
    duration: float,
    silences: List[Tuple[float, float]],
    max_seg: float = MAX_SEGMENT_SECONDS,
) -> List[Tuple[float, float]]:
    """Greedy segmentation: each segment <= max_seg, cutting at the midpoint of
    the last silence within the window.

    - duration <= max_seg -> single segment
    - no silence in the window -> hard-cut at max_seg (better to cut mid-word
      than to feed >60s and trigger hallucination)
    - the silence midpoint sits at the center of the pause, leaving the speech
      on both sides intact
    """
    if duration <= max_seg:
        return [(0.0, duration)]
    cuts = sorted((s + e) / 2.0 for s, e in silences)
    segments: List[Tuple[float, float]] = []
    start = 0.0
    while duration - start > max_seg:
        window_end = start + max_seg
        candidates = [c for c in cuts if start < c <= window_end]
        cut = candidates[-1] if candidates else window_end
        segments.append((start, cut))
        start = cut
    segments.append((start, duration))
    return segments


# Audio preprocessing (ffmpeg)

def _resample_to_wav16k(src: str, dst: Path) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", src,
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {proc.stderr[-500:]}")


def _detect_silences(wav_path: Path) -> List[Tuple[float, float]]:
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(wav_path),
         "-af", _SILENCE_FILTER, "-f", "null", "-"],
        capture_output=True, text=True, timeout=600,
    )
    # silencedetect writes to stderr; parse even on a non-zero exit (an empty
    # list falls back to hard-cutting rather than blocking transcription).
    return parse_silences(proc.stderr)


def _slice_wav(wav_path: Path, segments: List[Tuple[float, float]], out_dir: Path) -> List[Path]:
    """Slice with the stdlib wave module (16k mono pcm16; read once and split
    into N pieces, faster than N ffmpeg invocations)."""
    paths: List[Path] = []
    with wave.open(str(wav_path), "rb") as w:
        rate = w.getframerate()
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    width = params.sampwidth  # pcm_s16le → 2
    for i, (start, end) in enumerate(segments):
        lo = int(start * rate) * width
        hi = int(end * rate) * width
        p = out_dir / f"seg_{i:04d}.wav"
        with wave.open(str(p), "wb") as out:
            out.setparams(params)
            out.writeframes(frames[lo:hi])
        paths.append(p)
    return paths


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
