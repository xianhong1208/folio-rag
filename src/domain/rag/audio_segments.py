"""Audio segmentation shared by the segment-based ASR providers (FireRedASR, cloud).

Pure stdlib + the ffmpeg binary: resample any source to 16 kHz mono pcm16 WAV, find
silences with ``silencedetect``, plan segments that end at the midpoint of the last
silence inside a length window, and slice the WAV without re-encoding. No model
library is imported here, so the cloud profile can use it.
"""

from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path
from typing import List, Tuple

# FireRedASR's limit (60 s hard cap minus a 5 s margin); cloud providers pass their own.
MAX_SEGMENT_SECONDS = 55.0
# silencedetect parameters: below -35dB for 0.3s counts as silence (a natural
# pause in meeting speech).
_SILENCE_FILTER = "silencedetect=noise=-35dB:d=0.3"


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


def wav_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def speech_ratio(start: float, end: float, silences: List[Tuple[float, float]]) -> float:
    """Fraction of [start, end) not covered by detected silences (1.0 = all speech)."""
    span = max(end - start, 1e-9)
    silent = sum(max(0.0, min(e, end) - max(s, start)) for s, e in silences)
    return max(0.0, min(1.0, 1.0 - silent / span))
