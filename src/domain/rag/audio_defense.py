
"""Whisper audio defense — anti-hallucination patch, hallucination filtering, leading-silence trimming.

These are read-file-layer audio pre/post-processing steps, independent of indexing logic, shared by
both the indexing path (hierarchical_indexer) and REST STT (media router).

The whisper anti-hallucination monkey-patch (_enable_whisper_anti_hallucination_defaults, idempotent)
is applied at module import time — it must run before the Docling AudioPipeline first loads the
whisper model.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from src.log import get_api_logger

logger = get_api_logger()

# Whisper anti-hallucination defaults via monkey-patch of whisper.load_model.
# Docling 4.x calls model.transcribe() without hallucination knobs, leaving
# whisper's overly permissive defaults. On non-speech audio the decoder falls
# into outro/repetition hallucination loops (worse under AMD ROCm's experimental
# attention). We bake tighter defaults into every loaded model:
#   - condition_on_previous_text=False: each 30s decodes alone, no cascade
#   - no_speech_threshold=0.7: drop more silence than the 0.6 default
# We don't hardcode language=zh — the leading-silence trim below gives whisper a
# real speech window so auto-detect stays reliable across languages.
# Idempotent; runs at import before Docling lazily builds the audio pipeline.


def _enable_whisper_anti_hallucination_defaults() -> None:
    """Bake anti-hallucination defaults into every whisper.load_model() result. Idempotent."""
    try:
        import whisper
    except ImportError:
        logger.info("whisper not installed; anti-hallucination patch skipped")
        return

    if getattr(whisper.load_model, "_anti_hallucination_wrapped", False):
        return

    _original_load = whisper.load_model
    _DEFAULTS = {
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.7,
    }

    def _patched_load(*args, **kwargs):
        model = _original_load(*args, **kwargs)
        _original_transcribe = model.transcribe

        def _patched_transcribe(audio, **kw):
            for k, v in _DEFAULTS.items():
                kw.setdefault(k, v)
            return _original_transcribe(audio, **kw)

        model.transcribe = _patched_transcribe
        return model

    _patched_load._anti_hallucination_wrapped = True
    whisper.load_model = _patched_load
    logger.info(
        "Whisper anti-hallucination defaults enabled: "
        "condition_on_previous_text=False, no_speech_threshold=0.7"
    )


_enable_whisper_anti_hallucination_defaults()


# Trim leading silence before Whisper. Whisper auto-detects language on the first
# 30s mel window; if those seconds are silent, the encoder gets near-zero
# activations and detection falls back to the highest-prior language (English),
# poisoning the whole transcript. Pre-trimming gives it a real speech window.

_AUDIO_TRIM_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".webm"}
_SILENCE_THRESHOLD_DB = -35.0
_SILENCE_MIN_DURATION = 2.0
_TRIM_SKIP_BELOW_SECONDS = 0.5  # don't bother re-encoding for <0.5s of leading silence

# Per-format encoder settings for trim re-encode. Defaults are reasonable quality
# for archival/transcription use cases (not music mastering). Unknown extensions
# fall back to `-c:a copy` which is fastest + lossless but only works if ffmpeg
# can seek precisely (most modern containers can).
_TRIM_CODEC_ARGS: dict[str, list[str]] = {
    ".mp3":  ["-c:a", "libmp3lame", "-q:a", "2"],
    ".wav":  ["-c:a", "pcm_s16le"],
    ".m4a":  ["-c:a", "aac", "-b:a", "192k"],
    ".aac":  ["-c:a", "aac", "-b:a", "192k"],
    ".flac": ["-c:a", "flac"],
    ".ogg":  ["-c:a", "libvorbis"],
    ".opus": ["-c:a", "libopus"],
}


# Whisper hallucination patterns — Chinese YouTube training-data leakage that
# surfaces when Whisper sees non-speech audio (intro music, chimes, silence) and
# falls back to high-prior token sequences. Audio-only; applying to PDFs could
# false-positive on legitimately quoted content.
_WHISPER_HALLUCINATION_PATTERNS = [
    # Song credits — "詞・作曲・編曲 李宗盛" pattern (Whisper thinks it's a music video intro)
    re.compile(r"[词詞][・·•\s]{0,3}作曲[・·•\s]{0,3}[编編]曲\s*[一-鿿]{0,15}"),
    # "明镜与点点" YouTube channel outro — `+` matches consecutive repeats as one block
    re.compile(
        r"(?:[请請]不吝[点點][赞讚][^。\n]{0,40}?明[镜鏡][与與][点點][点點][^。\n]{0,15}?[栏欄]目[\s,、。]*)+"
    ),
    # Generic subscribe/donate outros
    re.compile(r"打[赏賞]支持[^。\n]{0,20}?[栏欄]目[\s,、。]*"),
    re.compile(r"[请請][订訂][阅閱][^。\n]{0,20}?(?:[频頻][道台]|频道)[\s,、。]*"),
    re.compile(r"[感][谢謝][观觀]看[本这這的][^。\n]{0,15}"),
    # Fan-subbing credits
    re.compile(r"字幕由[^。\n]{0,30}?Amara[^。\n]{0,30}"),
    re.compile(r"由志[愿願]者[^。\n]{0,15}?[提組]供"),
    re.compile(r"字幕志[愿願]者[^。\n]{0,30}"),
]


# Generic "same phrase repeats N+ times in a row" detector for Whisper repetition
# hallucinations (the model stuck in a token loop): consecutive duplicates of a
# 3-30 char Chinese phrase separated by whitespace/CJK punctuation.
_REPETITION_PATTERN = re.compile(
    r"([一-鿿]{3,30})"        # phrase: 3-30 Chinese chars
    r"(?:[\s,，。、!?！?]*\1){2,}"      # plus ≥ 2 more identical copies → total ≥ 3
)


def _strip_consecutive_repetitions(text: str) -> tuple[str, int]:
    """Collapse a 3-30 char Chinese phrase repeated 3+ times in a row into one copy.

    Returns ``(cleaned, n_excess_copies_removed)``.
    """
    if not text:
        return text, 0
    removed = 0

    def _collapse(m: "re.Match[str]") -> str:
        nonlocal removed
        phrase = m.group(1)
        # rough copy count from match span — phrase + separators repeat
        n_copies = m.group(0).count(phrase)
        removed += max(0, n_copies - 1)
        return phrase

    cleaned = _REPETITION_PATTERN.sub(_collapse, text)
    return cleaned, removed


def _filter_whisper_hallucinations(text: str) -> tuple[str, int]:
    """Two-layer Whisper hallucination filter: known regex patterns plus a generic
    consecutive-repetition detector. Returns ``(cleaned, n_stripped)``.
    """
    if not text:
        return text, 0
    total = 0
    for pat in _WHISPER_HALLUCINATION_PATTERNS:
        text, n = pat.subn("", text)
        total += n
    # Generic repetition catches loops without enumerating every phrase
    text, n = _strip_consecutive_repetitions(text)
    total += n
    return text, total


def _detect_leading_silence_end(src: str) -> float:
    """Detect the end time (seconds) of leading silence via ffmpeg silencedetect,
    or 0.0 if there is no significant leading silence.

    Only catches true silence; intro music/chimes are handled downstream by
    _strip_consecutive_repetitions on the Whisper output.
    """
    af = f"silencedetect=noise={_SILENCE_THRESHOLD_DB}dB:duration={_SILENCE_MIN_DURATION}"
    cmd = ["ffmpeg", "-hide_banner", "-i", src, "-vn", "-af", af, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    starts = re.findall(r"silence_start:\s*([\d.]+)", result.stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)", result.stderr)
    if not starts or not ends or float(starts[0]) > 1.0:
        return 0.0
    return float(ends[0])


def _trim_audio_leading_silence(input_path: str, file_name: str) -> Optional[str]:
    """Trim leading silence from an audio file, returning a temp file path (None if none).

    Uses silencedetect + an `-ss` seek + re-encode because ffmpeg 4.x's silenceremove
    is unreliable on long files. The caller must `unlink()` the result when done.
    Raises RuntimeError on ffmpeg failure.
    """
    seek_ts = _detect_leading_silence_end(input_path)
    detector = "ffmpeg silencedetect"

    if seek_ts <= _TRIM_SKIP_BELOW_SECONDS:
        return None

    suffix = Path(file_name).suffix.lower() or ".mp3"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    out_path = tmp.name

    # Unknown suffix → -c:a copy (fastest, lossless; ffmpeg can keyframe-align seek for most containers)
    codec_args = _TRIM_CODEC_ARGS.get(suffix, ["-c:a", "copy"])

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(seek_ts),
        "-i", input_path, "-vn",
        *codec_args,
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        Path(out_path).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg seek-trim failed: {result.stderr[:500]}")

    logger.info(
        f"Trimmed leading non-speech: {file_name} -- skipped first {seek_ts:.1f}s "
        f"via {detector} before Whisper transcription"
    )
    return out_path

