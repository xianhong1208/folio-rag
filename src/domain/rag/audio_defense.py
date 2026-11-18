
"""Whisper audio defense — hallucination filtering and leading-silence trimming.

Model-agnostic audio pre/post-processing around the speech-to-text provider (document_loader's
audio path): trimming leading silence keeps whisper's language detection on real speech, and
the filter drops the stock phrases whisper invents over silence. Both apply to any
OpenAI-compatible whisper endpoint (Forgeyard, speaches, OpenAI).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Optional

from src.log import get_api_logger
from src.utils.ffmpeg import run_ffmpeg

logger = get_api_logger()

# Trim leading silence before Whisper. Whisper auto-detects language on the first
# 30s mel window; if those seconds are silent, the encoder gets near-zero
# activations and detection falls back to the highest-prior language (English),
# poisoning the whole transcript. Pre-trimming gives it a real speech window.

_AUDIO_TRIM_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".webm"}
_SILENCE_THRESHOLD_DB = -35.0
_SILENCE_MIN_DURATION = 2.0
_MIN_SPEECH_LEFT_SECONDS = 1.0  # never trim a file down to less than this
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
    result = run_ffmpeg(cmd, capture_output=True, text=True, errors="replace")
    starts = re.findall(r"silence_start:\s*([\d.]+)", result.stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)", result.stderr)
    if not starts or not ends or float(starts[0]) > 1.0:
        return 0.0
    end = float(ends[0])
    # A "silence" that runs to the end of the file means the whole recording is below the
    # threshold (a quiet but real recording): trimming would leave nothing to transcribe.
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
    if m:
        total = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        if end >= total - _MIN_SPEECH_LEFT_SECONDS:
            return 0.0
    return end


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
    try:
        result = run_ffmpeg(cmd, capture_output=True, text=True, errors="replace")
    except RuntimeError:
        Path(out_path).unlink(missing_ok=True)
        raise
    if result.returncode != 0:
        Path(out_path).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg seek-trim failed: {result.stderr[:500]}")

    logger.info(
        f"Trimmed leading non-speech: {file_name} -- skipped first {seek_ts:.1f}s "
        f"via {detector} before Whisper transcription"
    )
    return out_path

