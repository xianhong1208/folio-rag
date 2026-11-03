"""Run the ffmpeg binary, turning "not installed" into a clear error.

Audio transcription (local Whisper / FireRedASR and the cloud ASR path alike) shells
out to ffmpeg for resampling, silence detection and trimming. Without the binary,
``subprocess.run`` raises a bare ``FileNotFoundError: 'ffmpeg'`` from deep inside the
pipeline, which reads like a missing upload. Callers already handle ``RuntimeError``
(an ffmpeg failure), so a missing binary is reported as one too, with the fix in the
message.
"""

from __future__ import annotations

import subprocess
from typing import Any, List

FFMPEG_MISSING = (
    "ffmpeg is not installed or not on PATH; audio transcription needs it "
    "(e.g. `apt install ffmpeg`)"
)


class FfmpegNotFoundError(RuntimeError):
    """The ffmpeg executable could not be found."""


def run_ffmpeg(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """``subprocess.run(cmd, **kwargs)`` for an ffmpeg command line (``cmd[0]`` is the binary)."""
    try:
        return subprocess.run(cmd, **kwargs)
    except FileNotFoundError as e:
        raise FfmpegNotFoundError(FFMPEG_MISSING) from e
