"""src.utils.ffmpeg: a missing ffmpeg binary is reported as a clear RuntimeError."""
import shutil

import pytest

from src.utils.ffmpeg import FFMPEG_MISSING, FfmpegNotFoundError, run_ffmpeg


def test_missing_binary_raises_clear_runtime_error(monkeypatch, tmp_path):
    # An empty PATH is a machine without ffmpeg: the real FileNotFoundError, no mocking.
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FfmpegNotFoundError) as exc:
        run_ffmpeg(["ffmpeg", "-version"], capture_output=True)
    assert isinstance(exc.value, RuntimeError)  # callers catch RuntimeError
    assert str(exc.value) == FFMPEG_MISSING
    assert isinstance(exc.value.__cause__, FileNotFoundError)


def test_resample_without_ffmpeg_names_the_fix(monkeypatch, tmp_path):
    from src.domain.rag.audio_segments import _resample_to_wav16k

    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
        _resample_to_wav16k(str(tmp_path / "in.mp3"), tmp_path / "out.wav")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
def test_runs_ffmpeg_when_present():
    proc = run_ffmpeg(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stdout.startswith("ffmpeg version")
