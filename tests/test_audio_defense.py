
"""audio_defense -- Whisper hallucination filtering (pure text) + silence detection/trimming (mock ffmpeg)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.domain.rag import audio_defense as ad


class TestHallucinationFilter:
    def test_strip_consecutive_repetitions(self):
        text = "會議開始 請回座 請回座 請回座 請回座 然後繼續"
        cleaned, removed = ad._strip_consecutive_repetitions(text)
        assert cleaned.count("請回座") == 1
        assert removed == 3  # 4 copies collapsed to 1, 3 redundant removed

    def test_no_repetition_untouched(self):
        text = "正常的會議紀錄內容,沒有任何重複段落。"
        cleaned, removed = ad._strip_consecutive_repetitions(text)
        assert cleaned == text and removed == 0

    def test_empty_text(self):
        assert ad._strip_consecutive_repetitions("") == ("", 0)
        assert ad._filter_whisper_hallucinations("") == ("", 0)

    def test_known_pattern_mingjing_outro(self):
        """The "明镜与点点" YouTube outro -- a real-world hallucination sample."""
        text = "各位貴賓請就座。請不吝点赞订阅转发打赏支持明镜与点点栏目。正式開始。"
        cleaned, n = ad._filter_whisper_hallucinations(text)
        assert "明镜" not in cleaned and n >= 1
        assert "各位貴賓請就座" in cleaned and "正式開始" in cleaned

    def test_known_pattern_song_credit(self):
        text = "詞・作曲・編曲 李宗盛 今天的議程如下"
        cleaned, n = ad._filter_whisper_hallucinations(text)
        assert "李宗盛" not in cleaned and n >= 1
        assert "今天的議程如下" in cleaned

    def test_known_pattern_amara_subtitle(self):
        text = "字幕由社群提供 Amara.org 字幕組 內文開始"
        cleaned, n = ad._filter_whisper_hallucinations(text)
        assert "Amara" not in cleaned and n >= 1

    def test_both_layers_combined(self):
        """Both layers -- known pattern + generic repetition -- fire, counts summed."""
        text = "打赏支持本栏目。重要決議 重要決議 重要決議 結束"
        cleaned, n = ad._filter_whisper_hallucinations(text)
        assert cleaned.count("重要決議") == 1
        assert "打赏" not in cleaned
        assert n >= 3


def _ffmpeg_result(stderr: str, returncode: int = 0):
    r = MagicMock()
    r.stderr = stderr
    r.returncode = returncode
    return r


class TestSilenceTrim:
    @pytest.mark.parametrize("end, trimmed", [(4.3, 0.0), (3.4, 0.0), (3.2, 3.2)])
    def test_silence_to_the_end_of_a_quiet_file_is_not_trimmed(self, end, trimmed):
        """A quiet recording reads as silence from start to end; trimming it left 0.02 s and the
        endpoint returned nothing (found on a real FLEURS clip). Keep at least 1 s."""
        stderr = (f"  Duration: 00:00:04.32, bitrate: 512 kb/s\n"
                  f"[silencedetect] silence_start: 0\n[silencedetect] silence_end: {end} | silence_duration: {end}")
        with patch.object(ad, "run_ffmpeg", return_value=_ffmpeg_result(stderr)):
            assert ad._detect_leading_silence_end("x.wav") == trimmed

    def test_detect_leading_silence(self):
        stderr = "[silencedetect] silence_start: 0.0\n[silencedetect] silence_end: 12.5 | silence_duration: 12.5"
        with patch.object(ad, "run_ffmpeg", return_value=_ffmpeg_result(stderr)):
            assert ad._detect_leading_silence_end("x.mp3") == 12.5

    def test_detect_no_silence(self):
        with patch.object(ad, "run_ffmpeg", return_value=_ffmpeg_result("no silence lines")):
            assert ad._detect_leading_silence_end("x.mp3") == 0.0

    def test_detect_silence_not_at_start_ignored(self):
        """silence_start > 1.0 = not "leading" silence -> do not trim."""
        stderr = "silence_start: 30.0\nsilence_end: 45.0"
        with patch.object(ad, "run_ffmpeg", return_value=_ffmpeg_result(stderr)):
            assert ad._detect_leading_silence_end("x.mp3") == 0.0

    def test_trim_skipped_below_threshold(self):
        """<0.5s of leading silence is not worth a re-encode -> None."""
        with patch.object(ad, "_detect_leading_silence_end", return_value=0.3):
            assert ad._trim_audio_leading_silence("in.mp3", "in.mp3") is None

    def test_trim_success_returns_temp(self, tmp_path):
        with patch.object(ad, "_detect_leading_silence_end", return_value=10.0), \
             patch.object(ad, "run_ffmpeg", return_value=_ffmpeg_result("", 0)) as m:
            out = ad._trim_audio_leading_silence("in.mp3", "meeting.mp3")
        assert out is not None and out.endswith(".mp3")
        cmd = m.call_args[0][0]
        assert "-ss" in cmd and "10.0" in cmd
        assert "libmp3lame" in cmd  # .mp3 -> corresponding codec args
        Path(out).unlink(missing_ok=True)

    def test_trim_unknown_suffix_uses_copy_codec(self):
        with patch.object(ad, "_detect_leading_silence_end", return_value=5.0), \
             patch.object(ad, "run_ffmpeg", return_value=_ffmpeg_result("", 0)) as m:
            out = ad._trim_audio_leading_silence("in.xyz", "odd.xyz")
        cmd = m.call_args[0][0]
        assert "copy" in cmd
        if out:
            Path(out).unlink(missing_ok=True)

    def test_trim_ffmpeg_failure_raises_and_cleans_temp(self):
        with patch.object(ad, "_detect_leading_silence_end", return_value=10.0), \
             patch.object(ad, "run_ffmpeg", return_value=_ffmpeg_result("boom", 1)):
            with pytest.raises(RuntimeError, match="ffmpeg seek-trim failed"):
                ad._trim_audio_leading_silence("in.mp3", "x.mp3")


