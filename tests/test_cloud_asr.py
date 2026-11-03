"""Cloud speech-to-text: segmented upload, retries and the runaway guard.

Audio is synthesised with the stdlib `wave` module; the endpoint is the in-process fake
server. ffmpeg (resample + silencedetect) is used when present, otherwise those tests skip.
"""

import math
import shutil
import struct
import wave
from pathlib import Path

import httpx
import pytest

from src.domain.rag import asr_provider as ap
from src.domain.rag.asr_provider import (
    OpenAICompatibleAsrProvider,
    TranscriptRejected,
    transcript_problem,
)
from src.domain.rag.audio_segments import speech_ratio, wav_duration
from tests.fake_model_server import FakeModelServer

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _wav(path: Path, pattern, rate=16000):
    """pattern: list of (seconds, tone: bool). Tone = 440 Hz sine, else digital silence."""
    frames = bytearray()
    for seconds, tone in pattern:
        for i in range(int(seconds * rate)):
            v = int(12000 * math.sin(2 * math.pi * 440 * i / rate)) if tone else 0
            frames += struct.pack("<h", v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return path


def _provider(srv, **kw):
    base = dict(base_url=srv.url + "/v1", api_key="k", model="whisper-1", sleep=lambda s: None)
    base.update(kw)
    return OpenAICompatibleAsrProvider(**base)


class TestGuard:
    def test_sane_text_passes(self):
        assert transcript_problem("hello world", 5, 1.0) is None

    def test_empty_with_speech_is_rejected(self):
        assert "empty" in transcript_problem("", 30, 0.9)

    def test_empty_silence_is_fine(self):
        assert transcript_problem("", 30, 0.05) is None

    def test_runaway_length(self):
        assert "runaway" in transcript_problem("x" * 1300, 30, 1.0)

    def test_short_segments_get_a_floor(self):
        assert transcript_problem("a b c " * 30, 1, 1.0) is None  # 180 chars in 1 s is under the floor

    def test_repetition_loop(self):
        assert "repetitive" in transcript_problem("thank you for watching. " * 40, 600, 1.0)

    def test_speech_ratio(self):
        assert speech_ratio(0, 10, []) == 1.0
        assert speech_ratio(0, 10, [(2, 4), (8, 20)]) == pytest.approx(0.6)
        assert speech_ratio(0, 10, [(-5, 50)]) == 0.0


@needs_ffmpeg
class TestSegmentedUpload:
    def test_short_audio_is_one_request(self, tmp_path):
        audio = _wav(tmp_path / "a.wav", [(2, True)])
        with FakeModelServer() as srv:
            r = _provider(srv).transcribe(audio_path=str(audio), file_name="talk.mp3")
            posts = srv.posts("/audio/transcriptions")
            assert r.text == "hello" and len(posts) == 1
            assert r.chunks == [{"text": "hello", "headings": ["00:00:00–00:00:02"]}]
            form = posts[0]
            assert form["model"] == "whisper-1" and form["response_format"] == "json"
            assert form["file"]["filename"] == "talk.mp3"
            assert srv.requests[-1][1] == "/v1/audio/transcriptions"
            assert srv.requests[-1][2]["Authorization"] == "Bearer k"

    def test_long_audio_is_split_at_silences_in_order(self, tmp_path):
        # 3 × (25 s tone + 2 s silence) = 81 s; 30 s segments must cut inside the silences
        audio = _wav(tmp_path / "long.wav", [(25, True), (2, False)] * 3)
        with FakeModelServer() as srv:
            srv.transcribe_reply = lambda f: {"text": f"part-{f['file']['filename'][-8:-4]}"}
            progress = []
            p = _provider(srv, segment_seconds=30, language="en")
            r = p.transcribe(audio_path=str(audio), file_name="meeting.m4a",
                             progress_cb=lambda *a: progress.append(a))
            posts = srv.posts("/audio/transcriptions")
        assert len(posts) == 3
        assert r.text.split("\n") == ["part-0000", "part-0001", "part-0002"]
        # timed records: all three segments start in the first 5-minute block
        assert [c["text"] for c in r.chunks] == ["part-0000", "part-0001", "part-0002"]
        assert len({tuple(c["headings"]) for c in r.chunks}) == 1
        assert r.chunks[0]["headings"][0].startswith("00:00:00–00:01:2")
        assert all(p["language"] == "en" for p in posts)
        # every upload stays well under the per-request limit (30 s of 16k pcm16 ≈ 1 MB)
        assert all(p["file"]["size"] < 1_100_000 for p in posts)
        assert progress[-1] == ("loading", 3, 3)

    def test_empty_transcript_for_speech_is_retried(self, tmp_path):
        audio = _wav(tmp_path / "a.wav", [(3, True)])
        replies = iter([{"text": ""}, {"text": "  recovered "}])
        with FakeModelServer() as srv:
            srv.transcribe_reply = lambda f: next(replies)
            r = _provider(srv, max_retries=2).transcribe(audio_path=str(audio), file_name="a.wav")
            assert r.text == "recovered" and len(srv.posts("/audio/transcriptions")) == 2

    def test_runaway_every_time_fails_the_file(self, tmp_path):
        audio = _wav(tmp_path / "a.wav", [(3, True)])
        with FakeModelServer() as srv:
            srv.transcribe_reply = lambda f: {"text": "la " * 500}
            with pytest.raises(TranscriptRejected, match="after 2 attempts"):
                _provider(srv, max_retries=1).transcribe(audio_path=str(audio), file_name="a.wav")

    def test_http_429_is_retried_with_backoff(self, tmp_path):
        audio = _wav(tmp_path / "a.wav", [(2, True)])
        slept = []
        with FakeModelServer() as srv:
            srv.fail_next = [(429, "1"), (502, None)]
            r = _provider(srv, max_retries=2, sleep=slept.append).transcribe(
                audio_path=str(audio), file_name="a.wav")
            assert r.text == "hello" and len(srv.posts("/audio/transcriptions")) == 3
        assert slept[0] == 1.0  # Retry-After honoured

    def test_auth_error_is_not_retried(self, tmp_path):
        audio = _wav(tmp_path / "a.wav", [(2, True)])
        with FakeModelServer() as srv:
            srv.fail_next = [(401, None)]
            with pytest.raises(httpx.HTTPStatusError):
                _provider(srv).transcribe(audio_path=str(audio), file_name="a.wav")
            assert len(srv.posts("/audio/transcriptions")) == 1

    def test_no_key_sends_no_auth_header(self, tmp_path):
        audio = _wav(tmp_path / "a.wav", [(1, True)])
        with FakeModelServer() as srv:
            _provider(srv, api_key=None, model=None).transcribe(audio_path=str(audio), file_name="a.wav")
            assert "Authorization" not in srv.requests[-1][2]
            assert "model" not in srv.posts("/audio/transcriptions")[0]

    def test_bad_progress_callback_is_ignored(self, tmp_path):
        audio = _wav(tmp_path / "a.wav", [(1, True)])

        def boom(*a):
            raise RuntimeError("ui gone")
        with FakeModelServer() as srv:
            assert _provider(srv).transcribe(audio_path=str(audio), file_name="a.wav",
                                             progress_cb=boom).text == "hello"


class TestWithoutFfmpeg:
    def _no_ffmpeg(self, monkeypatch):
        def missing(src, dst):
            raise FileNotFoundError("ffmpeg")
        monkeypatch.setattr("src.domain.rag.audio_segments._resample_to_wav16k", missing)

    def test_small_file_uploaded_whole(self, tmp_path, monkeypatch):
        self._no_ffmpeg(monkeypatch)
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"ID3" + b"\0" * 2000)
        with FakeModelServer() as srv:
            r = _provider(srv).transcribe(audio_path=str(audio), file_name="a.mp3")
            assert r.text == "hello"
            assert srv.posts("/audio/transcriptions")[0]["file"]["size"] == 2003

    def test_large_file_needs_ffmpeg(self, tmp_path, monkeypatch):
        self._no_ffmpeg(monkeypatch)
        monkeypatch.setattr(ap, "MAX_SINGLE_UPLOAD_BYTES", 10)
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x" * 100)
        with pytest.raises(RuntimeError, match="ffmpeg is required"):
            OpenAICompatibleAsrProvider(base_url="http://h/v1").transcribe(
                audio_path=str(audio), file_name="a.mp3")


def test_available_and_wav_duration(tmp_path):
    assert OpenAICompatibleAsrProvider(base_url="http://h/v1").available()
    assert wav_duration(_wav(tmp_path / "d.wav", [(1.5, False)])) == pytest.approx(1.5)
