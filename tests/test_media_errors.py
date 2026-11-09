"""/v1 media endpoints: honest model ids and 4xx for bad input (QA 2026-11-09).

- Speech-to-text reported "whisper-turbo" and /v1/models listed Whisper while
  FireRedASR did the work.
- A corrupt audio file or PDF got a 500, the audio one carrying ffmpeg's stderr.
- An unknown response_format was ignored silently (json came back).
"""

import io
import shutil
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router import media

_CFG = "src.config.config_manager.Config.get_config_model"


def _cfg(provider="fireredasr", enabled=True):
    return SimpleNamespace(rag=SimpleNamespace(asr=SimpleNamespace(
        enabled=enabled, provider=provider, base_url=None, api_key=None, model=None)))


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(media.router)
    return TestClient(app)


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def _pdf() -> bytes:
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def _png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, format="PNG")
    return buf.getvalue()


class TestModelIds:
    @pytest.mark.parametrize("provider, expected", [
        ("fireredasr", "fireredasr"), ("openai-compatible", "openai-compatible"),
        ("docling-whisper", "whisper-turbo")])
    def test_stt_model_id_follows_the_provider(self, provider, expected):
        with patch(_CFG, return_value=_cfg(provider)):
            assert media._asr_model_id() == expected

    def test_models_lists_the_configured_provider(self, client):
        converter = MagicMock(format_to_options={}, initialized_pipelines={})
        with patch(_CFG, return_value=_cfg("fireredasr")), \
                patch("src.api.router.media.get_converter", return_value=converter):
            ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
        assert "fireredasr" in ids and "whisper-turbo" not in ids


class TestResponseFormat:
    @pytest.mark.parametrize("path, field, name", [
        ("/v1/transcriptions/general", "audio", "a.wav"), ("/v1/ocr/general", "file", "a.png")])
    def test_unknown_format_is_400(self, client, path, field, name):
        r = client.post(path, files={field: (name, b"x")}, data={"response_format": "srt"})
        assert r.status_code == 400 and "response_format" in r.json()["detail"]


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
class TestAudioInput:
    def test_a_real_wav_is_decodable(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(_wav())
        assert media._decodable_audio(str(p)) is True

    def test_garbage_is_not(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(b"RIFF\x00\x00garbage" * 50)
        assert media._decodable_audio(str(p)) is False

    def test_a_readable_file_without_an_audio_stream_is_not(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(_png())  # ffprobe opens it fine, but there is nothing to listen to
        assert media._decodable_audio(str(p)) is False

    def test_corrupt_audio_is_422_without_ffmpeg_output(self, client):
        with patch(_CFG, return_value=_cfg("fireredasr")):
            r = client.post("/v1/transcriptions/general", files={"audio": ("a.mp3", b"\xff\xfb" + b"\x00" * 300)})
        assert r.status_code == 422
        assert "could not be decoded" in r.json()["detail"] and "ffmpeg" not in r.json()["detail"].lower()


class TestDocumentInput:
    @pytest.mark.parametrize("ext, good", [(".pdf", _pdf), (".png", _png)])
    def test_real_files_open(self, tmp_path, ext, good):
        p = tmp_path / f"a{ext}"
        p.write_bytes(good())
        assert media._readable_document(str(p), ext) is True

    @pytest.mark.parametrize("ext", [".pdf", ".png", ".jpg"])
    def test_corrupt_files_do_not(self, tmp_path, ext):
        p = tmp_path / f"a{ext}"
        p.write_bytes(b"%PDF-1.4 broken" if ext == ".pdf" else b"\x89PNG broken")
        assert media._readable_document(str(p), ext) is False

    def test_corrupt_pdf_is_422(self, client):
        with patch("src.domain.rag.docling_loader.ocr_enabled", return_value=True), \
                patch("src.api.router.media._cloud_parsing_selected", return_value=False):
            r = client.post("/v1/ocr/general", files={"file": ("a.pdf", b"%PDF-1.4 broken")})
        assert r.status_code == 422 and "could not be opened as a PDF" in r.json()["detail"]

    def test_an_internal_failure_does_not_echo_the_exception(self, client):
        with patch("src.domain.rag.docling_loader.ocr_enabled", return_value=True), \
                patch("src.api.router.media._cloud_parsing_selected", return_value=False), \
                patch("src.api.router.media.get_converter", side_effect=RuntimeError("secret /path/detail")):
            r = client.post("/v1/ocr/general", files={"file": ("a.png", _png())})
        assert r.status_code == 500 and "secret" not in r.json()["detail"]
