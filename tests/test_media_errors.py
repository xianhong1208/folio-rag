"""/v1 media endpoints: 4xx for bad input (QA 2026-11-09), and the removed speech-to-text.

- A corrupt PDF got a 500.
- An unknown response_format was ignored silently (json came back).
- 1.5: POST /v1/transcriptions/general answers 410 Gone with the migration guide, and
  /v1/models lists no speech-to-text model (Folio runs none).
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router import media

@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(media.router)
    return TestClient(app)


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


class TestSpeechToTextRemoved:
    def test_transcriptions_endpoint_is_gone_with_the_guide(self, client):
        r = client.post("/v1/transcriptions/general", files={"audio": ("a.wav", b"RIFF....WAVE")})
        assert r.status_code == 410
        assert "removed in Folio 1.5" in r.json()["detail"]
        assert "docs/migrate-asr.md" in r.json()["detail"] and "migrate-asr.md" in r.headers["link"]

    def test_gone_without_a_body_too(self, client):
        assert client.post("/v1/transcriptions/general").status_code == 410

    @pytest.mark.parametrize("ocr_formats", [True, False])
    def test_models_lists_no_speech_model(self, client, ocr_formats):
        from docling.datamodel.base_models import InputFormat
        opts = {InputFormat.PDF: object(), InputFormat.AUDIO: object()} if ocr_formats else {InputFormat.AUDIO: object()}
        converter = MagicMock(format_to_options=opts, initialized_pipelines={})
        with patch("src.api.router.media.get_converter", return_value=converter):
            data = client.get("/v1/models").json()["data"]
        assert [m["type"] for m in data] == (["ocr"] if ocr_formats else [])


class TestResponseFormat:
    def test_unknown_format_is_400(self, client):
        r = client.post("/v1/ocr/general", files={"file": ("a.png", b"x")}, data={"response_format": "srt"})
        assert r.status_code == 400 and "response_format" in r.json()["detail"]


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
