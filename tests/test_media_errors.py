"""/v1 media endpoints, removed in Folio 1.5: every one answers 410 Gone, says what to call
instead and links the migration guide. Nothing is read or run (no OCR, no speech model)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router import media


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(media.router)
    return TestClient(app)


GONE = [
    ("post", "/v1/ocr/general", "vision model"),
    ("post", "/v1/transcriptions/general", "transcription endpoint"),
    ("get", "/v1/models", "GET {base_url}/models"),
]


@pytest.mark.parametrize("method,path,replacement", GONE)
def test_each_endpoint_is_gone_with_the_guide(client, method, path, replacement):
    r = getattr(client, method)(path)
    assert r.status_code == 410
    detail = r.json()["detail"]
    assert f"{method.upper()} {path} was removed in Folio 1.5" in detail and replacement in detail
    assert "docs/migrate-1.5.md" in detail
    assert r.headers["link"] == f'<{media.MIGRATION_URL}>; rel="help"'


def test_an_upload_is_not_read(client):
    r = client.post("/v1/ocr/general", files={"file": ("scan.png", b"\x89PNG not really")},
                    data={"response_format": "nonsense"})
    assert r.status_code == 410


def test_no_deprecation_header_any_more(client):
    assert "deprecation" not in client.get("/v1/models").headers


def test_openapi_documents_the_410(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths["/v1/ocr/general"]["post"]["responses"]) == {"410"}
    assert set(paths["/v1/models"]["get"]["responses"]) == {"410"}
