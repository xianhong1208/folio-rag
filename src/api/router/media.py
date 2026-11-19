"""The ``/v1`` media endpoints, removed in Folio 1.5: every one answers 410 Gone.

Folio runs no OCR or speech-to-text model any more (docs/migrate-1.5.md). Callers of the old
endpoints call the OpenAI-compatible server that now does the work, e.g. Forgeyard:

- ``POST /v1/ocr/general``            → a vision model, ``POST {base_url}/chat/completions``
  with the page as an ``image_url`` part
- ``POST /v1/transcriptions/general`` → ``POST {base_url}/audio/transcriptions``
- ``GET  /v1/models``                 → that server's own ``GET {base_url}/models``

Each answer carries a ``Link`` header to the migration guide. The request body is not read.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.router.response import ErrorDetailResponse

router = APIRouter(tags=["Media"], prefix="/v1")

MIGRATION_URL = "https://github.com/xianhong1208/folio-rag/blob/main/docs/migrate-1.5.md"
# Kept for callers of the 1.5 speech-to-text removal (same guide).
ASR_MIGRATION_URL = MIGRATION_URL

_REPLACEMENT = {
    "POST /v1/ocr/general": "OCR no longer runs inside Folio. Send the page image to an "
                            "OpenAI-compatible vision model (POST {base_url}/chat/completions with an "
                            "image_url part, e.g. Qwen3-VL on Forgeyard) directly",
    "POST /v1/transcriptions/general": "speech-to-text no longer runs inside Folio. Call an "
                                       "OpenAI-compatible transcription endpoint "
                                       "(POST {base_url}/audio/transcriptions, e.g. Forgeyard's) directly",
    "GET /v1/models": "Folio serves no media model any more. List the models of the "
                      "OpenAI-compatible server you call instead (GET {base_url}/models)",
}


def _gone(endpoint: str) -> JSONResponse:
    return JSONResponse(status_code=410, content={
        "detail": f"{endpoint} was removed in Folio 1.5: {_REPLACEMENT[endpoint]}. See {MIGRATION_URL}",
    }, headers={"Link": f'<{MIGRATION_URL}>; rel="help"'})


_GONE = {410: {"model": ErrorDetailResponse, "description": "Removed in Folio 1.5"}}


@router.post("/ocr/general", summary="Removed in 1.5: image / PDF → text", status_code=410,
             description=f"{_REPLACEMENT['POST /v1/ocr/general']}; see {MIGRATION_URL}.", responses=_GONE)
async def ocr_removed():
    return _gone("POST /v1/ocr/general")


@router.post("/transcriptions/general", summary="Removed in 1.5: audio → text", status_code=410,
             description=f"{_REPLACEMENT['POST /v1/transcriptions/general']}; see {MIGRATION_URL}.",
             responses=_GONE)
async def transcribe_removed():
    return _gone("POST /v1/transcriptions/general")


@router.get("/models", summary="Removed in 1.5: list media models", status_code=410,
            description=f"{_REPLACEMENT['GET /v1/models']}; see {MIGRATION_URL}.", responses=_GONE)
async def models_removed():
    return _gone("GET /v1/models")
