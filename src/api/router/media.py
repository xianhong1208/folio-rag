
"""Media REST API — OCR via Docling's pre-loaded models.

Calls docling's singleton DocumentConverter directly, sharing the RapidOCR (PDF/Image OCR)
models already loaded by hierarchical_indexer. No extra loading, no proxying.

Endpoints:
- POST /v1/ocr/general             — image/pdf → text (Docling OCR pipeline)
- GET  /v1/models                  — list the loaded models
- POST /v1/transcriptions/general  — removed in 1.5: answers 410 Gone with a pointer to the
  migration guide (call the OpenAI-compatible transcription endpoint directly)
"""
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse

from src.api.router.response import ErrorDetailResponse
from src.domain.rag.docling_loader import get_converter, DOCLING_INFER_LOCK
from src.log import get_api_logger

logger = get_api_logger()

# The docling converter is a process-wide singleton that the indexing path also runs convert() on.
# Reuse docling_loader's same inference lock so "indexing" and "/v1/ocr" are serialized against
# each other too; otherwise concurrent forwards on the same instance CUDA crash / OOM.
_DOCLING_INFER_LOCK = DOCLING_INFER_LOCK

# Deliberately unauthenticated: OCR callers can't obtain a token, so auth is waived and only a
# 100MB upload cap guards against abuse. The cost is that anonymous callers can drive OCR
# GPU inference — suitable for internal-network deployment only. To lock it back down, re-add
# dependencies=[Depends(authenticate_request)].
router = APIRouter(
    tags=["Media"],
    prefix="/v1",
)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp", ".pdf"}

# Upload size cap — matches FileDB.MAX_FILE_SIZE (100MB). UploadFile streams to a temp file, so
# without a cap a single request could fill the temp disk.
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# Where callers of the removed speech-to-text endpoint learn what to call instead.
ASR_MIGRATION_URL = "https://github.com/xianhong1208/folio-rag/blob/main/docs/migrate-asr.md"


_RESPONSE_FORMATS = ("json", "text")


def _check_response_format(value: str) -> None:
    """Only json and text exist; anything else used to be ignored silently (json came back)."""
    if value not in _RESPONSE_FORMATS:
        raise HTTPException(400, f"Unsupported response_format: {value!r}. Use one of {list(_RESPONSE_FORMATS)}.")


def _readable_document(path: str, ext: str) -> bool:
    """Whether a PDF or image opens at all, before the OCR pipeline gets it."""
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            return len(PdfReader(path).pages) > 0
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _cloud_parsing_selected(file_name: str) -> bool:
    try:
        from src.config.config_manager import Config
        from src.domain.rag.cloud_parser import is_cloud_parsed
        rag = getattr(Config.get_config_model(), "rag", None)
        return bool(rag is not None and is_cloud_parsed(rag, file_name))
    except Exception:
        return False


async def _save_upload_to_temp(file: UploadFile) -> tuple[str, int]:
    """Stream-write to a temp file, avoiding loading the whole audio into memory at once.

    Args:
        file: The FastAPI UploadFile object.

    Returns:
        ``(temp_path, bytes_written)`` — the caller must unlink it when done.
    """
    suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    bytes_written = 0
    try:
        while chunk := await file.read(1 << 20):  # 1MB
            bytes_written += len(chunk)
            if bytes_written > _MAX_UPLOAD_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Upload exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit"
                    ),
                )
            tmp.write(chunk)
    except BaseException:
        # On the failure path (including CancelledError from client disconnect, and 413), clean up
        # the half-written temp so it doesn't linger on disk. The success path does not unlink —
        # the caller needs tmp.name.
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise
    finally:
        tmp.close()
    return tmp.name, bytes_written


def _unlink_safe(path: Optional[str]) -> None:
    if path:
        Path(path).unlink(missing_ok=True)


def _fmt_size(n: int) -> str:
    """Human-friendly byte size: 1234 → '1.2 KB'."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore
    return f"{n:.1f} TB"


@router.post(
    "/transcriptions/general",
    summary="Removed in 1.5: audio → text",
    description="Folio no longer runs speech-to-text itself. Call your OpenAI-compatible "
                "transcription endpoint (POST {base_url}/audio/transcriptions, e.g. Forgeyard's) "
                f"directly; see {ASR_MIGRATION_URL}.",
    status_code=410,
    responses={410: {"model": ErrorDetailResponse, "description": "Removed in Folio 1.5"}},
)
async def transcribe_removed():
    """410 Gone with the migration pointer (the request body is not read)."""
    return JSONResponse(status_code=410, content={
        "detail": "POST /v1/transcriptions/general was removed in Folio 1.5: speech-to-text no longer "
                  "runs inside Folio. Call an OpenAI-compatible transcription endpoint "
                  "(POST {base_url}/audio/transcriptions, e.g. Forgeyard's) directly. "
                  f"See {ASR_MIGRATION_URL}",
    }, headers={"Link": f'<{ASR_MIGRATION_URL}>; rel="help"'})


# Response = json {text, pages, mime_type}, or plain text (response_format=text)
@router.post(
    "/ocr/general",
    summary="Image / PDF → text via Docling OCR",
    description="Shares the OCR pipeline already loaded by Docling (RapidOCR torch backend).",
    responses={
        200: {
            "description": "json: {text, pages, mime_type} / text: plain extracted text",
            "content": {
                "application/json": {"example": {"text": "...", "pages": 3, "mime_type": "application/pdf"}},
                "text/plain": {"example": "..."},
            },
        },
        415: {"model": ErrorDetailResponse, "description": "Unsupported image / PDF format"},
        500: {"model": ErrorDetailResponse, "description": "OCR inference failed"},
    },
)
async def ocr(
    file: UploadFile = File(...),
    response_format: str = Form("json"),
):
    """Args:
        file:             Image or PDF (png / jpg / bmp / tiff / webp / pdf …)
        response_format:  json | text (default json)
    """
    t_start = time.time()
    file_name = file.filename or "image"
    ext = Path(file_name).suffix.lower()
    logger.info(
        f"[ocr] ⮕ request received | file={file_name} ext={ext} "
        f"content_type={file.content_type} response_format={response_format}"
    )
    _check_response_format(response_format)
    if ext not in _IMAGE_EXTS:
        logger.warning(f"[ocr] ✗ rejected unsupported ext={ext} for {file_name}")
        raise HTTPException(415, f"Unsupported format: {ext}. Allowed: {sorted(_IMAGE_EXTS)}")
    if _cloud_parsing_selected(file_name):
        # This endpoint is unauthenticated; forwarding anonymous uploads to the paid
        # vision model would let anyone spend the parsing budget, so it is not offered.
        raise HTTPException(
            503, "Local OCR is not loaded: rag.parsing sends PDFs/images to a cloud vision "
                 "model, which this unauthenticated endpoint does not expose")
    from src.domain.rag.docling_loader import ocr_enabled
    if not ocr_enabled():
        raise HTTPException(503, "OCR is disabled (rag.docling.ocr_enabled=false)")

    t_save = time.time()
    src_path, bytes_written = await _save_upload_to_temp(file)
    save_elapsed = time.time() - t_save
    logger.info(
        f"[ocr] · upload saved | size={_fmt_size(bytes_written)} "
        f"tmp={src_path} save_elapsed={save_elapsed:.2f}s"
    )

    try:
        if not await run_in_threadpool(_readable_document, src_path, ext):
            raise HTTPException(422, f"{file_name} could not be opened as {'a PDF' if ext == '.pdf' else 'an image'}.")
        queued = _DOCLING_INFER_LOCK.locked()
        if queued:
            logger.info("[ocr] · docling pipeline queued (another request holds docling lock)")
        else:
            logger.info("[ocr] · docling pipeline start")
        t_lock_request = time.time()

        # docling's PdfPipeline contains a TableFormer transformer + RapidOCR session,
        # with the same thread-safety concern, so serialize. Time lock-wait / compute separately.
        def _run_docling_ocr():
            from src.domain.rag.docling_guard import convert_checked, release_accelerator_cache
            spent = [0.0]

            def _convert():
                with _DOCLING_INFER_LOCK:
                    t_compute = time.time()
                    try:
                        return get_converter().convert(src_path)
                    finally:
                        spent[0] += time.time() - t_compute

            try:
                # Same guard as indexing: retries a partial / out-of-memory conversion.
                res, _warning = convert_checked(_convert, file_name)
            finally:
                release_accelerator_cache()
            return res, spent[0]

        result, compute_seconds = await run_in_threadpool(_run_docling_ocr)
        text = result.document.export_to_markdown()
        total_lock_phase = time.time() - t_lock_request
        wait_seconds = max(0.0, total_lock_phase - compute_seconds)
        logger.info(
            f"[ocr] · docling pipeline done | chars={len(text)} "
            f"compute={compute_seconds:.2f}s wait_for_lock={wait_seconds:.2f}s"
        )

        total_elapsed = time.time() - t_start
        logger.info(
            f"[ocr] ⮕ ✓ done | file={file_name} size={_fmt_size(bytes_written)} "
            f"chars={len(text)} total_elapsed={total_elapsed:.2f}s"
        )

        if response_format == "text":
            # Match STT endpoint: text/plain instead of JSON-encoded string
            return PlainTextResponse(content=text, media_type="text/plain; charset=utf-8")

        return {
            "text": text,
            "metadata": {
                "model": "docling-ocr",
                "filename": file_name,
                "file_size_bytes": bytes_written,
                "elapsed_seconds": round(total_elapsed, 2),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[ocr] ✗ failed | file={file_name} "
            f"size={_fmt_size(bytes_written)} error={e}"
        )
        raise HTTPException(500, "OCR failed; see the server log.") from e
    finally:
        _unlink_safe(src_path)


# Response = {object: "list", data: [{id, object, type, owned_by, backend, warmed}, ...]}
# Mirrors the OpenAI /v1/models style; type = "ocr" (speech-to-text left Folio in 1.5)
@router.get(
    "/models",
    summary="List loaded media models",
    description="Reports the status of the OCR pipeline loaded by Docling.",
    responses={500: {"model": ErrorDetailResponse}},
)
async def list_models():
    """OpenAI `/v1/models`-style response."""
    try:
        from docling.datamodel.base_models import InputFormat
        converter = get_converter()
        fmt_opts = getattr(converter, "format_to_options", {}) or {}
        # initialized_pipelines is a dict keyed by (PipelineClass, hash); check class names
        pipeline_classes = {
            getattr(key[0], "__name__", "") for key in
            getattr(converter, "initialized_pipelines", {}).keys()
        }

        data = []
        if InputFormat.PDF in fmt_opts or InputFormat.IMAGE in fmt_opts:
            data.append({
                "id": "docling-ocr",
                "object": "model",
                "type": "ocr",
                "owned_by": "ibm-docling",
                "backend": "rapidocr-torch",
                "warmed": "StandardPdfPipeline" in pipeline_classes,
            })
        return {"object": "list", "data": data}
    except Exception as e:
        logger.error(f"[list_models] failed: {e}")
        raise HTTPException(500, f"Failed to query models: {e}") from e
