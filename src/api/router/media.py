
"""Media REST API — STT / OCR via Docling's pre-loaded models.

Calls docling's singleton DocumentConverter directly, sharing the Whisper turbo (ASR) + RapidOCR
(PDF/Image OCR) models already loaded by hierarchical_indexer. No extra loading, no proxying to an
external container.

Endpoints:
- POST /v1/transcriptions/general  — audio → text (backend follows rag.asr.provider)
- POST /v1/ocr/general             — image/pdf → text (Docling OCR pipeline)
- GET  /v1/models                  — list the loaded models
"""
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from src.api.router.response import ErrorDetailResponse
from src.domain.rag.docling_loader import get_converter, DOCLING_INFER_LOCK
from src.domain.rag.audio_defense import (
    _AUDIO_TRIM_EXTENSIONS,
    _filter_whisper_hallucinations,
    _trim_audio_leading_silence,
)
from src.log import get_api_logger

logger = get_api_logger()

# PyTorch transformer models are not thread-safe — two concurrent transcribe() calls on the same
# model instance trigger a KV-cache race such as "cannot reshape tensor of 0 elements".
# A single GPU can't run two forwards in parallel anyway, so serializing costs no throughput.
# STT and OCR use different models, so their independent locks don't block each other.
_WHISPER_INFER_LOCK = threading.Lock()
# The docling converter is a process-wide singleton that the indexing path also runs convert() on.
# Reuse docling_loader's same inference lock so "indexing" and "/v1/ocr, /v1/transcriptions" are
# serialized against each other too; otherwise concurrent forwards on the same instance CUDA crash / OOM.
_DOCLING_INFER_LOCK = DOCLING_INFER_LOCK

# Deliberately unauthenticated: STT/OCR callers can't obtain a token, so auth is waived and only a
# 100MB upload cap guards against abuse. The cost is that anonymous callers can drive Whisper/OCR
# GPU inference — suitable for internal-network deployment only. To lock it back down, re-add
# dependencies=[Depends(authenticate_request)].
router = APIRouter(
    tags=["Media"],
    prefix="/v1",
)


_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".webm"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp", ".pdf"}

# Upload size cap — matches FileDB.MAX_FILE_SIZE (100MB). UploadFile streams to a temp file, so
# without a cap a single request could fill the temp disk.
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _get_config_asr_provider():
    """The speech source for media STT = config `rag.asr.provider`, the same one used for indexing —
    configured in one place, effective for all speech tasks.

    - docling-whisper (or config default) -> returns (None, None): use the original whisper-direct
      path (the *same* model chosen by config, just called directly to retain language/prompt control)
    - fireredasr / openai-compatible -> returns (provider, name); when unavailable the caller returns
      503, never silently falling back to a different model (whatever config chose is what runs)
    - non-whisper providers don't support language/prompt (FireRedASR is a Chinese/English AED with
      no prompt bias) — the endpoint logs and ignores them if received
    """
    try:
        from src.config.config_manager import Config as _Cfg
        asr_cfg = getattr(getattr(_Cfg.get_config_model(), "rag", None), "asr", None)
    except Exception:
        asr_cfg = None  # config unreadable -> treat as default (docling-whisper legacy semantics)
    if asr_cfg is None or not asr_cfg.enabled or asr_cfg.provider == "docling-whisper":
        return None, None
    from src.domain.rag.asr_provider import create_asr_provider
    return create_asr_provider(asr_cfg), asr_cfg.provider


def _get_loaded_whisper():
    """Extract the underlying whisper.Whisper instance from docling's initialized AsrPipeline.

    docling's _NativeWhisperModel.transcribe() doesn't pass the options' `language` through to
    whisper (it's ignored) and doesn't expose `initial_prompt`. Taking the raw whisper model and
    calling it directly is the only way to fully control these parameters.

    Returns whisper model instance (with `.transcribe(audio, language=..., initial_prompt=...)`),
    or None if the AsrPipeline isn't warm yet.
    """
    converter = get_converter()
    pipes = getattr(converter, "initialized_pipelines", {}) or {}
    for (klass, _hash), pipe in pipes.items():
        if klass.__name__ == "AsrPipeline":
            inner = getattr(pipe, "_model", None)
            return getattr(inner, "model", None)
    return None


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


# Extensions where docling's format dispatcher reliably routes to the
# AudioPipeline. Everything else (.webm, future containers) gets
# transcoded to .wav first — see _transcode_to_wav's docstring for details.
_DOCLING_AUDIO_SAFE_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}


def _transcode_to_wav(input_path: str) -> str:
    """Transcode an audio file to 16 kHz mono WAV via ffmpeg.

    Why this exists:
        docling's format dispatcher runs a three-stage waterfall —
        ``filetype.guess_mime`` (magic bytes) → extension lookup →
        content sniff — and the magic-byte stage is *content*-driven,
        not extension-driven. Renaming ``.webm`` to ``.ogg`` does not
        help: the magic-byte stage sees the EBML/Matroska header
        (``1A 45 DF A3``) and tags the file as webm regardless of the
        renamed extension. docling has no webm → audio mapping, so the
        dispatcher returns ``None`` and the convert call fails before
        AudioPipeline can ever see it.

        Rather than fight the dispatcher, run a tiny ffmpeg pass that
        re-packages whatever container the browser sent (typically
        webm/opus on Chrome, mp4/aac on Safari) into a plain
        ``RIFF....WAVE`` PCM-16 WAV at 16 kHz mono. docling's first
        waterfall stage recognizes WAV by magic bytes, routes to
        AudioPipeline, and Whisper runs — the same downstream pipeline
        that already works for ``.wav`` uploads.

    Performance:
        For a 60-second browser recording (~100 KB opus, ~1.9 MB as wav
        at 16k mono) the transcode runs in ~150 ms on a modern CPU —
        negligible compared to the 1-3 second Whisper inference that
        follows. The wav is bigger on disk but nobody keeps it — the
        caller unlinks it after the response.

    Args:
        input_path: Source audio path. ffmpeg auto-detects the
            container from content, so the extension can be wrong
            or missing.

    Returns:
        Path to a new ``.wav`` temp file. Caller owns it (unlink
        when done).

    Raises:
        RuntimeError: ffmpeg failure. Caller should surface this
            as a 502 so the frontend knows to retry.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_path,
        "-vn",           # drop any stray cover-art stream that would break WAV output
        "-ar", "16000",  # Whisper's native sample rate — avoids an internal resample
        "-ac", "1",      # mono
        "-f", "wav",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        Path(out_path).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr[:500]}")
    return out_path


def _fmt_size(n: int) -> str:
    """Human-friendly byte size: 1234 → '1.2 KB'."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore
    return f"{n:.1f} TB"


# Response = json (default) with {text, language, duration, segments[]}, or plain text (response_format=text)
@router.post(
    "/transcriptions/general",
    summary="Audio → text (follows rag.asr.provider)",
    description="OpenAI-compatible-ish media STT endpoint. Backend follows config rag.asr.provider: docling-whisper (shares the loaded Whisper turbo, supports language/prompt) or fireredasr / openai-compatible (same AsrProvider as indexing; language/prompt ignored).",
    responses={
        200: {
            "description": "json: {text, language, duration, segments} / text: plain transcript",
            "content": {
                "application/json": {"example": {"text": "The weather is nice today.", "language": "zh", "duration": 12.3}},
                "text/plain": {"example": "The weather is nice today."},
            },
        },
        415: {"model": ErrorDetailResponse, "description": "Unsupported audio format"},
        500: {"model": ErrorDetailResponse, "description": "Whisper inference failed"},
    },
)
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    trim_leading_silence: bool = Form(True),
    strip_hallucinations: bool = Form(True),
):
    """Args:
        audio:                 Audio file (mp3 / wav / m4a / flac / ogg / aac / opus) — field name aligned with mistt
        language:              ISO-639-1 language code (e.g. 'zh' / 'en' / 'ja'); None = Whisper auto-detect
        prompt:                Context hint for Whisper (bias toward proper nouns / domain terms); default None
        response_format:       json | text (default json)
        trim_leading_silence:  Whether to first strip leading silence with ffmpeg silencedetect (default True)
        strip_hallucinations:  Whether to apply Whisper hallucination filtering + repetition collapse (default True)
    """
    t_start = time.time()
    file_name = audio.filename or "audio"
    ext = Path(file_name).suffix.lower()
    logger.info(
        f"[transcribe] ⮕ request received | file={file_name} ext={ext} "
        f"content_type={audio.content_type} "
        f"lang={language or 'auto'} prompt={'set' if prompt else 'none'} "
        f"trim={trim_leading_silence} strip_hallucinations={strip_hallucinations} "
        f"response_format={response_format}"
    )
    if ext not in _AUDIO_EXTS:
        logger.warning(f"[transcribe] ✗ rejected unsupported ext={ext} for {file_name}")
        raise HTTPException(415, f"Unsupported audio format: {ext}. Allowed: {sorted(_AUDIO_EXTS)}")

    t_save = time.time()
    src_path, bytes_written = await _save_upload_to_temp(audio)
    save_elapsed = time.time() - t_save
    logger.info(
        f"[transcribe] · upload saved | size={_fmt_size(bytes_written)} "
        f"tmp={src_path} save_elapsed={save_elapsed:.2f}s"
    )

    trimmed_path: Optional[str] = None
    transcoded_path: Optional[str] = None
    effective_path = src_path
    try:
        # Transcode docling-unfriendly containers (e.g. browser .webm) to WAV first;
        # see _transcode_to_wav. Skip the safe extensions to keep the happy path overhead-free.
        if ext not in _DOCLING_AUDIO_SAFE_EXTS:
            t_transcode = time.time()
            try:
                transcoded_path = await run_in_threadpool(_transcode_to_wav, src_path)
                effective_path = transcoded_path
                logger.info(
                    f"[transcribe] · transcoded → wav | "
                    f"new={transcoded_path} elapsed={time.time() - t_transcode:.2f}s"
                )
            except Exception as e:
                logger.error(
                    f"[transcribe] ✗ transcode failed for {file_name}: {e}"
                )
                raise HTTPException(
                    status_code=502,
                    detail=f"Audio transcode failed: {e}",
                )

        if trim_leading_silence and ext in _AUDIO_TRIM_EXTENSIONS:
            t_trim = time.time()
            try:
                trimmed_path = await run_in_threadpool(
                    _trim_audio_leading_silence, src_path, file_name
                )
                trim_elapsed = time.time() - t_trim
                if trimmed_path:
                    effective_path = trimmed_path
                    logger.info(
                        f"[transcribe] · leading silence trimmed | "
                        f"new={trimmed_path} elapsed={trim_elapsed:.2f}s"
                    )
                else:
                    logger.info(
                        f"[transcribe] · no leading silence to trim "
                        f"(elapsed={trim_elapsed:.2f}s)"
                    )
            except Exception as e:
                trim_elapsed = time.time() - t_trim
                logger.warning(
                    f"[transcribe] ⚠ trim failed for {file_name} "
                    f"(elapsed={trim_elapsed:.2f}s, using original file): {e}"
                )

        cfg_provider, cfg_provider_name = _get_config_asr_provider()
        whisper_model = None if cfg_provider is not None else _get_loaded_whisper()
        if cfg_provider is not None:
            if not cfg_provider.available():
                # The config-selected model is unavailable -> explicit 503; do not swap in another model
                logger.error(
                    f"[transcribe] ✗ configured provider {cfg_provider_name} unavailable "
                    "(weights/endpoint not ready)"
                )
                raise HTTPException(
                    503,
                    f"rag.asr.provider={cfg_provider_name} unavailable (weights/endpoint not ready) — "
                    "check assets/ weights or the rag.asr configuration",
                )
            backend = f"asr-provider:{cfg_provider_name}"
            if language or prompt:
                logger.info(
                    f"[transcribe] · language/prompt not applicable to {cfg_provider_name}, ignored"
                )
        else:
            backend = "whisper-direct" if whisper_model is not None else "docling-asr"
        queued = _WHISPER_INFER_LOCK.locked()
        if queued:
            logger.info(
                f"[transcribe] · inference queued | backend={backend} "
                f"(another request holds whisper lock)"
            )
        else:
            logger.info(f"[transcribe] · inference start | backend={backend}")
        t_lock_request = time.time()

        if cfg_provider is not None:
            # The config-specified AsrProvider (fireredasr / openai-compatible) —
            # same implementation as the indexing path; fireredasr has its own internal inference lock (FIRERED_INFER_LOCK)
            def _run_cfg_provider():
                t_compute = time.time()
                r = cfg_provider.transcribe(
                    audio_path=effective_path, file_name=file_name)
                return r.text, time.time() - t_compute

            text, compute_seconds = await run_in_threadpool(_run_cfg_provider)
            text = text.strip()
            detected_lang = None
        elif whisper_model is not None:
            # Call the underlying whisper.Whisper.transcribe directly -> full language + initial_prompt control.
            # Serialize with a lock to avoid the KV-cache race (PyTorch transformer is not thread-safe).
            # Time lock-wait and actual inference separately so the metric distinguishes queue vs compute latency.
            def _run_whisper():
                with _WHISPER_INFER_LOCK:
                    t_compute = time.time()
                    result = whisper_model.transcribe(
                        effective_path,
                        language=language,
                        initial_prompt=prompt,
                        verbose=False,
                        word_timestamps=False,
                    )
                    return result, time.time() - t_compute

            result, compute_seconds = await run_in_threadpool(_run_whisper)
            text = result.get("text", "").strip()
            detected_lang = result.get("language")
        else:
            # AsrPipeline isn't warm yet -> fall back to docling.convert (loses language/prompt)
            def _run_docling_asr():
                with _DOCLING_INFER_LOCK:
                    t_compute = time.time()
                    return get_converter().convert(effective_path), time.time() - t_compute

            doc_result, compute_seconds = await run_in_threadpool(_run_docling_asr)
            text = doc_result.document.export_to_markdown()
            detected_lang = None

        total_lock_phase = time.time() - t_lock_request
        wait_seconds = max(0.0, total_lock_phase - compute_seconds)
        logger.info(
            f"[transcribe] · inference done | backend={backend} "
            f"raw_chars={len(text)} detected_lang={detected_lang or '?'} "
            f"compute={compute_seconds:.2f}s wait_for_lock={wait_seconds:.2f}s"
        )

        n_stripped = 0
        if strip_hallucinations:
            t_filter = time.time()
            text, n_stripped = _filter_whisper_hallucinations(text)
            filter_elapsed = time.time() - t_filter
            if n_stripped > 0:
                logger.info(
                    f"[transcribe] · hallucinations stripped | count={n_stripped} "
                    f"final_chars={len(text)} elapsed={filter_elapsed:.3f}s"
                )

        total_elapsed = time.time() - t_start
        # Preview head + tail of transcript so operators can verify content without
        # opening a separate log search. Truncate aggressively (60 + 60 chars) — full
        # transcript is in the response body; this is for quick sanity check.
        def _preview(s: str, n: int = 60) -> str:
            s = " ".join(s.split())  # collapse whitespace for log readability
            if len(s) <= n * 2 + 5:
                return s
            return f"{s[:n]} … {s[-n:]}"

        logger.info(
            f"[transcribe] ⮕ ✓ done | file={file_name} "
            f"size={_fmt_size(bytes_written)} chars={len(text)} "
            f"lang={language or detected_lang or 'auto'} "
            f"trimmed={trimmed_path is not None} "
            f"hallucinations_stripped={n_stripped} "
            f"backend={backend} total_elapsed={total_elapsed:.2f}s"
        )
        logger.info(f"[transcribe] · preview: {_preview(text)!r}")

        if response_format == "text":
            # Only PlainTextResponse sets Content-Type: text/plain correctly
            # Otherwise FastAPI's default JSON serialization wraps the str as "..." (whole thing quoted + escaped)
            return PlainTextResponse(content=text, media_type="text/plain; charset=utf-8")

        return {
            "text": text,
            "metadata": {
                "model": "whisper-turbo",
                "backend": backend,
                "filename": file_name,
                "file_size_bytes": bytes_written,
                "language": language or detected_lang,
                "prompt_used": bool(prompt),
                "trimmed_leading_silence": trimmed_path is not None,
                "hallucinations_stripped": n_stripped,
                "elapsed_seconds": round(total_elapsed, 2),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[transcribe] ✗ failed | file={file_name} "
            f"size={_fmt_size(bytes_written)} error={e}"
        )
        raise HTTPException(500, f"Transcription failed: {e}") from e
    finally:
        _unlink_safe(trimmed_path)
        _unlink_safe(transcoded_path)
        _unlink_safe(src_path)


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
    if ext not in _IMAGE_EXTS:
        logger.warning(f"[ocr] ✗ rejected unsupported ext={ext} for {file_name}")
        raise HTTPException(415, f"Unsupported format: {ext}. Allowed: {sorted(_IMAGE_EXTS)}")

    t_save = time.time()
    src_path, bytes_written = await _save_upload_to_temp(file)
    save_elapsed = time.time() - t_save
    logger.info(
        f"[ocr] · upload saved | size={_fmt_size(bytes_written)} "
        f"tmp={src_path} save_elapsed={save_elapsed:.2f}s"
    )

    try:
        queued = _DOCLING_INFER_LOCK.locked()
        if queued:
            logger.info("[ocr] · docling pipeline queued (another request holds docling lock)")
        else:
            logger.info("[ocr] · docling pipeline start")
        t_lock_request = time.time()

        # docling's PdfPipeline contains a TableFormer transformer + RapidOCR session,
        # with the same thread-safety concern, so serialize. Time lock-wait / compute separately.
        def _run_docling_ocr():
            with _DOCLING_INFER_LOCK:
                t_compute = time.time()
                return get_converter().convert(src_path), time.time() - t_compute

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
        raise HTTPException(500, f"OCR failed: {e}") from e
    finally:
        _unlink_safe(src_path)


# Response = {object: "list", data: [{id, object, type, owned_by, backend, warmed}, ...]}
# Mirrors the OpenAI /v1/models style; type ∈ "transcription" / "ocr"
@router.get(
    "/models",
    summary="List loaded media models",
    description="Reports the status of the STT / OCR pipelines loaded by Docling.",
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
        if InputFormat.AUDIO in fmt_opts:
            data.append({
                "id": "whisper-turbo",
                "object": "model",
                "type": "transcription",
                "owned_by": "openai",
                "backend": "docling+whisper",
                "warmed": "AsrPipeline" in pipeline_classes,
            })
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
