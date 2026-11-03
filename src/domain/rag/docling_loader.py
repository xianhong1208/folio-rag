
"""Docling-based document loader with structure-aware chunking

Uses IBM's Docling to parse PDF/DOCX/PPTX with full structure preservation:
- Tables are kept as complete Markdown chunks (never split)
- Document hierarchy (headings, sections) is respected
- HybridChunker ensures chunks fit within embedding model token limits

Replaces: SimpleDirectoryReader + SentenceSplitter for structured documents.

Device configuration comes from config.yaml:
    rag:
      docling:
        device: "cuda:4"    # "cpu", "cuda", "cuda:0", "cuda:4"
        ocr_enabled: true
        hf_offline: true     # recommended for packaged environments
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

from src.log import get_api_logger
from src.utils.runtime_paths import resolve_external_dir
from src.domain.rag.docling_guard import (
    convert_checked,
    missing_pages,
    release_accelerator_cache,
)

logger = get_api_logger()

# Supported file types for Docling. Documents / images / audio each map to a
# Docling FormatOption; the audio path only activates when the Whisper model
# exists, else it falls back to SimpleDirectoryReader. This set is kept aligned
# to docling's official format list by a programmatic invariant in
# tests/test_upload_formats.py (see its _EXCLUDED_FORMATS for exclusions).
DOCLING_EXTENSIONS = {
    # PDF / Office Open XML (including template/macro variants)
    ".pdf",
    ".docx", ".dotx", ".docm", ".dotm",
    ".pptx", ".ppsx", ".pptm", ".potm", ".ppsm",  # potx: docling can't open it in practice
    ".xlsx", ".xlsm",
    # Legacy Office (docling 2.119+; internally converted to modern formats via LibreOffice `soffice`
    # before parsing — the deployment image/machine must have libreoffice-writer installed, or indexing
    # fails explicitly. Quality verified: .doc and .docx of the same content give 51 chunks, 96% line overlap)
    ".doc", ".dot", ".xls", ".xlt", ".ppt", ".pot", ".pps",
    # OpenDocument (including template variants)
    ".odt", ".ods", ".odp",  # template variants ott/ots/otp: docling's odfdo backend can't open them in practice
    # Markup / scientific documents
    ".html", ".htm", ".xhtml", ".md", ".qmd", ".rmd",
    ".adoc", ".asciidoc", ".asc", ".tex", ".latex",
    # Email (.msg = python-oxmsg, pure Python; .eml = MIME) / ebooks / subtitles
    ".msg", ".eml", ".epub", ".vtt",
    # Images
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
    # Audio (docling's 6 official audio types; transcription goes through AsrProvider)
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac",
}

# Lazy-init flag (ensures the env is applied only once)
_docling_env_applied = False

# Singleton DocumentConverter (preloaded at startup to avoid rebuilding models).
_converter = None
# Construction lock: get_converter runs from multiple OS threads (asyncio.to_thread /
# run_in_threadpool); without it, the first concurrent wave would each build a
# converter (several-second GPU load ×N → wasted VRAM/OOM).
_converter_lock = threading.Lock()
# Inference lock. The converter is a process-wide singleton shared by indexing and
# media's /v1/ocr, /v1/transcriptions; concurrent forwards on the same
# TableFormer / RapidOCR / Whisper instance race the KV-cache → tensor reshape
# errors, CUDA illegal memory, OOM. Every converter.convert() must hold this lock.
# media.py imports this same lock — do not create separate ones.
DOCLING_INFER_LOCK = threading.Lock()

# Page-level parse progress (OCR progress bar). docling has no progress callback,
# so we observe pages as they complete and report ("loading", done, total).
# thread-local, since convert runs synchronously in one worker thread; media.py's
# OCR path leaves it unset for zero behavior change.
import threading as _threading

_page_progress_tls = _threading.local()
_page_patch_installed = False


class _PageNotifyList(list):
    """A list that reports page-level progress on append — swapped into ProcessingResult.pages/failed_pages.

    Assumption: in docling 2.119's threaded pipeline, the collection loop (proc.pages.append) runs on
    the same thread that called convert (worker stages only write to a queue), so reading ctx here
    needs no cross-thread synchronization. If this assumption breaks in a future version, the symptom
    is progress stuck at zero (ctx is TLS, unreadable from a worker thread), not corruption or a crash.
    """

    def __init__(self, ctx: dict):
        super().__init__()
        self._ctx = ctx

    def append(self, item):
        super().append(item)
        ctx = self._ctx
        ctx["done"] += 1
        cb = ctx.get("cb")
        if cb is not None:
            try:
                cb("loading", ctx["done"], ctx.get("total"))
            except Exception:  # progress reporting must never kill parsing
                pass


def _install_page_progress_patch() -> None:
    """Install the page-level progress observer (idempotent). Pure observation: zero behavior change when TLS is unset.

    Two docling generations have different hook points; install both, each guarded independently:
    - 2.118+: StandardPdfPipeline is a threaded staged pipeline whose collection loop appends pages
      into ProcessingResult on the caller's thread → replace it with _PageNotifyList
    - ≤2.111: StandardPdfPipeline inherits PaginatedPipeline, streaming page by page through the
      _apply_on_pages generator → wrap it with a counter
    """
    global _page_patch_installed
    if _page_patch_installed:
        return
    _page_patch_installed = True

    # --- Hook point 1: 2.118+ threaded pipeline ---
    try:
        from docling.pipeline.standard_pdf_pipeline import ProcessingResult

        _orig_init = ProcessingResult.__init__

        def _init_with_progress(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            ctx = getattr(_page_progress_tls, "ctx", None)
            if ctx is not None:
                # plain dataclass attribute replacement; success_count/failure_count use len(),
                # so the list subclass is fully compatible
                self.pages = _PageNotifyList(ctx)
                self.failed_pages = _PageNotifyList(ctx)

        ProcessingResult.__init__ = _init_with_progress
        logger.info("[DOCLING] page-progress observer installed (ProcessingResult)")
    except ImportError:
        pass

    # --- Hook point 2: ≤2.111 paginated pipeline ---
    try:
        from docling.pipeline.base_pipeline import PaginatedPipeline

        _orig_apply = PaginatedPipeline._apply_on_pages

        def _apply_with_progress(self, conv_res, page_batch):
            ctx = getattr(_page_progress_tls, "ctx", None)
            for page in _orig_apply(self, conv_res, page_batch):
                if ctx is not None:
                    ctx["done"] += 1
                    cb = ctx.get("cb")
                    if cb is not None:
                        try:
                            cb("loading", ctx["done"], ctx.get("total"))
                        except Exception:
                            pass
                yield page

        PaginatedPipeline._apply_on_pages = _apply_with_progress
        logger.info("[DOCLING] page-progress observer installed (PaginatedPipeline)")
    except (ImportError, AttributeError):
        pass


def _pdf_page_count(file_path: str) -> Optional[int]:
    """Compute a PDF's total page count instantly via pymupdf (progress-bar denominator); returns None for non-PDF / open failure."""
    try:
        import fitz
        with fitz.open(file_path) as doc:
            return len(doc)
    except Exception:
        return None


def resolve_assets_dir() -> Optional[Path]:
    """Resolve the assets/ resource directory (shared API; document_indexer.py also imports this).

    The resource directory holds ML models, tokenizers, nltk_data, and other files that "must ship
    together when packaged". The actual lookup logic lives in src.utils.runtime_paths.resolve_external_dir;
    this is just a thin wrapper binding `name="assets"` and dev_root.

    Note: after packaging, libs/ is occupied by .venv site-packages, so ML assets live separately
    under assets/ — never move them back to libs/.
    """
    return resolve_external_dir(
        "assets",
        dev_root=Path(__file__).resolve().parents[3],
    )


def _resolve_artifacts_path() -> Optional[Path]:
    """Resolve the Docling ML model directory: assets/docling_models

    Produced by `docling-tools models download -o assets/docling_models`. Returns None if not found.
    """
    assets = resolve_assets_dir()
    if assets:
        p = assets / "docling_models"
        if p.is_dir():
            return p
    return None


def _resolve_whisper_path() -> Optional[Path]:
    """Resolve the Whisper ASR model directory: assets/whisper_models

    `whisper.load_model('turbo', download_root=...)` saves the weights as `large-v3-turbo.pt`
    (OpenAI internally maps the alias 'turbo' to the formal filename). We don't hard-code the
    filename; a directory is considered "whisper ready" if it contains at least one .pt file.

    Auto-discover design: no directory or no .pt file → return None → the audio pipeline isn't
    installed → uploaded .wav/.mp3 naturally fall back to SimpleDirectoryReader (a silent fallback,
    consistent with the principle of least surprise).
    """
    assets = resolve_assets_dir()
    if not assets:
        return None
    p = assets / "whisper_models"
    if p.is_dir() and any(p.glob("*.pt")):
        return p
    return None


def ocr_enabled() -> bool:
    """config ``rag.docling.ocr_enabled`` (restart-level: the converter is a singleton).
    Unreadable config keeps the default (on)."""
    try:
        from src.config.config_manager import Config as _Cfg
        return bool(_Cfg.get_config_model().rag.docling.ocr_enabled)
    except Exception:
        return True


def asr_wants_local_whisper() -> bool:
    """Whether config `rag.asr` requires local docling-whisper.

    Installation (get_converter) and warmup (warmup_docling) must use the SAME condition. If they
    judge independently (warmup only checking whether the weight file exists), then with the provider
    set to fireredasr but a stray whisper .pt left on the machine, warmup would init a default docling
    audio pipeline with no artifacts_path → triggering a runtime download → SSL failure in an
    offline/intercepted environment. A missing config (None) keeps the old auto-discover semantics (yes).
    """
    try:
        from src.config.config_manager import Config as _Cfg
        _asr_cfg = getattr(getattr(_Cfg.get_config_model(), "rag", None), "asr", None)
        if _asr_cfg is not None:
            return bool(_asr_cfg.enabled and _asr_cfg.provider == "docling-whisper")
    except Exception:
        pass  # config unreadable → keep the old behavior
    return True


def _local_pdf_models_wanted() -> bool:
    """Whether rag.parsing keeps PDF or image parsing on the in-process docling models
    (config unreadable → the old behaviour: yes)."""
    try:
        from src.config.config_manager import Config as _Cfg
        from src.config.roles import needs_local_docling_models
        rag = getattr(_Cfg.get_config_model(), "rag", None)
        if rag is not None:
            return needs_local_docling_models(rag)
    except Exception:
        pass
    return True


def get_hf_tokenizer(tokenizer_name: Optional[str]):
    """Token counter for chunk sizing (kept under its old name for callers).

    Delegates to ``chunk_tokenizer.get_chunk_tokenizer``: local assets first,
    then the bundled tokenizer, never a download. ``tokenizer_name`` is the
    embedding model name, or None for providers without a HuggingFace name
    (OpenAI / Azure), which used to crash every docling-parsed file.
    """
    from src.domain.rag.chunk_tokenizer import get_chunk_tokenizer

    return get_chunk_tokenizer(tokenizer_name)


def _apply_docling_env():
    """Read Docling settings from config and apply them to environment variables.

    Deferred until Docling is first used, by which point Config is already loaded. Runs only once,
    guarded by the _docling_env_applied flag.

    Note: the device setting no longer goes through the DOCLING_DEVICE env var (Docling doesn't read
    it). The actual GPU enablement point is `_get_accelerator_options()`, which passes AcceleratorOptions
    into PdfPipelineOptions / AsrPipelineOptions.
    """
    global _docling_env_applied
    if _docling_env_applied:
        return
    _docling_env_applied = True

    try:
        from src.config.config_manager import Config
        config = Config.get_config_model()

        if config and config.rag:
            docling_config = getattr(config.rag, 'docling', None)

            if docling_config:
                device = docling_config.device or "cpu"
                # Keep the DOCLING_DEVICE env var (backward compatibility, for debug visibility).
                # The entry point that actually takes effect is _get_accelerator_options().
                os.environ["DOCLING_DEVICE"] = device
                logger.info(f"[DOCLING_INIT] device={device} (resolves to AcceleratorOptions)")

                # From docling 2.119, AcceleratorOptions supports "cuda:N" (see
                # _get_accelerator_options), and cuda:N genuinely pins to the Nth card.
                # If CUDA_VISIBLE_DEVICES is also set, N is the index *after remapping* —
                # using both together easily misfires (e.g. CUDA_VISIBLE_DEVICES=4 + device=cuda:4
                # → no visible 5th card, so it crashes). For shared-card scenarios, pick one:
                #   (a) device=cuda:4 without setting CUDA_VISIBLE_DEVICES (recommended, intuitive)
                #   (b) CUDA_VISIBLE_DEVICES=4 + device=cuda:0 (more thorough isolation)
                if device.startswith("cuda:") and os.environ.get("CUDA_VISIBLE_DEVICES"):
                    logger.warning(
                        f"[DOCLING_INIT] device='{device}' set together with "
                        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}: "
                        f"N in cuda:N is the post-remap index; confirm it points to the right card"
                    )

                # HuggingFace offline mode
                if docling_config.hf_offline:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    logger.info("[DOCLING_INIT] HuggingFace offline mode enabled")

                # torch.compile toggle (disabling speeds startup by ~120s, but inference is slightly slower)
                if not docling_config.compile_models:
                    os.environ["DOCLING_INFERENCE"] = '{"compile_torch_models": false}'
                    logger.info("[DOCLING_INIT] torch.compile disabled (faster startup)")

                # OCR toggle: applied as PdfPipelineOptions(do_ocr=...) in _build_converter
                # (docling reads no environment variable for it).
                if not docling_config.ocr_enabled:
                    logger.info("[DOCLING_INIT] OCR disabled")

                return

    except Exception as e:
        logger.warning(f"[DOCLING_INIT] Failed to read config, using defaults: {e}")

    # Fallback
    if "DOCLING_DEVICE" not in os.environ:
        os.environ["DOCLING_DEVICE"] = "cpu"
        logger.info("[DOCLING_INIT] device=cpu (default)")


def _get_rapidocr_options(artifacts_path, accelerator_options=None):
    """Build RapidOcrOptions (PP-OCRv6, torch backend) per config.rag.docling.ocr_model_scale.

    The OCR inference engine uniformly uses rapidocr's torch backend, no longer onnxruntime:
    - Shares the same torch as docling's other models (layout/tables) — both CUDA and ROCm hit the
      GPU directly (ROCm's torch masquerades as a cuda device, no separate handling needed)
    - pyproject no longer has to maintain onnxruntime variants per GPU type (gpu/rocm/cpu)
    - docling 2.111 wires up the torch backend's EngineConfig.torch.use_cuda correctly (the
      disconnect bug was onnxruntime-engine-only), retiring the old GPU-key patch alongside

    Models: PP-OCRv6 det/rec official torch weights (RapidAI conversion, accuracy on par with the
    onnx version; all three tiny/small/medium scales) + PP-OCRv4 v2.0 cls (currently the only torch
    cls weights; cls only judges 0/180 text-line orientation, so version differences are imperceptible).
    torch weights don't embed a dictionary, so rec must carry rec_keys_path (tiny uses a dedicated dict).

    Args:
        artifacts_path: assets/docling_models path (Path).
        accelerator_options: docling AcceleratorOptions. The GPU toggle isn't handled here —
            docling's RapidOcrModel derives EngineConfig.torch.use_cuda / device_id from the
            accelerator device itself; kept only for logging.

    Returns:
        RapidOcrOptions (backend="torch" + explicit model paths).
    """
    scale = "small"
    try:
        from src.config.config_manager import Config
        config = Config.get_config_model()
        if config and config.rag and config.rag.docling:
            scale = (config.rag.docling.ocr_model_scale or "small").lower()
    except Exception:
        pass

    if scale not in ("tiny", "small", "medium"):
        logger.warning(f"[DOCLING] unknown ocr_model_scale='{scale}', fallback to small")
        scale = "small"

    rapid_root = Path(artifacts_path) / "RapidOcr"
    det = rapid_root / "torch" / "PP-OCRv6" / "det" / f"PP-OCRv6_det_{scale}.pth"
    rec = rapid_root / "torch" / "PP-OCRv6" / "rec" / f"PP-OCRv6_rec_{scale}.pth"
    cls = rapid_root / "torch" / "PP-OCRv4" / "cls" / "ch_ptocr_mobile_v2.0_cls_mobile.pth"
    # tiny's rec dictionary differs from small/medium (smaller character set) and can't be mixed
    dict_name = "ppocrv6_tiny_dict.txt" if scale == "tiny" else "ppocrv6_dict.txt"
    rec_keys = rapid_root / "paddle" / "PP-OCRv6" / "rec" / dict_name

    missing = [str(p) for p in (det, rec, cls, rec_keys) if not p.exists()]
    if missing:
        # No fallback: docling's torch default is PP-OCRv4, whose files are likewise absent from
        # assets, so a silent downgrade only defers the error. Let rapidocr fail loudly at model
        # load so the deployment side knows assets are incomplete.
        logger.error(
            f"[DOCLING] OCR torch model files missing (scale={scale}): {missing} — "
            f"ensure assets/docling_models/RapidOcr/torch/ is shipped in the package"
        )

    device_str = str(getattr(accelerator_options, "device", "") or "").lower()
    from docling.datamodel.pipeline_options import RapidOcrOptions
    logger.info(
        f"[DOCLING] OCR = PP-OCRv6 {scale} det/rec + cls v2.0 mobile "
        f"(torch backend, device={device_str or 'auto'})"
    )
    return RapidOcrOptions(
        backend="torch",
        det_model_path=str(det),
        rec_model_path=str(rec),
        cls_model_path=str(cls),
        rec_keys_path=str(rec_keys),
    )


def _get_accelerator_options():
    """Compute Docling AcceleratorOptions from config.

    Returns:
        An AcceleratorOptions instance whose device corresponds to config.rag.docling.device.
        If config sets cuda* but PyTorch can't see a GPU, automatically downgrades to CPU and warns.
    """
    from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice

    device_str = os.environ.get("DOCLING_DEVICE", "cpu").lower()

    if device_str.startswith("cuda"):
        # Cross-check that the GPU is actually available, to avoid a silent fallback
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    f"[DOCLING_INIT] config device='{device_str}' but "
                    f"torch.cuda.is_available()=False — fallback CPU"
                )
                return AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=4)
        except ImportError:
            logger.warning("[DOCLING_INIT] torch not importable, fallback CPU")
            return AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=4)

        # docling 2.119's AcceleratorOptions.device accepts a "cuda:N" string directly (the validator
        # matches ^cuda(:\d+)?$). Passed through as-is → it genuinely pins to the specified card,
        # no longer collapsed to bare CUDA (GPU 0). This separates docling from vLLM in shared-card
        # scenarios, avoiding the CUBLAS_ALLOC_FAILED caused by gpt-oss saturating GPU 0.
        # Indexing semantics: docling uses "the Nth card PyTorch can see". If CUDA_VISIBLE_DEVICES is
        # also set, N is the index after remapping (see the _apply_docling_env warning).
        logger.info(
            f"[DOCLING_INIT] Using device='{device_str}' "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})"
        )
        return AcceleratorOptions(device=device_str, num_threads=4)

    if device_str == "mps":
        return AcceleratorOptions(device=AcceleratorDevice.MPS, num_threads=4)

    return AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=4)


# Audio extensions listed separately — only actually supported when the Whisper model is present
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}  # docling's 6 official types


def is_docling_supported(file_name: str) -> bool:
    """Determine whether a file can be processed by Docling.

    Audio files (.wav / .mp3) require assets/whisper_models/turbo.pt to exist; otherwise they're
    treated as unsupported, letting the caller fall back to SimpleDirectoryReader.

    Args:
        file_name: The filename including its extension.

    Returns:
        True = Docling can process it; False = use the fallback.
    """
    ext = Path(file_name).suffix.lower()
    if ext not in DOCLING_EXTENSIONS:
        return False
    if ext in _AUDIO_EXTENSIONS and _resolve_whisper_path() is None:
        return False
    return True


def get_converter():
    """Get the singleton DocumentConverter (thread-safe), building it if not yet initialized.

    The first call loads all ML models (layout, table structure, OCR); later calls return the
    already-built instance.

    Double-checked locking: if already built, take the lock-free fast path; if not, acquire the lock
    and let _build_converter's None check confirm a second time — concurrent first-touch builds only once.
    """
    global _converter
    if _converter is not None:
        return _converter
    with _converter_lock:
        return _build_converter()


def _build_converter():
    """Actually build the singleton (the caller must hold _converter_lock).

    Offline deployment (hf_offline=true): force use of the local models in assets/docling_models,
    raising immediately if not found rather than blowing up later on the first uploaded file.
    """
    global _converter
    if _converter is None:
        _apply_docling_env()  # must run first: hf_offline=true sets HF_HUB_OFFLINE=1

        # PDF/image layout + OCR models are only configured when rag.parsing keeps them
        # local; under cloud parsing this converter serves Office/Markdown/HTML only and
        # must not touch the model stack (or torch, via the CUDA check below).
        local_models = _local_pdf_models_wanted()
        artifacts_path = _resolve_artifacts_path() if local_models else None
        offline = os.environ.get("HF_HUB_OFFLINE") == "1"

        if local_models and not artifacts_path and offline:
            raise RuntimeError(
                "[DOCLING] hf_offline=true but assets/docling_models not found. "
                "Prepare via: `uv run docling-tools models download -o assets/docling_models`"
            )

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AsrPipelineOptions
        from docling.datamodel.asr_model_specs import WHISPER_TURBO_NATIVE
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
            ImageFormatOption,
            AudioFormatOption,
        )

        format_options: dict = {}

        # Audio pipeline: installed only when the Whisper model exists AND config permits.
        whisper_path = _resolve_whisper_path() if asr_wants_local_whisper() else None

        # Compute the GPU/CPU choice from config — passed to all pipeline options so Docling's
        # Layout / TableFormer / OCR / Whisper all run on the configured device
        accelerator_options = (
            _get_accelerator_options() if (local_models or whisper_path) else None)

        ocr_on = ocr_enabled()
        pdf_kwargs = {
            "accelerator_options": accelerator_options,
            # rag.docling.ocr_enabled=false must really skip OCR (the model is not even loaded).
            "do_ocr": ocr_on,
        }
        if artifacts_path:
            logger.info(f"[DOCLING] Using local artifacts_path: {artifacts_path}")
            pdf_kwargs["artifacts_path"] = str(artifacts_path)
            # OCR uses the rapidocr torch backend (PP-OCRv6, file selected per ocr_model_scale)
            ocr_options = _get_rapidocr_options(artifacts_path, accelerator_options) if ocr_on else None
            if ocr_options is not None:
                pdf_kwargs["ocr_options"] = ocr_options
        elif local_models:
            logger.warning(
                "[DOCLING] No local artifacts_path; models will be fetched from HuggingFace Hub "
                "on first use (online machines only)"
            )
        if local_models:
            if not ocr_on:
                logger.info("[DOCLING] OCR off (rag.docling.ocr_enabled=false): scanned pages yield no text")
            pdf_pipeline_options = PdfPipelineOptions(**pdf_kwargs)
            # PDF and Image share the same pipeline_options (IMAGE internally goes through StandardPdfPipeline)
            format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pdf_pipeline_options)
            format_options[InputFormat.IMAGE] = ImageFormatOption(pipeline_options=pdf_pipeline_options)
        else:
            logger.info("[DOCLING] PDF/image models not configured: rag.parsing sends them to the cloud")

        # Not installed when rag.asr.enabled=false or the provider isn't docling-whisper (e.g. cloud),
        # saving local whisper's VRAM/warmup; a missing config keeps the old auto-discover semantics.
        if whisper_path is None and not asr_wants_local_whisper():
            logger.info("[DOCLING] Audio pipeline skipped by rag.asr config (disabled or non-local provider)")
        if whisper_path:
            # Patch whisper.load_model before docling's audio pipeline can load the model.
            from src.domain.rag.audio_defense import _enable_whisper_anti_hallucination_defaults
            _enable_whisper_anti_hallucination_defaults()
            logger.info(f"[DOCLING] Audio pipeline enabled, whisper_models at: {whisper_path}")
            # word_timestamps=False: avoids whisper/timing.py's hard dependency on Triton
            # (`triton_ops.dtw_kernel` / `median_filter_cuda`) — that API is unstable across Triton/ROCm
            # versions, and we don't need word-level timestamps. From docling 2.111 this is an official
            # option (older versions hardcoded True, requiring a whole-class PatchedAsrPipeline override,
            # since removed with the 2.119 upgrade).
            asr_pipeline_options = AsrPipelineOptions(
                artifacts_path=str(whisper_path),
                asr_options=WHISPER_TURBO_NATIVE.model_copy(
                    update={"word_timestamps": False}
                ),
                accelerator_options=accelerator_options,
            )
            format_options[InputFormat.AUDIO] = AudioFormatOption(
                pipeline_options=asr_pipeline_options,
            )
        elif asr_wants_local_whisper():
            logger.info("[DOCLING] assets/whisper_models/*.pt not found: local whisper transcription unavailable")

        converter_kwargs = {"format_options": format_options} if format_options else {}

        logger.info("[DOCLING] Creating DocumentConverter (loading ML models)...")
        t0 = time.time()
        _converter = DocumentConverter(**converter_kwargs)
        elapsed = time.time() - t0
        logger.info(f"[DOCLING] DocumentConverter ready ({elapsed:.1f}s)")
    return _converter


def warmup_docling():
    """Preload Docling models at startup.

    Called during app startup to load ML models into GPU/CPU ahead of time. Uses
    DocumentConverter.initialize_pipeline() to initialize the PDF pipeline directly, forcing all ML
    models (Layout Detection, TableFormer, OCR) to load without needing a dummy PDF.
    """
    logger.info("[DOCLING_WARMUP] Pre-loading Docling models...")
    t0 = time.time()
    try:
        converter = get_converter()

        # Force-initialize every pipeline to be exposed via /v1/transcriptions/general and
        # /v1/ocr/general, so the first REST request after startup doesn't pay cold-start model load cost.
        from docling.datamodel.base_models import InputFormat
        warmup_formats = [InputFormat.PDF, InputFormat.DOCX, InputFormat.IMAGE]
        # The AUDIO warmup condition must match get_converter's install condition (config + weights) —
        # checking only the weight file would, when the provider isn't docling-whisper, warm up
        # docling's default audio pipeline (no artifacts_path → runtime download → SSL failure offline)
        _local_whisper = asr_wants_local_whisper() and _resolve_whisper_path() is not None
        if _local_whisper:
            warmup_formats.append(InputFormat.AUDIO)
        for fmt in warmup_formats:
            try:
                logger.info(f"[DOCLING_WARMUP] Initializing pipeline for {fmt.value}...")
                converter.initialize_pipeline(fmt)
            except Exception as e:
                logger.warning(f"[DOCLING_WARMUP] Pipeline init for {fmt.value} failed: {e}")

        # Verify Whisper anti-hallucination monkey-patch actually took effect.
        # If Docling imported whisper before hierarchical_indexer loaded, the patch misses that copy
        # → Layer 1 defense silently fails. Assert right after audio pipeline warmup, logging an error
        # on failure so ops sees it at a glance (no raise — avoids blocking server startup; degraded is still usable).
        if _local_whisper:
            try:
                import whisper as _w
                if not getattr(_w.load_model, "_anti_hallucination_wrapped", False):
                    logger.error(
                        "[DOCLING_WARMUP] ❌ Whisper anti-hallucination monkey-patch NOT applied — "
                        "Layer 1 defense disabled. Check import order in hierarchical_indexer.py"
                    )
                else:
                    logger.info(
                        "[DOCLING_WARMUP] ✓ Whisper anti-hallucination monkey-patch verified"
                    )
            except Exception as exc:
                logger.warning(f"[DOCLING_WARMUP] Could not verify whisper monkey-patch: {exc}")

        # GPU self-diagnosis: an OCR session's CUDA provider init failure silently falls back to CPU
        # (ORT only grumbles to stderr), so print the truth into the real log — on deployment, the
        # top of docker logs shows which device OCR/Layout actually landed on (cf. scripts/probe_gpu_pipeline.py)
        try:
            _log_pipeline_devices(converter)
        except Exception as exc:
            logger.warning(f"[DOCLING_WARMUP] GPU self-check skipped: {exc}")

        elapsed = time.time() - t0
        logger.info(f"[DOCLING_WARMUP] All models loaded successfully ({elapsed:.1f}s)")
    except Exception as e:
        logger.warning(f"[DOCLING_WARMUP] Failed to pre-load models: {e}")
        logger.warning("[DOCLING_WARMUP] Models will be loaded on first document processing")


def _log_pipeline_devices(converter) -> None:
    """Log the actual device of the OCR sessions / Layout model (warmup self-diagnosis)."""
    from docling.datamodel.base_models import InputFormat

    pdf_pipeline = converter._get_pipeline(InputFormat.PDF)

    ocr_model = getattr(pdf_pipeline, "ocr_model", None)
    reader = getattr(ocr_model, "reader", None)
    if reader is not None:
        parts = []
        all_gpu = True
        for part_name in ("text_det", "text_cls", "text_rec"):
            part = getattr(reader, part_name, None)
            sess = getattr(getattr(part, "session", None), "session", None) or getattr(part, "session", None)
            dev = None
            if hasattr(sess, "get_providers"):  # onnxruntime engine (leftover from old deployments)
                dev = "cuda" if "CUDAExecutionProvider" in sess.get_providers() else "cpu"
            elif getattr(sess, "device", None) is not None:  # torch engine
                dev = str(sess.device)
            if dev is None:
                continue
            on_gpu = "cuda" in dev.lower()
            all_gpu &= on_gpu
            parts.append(f"{part_name}={'GPU' if on_gpu else 'CPU'}")
        if parts:
            line = f"[DOCLING_WARMUP] OCR sessions: {', '.join(parts)}"
            if all_gpu:
                logger.info(f"{line} ✓")
            else:
                logger.warning(
                    f"{line} — ❌ a session landed on CPU! Scanned PDFs will be much slower. "
                    f"Under the torch engine this means the accelerator device resolved to cpu "
                    f"(check config docling.device and torch.cuda.is_available)"
                )

    layout = getattr(pdf_pipeline, "layout_model", None)
    if layout is not None:
        dev = getattr(layout, "device", None) or getattr(layout, "_device", None)
        inner = getattr(layout, "layout_predictor", None) or getattr(layout, "model", None)
        if dev is None and inner is not None:
            dev = getattr(inner, "device", None) or getattr(inner, "_device", None)
        if dev is not None:
            msg = f"[DOCLING_WARMUP] Layout model device: {dev}"
            if "cuda" in str(dev).lower():
                logger.info(f"{msg} ✓")
            else:
                logger.warning(f"{msg} — ❌ landed on CPU")


# Tokenizer-aware chunking helpers. We deliberately do NOT use HybridChunker: on a
# single huge chunk (e.g. an xlsx 5000-row table from HierarchicalChunker) its
# slide-window tokenization runs for tens of minutes and trips the watchdog and
# per-file timeout. Instead we re-implement its contract per-chunk — structural
# pre-split via HierarchicalChunker, then one tokenizer.encode() per chunk (bounded
# by chunk size), splitting oversized chunks by natural boundaries and merging small
# ones (mirrors merge_peers / repeat_table_header). Output is text-only, which is all
# HierarchicalIndexer needs.


# Markdown table separator (`| --- | :---: | ... |`), used to detect a table chunk
# whose first 2 lines are the header to repeat on splits.
_MD_TABLE_SEPARATOR_RE = re.compile(r'^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$')

# Sentence ends. CJK terminators (。！？；) end a sentence on their own: Chinese and
# Japanese put no space after them, so requiring whitespace (as the old pattern did)
# left a whole Chinese paragraph as one "sentence" that fell through to the character
# split and was cut mid-clause. Latin terminators need following whitespace, so
# decimals ("3.5"), version numbers and URLs stay whole. Closing quotes/brackets stay
# with the sentence they close; trailing whitespace stays with the sentence too, so
# concatenating the pieces gives back the original text.
_SENTENCE_END_RE = re.compile(
    r'[。！？；]+[」』）】》〉”’"\')]*\s*'
    r'|[.!?]+["”’\')\]]*\s+'
)
# Clause ends, for a single sentence that is still over budget: CJK commas / colons
# (no space needed) and Latin ones followed by whitespace.
_CLAUSE_END_RE = re.compile(r'[，、：]\s*|[,:;]\s+')


def _split_keep_ends(text: str, pattern: "re.Pattern[str]") -> List[str]:
    """Split ``text`` after every match of ``pattern``, keeping the separator (and
    its trailing whitespace) on the left piece. ``"".join(result) == text``."""
    out: List[str] = []
    start = 0
    for m in pattern.finditer(text):
        if m.end() > start:
            out.append(text[start:m.end()])
            start = m.end()
    if start < len(text):
        out.append(text[start:])
    return out


def _pack_balanced(counts: List[int], max_tokens: int) -> List[List[int]]:
    """Group consecutive units (token counts given) into pieces of at most
    ``max_tokens``, as evenly sized as the unit boundaries allow.

    Plain greedy packing fills every piece to the brim and leaves a runt at the end
    (242 + 45 tokens); here each piece aims at ``remaining / pieces still needed`` and
    a unit goes to the next piece when that lands closer to the target. A unit larger
    than ``max_tokens`` gets a piece of its own (callers pre-split those).
    """
    groups: List[List[int]] = []
    cur: List[int] = []
    cur_tokens = 0
    remaining = sum(counts)

    def _target() -> float:
        pieces = max(1, -(-remaining // max(max_tokens, 1)))  # ceil
        return remaining / pieces

    target = _target()
    for i, c in enumerate(counts):
        if cur:
            over_budget = cur_tokens + c > max_tokens
            overshoot = cur_tokens + c - target
            undershoot = target - cur_tokens
            if over_budget or (overshoot > 0 and overshoot > undershoot):
                groups.append(cur)
                remaining -= cur_tokens
                target = _target()
                cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += c
    if cur:
        groups.append(cur)
    return groups


def _detect_table_header(lines: List[str]) -> Optional[str]:
    """If ``lines`` is a markdown table, return the header + separator
    block to prepend on continuation chunks. ``None`` otherwise."""
    if len(lines) < 2:
        return None
    first = lines[0].strip()
    second = lines[1].strip()
    if "|" in first and _MD_TABLE_SEPARATOR_RE.match(second):
        return f"{lines[0]}\n{lines[1]}"
    return None


def _count_tokens(tokenizer, text: str) -> int:
    """Single-call token count. ``add_special_tokens=False`` matches the
    HuggingFace BPE-level count; HybridChunker uses the same flag for
    its own ≤ max_tokens checks."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_by_lines(
    text: str,
    tokenizer,
    max_tokens: int,
    header: Optional[str],
) -> List[str]:
    """Split oversized text by ``\\n`` boundaries (table rows / paragraphs).

    Each emitted sub-chunk is ≤ max_tokens (verified via tokenizer).
    When ``header`` is set, continuation sub-chunks get it prepended for
    context (the first sub-chunk already contains it as part of the
    original lines). The header's token cost is reserved upfront for
    every chunk that needs it, so the final assembled chunk stays
    under max_tokens — otherwise the header bytes overflow continuation
    chunks by header_tokens (typically 20-30 tokens for a markdown
    table header).
    """
    lines = text.split("\n")
    if header is None:
        header = _detect_table_header(lines)

    # Reserve header budget for continuation chunks. The first chunk
    # already includes the header lines as part of the natural line
    # sequence, so its effective budget is the full max_tokens;
    # continuations get max_tokens - header_tokens to leave room for
    # the prepended header at flush time.
    header_tokens = _count_tokens(tokenizer, header) if header else 0

    out: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    def _budget() -> int:
        return max_tokens - header_tokens if (header and out) else max_tokens

    def _flush():
        nonlocal buf, buf_tokens
        if not buf:
            return
        body = "\n".join(buf)
        # Continuation chunks (output index > 0) get the header back if
        # the header isn't already the first lines of this buffer.
        if header and out and not body.startswith(header):
            body = f"{header}\n{body}"
        # Tokenizer-portability safeguard: the budget check uses "per-line token sum + separator
        # reserve", which is exact under bge-m3 (sentencepiece). If a tokenizer that counts \n as
        # more than 1 token is ever used, this is a final whole-chunk verification that logs on
        # overflow (no raise — the embedding context is far larger than max_tokens, so a slight
        # overshoot isn't fatal, but it must be visible).
        final_tokens = _count_tokens(tokenizer, body)
        if final_tokens > max_tokens:
            logger.warning(
                f"[CHUNK] assembled chunk exceeds budget after join: "
                f"{final_tokens} > {max_tokens} tokens (separator drift); "
                f"check tokenizer newline behaviour"
            )
        out.append(body)
        buf = []
        buf_tokens = 0

    for line in lines:
        if not line.strip():
            continue
        line_tokens = _count_tokens(tokenizer, line)

        # Single line still too big — fall to sentence split.
        # In the table case, continuations also need the header: reserve header budget,
        # then prepend the header to each split piece.
        if line_tokens > _budget():
            _flush()
            sent_budget = max_tokens - header_tokens if header else max_tokens
            for piece in _split_by_sentence(line, tokenizer, sent_budget):
                if header and out and not piece.startswith(header):
                    piece = f"{header}\n{piece}"
                out.append(piece)
            continue

        # Would overflow if we add this line — flush first.
        # `len(buf)` = the number of \n separators the join needs after adding this line, reserving
        # budget for a tokenizer that counts \n as 1 token (slightly conservative under sentencepiece, harmless).
        if buf_tokens + line_tokens + len(buf) > _budget():
            _flush()

        buf.append(line)
        buf_tokens += line_tokens

    _flush()
    return out


def _split_by_sentence(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Mid-tier split: break a single line at sentence ends, then pack the
    sentences into balanced pieces of at most ``max_tokens``.

    Reached when a single newline-bounded line is itself > max_tokens —
    typical for long paragraphs without internal structure. A sentence that is
    still too big is cut at clause ends (，、： / ", "), and only a clause that is
    still too big falls to the character split.
    """
    units: List[str] = []
    for sent in _split_keep_ends(text, _SENTENCE_END_RE):
        if not sent.strip():
            continue
        if _count_tokens(tokenizer, sent.strip()) <= max_tokens:
            units.append(sent)
            continue
        for clause in _split_keep_ends(sent, _CLAUSE_END_RE):
            if not clause.strip():
                continue
            if _count_tokens(tokenizer, clause.strip()) <= max_tokens:
                units.append(clause)
            else:
                units.extend(_split_by_chars(clause, tokenizer, max_tokens))

    counts = [_count_tokens(tokenizer, u.strip()) for u in units]
    out: List[str] = []

    def _emit(group: List[int]) -> None:
        piece = "".join(units[i] for i in group).strip()
        # The token count of a concatenation can exceed the sum of its parts by a
        # token or two (word-start markers); halve such a group instead of overflowing.
        if len(group) > 1 and _count_tokens(tokenizer, piece) > max_tokens:
            mid = len(group) // 2
            _emit(group[:mid])
            _emit(group[mid:])
            return
        if piece:
            out.append(piece)

    for group in _pack_balanced(counts, max_tokens):
        _emit(group)
    return out


def _split_by_chars(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Last-resort character split with tokenizer verification.

    Reached for pathological inputs: single sentences > max_tokens
    (rare in zh / en, common for poorly punctuated dumps or numeric
    sequences). We slice by an estimated char-per-token ratio, then
    verify each slice; if the estimate undershot we halve and retry.
    """
    out: List[str] = []
    # Conservative CJK-friendly estimate: 1.5 chars per token. English
    # would be ~3-4, but undershooting just causes one extra verify cycle.
    chars_per_chunk = max(64, int(max_tokens * 1.5))
    i = 0
    while i < len(text):
        piece = text[i:i + chars_per_chunk]
        if _count_tokens(tokenizer, piece) > max_tokens:
            if chars_per_chunk <= 32:
                # Progress guarantee: even at the 32-char floor it still exceeds budget (a pathological
                # combination of a tiny max_tokens + high-token-density characters). Hard-cut, emit,
                # and advance — better a slightly oversized single chunk than an infinite loop.
                logger.warning(
                    f"[CHUNK] 32-char piece still exceeds {max_tokens} tokens; "
                    f"emitting oversized piece to guarantee progress"
                )
                out.append(piece)
                i += chars_per_chunk
                continue
            # Estimate too generous — halve and retry without advancing
            chars_per_chunk = max(32, chars_per_chunk // 2)
            continue
        out.append(piece)
        i += chars_per_chunk
    return out


def _doc_item_kind(item: Any) -> Optional[str]:
    """Map a Docling doc_item's label to a coarse content kind, or None.

    Docling labels items (DocItemLabel: TABLE / PICTURE / TEXT / …). We only care
    about the multi-modal ones. Defensive across Docling versions — the label may be
    an enum (``.value``/``.name``) or a bare string.
    """
    label = getattr(item, "label", None)
    if label is None:
        return None
    name = str(getattr(label, "value", None) or getattr(label, "name", None) or label).lower()
    if "table" in name:
        return "table"
    if "picture" in name or "image" in name or "figure" in name:
        return "picture"
    return None


def _container_headings(item: Any, doc: Any) -> List[str]:
    """Names of the named containers a doc_item sits in, outermost first.

    Only xlsx sheets for now: docling puts each sheet's tables in a GroupItem
    labelled ``sheet`` named after the sheet, but no heading — so a table on the
    "Borealis plan" sheet never says "Borealis". Defensive: any resolution failure
    yields no extra heading.
    """
    names: List[str] = []
    ref = getattr(item, "parent", None)
    hops = 0
    while ref is not None and doc is not None and hops < 32:
        hops += 1
        try:
            node = ref.resolve(doc)
        except Exception:
            break
        label = getattr(node, "label", None)
        label = str(getattr(label, "value", label) or "").lower()
        name = getattr(node, "name", None)
        if label == "sheet" and name:
            names.append(str(name))
        ref = getattr(node, "parent", None)
    return list(reversed(names))


def _chunk_provenance(chunk: Any, doc: Any = None) -> dict:
    """Extract headings / page_no / content_type from a docling chunk object (bare strings return empty meta).

    - headings: HierarchicalChunker's meta.headings (the section path, list[str]),
      prefixed with the xlsx sheet name when ``doc`` is given (see _container_headings)
    - page_no: the page_no of the first prov of the first doc_item (present for PDF; usually absent for docx etc.)
    - content_type: "table" when the chunk carries a table doc_item (Docling keeps the
      table as Markdown in the chunk text; this flag lets the UI badge / render it).
      Defaults to None for plain text.
    Extraction is defensive throughout — any missing field returns None and doesn't affect chunking.
    """
    headings = None
    page_no = None
    content_type = None
    meta = getattr(chunk, "meta", None)
    if meta is not None:
        h = getattr(meta, "headings", None)
        if h:
            headings = list(h)
        items = list(getattr(meta, "doc_items", None) or [])
        if items and doc is not None:
            containers = _container_headings(items[0], doc)
            if containers:
                headings = containers + [x for x in (headings or []) if x not in containers]
        for item in items:
            kind = _doc_item_kind(item)
            if kind == "table":  # a table anywhere in the chunk wins
                content_type = "table"
            elif kind == "picture" and content_type is None:
                content_type = "picture"
            if page_no is None:  # keep the first doc_item's page (original semantics)
                for prov in (getattr(item, "prov", None) or []):
                    p = getattr(prov, "page_no", None)
                    if p is not None:
                        page_no = int(p)
                        break
            if page_no is not None and content_type == "table":
                break
    return {"headings": headings, "page_no": page_no, "content_type": content_type}


def _is_triplet_table(text: str) -> bool:
    """Docling's table serialization: one line of ``row, col = value`` cells joined by ". "."""
    return "\n" not in text.strip() and " = " in text and ", " in text


def _split_triplet_table(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Split an oversized triplet table at row boundaries (a row = consecutive cells
    with the same row label), so every piece is itself a valid table. A single row
    larger than the budget falls back to cell boundaries."""
    cells = [c.strip() for c in text.strip().rstrip(".").split(". ") if c.strip()]
    rows: List[List[str]] = []
    last_label = None
    for cell in cells:
        label = cell.split(", ", 1)[0] if ", " in cell else None
        if rows and label == last_label:
            rows[-1].append(cell)
        else:
            rows.append([cell])
        last_label = label
    units: List[str] = []
    for row in rows:
        joined = ". ".join(row)
        if _count_tokens(tokenizer, joined) <= max_tokens:
            units.append(joined)
        else:
            units.extend(row)
    out: List[str] = []
    buf: List[str] = []
    for unit in units:
        candidate = ". ".join(buf + [unit])
        if buf and _count_tokens(tokenizer, candidate) > max_tokens:
            out.append(". ".join(buf))
            buf = [unit]
        else:
            buf.append(unit)
    if buf:
        out.append(". ".join(buf))
    # A cell longer than the budget on its own still has to be cut.
    final: List[str] = []
    for piece in out:
        if _count_tokens(tokenizer, piece) > max_tokens:
            final.extend(_split_by_chars(piece, tokenizer, max_tokens))
        else:
            final.append(piece)
    return final


def _refine_chunks_for_token_budget(
    raw_chunks: List[Any],
    tokenizer,
    max_tokens: int,
    doc: Any = None,
    table_max_tokens: Optional[int] = None,
) -> List[dict]:
    """Refine HierarchicalChunker output into token-bounded chunk records.

    Same contract as HybridChunker (token-bounded, merge_peers, repeat_table_header)
    but with per-chunk tokenize calls, and section-aligned: small chunks merge only
    with neighbours of the same section (chunk_identity.section_key), and a merged
    chunk is labelled with the headings all its pieces share. Returns records
    ``{"text", "headings", "page_no", "content_type"}``; split pieces inherit the
    source chunk's meta, bare-string input has meta None.

    ``table_max_tokens`` (default ``max_tokens``) lets a table stay whole up to a
    larger budget than prose; an oversized docling table is split at row
    boundaries so each piece remains a readable table.
    """
    from src.domain.rag.chunk_identity import (
        common_headings,
        merge_content_type,
        section_key,
    )

    table_budget = max(max_tokens, table_max_tokens or max_tokens)
    # merge_peers heuristic — flush the buffer when the token budget reaches
    # half of max_tokens. Matches HybridChunker default behavior:
    # produce chunks at roughly 50-100% of max_tokens for retrieval.
    merge_flush_threshold = max(max_tokens // 2, 1)

    output: List[dict] = []
    pending: List[tuple] = []  # (text, meta)
    pending_tokens = 0

    def _flush_pending():
        nonlocal pending, pending_tokens
        if not pending:
            return
        metas = [m for _, m in pending]
        page = next((m.get("page_no") for m in metas if m.get("page_no") is not None), None)
        output.append({
            "text": "\n".join(t for t, _ in pending),
            "headings": common_headings(m.get("headings") for m in metas),
            "page_no": page,
            "content_type": merge_content_type(m.get("content_type") for m in metas),
        })
        pending = []
        pending_tokens = 0

    for chunk in raw_chunks:
        # Accept a HierarchicalChunker chunk object or a bare string (used by the fallback splitting path)
        text = (chunk.text if hasattr(chunk, "text") else chunk).strip()
        if not text:
            continue
        meta = _chunk_provenance(chunk, doc)

        token_count = _count_tokens(tokenizer, text)
        is_table = meta.get("content_type") == "table"
        budget = table_budget if is_table else max_tokens

        # Never merge across a section boundary.
        if pending and section_key(pending[0][1].get("headings")) != section_key(meta.get("headings")):
            _flush_pending()

        if token_count > budget:
            # Oversized — flush any pending small chunks first
            _flush_pending()
            if is_table and _is_triplet_table(text):
                parts = _split_triplet_table(text, tokenizer, budget)
            else:
                parts = _split_by_lines(text, tokenizer, budget, header=None)
            for part in parts:
                output.append({"text": part, **meta})  # split pieces inherit the source chunk's meta
        elif token_count < merge_flush_threshold:
            # Small — accumulate (both halves stay under max_tokens), flush past the threshold
            pending.append((text, meta))
            pending_tokens += token_count
            if pending_tokens >= merge_flush_threshold:
                _flush_pending()
        else:
            # Reasonable size — flush pending then add this chunk as-is
            _flush_pending()
            output.append({"text": text, **meta})

    _flush_pending()
    return output


_HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
_UTF8_BOM = b"\xef\xbb\xbf"
_CHARSET_DECL = re.compile(
    rb"<meta[^>]+charset\s*=|<\?xml[^>]+encoding\s*=", re.IGNORECASE)


def _declare_html_utf8(data: bytes) -> bytes:
    """Mark undeclared UTF-8 HTML as UTF-8 with a byte-order mark.

    docling hands the raw bytes to BeautifulSoup, which guesses the encoding when
    the page declares none; for short CJK text it guessed cp874 (Thai) and the
    whole page came out garbled. Only bytes that are valid, non-ASCII UTF-8 with no
    BOM and no charset / XML encoding declaration are touched; everything else is
    left to the page's own declaration or to detection.
    """
    if data.startswith(_UTF8_BOM) or _CHARSET_DECL.search(data[:4096]):
        return data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    if data.isascii():
        return data
    return _UTF8_BOM + data


def chunk_docling_document(
    doc: Any,
    file_name: str,
    max_tokens: int = 512,
    tokenizer: Optional[str] = None,
    table_max_tokens: Optional[int] = None,
) -> tuple[str, List[dict]]:
    """Export a DoclingDocument to Markdown and chunk it into records.

    The single chunking entry point for every parse path: docling_convert_once (local
    models) and cloud_parser (vision-model Markdown re-assembled into one document, with
    page provenance attached) both end here, so they produce the same record shape
    ``[{"text", "headings", "page_no", "content_type"}, ...]``.
    """
    from docling.chunking import HierarchicalChunker

    # Export full text as Markdown
    full_text = doc.export_to_markdown()
    logger.debug(f"Docling loaded {file_name}: {len(full_text)} chars")

    # Chunk via HierarchicalChunker + per-chunk tokenizer refinement
    # (see helpers above; we bypass HybridChunker for the large-table case).
    hf_tok = get_hf_tokenizer(tokenizer)

    raw_chunks = list(HierarchicalChunker().chunk(dl_doc=doc))
    texts = _refine_chunks_for_token_budget(
        raw_chunks, hf_tok, max_tokens, doc=doc, table_max_tokens=table_max_tokens)

    logger.info(
        f"Docling processed {file_name}: {len(texts)} chunks "
        f"(max_tokens={max_tokens})"
    )
    return full_text, texts


def docling_convert_once(
    file_path: str,
    file_name: str,
    max_tokens: int = 512,
    tokenizer: Optional[str] = None,
    progress_cb: Optional[Any] = None,
    report: Optional[dict] = None,
    table_max_tokens: Optional[int] = None,
) -> tuple[str, List[dict]]:
    """Parse with Docling once, returning both the full text and chunk records (with headings/page_no).

    Consolidates the previously duplicated parsing of docling_load_as_text + docling_load_and_chunk,
    reducing PDF parse time from 2 passes → 1.

    Args:
        file_path: Path to the document file
        file_name: Original file name
        max_tokens: Maximum tokens per chunk
        tokenizer: HuggingFace tokenizer name
        progress_cb: ``(stage, done, total)`` optional page-level parse progress callback.
            PDF reports ("loading", pages_done, total_pages) — OCR-heavy files finally get a progress
            bar; non-paginated formats (docx etc.) don't report counts, with unchanged behavior.
        report: optional dict filled with ``warning`` (pages that could not be parsed,
            None when complete) and ``missing_pages``.
        table_max_tokens: budget a table may use before it is split (default max_tokens).

    Returns:
        (full_text, chunk_records) — full Markdown text + chunk records
            ``[{"text", "headings", "page_no"}, ...]`` (citation provenance)
    """
    try:
        # 1. Convert once (using the singleton converter; models already preloaded)
        converter = get_converter()

        # Page-level progress (observer, see _install_page_progress_patch):
        # the denominator is computed instantly via pymupdf; if it can't (non-PDF), only report
        # done, and the frontend shows the bare stage
        if progress_cb is not None:
            _install_page_progress_patch()
            _page_progress_tls.ctx = {
                "done": 0,
                "total": _pdf_page_count(file_path),
                "cb": progress_cb,
            }

        # FileStorage saves files as extensionless UUIDs. docling's _guess_format falls
        # back on extension when there are no magic bytes, so plain-text formats
        # (.md / .html / .csv / .vtt / .latex / .adoc) fail to detect. Wrapping into a
        # DocumentStream carrying the original file_name restores extension detection.
        path_obj = Path(file_path)
        is_html = Path(file_name).suffix.lower() in _HTML_EXTENSIONS
        # Only convert() holds the inference lock; keep file reading outside to shorten hold time.
        if (path_obj.suffix == "" and Path(file_name).suffix != "") or is_html:
            from docling.datamodel.base_models import DocumentStream
            from io import BytesIO
            with path_obj.open("rb") as f:
                data = f.read()
            if is_html:
                data = _declare_html_utf8(data)

            def _source():  # a stream is consumed by convert(): build a fresh one per attempt
                return DocumentStream(name=file_name, stream=BytesIO(data))
        else:
            def _source():
                return file_path

        def _convert():
            if progress_cb is not None:
                _page_progress_tls.ctx["done"] = 0  # a retry restarts the page count
            with DOCLING_INFER_LOCK:
                return converter.convert(source=_source())

        # Status is checked: a partial result is retried (smaller stage batch after an
        # OOM) and pages still missing come back as a warning for the file record.
        result, parse_warning = convert_checked(_convert, file_name)
        if report is not None:
            report["warning"] = parse_warning
            report["missing_pages"] = missing_pages(result) if parse_warning else []
        doc = result.document

        # 2-3. Markdown export + chunking (shared with the cloud parser's documents)
        return chunk_docling_document(doc, file_name, max_tokens=max_tokens, tokenizer=tokenizer,
                                      table_max_tokens=table_max_tokens)

    except Exception as e:
        # Detect common "the document itself is the problem" errors and give the user a clearer message.
        # These aren't server bugs; the file needs special handling.
        err_msg = str(e).lower()
        if "incorrect password" in err_msg or "encrypted" in err_msg:
            logger.error(
                f"❌ {file_name} is password-protected. "
                f"Remove password before uploading. "
                f"(original error: {e})"
            )
            raise ValueError(
                f"PDF is password-protected: '{file_name}'. "
                f"Please decrypt it before uploading."
            ) from e
        if "is not valid" in err_msg or "corrupt" in err_msg:
            logger.error(f"❌ {file_name} is corrupted or unreadable: {e}")
            raise ValueError(
                f"File appears corrupted or unreadable: '{file_name}'."
            ) from e
        logger.error(f"Docling processing failed for {file_name}: {e}")
        raise
    finally:
        # Page-progress TLS cleanup — the next file on the same thread must not inherit stale context
        if getattr(_page_progress_tls, "ctx", None) is not None:
            _page_progress_tls.ctx = None
        # Hand cached GPU memory back after every file (the card is often shared with vLLM).
        release_accelerator_cache()

