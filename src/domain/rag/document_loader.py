
"""Document Loader — three-way loading of a file into a llama_index Document.

Three paths:
1. Plain-text family (.txt/.json/.csv/...): read directly via read_text_robust with encoding
   tolerance, bypassing docling.
2. docling-supported formats: docling_convert_once (audio is first silence-trimmed, then passed
   through Whisper hallucination filtering).
3. Fallback: SimpleDirectoryReader.

Import ordering: this module imports audio_defense at the top — the whisper anti-hallucination
patch is applied at import time and must run before docling's get_converter(). Since
hierarchical_indexer imports this module at its top, the chain becomes
… → hierarchical_indexer → document_loader → audio_defense, preserving the required ordering.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from llama_index.core import Document, SimpleDirectoryReader

from src.log import get_api_logger
from src.domain.rag.docling_loader import (
    is_docling_supported,
    docling_convert_once,
    ocr_enabled,
)
from src.domain.rag.asr_provider import AsrProvider, create_asr_provider
from src.domain.rag.chunk_identity import token_budgets
from src.domain.rag.han_script import strip_invisible
from src.domain.rag.audio_defense import (
    _AUDIO_TRIM_EXTENSIONS,
    _filter_whisper_hallucinations,
    _trim_audio_leading_silence,
)
from src.storage.file_storage import FileStorage

logger = get_api_logger()

# Formats whose text can come only from OCR when the file has no text layer.
_OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def clean_text(text: str) -> str:
    """Strip NUL and control characters (rejected by PostgreSQL; commonly sourced from PDF
    font tables / ICC profiles), zero-width characters (they split keyword tokens), and
    turn no-break / thin spaces into plain spaces."""
    if not text:
        return text
    cleaned = strip_invisible(text).replace('\x00', '')
    cleaned = ''.join(
        char for char in cleaned
        if char in ('\n', '\t', '\r') or ord(char) >= 32
    )
    return cleaned


class DocumentLoader:
    """File loader (composed into HierarchicalIndexer; logic factored out of its load_document_from_file)."""

    def __init__(
        self,
        leaf_chunk_size: int,
        embedding_model_name: Optional[str] = None,
        has_context_generator: bool = False,
        asr_provider: Optional[AsrProvider] = None,
        asr_enabled: bool = True,
    ):
        self._leaf_chunk_size = leaf_chunk_size
        self._embedding_model_name = embedding_model_name
        # Only affects the docling_max_tokens formula (the contextual prefix consumes token budget)
        self._has_context_generator = has_context_generator
        # Audio transcription uses the injected provider; when not injected, defaults to
        # docling-whisper (available() follows the existing auto-discover semantics)
        self._asr_provider = asr_provider or create_asr_provider(None)
        self._asr_enabled = asr_enabled

    def load(
        self,
        file_path: str,
        file_id: str,
        file_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_cb: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
    ) -> Document:
        """Load a file into a Document, matching the behavior of the flat-RAG load_document_from_file.

        progress_cb: optional page-level parse progress ``("loading", done, total)`` (for PDF/OCR),
        letting large scanned files show "parsing document 12/161" instead of appearing to hang.
        """
        # Plain-text family: read directly (read_text_robust for encoding tolerance), bypassing docling
        text_extensions = {
            '.txt', '.text', '.json', '.csv',
            '.yaml', '.yml', '.xml', '.conf', '.log',
        }
        file_extension = Path(file_name).suffix.lower()

        # The DB stores a relative `storage/...` string. A bare open() would resolve against cwd,
        # mismatching save_file's _PROJECT_ROOT-anchored writes → ENOENT. Always resolve via FileStorage.
        resolved_path = str(FileStorage.resolve_path(file_path))

        base_metadata: Dict[str, Any] = {
            "file_id": str(file_id),
            "file_name": file_name,
            "file_path": file_path,  # keep the DB relative string for metadata consumers
        }
        if metadata:
            base_metadata.update({k: v for k, v in metadata.items() if v is not None})

        doc_id = f"file_{file_id}"

        if file_extension in text_extensions:
            # Encoding tolerance: UTF-16 (Windows Notepad "Unicode") / Big5 txt·csv are common in
            # practice, and hardcoding utf-8 would fail indexing outright with UnicodeDecodeError.
            # When it can't identify the encoding, read_text_robust raises a ValueError with
            # remediation guidance (surfaced in FileIndex.error_message) rather than silently
            # indexing garbled text.
            from src.utils.text_io import read_text_robust
            content = clean_text(read_text_robust(resolved_path, file_name))
            return Document(text=content, metadata=base_metadata, doc_id=doc_id)

        # Audio uses the injected AsrProvider (trim → transcribe → hallucination filter).
        # The config-selected provider is the ONLY speech source: if it can't serve, raise
        # an explicit error (file marked failed) rather than silently switching models or
        # skipping. Only rag.asr.enabled=false means "explicitly do not transcribe".
        if file_extension in _AUDIO_TRIM_EXTENSIONS and not self._asr_enabled:
            # Off means off: never fall through to docling's whisper or SimpleDirectoryReader.
            raise ValueError(
                f"Speech-to-text is disabled (rag.asr.enabled=false), so '{file_name}' was "
                "not transcribed. Enable it and re-index the file to include it."
            )
        if file_extension in _AUDIO_TRIM_EXTENSIONS:
            if not self._asr_provider.available():
                raise RuntimeError(
                    f"The configured rag.asr provider is not available (weights or endpoint "
                    f"not ready), so '{file_name}' cannot be transcribed. Check assets/ or "
                    "the rag.asr settings; set rag.asr.enabled=false to skip audio on purpose."
                )
            return self._load_audio(
                resolved_path=resolved_path,
                file_name=file_name,
                base_metadata=base_metadata,
                doc_id=doc_id,
                progress_cb=progress_cb,
            )

        if is_docling_supported(file_name):
            docling_max_tokens, table_max_tokens = token_budgets(
                self._leaf_chunk_size, self._has_context_generator)
            report: Dict[str, Any] = {}
            full_text, chunk_texts = docling_convert_once(
                file_path=resolved_path,
                file_name=file_name,
                max_tokens=docling_max_tokens,
                tokenizer=self._embedding_model_name,
                progress_cb=progress_cb,
                report=report,
                table_max_tokens=table_max_tokens,
            )
            content = clean_text(full_text)
            if not content.strip():
                if file_extension in _OCR_EXTENSIONS and not ocr_enabled():
                    raise ValueError(
                        f"No text found in '{file_name}' and OCR is disabled "
                        "(rag.docling.ocr_enabled=false). Enable OCR to index scanned pages."
                    )
                raise ValueError(f"Empty content extracted from {file_name}")

            doc = Document(text=content, metadata=base_metadata, doc_id=doc_id)
            doc.metadata["_docling_chunks"] = chunk_texts
            if report.get("warning"):
                # Underscore keys never reach chunk metadata; the indexer hands it to the file record.
                doc.metadata["_parse_warning"] = report["warning"]
            return doc

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file_name
            shutil.copy2(resolved_path, temp_file_path)
            return self._load_via_simple_reader(
                temp_dir, file_name, base_metadata, doc_id)

    def _load_via_simple_reader(self, temp_dir, file_name, base_metadata, doc_id) -> Document:
        reader = SimpleDirectoryReader(
            input_dir=temp_dir,
            filename_as_id=True,
            recursive=False,
        )
        docs = reader.load_data()
        if not docs:
            raise ValueError(f"No content extracted from {file_name}")
        original = docs[0]
        cleaned = clean_text(original.text)
        merged_meta = {**original.metadata, **base_metadata}
        return Document(text=cleaned, metadata=merged_meta, doc_id=doc_id)

    def _load_audio(
        self, *, resolved_path: str, file_name: str,
        base_metadata: Dict[str, Any], doc_id: str,
        progress_cb: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
    ) -> Document:
        """Audio: leading-silence trim → AsrProvider transcription → Whisper hallucination filter → Document.

        Trimming/filtering are model-agnostic pre/post-processing (audio_defense), deliberately kept
        outside the provider so both defenses survive swapping in any provider.
        """
        docling_max_tokens, _ = token_budgets(self._leaf_chunk_size, self._has_context_generator)

        # 1. Trim leading silence (on failure, fall back to the original file; don't block transcription)
        effective_path = resolved_path
        trimmed_temp: Optional[str] = None
        try:
            trimmed_temp = _trim_audio_leading_silence(resolved_path, file_name)
            if trimmed_temp:
                effective_path = trimmed_temp
        except Exception as e:
            logger.warning(f"Audio trim failed for {file_name}, using original: {e}")
            trimmed_temp = None

        try:
            # 2. Transcribe (provider failures re-raise as-is → the existing failure path writes a failed tag)
            result = self._asr_provider.transcribe(
                audio_path=effective_path,
                file_name=file_name,
                max_tokens=docling_max_tokens,
                tokenizer=self._embedding_model_name,
                progress_cb=progress_cb,
            )
            full_text = result.text
            chunk_texts = result.chunks

            # 3. Strip known Whisper hallucinations
            full_text, n_full = _filter_whisper_hallucinations(full_text)
            n_chunks = 0
            if chunk_texts is not None:
                cleaned_chunks = []
                for c in chunk_texts:
                    cleaned, n = _filter_whisper_hallucinations(c)
                    cleaned_chunks.append(cleaned)
                    n_chunks += n
                chunk_texts = cleaned_chunks
            if n_full or n_chunks:
                logger.info(
                    f"Filtered Whisper hallucinations from {file_name}: "
                    f"{n_full} matches in full text, {n_chunks} across chunks"
                )

            content = clean_text(full_text)
            if not content.strip():
                raise ValueError(f"Empty content extracted from {file_name}")

            doc = Document(text=content, metadata=base_metadata, doc_id=doc_id)
            if chunk_texts is not None:
                # the docling provider pre-splits chunks; cloud returns None → leaf_splitter plain-text path
                doc.metadata["_docling_chunks"] = chunk_texts
            return doc
        finally:
            if trimmed_temp:
                Path(trimmed_temp).unlink(missing_ok=True)
