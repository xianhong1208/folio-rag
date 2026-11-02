
"""Document Loader — three-way loading of a file into a llama_index Document.

Paths:
1. Plain-text family (.txt/.json/.csv/...): read directly via read_text_robust with encoding
   tolerance, bypassing docling.
2. PDFs / images when rag.parsing selects cloud (the default under profile: cloud):
   cloud_parser — a vision model transcribes every page; the pages are re-assembled into
   one document and chunked by the same chunker as the docling path.
3. docling-supported formats: docling_convert_once (audio is first silence-trimmed, then passed
   through Whisper hallucination filtering).
4. Fallback: SimpleDirectoryReader.

The whisper anti-hallucination patch (audio_defense) is applied by docling_loader._build_converter
right before it installs the local Whisper pipeline, so importing this module never imports
whisper/torch.
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
)
from src.domain.rag.asr_provider import AsrProvider, create_asr_provider
from src.domain.rag.audio_defense import (
    _AUDIO_TRIM_EXTENSIONS,
    _filter_whisper_hallucinations,
    _trim_audio_leading_silence,
)
from src.storage.file_storage import FileStorage

logger = get_api_logger()


def clean_text(text: str) -> str:
    """Strip NUL and control characters (rejected by PostgreSQL; commonly sourced from PDF font tables / ICC profiles)."""
    if not text:
        return text
    cleaned = text.replace('\x00', '')
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
        if file_extension in _AUDIO_TRIM_EXTENSIONS and self._asr_enabled:
            if not self._asr_provider.available():
                raise RuntimeError(
                    f"rag.asr provider unavailable (weights/endpoint not ready), cannot transcribe {file_name} — "
                    "Check the weights under assets/ or the rag.asr settings; to skip audio transcription set rag.asr.enabled=false"
                )
            return self._load_audio(
                resolved_path=resolved_path,
                file_name=file_name,
                base_metadata=base_metadata,
                doc_id=doc_id,
                progress_cb=progress_cb,
            )

        cloud_parser = self._cloud_parser_for(file_name)
        if cloud_parser is not None:
            docling_max_tokens = min(
                self._leaf_chunk_size, 380 if self._has_context_generator else 450
            )
            full_text, chunk_texts = cloud_parser.parse(
                resolved_path, file_name,
                max_tokens=docling_max_tokens,
                tokenizer=self._embedding_model_name,
                progress_cb=progress_cb,
            )
            content = clean_text(full_text)
            if not content.strip():
                raise ValueError(f"Empty content extracted from {file_name}")
            doc = Document(text=content, metadata=base_metadata, doc_id=doc_id)
            doc.metadata["_docling_chunks"] = chunk_texts
            return doc

        if is_docling_supported(file_name):
            docling_max_tokens = min(
                self._leaf_chunk_size, 380 if self._has_context_generator else 450
            )
            full_text, chunk_texts = docling_convert_once(
                file_path=resolved_path,
                file_name=file_name,
                max_tokens=docling_max_tokens,
                tokenizer=self._embedding_model_name,
                progress_cb=progress_cb,
            )
            content = clean_text(full_text)
            if not content.strip():
                raise ValueError(f"Empty content extracted from {file_name}")

            doc = Document(text=content, metadata=base_metadata, doc_id=doc_id)
            doc.metadata["_docling_chunks"] = chunk_texts
            return doc

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file_name
            shutil.copy2(resolved_path, temp_file_path)
            return self._load_via_simple_reader(
                temp_dir, file_name, base_metadata, doc_id)

    @staticmethod
    def _cloud_parser_for(file_name: str):
        """A CloudDocumentParser when the live config routes this file type to the cloud
        (read per file, so a console change to rag.parsing applies to the next file)."""
        from src.config.config_manager import Config
        from src.domain.rag.cloud_parser import CloudDocumentParser, is_cloud_parsed
        rag = getattr(Config.get_config_model(), "rag", None)
        if rag is None or not is_cloud_parsed(rag, file_name):
            return None
        return CloudDocumentParser.from_config(rag)

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
        docling_max_tokens = min(
            self._leaf_chunk_size, 380 if self._has_context_generator else 450
        )

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
