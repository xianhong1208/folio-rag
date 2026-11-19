
"""Document Loader — loading a file into a llama_index Document.

Paths:
1. Plain-text family (.txt/.json/...): read directly via read_text_robust with encoding
   tolerance, bypassing docling. A .csv is read the same way and turned into a Markdown
   table chunked as a table, so every chunk repeats the header row.
2. Audio: the injected AsrProvider (the rag.asr OpenAI-compatible endpoint), with
   leading-silence trimming before and Whisper hallucination filtering after.
3. PDFs and images with a vision model (rag.parsing.vision; rag.parsing.pdf auto or vision):
   cloud_parser — the model transcribes every page; the pages are re-assembled into one
   document and chunked by the shared chunker.
4. PDFs without one (or rag.parsing.pdf: text): pdf_text_layer — the PDF's own text layer,
   a marker for each page without one; a PDF with no text at all fails. Images without a
   vision model fail with what to set (uploads are refused before that).
5. Other docling formats (Office, OpenDocument, HTML, Markdown, ...): docling_convert_once,
   docling's model-free backends.
6. Fallback: SimpleDirectoryReader.
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
from src.domain.rag.chunk_identity import token_budgets
from src.domain.rag.han_script import strip_invisible
from src.domain.rag.audio_defense import (
    _AUDIO_TRIM_EXTENSIONS,
    _filter_whisper_hallucinations,
    _trim_audio_leading_silence,
)
from src.storage.file_storage import FileStorage

logger = get_api_logger()

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


def _md_cell(value: str) -> str:
    return " ".join((value or "").split()).replace("|", "\\|")


def csv_to_markdown(content: str) -> Optional[str]:
    """CSV text -> Markdown table (first row = header), or None when it is not a table
    (fewer than two rows or columns, or unparseable)."""
    import csv
    import io

    text = (content or "").lstrip("\ufeff")
    if not text.strip():
        return None
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    except csv.Error:
        return None
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    if width < 2:
        return None
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(_md_cell(c) for c in rows[0]) + " |",
             "| " + " | ".join("---" for _ in range(width)) + " |"]
    lines += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


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
        # Audio transcription uses the injected provider; when not injected, speech-to-text
        # is unconfigured and audio fails with a message saying what to set
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

        if file_extension == ".csv":
            from src.utils.text_io import read_text_robust
            content = clean_text(read_text_robust(resolved_path, file_name))
            return self._load_csv(content, base_metadata, doc_id, file_name)

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
        # It is the ONLY speech source: if it can't serve, raise an explicit error (file
        # marked failed) rather than indexing an empty file. Uploads are already refused
        # in that state (asr_upload_refusal); this covers files stored before.
        if file_extension in _AUDIO_TRIM_EXTENSIONS and not self._asr_enabled:
            # Off means off: never fall through to docling or SimpleDirectoryReader.
            raise ValueError(
                f"Speech-to-text is disabled (rag.asr.enabled=false), so '{file_name}' was "
                "not transcribed. Enable it and re-index the file to include it."
            )
        if file_extension in _AUDIO_TRIM_EXTENSIONS:
            if not self._asr_provider.available():
                from src.domain.rag.asr_provider import NOT_CONFIGURED
                raise RuntimeError(f"{NOT_CONFIGURED}, so '{file_name}' cannot be transcribed.")
            return self._load_audio(
                resolved_path=resolved_path,
                file_name=file_name,
                base_metadata=base_metadata,
                doc_id=doc_id,
                progress_cb=progress_cb,
            )

        page_parser = self._page_parser_for(file_name)
        if page_parser is not None:
            # Vision model or text layer: the same budgets as the docling path (prefix +
            # breadcrumb + text must fit the embedding window), and pages that could not be
            # read surface as the file's warning.
            docling_max_tokens, table_max_tokens = token_budgets(
                self._leaf_chunk_size, self._has_context_generator)
            warnings: Dict[str, Any] = {}
            full_text, chunk_texts = page_parser(
                resolved_path, file_name,
                max_tokens=docling_max_tokens,
                tokenizer=self._embedding_model_name,
                progress_cb=progress_cb,
                table_max_tokens=table_max_tokens,
                warnings=warnings,
            )
            content = clean_text(full_text)
            if not content.strip():
                raise ValueError(f"Empty content extracted from {file_name}")
            doc = Document(text=content, metadata=base_metadata, doc_id=doc_id)
            doc.metadata["_docling_chunks"] = chunk_texts
            if warnings.get("warning"):
                doc.metadata["_parse_warning"] = warnings["warning"]
            return doc

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

    def _load_csv(self, content: str, base_metadata: Dict[str, Any], doc_id: str,
                  file_name: str) -> Document:
        """CSV -> a Markdown table chunked as one table.

        Read as plain text, a CSV was split into chunks of which only the first carried
        the header row, so every later chunk was a list of bare values. As a table, an
        oversized CSV is split at row boundaries with the header repeated on every piece
        (and its pieces share one contextual prefix). Content that does not parse as a
        table with a header and at least one row stays plain text.
        """
        table = csv_to_markdown(content)
        doc = Document(text=table or content, metadata=base_metadata, doc_id=doc_id)
        if not table:
            return doc
        from src.domain.rag.docling_loader import (
            _refine_chunks_for_token_budget,
            get_hf_tokenizer,
        )
        budget, table_budget = token_budgets(self._leaf_chunk_size, self._has_context_generator)
        doc.metadata["_docling_chunks"] = _refine_chunks_for_token_budget(
            [{"text": table, "content_type": "table"}],
            get_hf_tokenizer(self._embedding_model_name), budget,
            table_max_tokens=table_budget,
        )
        logger.info(f"CSV {file_name}: {len(doc.metadata['_docling_chunks'])} table chunk(s)")
        return doc

    def _size_transcript_records(self, records):
        """Timed transcript records -> leaf-sized records: segments merge and split
        within their time block (never across), sized in embedding tokens."""
        from src.domain.rag.docling_loader import (
            _refine_chunks_for_token_budget,
            get_hf_tokenizer,
        )
        budget, _ = token_budgets(self._leaf_chunk_size, self._has_context_generator)
        return _refine_chunks_for_token_budget(
            [r if isinstance(r, dict) else {"text": r} for r in records],
            get_hf_tokenizer(self._embedding_model_name), budget)

    @staticmethod
    def _page_parser_for(file_name: str):
        """The page parser for a PDF or an image under the live config (read per file, so a
        console change to rag.parsing applies to the next file): the vision model's
        ``parse``, the text layer's ``parse``, or None for every other format. An image
        without a vision model raises ImagesNeedVisionError."""
        from src.config.config_manager import Config
        from src.config.roles import pdf_parsing
        from src.domain.rag import pdf_text_layer
        from src.domain.rag.cloud_parser import (
            IMAGE_EXTENSIONS,
            PDF_EXTENSIONS,
            CloudDocumentParser,
            ImagesNeedVisionError,
            images_need_vision_message,
            uses_vision,
        )
        ext = Path(file_name).suffix.lower()
        if ext not in PDF_EXTENSIONS and ext not in IMAGE_EXTENSIONS:
            return None
        rag = getattr(Config.get_config_model(), "rag", None)
        if rag is not None and uses_vision(rag, file_name):
            return CloudDocumentParser.from_config(rag).parse
        if ext in IMAGE_EXTENSIONS:
            raise ImagesNeedVisionError(images_need_vision_message(file_name))
        if rag is not None and pdf_parsing(rag) == "vision":
            # rag.parsing.pdf: vision without a usable model (refused at load and in the
            # console; this covers a model removed since): never fall back silently.
            raise ValueError(f"rag.parsing.pdf is 'vision' but rag.parsing.vision is not configured, "
                             f"so '{file_name}' was not read. Configure the vision model or set "
                             "rag.parsing.pdf to auto.")
        return pdf_text_layer.parse

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
                    if isinstance(c, dict):  # timed record (segment text + time-block heading)
                        cleaned, n = _filter_whisper_hallucinations(c.get("text") or "")
                        cleaned_chunks.append({**c, "text": cleaned})
                    else:
                        cleaned, n = _filter_whisper_hallucinations(c)
                        cleaned_chunks.append(cleaned)
                    n_chunks += n
                chunk_texts = cleaned_chunks
                if any(isinstance(c, dict) for c in chunk_texts):
                    chunk_texts = self._size_transcript_records(chunk_texts)
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
                # docling pre-splits chunks; segment-based providers return timed records
                # (sized above); None (no timing) → leaf_splitter plain-text path
                doc.metadata["_docling_chunks"] = chunk_texts
            return doc
        finally:
            if trimmed_temp:
                Path(trimmed_temp).unlink(missing_ok=True)
