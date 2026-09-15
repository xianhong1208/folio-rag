
"""File record database model.

Provides the file-record table definition and related database operations,
supporting hierarchical management (Folder -> File).
"""

import hashlib
import os
import mimetypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import re
import unicodedata

from .db import File
from db.db import Session as DBSession
from .baseDB import BaseDB
from src.log import get_db_logger

log = get_db_logger()


class ToolError(Exception):
    """Tool-related error."""
    pass


class FileDB(BaseDB):
    """File-record database operations class.

    Provides file upload, validation, and storage functionality.
    """

    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {
        # Plain text (read directly; text_io.read_text_robust auto-handles UTF-8/16/Big5, etc.)
        'txt', 'text', 'json', 'csv',
        'yaml', 'yml', 'xml', 'conf', 'log',
        # Office Open XML (including template/macro variants; parsed structurally via Docling)
        'pdf', 'docx', 'dotx', 'docm', 'dotm',
        'pptx', 'ppsx', 'pptm', 'potm', 'ppsm',  # potx excluded: docling cannot open it
        'xlsx', 'xlsm',
        # OpenDocument (including templates) / e-books / LaTeX / MIME email / subtitles;
        # conformance tests live in test_upload_formats
        'odt', 'ods', 'odp',  # template variants ott/ots/otp excluded: docling cannot open them
        'epub', 'tex', 'latex', 'eml', 'vtt', 'qmd', 'rmd',
        # Legacy Office (docling 2.119+ parses these via LibreOffice conversion).
        # Deployment prerequisite: libreoffice-writer must be installed on the
        # image/machine; when missing, indexing fails explicitly rather than
        # silently producing garbage.
        'doc', 'dot', 'xls', 'xlt', 'ppt', 'pot', 'pps',
        # Outlook mail (docling 2.119+; python-oxmsg is pure Python, no system deps)
        'msg',
        # Markup languages
        'md', 'html', 'htm', 'xhtml', 'adoc', 'asciidoc', 'asc',
        # Images (Docling + RapidOcr)
        'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'webp',
        # Audio (AsrProvider: local docling whisper or cloud; aligned with docling's official 6 formats)
        'wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac',
    }

    @classmethod
    def get_orm_class(cls) -> type:
        """Get the ORM class."""
        return File

    @classmethod
    def get_log(cls):
        """Get the logger instance."""
        return log

    @classmethod
    def get_by_ids(cls, file_ids: List[str]) -> Dict[str, "File"]:
        """Batch-fetch file records to avoid N+1 queries.

        Args:
            file_ids: str list from vector store metadata; converted to UUIDs for
                filtering. Malformed IDs are skipped with a logged warning.

        Returns:
            ``{str(uuid): File}`` dict; file_ids with no match are absent from the dict.
        """
        if not file_ids:
            return {}

        uuid_ids = []
        for fid in file_ids:
            try:
                uuid_ids.append(uuid.UUID(fid) if not isinstance(fid, uuid.UUID) else fid)
            except (ValueError, AttributeError):
                log.warning(f"Invalid UUID format, skipping: {fid}")
        if not uuid_ids:
            return {}

        unique_ids = list(set(uuid_ids))

        with DBSession() as session:
            rows = (
                session.query(cls.get_orm_class())
                .filter(cls.get_orm_class().id.in_(unique_ids))
                .all()
            )

        # Key by str so callers can look up directly with the string file_id from metadata
        return {str(row.id): row for row in rows}


    @classmethod
    def delete_by_folder_id(
        cls, 
        folder_id: int
        ) -> int:
        """Delete all files in a folder by folder ID
        
        Args:
            folder_id: The ID of the folder whose files should be deleted
            
        Returns:
            Number of files deleted
        """
        try:
            with DBSession() as session:
                deleted_count = session.query(cls.get_orm_class()).filter_by(folder_id=folder_id).delete(synchronize_session=False)
                session.commit()
                cls.get_log().info(f"Successfully deleted {deleted_count} files from folder {folder_id}")
                return deleted_count
        except Exception as e:
            cls.get_log().error(f"Failed to delete files from folder {folder_id}: {e}")
            raise
    
    @classmethod
    def _validate_file_content(cls, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Validate file content.

        Args:
            file_content: Binary content of the file.
            file_name: File name.

        Returns:
            A dict containing file information.

        Raises:
            ToolError: Raised when file validation fails.
        """
        file_size = len(file_content)
        if file_size > cls.MAX_FILE_SIZE:
            max_size_mb = cls.MAX_FILE_SIZE / (1024 * 1024)
            raise ValueError(f"File size exceeds limit ({max_size_mb:.1f}MB)")

        ext = Path(file_name).suffix.lower().lstrip('.')
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise ValueError(f"Not supported file type: {ext}")

        mime_type = mimetypes.guess_type(file_name)[0]

        # sha256 content hash for idempotent re-indexing. Computed once
        # at upload so re-index can skip when File.content_hash == FileIndex.content_hash.
        content_hash = hashlib.sha256(file_content).hexdigest()

        return {
            'file_size': file_size,
            'mime_type': mime_type,
            'extension': ext,
            'content_hash': content_hash,
        }
