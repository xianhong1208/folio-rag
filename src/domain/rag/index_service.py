
"""Indexing Service

Encapsulates all folder/file/index database access and cache invalidation needed by the RAG adapter.
This keeps persistence and caching concerns out of the adapter layer.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from db.cached_folderdb import CachedFolderDB
from db.db import FileIndex
from db.filedb import FileDB
from db.fileindexdb import FileIndexDB
from src.domain.rag.dto import FileRequest
from src.domain.exceptions import (
    FileIndexNotFoundError,
    FolderNotFoundError,
    UnauthorizedAccessError,
)
from src.infrastructure.cache.cache_service import CacheKeys, CacheService
from src.log import get_api_logger

logger = get_api_logger()


class IndexingService:
    """Provides folder/file/index helpers used during indexing workflows."""

    def __init__(self, cache_service: Optional[CacheService] = None):
        self._cache = cache_service or CacheService.get_instance()

    # Folder helpers
    def get_folder(self, folder_id: int, token: str):
        """Return folder after validating ownership."""
        folder = CachedFolderDB.get_by_id(folder_id)
        if not folder:
            raise FolderNotFoundError(folder_id=folder_id)

        # fail-closed: reject any ownerless folder whose user_token is NULL/empty,
        # consistent with the check in api/dependencies/auth.py
        if folder.user_token != token:
            raise UnauthorizedAccessError(
                resource_type="folder",
                resource_id=folder_id,
                reason="Token does not have permission for this folder",
            )
        return folder
    
    def get_folder_by_id_unsafe(self, folder_id: int):
        """Return folder without validating ownership (for internal use only)."""
        return CachedFolderDB.get_by_id(folder_id)

    def get_folder_by_name(self, folder_name: str, token: str):
        """Return folder by name scoped to token."""
        folder = CachedFolderDB.get_by_name_and_token(folder_name, token)
        if not folder:
            raise FolderNotFoundError(folder_name=folder_name)
        return folder

    def invalidate_folder_caches(self, folder_id: int) -> None:
        """Clear folder-related caches."""
        logger.debug(f"Invalidating caches for folder {folder_id}")
        self._cache.delete(CacheKeys.files_in_folder(folder_id))
        self._cache.clear_pattern(f"query:{folder_id}:*")

    # File helpers
    def list_files(self, folder_id: int) -> List[FileRequest]:
        """Return all files for a folder as FileRequest models."""
        files = FileDB.get(folder_id=folder_id)
        return [self._to_file_request(record) for record in files]

    def get_files_by_ids(self, file_ids: Iterable[int]):
        """Return mapping of file_id -> file record for the given ids."""
        return FileDB.get_by_ids(list(file_ids))

    @staticmethod
    def _to_file_request(file_index: FileIndex) -> FileRequest:
        excluded_fields = {"upload_at", "updated_at"}
        payload = {
            column.name: getattr(file_index, column.name)
            for column in file_index.__table__.columns
            if column.name not in excluded_fields
        }
        return FileRequest(**payload)

    # File index helpers
    def get_index_for_file(self, file_id):
        return FileIndexDB.get_by_file(file_id)

    def list_indices_for_folder(self, folder_id: int):
        return FileIndexDB.get_by_folder(folder_id)

    def record_index_success(
        self,
        *,
        file_id,
        folder_id: int,
        index_id: str,
        vector_store_table: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        num_chunks: int,
        content_hash: Optional[str] = None,
    ):
        """Record a successful index (upsert, handling reindex of an existing row).

        content_hash is File.content_hash at index time; it lets the next reindex
        short-circuit.
        """
        return FileIndexDB.upsert_indexed(
            file_id=file_id,
            folder_id=folder_id,
            index_id=index_id,
            vector_store_table=vector_store_table,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            num_chunks=num_chunks,
            content_hash=content_hash,
        )

    def mark_index_failed(
        self,
        file_id,
        reason: str,
        folder_id: Optional[int] = None,
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Record an index failure (from the exception-handling path).

        folder_id is required to create a new row when none exists; otherwise this
        only logs a warning.
        """
        FileIndexDB.mark_failed(
            file_id,
            reason,
            folder_id=folder_id,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def delete_index_for_file(self, file_id):
        deleted = FileIndexDB.delete_by_file(file_id)
        if not deleted:
            raise FileIndexNotFoundError(file_id=str(file_id))
        return deleted

    def delete_indices_for_folder(self, folder_id: int):
        """Delete all index records for a folder."""
        return FileIndexDB.delete_by_folder(folder_id)

    def delete_files_for_folder(self, folder_id: int):
        """Delete all files for a folder."""
        return FileDB.delete_by_folder_id(folder_id)
