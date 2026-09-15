
"""RAG maintenance service — index deletion, orphan cleanup, and index listing.

Shares RAGContext.indexing_service / vector_store_manager.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from src.domain.rag.vector_store_manager import VectorStoreManager  # physical-table deletion is centralized here
from src.api.router.response import DeleteFolderIndexResponse, FileIndexResult
from src.domain.exceptions import (
    DomainException,
    FileIndexNotFoundError,
    RAGOperationError,
)
from src.infrastructure.cache.cache_service import CacheKeys, CacheService, CacheTTL
from src.log import get_adapter_logger, log_err, log_op, log_warn

if TYPE_CHECKING:
    from src.adapter.rag_context import RAGContext

logger = get_adapter_logger()


class RAGMaintenanceService:
    """Service handling index maintenance (deletion, listing)."""

    def __init__(self, ctx: "RAGContext"):
        self._ctx = ctx

    async def delete_document_index(self, file_id: int, token: str) -> bool:
        """Delete a file's index from the vector store.

        Args:
            file_id: File ID
            token: API token, required — drives the folder-ownership check inside
                _get_vector_store

        Returns:
            True if deletion succeeded

        Security: token must be non-empty — the former "skip the access check when
        no token is supplied" branch was removed to close an IDOR. An orphan index
        (folder already deleted) makes get_vector_store raise; that is caught and the
        DB record is deleted directly.
        """
        if not token:
            raise ValueError("delete_document_index requires a non-empty token")

        indexing_service = self._ctx.indexing_service

        try:
            index_record = indexing_service.get_index_for_file(file_id)
            if not index_record:
                log_warn(
                    logger, "DELETE_INDEX_SKIP", file_id=file_id, token=token,
                    msg="no index found",
                )
                return False

            folder_id = index_record.folder_id

            # Vector store ownership check — an orphan index just deletes its DB row
            try:
                vector_store = self._ctx.get_vector_store(folder_id, token)
            except (RAGOperationError, DomainException) as folder_err:
                log_warn(
                    logger, "DELETE_INDEX_ORPHAN",
                    folder_id=folder_id, file_id=file_id, token=token,
                    msg=f"folder access failed, treating as orphan index: {folder_err}",
                )
                indexing_service.delete_index_for_file(file_id)
                indexing_service.invalidate_folder_caches(folder_id)
                return True

            deletion_successful = self._delete_chunks_from_vector_store(
                vector_store=vector_store,
                index_record=index_record,
                folder_id=folder_id,
                file_id=file_id,
            )
            if not deletion_successful:
                log_warn(logger, "DELETE_NO_CHUNKS", folder_id=folder_id, file_id=file_id)

            # Even if chunk deletion failed, still clear the DB record so nothing is
            # left pointing at nonexistent chunks
            indexing_service.delete_index_for_file(file_id)
            indexing_service.invalidate_folder_caches(folder_id)

            log_op(
                logger, "DELETE_INDEX_DONE",
                folder_id=folder_id, file_id=file_id, token=token,
            )
            return True

        except FileIndexNotFoundError:
            log_warn(
                logger, "DELETE_INDEX_ALREADY", file_id=file_id,
                msg="index record already removed",
            )
            return False
        except DomainException:
            raise
        except Exception as e:
            log_err(logger, "DELETE_INDEX_FAIL", e, file_id=file_id, token=token)
            raise ValueError(f"Failed to delete document index: {str(e)}")

    async def delete_folder_index(
        self,
        folder_id: int,
        token: str,
    ) -> DeleteFolderIndexResponse:
        """Delete all of a folder's indexes (keeping the folder and files themselves).

        Deletes every related row in FileIndexDB and DROPs the corresponding pgvector
        table; the Folder / File records are untouched. Cancels all in-flight indexing
        jobs first to avoid races.

        Args:
            folder_id: The folder whose indexes to clear.
            token: User token (access control).

        Returns:
            DeleteFolderIndexResponse (deletion stats + per-file detail list).
        """
        indexing_service = self._ctx.indexing_service

        try:
            folder = indexing_service.get_folder(folder_id, token)
            folder_name = folder.name
            vector_table_uuid = str(folder.vector_table_uuid)

            log_op(
                logger, "DELETE_FOLDER_IDX_START",
                folder_id=folder_id, token=token, msg=f"name={folder_name}",
            )

            # Cancel in-flight jobs first; otherwise dropping the table while another
            # job is still embedding will blow up
            try:
                from src.domain.rag.index_job_manager import IndexingJobManager
                cancelled = await IndexingJobManager.get_instance().cancel_jobs_for_folder(folder_id)
                if cancelled:
                    log_op(
                        logger, "DELETE_FOLDER_IDX_CANCEL",
                        folder_id=folder_id, token=token,
                        msg=f"cancelled {len(cancelled)} in-flight job(s)",
                    )
            except Exception as cancel_err:
                log_warn(
                    logger, "DELETE_FOLDER_IDX_CANCEL_FAIL",
                    folder_id=folder_id, token=token, msg=str(cancel_err),
                )

            index_records = indexing_service.list_indices_for_folder(folder_id)
            total_files = len(index_records)

            # Order matters: delete FileIndex rows first, then DROP the table, raising
            # if either step fails. Dropping first would leave rows with status=indexed
            # but no vectors — a later reindex then short-circuits on content_hash and
            # never self-heals. This order self-heals: a retry re-deletes rows
            # idempotently and re-runs the DROP.
            deleted_indices = indexing_service.delete_indices_for_folder(folder_id)
            log_op(
                logger, "DELETE_INDICES",
                folder_id=folder_id, msg=f"{deleted_indices} records",
            )

            vsm = self._ctx.vector_store_manager
            vector_store_table = vsm.physical_table_name(folder_id, vector_table_uuid)
            if vsm.drop_table(folder_id, vector_table_uuid):
                log_op(logger, "DROP_TABLE", folder_id=folder_id, msg=f"table={vector_store_table}")
            else:
                # A failed DROP must not be swallowed: leaving the old vector table
                # behind with no FileIndex means the next reindex overwrites it with
                # duplicate vectors. Raise so the caller (reindex endpoint) aborts and
                # reports it.
                raise ValueError(f"Failed to drop vector table {vector_store_table}")

            indexing_service.invalidate_folder_caches(folder_id)
            self._ctx.invalidate_query_engine_cache(folder_id)
            # drop_table() above already cleared the VSM cache entry.

            # Per-file rows: files are preserved even though their indices are gone.
            files = indexing_service.list_files(folder_id)
            results = [
                FileIndexResult(
                    file_id=str(f.id),
                    filename=f.file_name,
                    status="success",
                    message="Index deleted successfully",
                    num_chunks=None,
                )
                for f in files
            ]

            summary_message = (
                f"Index deletion completed: deleted {deleted_indices} file indices and "
                f"dropped vector store table (folder and files preserved)"
            )
            log_op(
                logger, "DELETE_FOLDER_IDX_DONE",
                folder_id=folder_id, token=token, msg=summary_message,
            )

            return DeleteFolderIndexResponse(
                folder_id=folder_id,
                total_files=total_files,
                successful=deleted_indices,
                failed=0,
                results=results,
                message=summary_message,
            )

        except DomainException:
            raise
        except Exception as e:
            log_err(logger, "DELETE_FOLDER_IDX_FAIL", e, folder_id=folder_id, token=token)
            raise ValueError(f"Failed to delete folder index: {str(e)}")

    async def get_indexed_files(self, folder_id: int) -> List[Dict[str, Any]]:
        """Get all indexed files in the given folder (cached)."""
        indexing_service = self._ctx.indexing_service

        try:
            cache = CacheService.get_instance()
            cache_key = CacheKeys.files_in_folder(folder_id)
            cached = cache.get(cache_key)

            if cached is not None:
                logger.debug("Indexed files cache HIT: %s", cache_key)
                return cached

            index_records = indexing_service.list_indices_for_folder(folder_id)

            if not index_records:
                cache.set(cache_key, [], ttl=CacheTTL.FILE_LIST)
                return []

            file_ids = [record.file_id for record in index_records]
            files_by_id = indexing_service.get_files_by_ids(file_ids)

            results = []
            for record in index_records:
                # FileDB.get_by_ids keys by str(uuid); passing a UUID object would
                # always miss and fall back to "Unknown"
                file_record = files_by_id.get(str(record.file_id))
                results.append({
                    "file_id": str(record.file_id),
                    "file_name": file_record.file_name if file_record else "Unknown",
                    "num_chunks": record.num_chunks,
                    "indexed_at": record.indexed_at.isoformat(),
                    "status": record.status,
                    "embedding_model": record.embedding_model,
                    "chunk_size": record.chunk_size,
                    "chunk_overlap": record.chunk_overlap,
                })

            cache.set(cache_key, results, ttl=CacheTTL.FILE_LIST)
            logger.debug(
                "Indexed files cache MISS: %s (cached %d records)",
                cache_key, len(results),
            )
            return results

        except Exception as e:
            log_err(logger, "GET_INDEXED_FILES_FAIL", e, folder_id=folder_id)
            return []

    def _delete_chunks_from_vector_store(
        self,
        *,
        vector_store,
        index_record,
        folder_id: int,
        file_id: int,
    ) -> bool:
        """Physically delete a file's chunks from the vector store.

        Delegates to VectorStoreManager.delete_chunks_by_file (matches on
        metadata_->>'file_id'). LlamaIndex's vector_store.delete(ref_doc_id) is not
        used: it frequently fails to match the ref_doc_id written for hierarchical
        chunks, so we go straight to raw SQL.

        Returns True if at least one chunk was deleted.
        """
        try:
            table_name = index_record.vector_store_table

            if not table_name or not table_name.startswith("data_"):
                folder = self._ctx.indexing_service.get_folder_by_id_unsafe(folder_id)
                if folder:
                    table_name = VectorStoreManager.physical_table_name(
                        folder_id, folder.vector_table_uuid)
                    logger.debug(
                        f"[DELETE_META] fid={folder_id} file={file_id} | "
                        f"using table={table_name}"
                    )

            # DELETE is centralized in VectorStoreManager.delete_chunks_by_file
            # (bind params guard against injection)
            deleted_count = VectorStoreManager.delete_chunks_by_file(table_name, file_id)
            if deleted_count > 0:
                log_op(
                    logger, "DELETE_BY_META",
                    folder_id=folder_id, file_id=file_id,
                    msg=f"{deleted_count} chunks removed",
                )
                return True
            return False

        except Exception as e:
            log_err(logger, "DELETE_META_FAIL", e, folder_id=folder_id, file_id=file_id)
            return False
