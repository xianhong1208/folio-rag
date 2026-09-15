
"""
CRUD operations for file upload, download, update and deletion.

Supports multiple file types ('txt', 'pdf', 'doc', 'docx', 'csv', 'md', 'json')
with file validation and security checks.
"""

from typing import Dict, Any, Optional, List, Union
from db.db import File, Session as DBSession
from db.filedb import FileDB
from db.folderdb import FolderDB
from sqlalchemy.exc import NoResultFound
from uuid import UUID, uuid4
from src.log import get_adapter_logger, log_op, log_err, log_warn, mask_token
from src.adapter.model import FileConfigData, FileDownloadData
from src.storage.file_storage import FileStorage
logger = get_adapter_logger()



class FileAdapter():

    @staticmethod
    def _verify_folder_ownership(folder_id: int, user_token: str):
        """Verify that a folder belongs to the given user.

        Args:
            folder_id: Folder ID
            user_token: User token

        Returns:
            Folder object if authorized

        Raises:
            NoResultFound: If the folder does not exist or does not belong to the user
        """
        folder_check = FolderDB.get(id=folder_id, user_token=user_token)
        if not folder_check:
            raise NoResultFound(f"Folder id {folder_id} does not exist or you don't have permission to access it")
        return folder_check[0]

    @staticmethod
    def _update_folder_stats(folder_id: int):
        """Update folder statistics."""
        with DBSession() as session:
            files_in_folder = session.query(File).filter(File.folder_id == folder_id).all()
            actual_file_count = len(files_in_folder)
            actual_total_size = sum(f.file_size for f in files_in_folder)
            FolderDB.update(folder_id, file_count=actual_file_count, total_size=actual_total_size)

    @staticmethod
    def _build_upload_response(is_batch: bool, successful_uploads: list, failed_uploads: list, 
                             skipped_uploads: list, files_list: list, total_size_added: int, 
                             single_file_result: Optional[FileConfigData]) -> Union[list[FileConfigData], Dict[str, Any]]:
        """Construct upload response"""
        if is_batch:
            return {
                "message": f"Batch upload completed: {len(successful_uploads)} succeeded, {len(failed_uploads)} failed, {len(skipped_uploads)} skipped",
                "summary": {
                    "total_files": len(files_list),
                    "successful_count": len(successful_uploads),
                    "failed_count": len(failed_uploads),
                    "skipped_count": len(skipped_uploads),
                    "total_size_added": total_size_added
                },
                "successful_uploads": successful_uploads,
                "failed_uploads": failed_uploads,
                "skipped_uploads": skipped_uploads
            }
        else:
            return [single_file_result]

    @staticmethod
    async def upload_file(
        files_data: Union[Dict[str, Any], List[Dict[str, Any]]],
        folder_id: int,
        user_token: str,
    ) -> Optional[Union[list[FileConfigData], Dict[str, Any]]]:
        """Upload files to a folder (single or batch).

        Args:
            files_data: A single file-data dict or a list of file-data dicts.
                      Single-file format: {
                          'file_content': bytes,
                          'file_name': str,
                          'description': Optional[str],
                          'tags': Optional[List[str]]
                      }
            folder_id: Folder ID
            user_token: User token

        Returns:
            Single file: list[FileConfigData] or an error dict.
            Batch upload: a dict with upload result statistics.
        """

        folder = FileAdapter._verify_folder_ownership(folder_id, user_token)
        is_batch = isinstance(files_data, list)
        files_list = files_data if is_batch else [files_data]

        successful_uploads = []
        failed_uploads = []
        skipped_uploads = []
        total_size_added = 0

        for file_data in files_list:
            try:
                file_content = file_data['file_content']
                file_name = file_data['file_name']
                description = file_data.get('description')
                tags = file_data.get('tags')

                logger.info(f"Processing file: {file_name}, description: {description}, tags: {tags}")

                validation_result = FileDB._validate_file_content(file_content, file_name)

                # Generate the UUID up front so the file is saved and the DB
                # record created against the same id in a single pass.
                file_id = uuid4()
                file_path = FileStorage.save_file(folder.user_token, folder.name, str(file_id), file_content)

                file_record = FileDB.create(
                    id=file_id,
                    folder_id=folder.id,
                    file_name=file_name,
                    file_path=file_path,
                    file_size=validation_result['file_size'],
                    mime_type=validation_result['mime_type'],
                    description=description,
                    tags=tags,
                    content_hash=validation_result.get('content_hash'),
                )

                total_size_added += validation_result['file_size']

                upload_info = {
                    "id": file_record.id,
                    "folder_id": file_record.folder_id,
                    "file_name": file_record.file_name,
                    "file_path": file_record.file_path,
                    "file_size": file_record.file_size,
                    "mime_type": file_record.mime_type,
                    "description": file_record.description,
                    "tags": file_record.tags,
                    "upload_time": file_record.upload_time.isoformat() if is_batch else file_record.upload_time,
                    "updated_time": file_record.updated_time
                }
                
                if is_batch:
                    successful_uploads.append({k: v for k, v in upload_info.items() if k != "updated_time"})
                else:
                    single_file_result = FileConfigData(**{
                        k: v for k, v in upload_info.items()
                        if k in FileConfigData.model_fields
                    })
                    
            except Exception as e:
                if is_batch:
                    failed_uploads.append({
                        'file_name': file_data.get('file_name', 'Unknown'),
                        'error': str(e)
                    })
                else:
                    raise

        if len(successful_uploads) > 0 or not is_batch:
            FileAdapter._update_folder_stats(folder.id)

        return FileAdapter._build_upload_response(
            is_batch, successful_uploads, failed_uploads, skipped_uploads, 
            files_list, total_size_added, single_file_result if not is_batch else None
        )

    @staticmethod
    async def get_file(user_token: str, **kwargs) -> Optional[list[FileConfigData]]:
        """Retrieve file information.

        Args:
            user_token: User token
            **kwargs: Query filters (including folder_id)
        """
        folder = FileAdapter._verify_folder_ownership(kwargs.get('folder_id'), user_token)

        file_dict = FileDB.get(**kwargs)
            
        return [FileConfigData(
            id=file.id,
            folder_id=file.folder_id,
            file_name=file.file_name,
            file_path=file.file_path,
            file_size=file.file_size,
            mime_type=file.mime_type,
            description=file.description,
            tags=file.tags,
            upload_time=file.upload_time,
            updated_time=file.updated_time
        ) for file in file_dict]

    @staticmethod
    async def download_file(
        folder_id: int,
        file_id: UUID,
        user_token: str,
    ) -> Optional[list[FileDownloadData]]:
        """Download a file.

        Args:
            folder_id: Folder ID
            file_id: File ID
            user_token: User token

        Returns:
            A record containing the file content and metadata.
        """
        try:
            folder = FileAdapter._verify_folder_ownership(folder_id, user_token)

            file_record = FileDB.get(folder_id=folder_id, id=file_id)

            if not file_record:
                raise NoResultFound(f"File {file_id} not found in folder {folder_id}")

            file_content = FileStorage.read_file(file_record[0].file_path)
            
            return [FileDownloadData(
                id=file_record[0].id,
                folder_id=file_record[0].folder_id,
                file_name=file_record[0].file_name,
                file_content=file_content,
                file_size=file_record[0].file_size,
                mime_type=file_record[0].mime_type,
                description=file_record[0].description,
                tags=file_record[0].tags,
                upload_time=file_record[0].upload_time,
                updated_time=file_record[0].updated_time
            )]
                    
        except Exception as e:
            logger.error(f"File download failed: {str(e)}")
            raise

    @staticmethod
    async def update_file_metadata(
        folder_id: int,
        file_id: UUID,
        user_token: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> list[FileConfigData]:
        """Update file metadata.

        Args:
            folder_id: Folder ID
            file_id: File ID
            user_token: User token
            description: New file description
            tags: New list of file tags

        Returns:
            List of updated file configuration data.
        """
        folder = FileAdapter._verify_folder_ownership(folder_id, user_token)

        file_list = FileDB.get(folder_id=folder.id, id=file_id)
        if not file_list:
            raise NoResultFound(f"File id {file_id} does not exist in folder {folder_id}")

        update_data = {}
        if description is not None:
            update_data['description'] = description
        if tags is not None:
            update_data['tags'] = tags
        updated_file = FileDB.update(file_id, **update_data)
        logger.info(f"Successfully updated file metadata: file_id={updated_file.id}")
        return [FileConfigData(
            id=updated_file.id,
            folder_id=updated_file.folder_id,
            file_name=updated_file.file_name,
            file_path=updated_file.file_path,
            file_size=updated_file.file_size,
            mime_type=updated_file.mime_type,
            description=updated_file.description,
            tags=updated_file.tags,
            upload_time=updated_file.upload_time,
            updated_time=updated_file.updated_time
        )]
        
    @staticmethod
    async def delete_file(
        file_ids: Optional[Union[UUID, List[UUID]]],
        folder_id: int,
        user_token: str,
        delete_all: bool = False
    ) -> Union[bool, Dict[str, Any]]:
        """Delete files (single or batch).

        Args:
            file_ids: A single file ID or a list of file IDs
            folder_id: Folder ID
            user_token: User token
            delete_all: Whether to delete all files in the folder (ignores file_ids when True)

        Returns:
            Single deletion: bool
            Batch deletion: a dict with deletion result statistics
        """
        from src.adapter.rag import get_rag_adapter

        folder = FileAdapter._verify_folder_ownership(folder_id, user_token)
        is_batch = isinstance(file_ids, list) or delete_all

        if delete_all:
            with DBSession() as session:
                files_to_delete = session.query(File).filter(
                    File.folder_id == folder.id
                ).all()
            files_to_delete_ids = [f.id for f in files_to_delete]
        elif isinstance(file_ids, list):
            files_to_delete_ids = file_ids
        else:
            files_to_delete_ids = [file_ids]

        try:
            rag_adapter = get_rag_adapter()
        except Exception as e:
            logger.warning(f"RAG adapter initialization failed, skipping index cleanup: {e}")
            rag_adapter = None
        
        deleted_files = []
        failed_deletions = []
        total_size_freed = 0
        
        for file_id in files_to_delete_ids:
            try:
                file_record = FileDB.get(folder_id=folder_id, id=file_id)

                # Pass user_token so the index delete goes through ACL rather
                # than bypassing permission checks.
                if rag_adapter:
                    try:
                        await rag_adapter.delete_document_index(file_id=file_id, token=user_token)
                        logger.info(f"Deleted RAG index for file {file_id}")
                    except Exception as idx_error:
                        # Index cleanup failure must not fail the whole deletion.
                        logger.warning(f"Failed to delete RAG index for file {file_id}: {idx_error}")

                FileStorage.delete_file(file_record[0].file_path)

                FileDB.delete(folder_id=folder_id, id=file_id)

                # Best-effort: abort any in-flight indexing of this file so it
                # stops at the next checkpoint instead of wasting context-gen /
                # embedding work on a file that no longer exists.
                try:
                    from src.domain.rag.index_job_manager import IndexingJobManager
                    aborted = await IndexingJobManager.get_instance().abort_indexing_for_file(
                        str(file_id), folder_id=folder_id,
                    )
                    if aborted:
                        logger.info(
                            f"In-flight indexing abort requested for deleted "
                            f"file {file_id}: {aborted}"
                        )
                except Exception as abort_err:
                    logger.warning(
                        f"Indexing-abort notify failed for {file_id} "
                        f"(existence gates will still catch it): {abort_err}"
                    )
                
                if is_batch:
                    deleted_files.append({
                        "id": file_record[0].id,
                        "file_name": file_record[0].file_name,
                        "file_size": file_record[0].file_size
                    })
                
                total_size_freed += file_record[0].file_size
                
            except Exception as e:
                if is_batch:
                    failed_deletions.append({
                        "id": file_id,
                        "error": str(e)
                    })
                else:
                    raise

        if total_size_freed > 0:
            FolderDB.update(folder.id,
                        file_count=max(0, folder.file_count - len(deleted_files if is_batch else [file_ids])),
                        total_size=max(0, folder.total_size - total_size_freed))

        if is_batch:
            return {
                "message": f"Batch deletion completed: {len(deleted_files)} succeeded, {len(failed_deletions)} failed",
                "summary": {
                    "total_requested": len(files_to_delete_ids),
                    "successfully_deleted": len(deleted_files),
                    "failed_deletions": len(failed_deletions),
                    "total_size_freed": total_size_freed
                },
                "deleted_files": deleted_files,
                "failed_deletions": failed_deletions
            }
        else:
            return True
        