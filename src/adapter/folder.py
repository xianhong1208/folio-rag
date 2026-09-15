
"""Folder adapter module.

CRUD operations for folders (create, query, update, delete), plus folder
metadata management and file statistics. Also handles dynamic registration
and removal of MCP tools.
"""

from typing import Optional

from db.folderdb import FolderDB
from db.filedb import FileDB
from db.db import get_engine
from sqlalchemy import text
from sqlalchemy.exc import NoResultFound
from src.log import get_adapter_logger, log_op, log_err, log_warn, mask_token
from src.adapter.model import FolderConfigData
from src.domain.rag.vector_store_manager import VectorStoreManager  # physical-table drop is centralized here
from src.storage.file_storage import FileStorage
# MCP tool (un)registration goes through the folder_events hooks (wired by app.py
# at startup), keeping the dependency direction one-way and avoiding an import cycle.
from src.adapter.folder_events import (
    notify_folder_created,
    notify_folder_deleted,
    notify_folder_renamed,
)


logger = get_adapter_logger()


class FolderAdapter:
    @staticmethod
    def create_folder(
        name: str,
        user_token: str,
        description: Optional[str] = None
    ) -> Optional[list[FolderConfigData]]:
        """Create a new folder.

        Args:
            name: Folder name
            user_token: User token
            description: Folder description

        Returns:
            List of folder configuration data for the created folder.
        """
        logger.info(f"Creating folder: {name} for user: {user_token[:8] if user_token else 'N/A'}...")
        try:
            # Fail fast: validate name/token against the same safe-name whitelist
            # used for path safety, before creating any DB row. Otherwise an
            # invalid name would leave a ghost row that cannot even be deleted.
            from src.storage.file_storage import _validate_safe_name
            _validate_safe_name(user_token, "user_token")
            _validate_safe_name(name, "folder_name")

            folder = FolderDB.get(name=name, user_token=user_token)
            if folder:
                raise ValueError(f"Folder with name '{name}' already exists for this user.")
            folder_dict = FolderDB.create(
                user_token=user_token,
                name=name,
                description=description)
            
            FileStorage._get_folder_storage_path(user_token, name)

            logger.info(f"Successfully created folder: {folder_dict.name}")

            # Register the dynamic MCP tool. The hooks fail safe internally:
            # a failure is logged only and never blocks the CRUD operation.
            notify_folder_created(name, description, user_token)
            
            return [FolderConfigData(
                id=folder_dict.id,
                user_token=str(folder_dict.user_token) if folder_dict.user_token else None,
                name=folder_dict.name,
                description=folder_dict.description,
                file_count=folder_dict.file_count,
                total_size=folder_dict.total_size,
                created_at=folder_dict.created_at,
                updated_at=folder_dict.updated_at,
            )]
        except Exception as e:
            logger.error(f"Failed to create folder {name}: {e}")
            raise

    @staticmethod
    async def get_folder(user_token: str, **kwargs) -> Optional[list[FolderConfigData]]:
        """Query folders.

        Args:
            user_token: User token (required)
            **kwargs: Additional query filters (e.g. folder_id, name)

        Returns:
            List of folder configuration data.
        """
        logger.info(f"Querying folder with parameters: {kwargs}")
        try:
            # Ensure every query is scoped by user_token
            kwargs['user_token'] = user_token
            folder_dicts = FolderDB.get(**kwargs)

            return [FolderConfigData(
                id=folder.id,
                user_token=str(folder.user_token) if folder.user_token else None,
                name=folder.name,
                description=folder.description,
                file_count=folder.file_count,
                total_size=folder.total_size,
                created_at=folder.created_at,
                updated_at=folder.updated_at,
            ) for folder in folder_dicts]
            
        except Exception as e:
            logger.error(f"Failed to query folder with filters: {e}")
            raise
        
    @staticmethod
    async def update_folder(
        folder_id: int,
        user_token: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[list[FolderConfigData]]:
        """Update a folder's name and description.

        Args:
            folder_id: Folder ID
            user_token: User token
            name: New folder name (optional)
            description: New folder description (optional)

        Returns:
            List of folder configuration data for the updated folder.
        """
        logger.info(f"Updating folder: folder_id: {folder_id}, name: {name}, description: {description}")
        try:
            folder_check = FolderDB.get(id=folder_id, user_token=user_token)
            if not folder_check:
                raise NoResultFound(f"Folder id {folder_id} does not exist")

            old_folder = folder_check[0]
            old_name = old_folder.name

            # On rename, keep storage directory and stored file paths in sync.
            if name is not None and name != old_name:
                if old_folder.user_token:
                    renamed = FileStorage.rename_folder(old_folder.user_token, old_name, name)
                    if renamed:
                        logger.info(f"Storage folder renamed: {old_name} -> {name}")
                    else:
                        logger.warning(f"Storage folder rename failed or not exists: {old_name}")

                try:
                    engine = get_engine()
                    with engine.connect() as conn:
                        conn.execute(
                            text('''
                                UPDATE "Files" 
                                SET file_path = REPLACE(file_path, :old_path, :new_path)
                                WHERE folder_id = :folder_id
                            '''),
                            {
                                "old_path": f"storage/{user_token}/{old_name}/",
                                "new_path": f"storage/{user_token}/{name}/",
                                "folder_id": folder_id
                            }
                        )
                        conn.commit()
                        logger.info(f"Updated file paths for folder {folder_id}: {old_name} -> {name}")
                except Exception as e:
                    logger.error(f"Failed to update file paths: {e}")
            
            update_kwargs = {"id": folder_id, "user_token": user_token}
            if name is not None:
                update_kwargs["name"] = name
            if description is not None:
                update_kwargs["description"] = description
            
            updated_folder = FolderDB.update(**update_kwargs)
            
            logger.info(f"Successfully updated folder: {folder_id}")

            # Re-register the MCP tool on name/description change. The hook always
            # attempts the register even if unregister fails, so the tool surface
            # converges to the new state.
            if notify_folder_renamed(
                old_name, updated_folder.name,
                updated_folder.description, updated_folder.user_token,
            ):
                logger.info(f"Successfully refreshed MCP tool for folder: {updated_folder.name}")

            return [FolderConfigData(
                id=updated_folder.id,
                user_token=str(updated_folder.user_token) if updated_folder.user_token else None,
                name=updated_folder.name,
                description=updated_folder.description,
                file_count=updated_folder.file_count,
                total_size=updated_folder.total_size,
                created_at=updated_folder.created_at,
                updated_at=updated_folder.updated_at,
            )]
        
        except Exception as e:
            logger.error(f"Failed to update folder description: {e}")
            raise 

    @staticmethod
    async def delete_folder(
        folder_id: int,
        user_token: Optional[str] = None
    ) -> bool:
        """Delete a folder.

        Deleting a folder automatically deletes all files within it.

        Args:
            folder_id: Folder ID

        Returns:
            True on successful deletion.
        """
        logger.info(f"Deleting folder: {folder_id}")

        try:
            folder_check = FolderDB.get(id=folder_id, user_token=user_token)
            if not folder_check:
                raise NoResultFound(f"Folder id {folder_id} does not exist")

            folder = folder_check[0]

            # Cancel this folder's in-flight indexing jobs first; otherwise the
            # DROP TABLE below would leave a job writing to a table that no
            # longer exists.
            try:
                from src.domain.rag.index_job_manager import IndexingJobManager
                job_manager = IndexingJobManager.get_instance()
                cancelled = await job_manager.cancel_jobs_for_folder(folder_id)
                if cancelled:
                    logger.info(
                        f"Cancelled {len(cancelled)} in-flight indexing job(s) "
                        f"before deleting folder {folder_id}"
                    )
            except Exception as e:
                logger.warning(f"Failed to cancel jobs for folder {folder_id}: {e}")

            try:
                files = FileDB.get(folder_id=folder_id)

                # This adapter is stateless (no VSM instance) and drops the whole
                # folder, so cache invalidation is moot — use the static
                # drop_table_by_name, which does not touch the cache.
                vector_table_name = VectorStoreManager.physical_table_name(
                    folder.id, folder.vector_table_uuid)
                VectorStoreManager.drop_table_by_name(vector_table_name)

                if files:
                    for file in files:
                        try:
                            from db.fileindexdb import FileIndexDB
                            FileIndexDB.delete_by_file(file.id)
                        except Exception as e:
                            logger.warning(f"Failed to delete FileIndex record for file {file.id}: {e}")

                        FileStorage.delete_file(file.file_path)

                    deleted_count = FileDB.delete_by_folder_id(folder_id)
                    logger.info(f"Successfully deleted {deleted_count} files from folder {folder_id}")
            except Exception as e:
                # Log a file-deletion failure but continue deleting the folder
                logger.error(f"Failed to delete files in folder {folder_id}: {e}")

            # Remove the dynamic MCP tool before deleting the DB record; the
            # hooks fail safe internally and never block deletion.
            notify_folder_deleted(folder.name, folder.user_token)

            # Best-effort storage cleanup: an invalid name (legacy ghost row) or a
            # missing directory must not block deletion of the DB record, otherwise
            # the folder could never be removed.
            try:
                FileStorage.delete_folder(folder.user_token, folder.name)
            except ValueError as e:
                logger.warning(
                    f"Storage cleanup skipped for folder {folder_id} "
                    f"(invalid name, likely ghost row): {e}")

            FolderDB.delete(id=folder_id)
            logger.info(f"Successfully deleted folder: {folder_id} ({folder.name})")

            # Finally, purge this folder's job records (in-memory + DB). Without
            # this, orphan jobs remain in the /index/jobs listing and the frontend
            # keeps polling a deleted folder, endlessly hitting FOLDER_NOT_FOUND.
            try:
                from src.domain.rag.index_job_manager import IndexingJobManager
                IndexingJobManager.get_instance().purge_jobs_for_folder(folder_id)
            except Exception as e:
                logger.warning(f"Failed to purge job records for folder {folder_id}: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete folder {folder_id}: {e}")
            raise
