
"""FileIndex database operations for tracking file indexing in the RAG system."""

from datetime import datetime, timezone
from typing import Optional

from .db import FileIndex
from db.db import Session as DBSession
from .baseDB import BaseDB
from src.log import get_db_logger

log = get_db_logger()


class FileIndexDB(BaseDB):
    """CRUD operations for file index tracking."""

    @classmethod
    def get_orm_class(cls) -> type:
        return FileIndex

    @classmethod
    def get_log(cls):
        return log

    @classmethod
    def get_indexed_models_for_folder(cls, folder_id: int) -> list:
        """Return the distinct embedding model names used by this folder's indexed records.

        Guards against model swaps: mixing vectors from two models in one table
        silently degrades retrieval (matching dimensions don't even fail the insert),
        so callers compare before indexing. Returns an empty list on DB failure
        (fail-open, does not block indexing).
        """
        try:
            with DBSession() as session:
                rows = (
                    session.query(FileIndex.embedding_model)
                    .filter(FileIndex.folder_id == folder_id)
                    .filter(FileIndex.status == "indexed")
                    .filter(FileIndex.embedding_model.isnot(None))
                    .distinct()
                    .all()
                )
                return [r[0] for r in rows]
        except Exception as e:
            cls.get_log().warning(
                f"get_indexed_models_for_folder({folder_id}) failed: {e}"
            )
            return []

    @classmethod
    def get_by_file(cls, file_id):
        """Return the index record for a file, or None."""
        try:
            result = cls.get(file_id=file_id)
            return result[0] if result else None
        except Exception as e:
            cls.get_log().error(f"Failed to get FileIndex by file_id {file_id}: {e}")
            raise

    @classmethod
    def get_by_folder(cls, folder_id: int):
        """Return all index records in a folder."""
        try:
            return cls.get(folder_id=folder_id)
        except Exception as e:
            cls.get_log().error(f"Failed to get FileIndex by folder_id {folder_id}: {e}")
            raise

    @classmethod
    def ensure_content_hash_columns(cls) -> None:
        """Ensure the content_hash column exists on Files / FileIndices (idempotent ALTER).

        A runtime self-healing fallback for tables created before this column
        existed; the canonical schema is Alembic. Normally a no-op, so actually
        adding the column means the migration was not applied and it warns about
        the schema drift.
        """
        try:
            from db.db import get_engine
            from sqlalchemy import text
            eng = get_engine()
            with eng.connect() as conn:
                missing = conn.execute(text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name IN ('Files', 'FileIndices') AND column_name = 'content_hash'"
                )).scalar()
                # Both tables present -> 2; either missing -> < 2
                if missing is not None and missing < 2:
                    cls.get_log().warning(
                        "content_hash column missing, added by runtime ensure — the "
                        "alembic migration may not have been applied (schema drift). "
                        "Confirm startup auto_migrate ran successfully."
                    )
                conn.execute(text(
                    'ALTER TABLE "Files" ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)'
                ))
                conn.execute(text(
                    'ALTER TABLE "FileIndices" ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)'
                ))
                conn.commit()
        except Exception as e:
            cls.get_log().warning(f"content_hash columns migration skipped: {e}")

    @classmethod
    def upsert_indexed(
        cls,
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
        """Idempotent version of `create(... status="indexed")`, for reindex.

        Updates a row in place when one exists (flip status to 'indexed', clear
        error_message, refresh indexed_at), else creates it. Avoids a
        UniqueViolation from calling create() on reindex. BaseDB.update is not
        reused because it skips None values and so cannot clear error_message.
        DB-layer exceptions are not swallowed, so the caller can decide.
        """
        try:
            with DBSession() as session:
                existing = (
                    session.query(cls.get_orm_class())
                    .filter_by(file_id=file_id)
                    .first()
                )
                if existing:
                    existing.folder_id = folder_id
                    existing.index_id = index_id
                    existing.vector_store_table = vector_store_table
                    existing.chunk_size = chunk_size
                    existing.chunk_overlap = chunk_overlap
                    existing.embedding_model = embedding_model
                    existing.num_chunks = num_chunks
                    existing.status = "indexed"
                    existing.error_message = None
                    existing.content_hash = content_hash
                    existing.indexed_at = datetime.now(timezone.utc)
                    session.commit()
                    session.refresh(existing)
                    session.expunge(existing)
                    cls.get_log().info(
                        f"Upserted (update) FileIndex for file_id={file_id} "
                        f"status=indexed (cleared prior error_message)"
                    )
                    return existing
            # Insert path (separate session for a clean state)
            return cls.create(
                file_id=file_id,
                folder_id=folder_id,
                index_id=index_id,
                vector_store_table=vector_store_table,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                num_chunks=num_chunks,
                status="indexed",
                content_hash=content_hash,
            )
        except Exception as e:
            cls.get_log().error(f"upsert_indexed failed for file_id={file_id}: {e}")
            raise

    @classmethod
    def mark_failed(
        cls,
        file_id,
        error_message: str,
        folder_id: Optional[int] = None,
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Mark a file's indexing as failed.

        Three paths: row exists -> update to status='failed'; no row but
        folder_id given -> create a failed row (so /indexed-files can see it);
        no row and no folder_id -> warn and return None (legacy behavior).
        """
        try:
            index_record = cls.get_by_file(file_id)
            if not index_record:
                if folder_id is None:
                    cls.get_log().warning(f"No FileIndex found for file_id {file_id}")
                    return None

                # Create a failed row so the failure is visible downstream.
                cls.get_log().info(
                    f"Creating failed FileIndex record for file_id {file_id} "
                    f"(folder_id={folder_id}) — no prior record existed"
                )
                create_kwargs = {
                    "file_id": file_id,
                    "folder_id": folder_id,
                    "index_id": f"file_{file_id}",
                    "vector_store_table": "",
                    "num_chunks": 0,
                    "status": "failed",
                    "error_message": error_message,
                }
                # Only set fields the caller supplied (backward compat).
                if embedding_model is not None:
                    create_kwargs["embedding_model"] = embedding_model
                if chunk_size is not None:
                    create_kwargs["chunk_size"] = chunk_size
                if chunk_overlap is not None:
                    create_kwargs["chunk_overlap"] = chunk_overlap
                return cls.create(**create_kwargs)

            # Only overwrite fields the caller explicitly passed; keep the rest.
            update_kwargs = {"status": "failed", "error_message": error_message}
            if embedding_model is not None:
                update_kwargs["embedding_model"] = embedding_model
            if chunk_size is not None:
                update_kwargs["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                update_kwargs["chunk_overlap"] = chunk_overlap
            return cls.update(index_record.id, **update_kwargs)
        except Exception as e:
            cls.get_log().error(f"Failed to mark FileIndex as failed for file_id {file_id}: {e}")
            raise

    @classmethod
    def delete_by_file(cls, file_id):
        """Delete a file's index record; return True if one was deleted."""
        try:
            result = cls.delete(file_id=file_id)
            if result:
                cls.get_log().info(f"Deleted FileIndex for file_id {file_id}")
            return result
        except Exception as e:
            cls.get_log().error(f"Failed to delete FileIndex by file_id {file_id}: {e}")
            raise

    @classmethod
    def delete_by_folder(cls, folder_id: int) -> int:
        """Delete all index records for a folder; return the count deleted."""
        try:
            with DBSession() as session:
                deleted_count = session.query(cls.get_orm_class()).filter_by(folder_id=folder_id).delete(synchronize_session=False)
                session.commit()
                cls.get_log().info(f"Deleted {deleted_count} FileIndex records for folder_id {folder_id}")
                return deleted_count
        except Exception as e:
            cls.get_log().error(f"Failed to delete FileIndex records by folder_id {folder_id}: {e}")
            raise
