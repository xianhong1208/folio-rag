
"""Optimized database queries with JOINs to avoid N+1 query problem

This module provides methods that use SQL JOINs to efficiently fetch
related data in a single query, significantly improving performance
for listing files with their index status.
"""

from typing import List, Dict, Optional, Any
from sqlalchemy import text
from sqlalchemy.orm import Session
from db.db import get_engine


class FileWithIndexDB:
    """Database operations for files with their index status (using JOINs)"""

    @classmethod
    def get_files_with_index_status(
        cls,
        folder_id: int,
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Get all files in a folder with their index status using a single JOIN query.

        This method solves the N+1 query problem by fetching files and their
        index status in one query instead of querying FileIndex for each file.

        Args:
            folder_id: The folder ID to query files from
            session: Optional SQLAlchemy session (creates new one if None)

        Returns:
            List of dictionaries containing file info and index status:
            [
                {
                    'id': UUID,
                    'folder_id': int,
                    'file_name': str,
                    'file_path': str,
                    'file_size': int,
                    'mime_type': str,
                    'description': str,
                    'tags': str,
                    'upload_time': datetime,
                    'updated_time': datetime,
                    'index_status': str | None,
                    'num_chunks': int | None,
                    'indexed_at': datetime | None,
                    'embedding_model': str | None,
                    'vector_store_table': str | None
                },
                ...
            ]

        Performance:
            - Before: O(n) queries (1 for files + n for each file's index)
            - After: O(1) single JOIN query
            - Expected speedup: 10-100x for large folders
        """
        query = text("""
            SELECT
                f.id,
                f.folder_id,
                f.file_name,
                f.file_path,
                f.file_size,
                f.mime_type,
                f.description,
                f.tags,
                f.upload_time,
                f.updated_time,
                fi.status as index_status,
                fi.num_chunks,
                fi.indexed_at,
                fi.embedding_model,
                fi.vector_store_table
            FROM "Files" f
            LEFT JOIN "FileIndices" fi ON f.id = fi.file_id
            WHERE f.folder_id = :folder_id
            ORDER BY f.upload_time DESC
        """)

        # Create session if not provided
        should_close = False
        if session is None:
            engine = get_engine()
            session = Session(bind=engine)
            should_close = True

        try:
            result = session.execute(query, {"folder_id": folder_id})
            rows = result.fetchall()

            files_with_index = []
            for row in rows:
                # Convert Row to dict
                file_dict = {
                    'id': str(row.id),  # Convert UUID to string
                    'folder_id': row.folder_id,
                    'file_name': row.file_name,
                    'file_path': row.file_path,
                    'file_size': row.file_size,
                    'mime_type': row.mime_type,
                    'description': row.description,
                    'tags': row.tags,
                    'upload_time': row.upload_time.isoformat() if row.upload_time else None,
                    'updated_time': row.updated_time.isoformat() if row.updated_time else None,
                    'index_status': row.index_status,
                    'num_chunks': row.num_chunks,
                    'indexed_at': row.indexed_at.isoformat() if row.indexed_at else None,
                    'embedding_model': row.embedding_model,
                    'vector_store_table': row.vector_store_table,
                    'is_indexed': row.index_status == 'indexed'
                }
                files_with_index.append(file_dict)

            return files_with_index

        finally:
            if should_close:
                session.close()
