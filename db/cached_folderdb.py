
"""Cached folder database operations

This module wraps FolderDB with caching to improve performance
for frequently accessed folders.
"""

from typing import List, Optional
from .folderdb import FolderDB
from .db import Folder
from src.infrastructure.cache.cache_service import CacheService, CacheKeys, CacheTTL
from src.log import get_db_logger

log = get_db_logger()


class CachedFolderDB:
    """Folder database operations with caching

    This class wraps FolderDB methods with cache-aside pattern:
    1. Check cache first
    2. If miss, query database
    3. Store result in cache
    4. Return result

    Cache invalidation happens on updates and deletes.
    """

    _cache = CacheService.get_instance()

    @classmethod
    def get_by_name_and_token(cls, name: str, user_token: str) -> Optional[Folder]:
        """Get folder by name and user token (with caching)

        Args:
            name: Folder name
            user_token: User token

        Returns:
            Folder instance or None if not found

        Performance:
            - Cache hit: < 1ms
            - Cache miss: 5-10ms (database query)
        """
        # Check cache first
        cache_key = CacheKeys.folder_by_name(user_token, name)
        cached_folder = cls._cache.get(cache_key)

        if cached_folder is not None:
            log.debug(f"Cache HIT: {cache_key}")
            return cached_folder

        # Cache miss - query database
        log.debug(f"Cache MISS: {cache_key}")
        folders = FolderDB.get(name=name, user_token=user_token)

        if not folders:
            # Cache negative result (prevents repeated DB queries for non-existent folders)
            cls._cache.set(cache_key, None, ttl=60)  # Short TTL for negative results
            return None

        folder = folders[0]

        # Cache the result
        cls._cache.set(cache_key, folder, ttl=CacheTTL.FOLDER)

        # Also cache by ID for faster lookups
        id_cache_key = CacheKeys.folder_by_id(folder.id)
        cls._cache.set(id_cache_key, folder, ttl=CacheTTL.FOLDER)

        return folder

    @classmethod
    def get_by_id(cls, folder_id: int) -> Optional[Folder]:
        """Get folder by ID (with caching)

        Args:
            folder_id: Folder ID

        Returns:
            Folder instance or None if not found
        """
        # Check cache
        cache_key = CacheKeys.folder_by_id(folder_id)
        cached_folder = cls._cache.get(cache_key)

        if cached_folder is not None:
            log.debug(f"Cache HIT: {cache_key}")
            return cached_folder

        # Cache miss
        log.debug(f"Cache MISS: {cache_key}")
        folders = FolderDB.get(id=folder_id)

        if not folders:
            cls._cache.set(cache_key, None, ttl=60)
            return None

        folder = folders[0]
        cls._cache.set(cache_key, folder, ttl=CacheTTL.FOLDER)

        return folder

    @classmethod
    def get_by_user_token(cls, user_token: str, use_cache: bool = True) -> List[Folder]:
        """Get all folders for a user (optionally cached)

        Args:
            user_token: User token
            use_cache: Whether to use cache (default True)

        Returns:
            List of Folder instances

        Note: List queries are cached for shorter TTL since they change frequently
        """
        if not use_cache:
            return FolderDB.get(user_token=user_token)

        cache_key = f"folders:user:{user_token}"
        cached_folders = cls._cache.get(cache_key)

        if cached_folders is not None:
            log.debug(f"Cache HIT: {cache_key}")
            return cached_folders

        log.debug(f"Cache MISS: {cache_key}")
        folders = FolderDB.get(user_token=user_token)

        # Cache folder list with shorter TTL
        cls._cache.set(cache_key, folders, ttl=180)  # 3 minutes

        return folders

    @classmethod
    def create(cls, **data) -> Folder:
        """Create a new folder and invalidate relevant caches

        Args:
            **data: Folder data (name, description, user_token, etc.)

        Returns:
            Created Folder instance
        """
        folder = FolderDB.create(**data)

        # Prime caches for the newly created folder
        if folder:
            cls._cache.set(
                CacheKeys.folder_by_id(folder.id),
                folder,
                ttl=CacheTTL.FOLDER
            )
            if folder.user_token:
                cls._cache.set(
                    CacheKeys.folder_by_name(folder.user_token, folder.name),
                    folder,
                    ttl=CacheTTL.FOLDER
                )
                user_cache_key = f"folders:user:{folder.user_token}"
                cls._cache.delete(user_cache_key)

        return folder

    @classmethod
    def update(cls, folder_id: int, **data) -> Optional[Folder]:
        """Update folder and invalidate caches

        Args:
            folder_id: Folder ID to update
            **data: Fields to update

        Returns:
            Updated Folder instance or None if not found
        """
        # Invalidate BEFORE DB write to prevent stale reads during update
        old_folder = cls.get_by_id(folder_id)
        if old_folder:
            cls.invalidate_folder_caches(old_folder)

        folder = FolderDB.update(id=folder_id, **data)

        if not folder:
            return None

        # Invalidate again with new data (name/token may have changed)
        cls.invalidate_folder_caches(folder)

        return folder

    @classmethod
    def delete(cls, folder_id: int) -> bool:
        """Delete folder and invalidate caches

        Args:
            folder_id: Folder ID to delete

        Returns:
            True if deleted, False otherwise
        """
        # Get folder first so we have data for cache invalidation
        folder = cls.get_by_id(folder_id)

        if not folder:
            return False

        # Invalidate BEFORE delete to prevent stale reads
        cls.invalidate_folder_caches(folder)

        # Delete from database
        return FolderDB.delete(id=folder_id)

    @classmethod
    def invalidate_folder_caches(cls, folder: Folder):
        """Invalidate all caches related to a folder

        Args:
            folder: Folder instance

        This is called after updates or deletes to ensure cache consistency.
        """
        # Delete folder-specific caches
        id_key = CacheKeys.folder_by_id(folder.id)
        name_key = CacheKeys.folder_by_name(folder.user_token, folder.name)
        files_key = CacheKeys.files_in_folder(folder.id)
        user_list_key = f"folders:user:{folder.user_token}"

        cls._cache.delete(id_key)
        cls._cache.delete(name_key)
        cls._cache.delete(files_key)
        cls._cache.delete(user_list_key)

        # Also clear any query caches for this folder
        query_pattern = f"query:{folder.id}:*"
        cls._cache.clear_pattern(query_pattern)

        log.debug(f"Invalidated caches for folder {folder.id}")

    @classmethod
    def get_cache_stats(cls) -> dict:
        """Get cache statistics

        Returns:
            Dictionary with cache metrics
        """
        return cls._cache.get_stats()
