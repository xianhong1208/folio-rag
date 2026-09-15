"""In-memory caching service for query results and frequently accessed data.

Simple TTL-based in-memory cache with LRU eviction and hit/miss statistics.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, List
import time
import fnmatch
from threading import Lock


class ICacheService(ABC):
    """Interface for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL in seconds."""
        pass

    @abstractmethod
    def delete(self, key: str):
        """Delete key from cache."""
        pass

    @abstractmethod
    def clear_pattern(self, pattern: str):
        """Delete all keys matching pattern (supports * wildcards)."""
        pass

    @abstractmethod
    def clear_all(self):
        """Clear entire cache."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get cache statistics."""
        pass


class InMemoryCache(ICacheService):
    """Simple in-memory cache with TTL support.

    Uses a dict with manual TTL checking. For larger or distributed workloads,
    consider cachetools.TTLCache or Redis.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self._cache: dict[str, tuple[Any, float, float]] = {}  # key -> (value, expiry_time, last_access_time)
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = Lock()

        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value, or None if missing or expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expiry_time, _ = self._cache[key]

            if time.time() > expiry_time:
                del self._cache[key]
                self._misses += 1
                return None

            # Update last access time for LRU
            self._cache[key] = (value, expiry_time, time.time())
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL in seconds (None uses the default)."""
        if ttl is None:
            ttl = self._default_ttl

        expiry_time = time.time() + ttl

        with self._lock:
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_oldest()

            self._cache[key] = (value, expiry_time, time.time())
            self._sets += 1

    def delete(self, key: str):
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._deletes += 1

    def clear_pattern(self, pattern: str):
        """Delete all keys matching the fnmatch pattern (e.g. ``"folder:*"``)."""
        with self._lock:
            keys_to_delete = [
                key for key in self._cache.keys()
                if fnmatch.fnmatch(key, pattern)
            ]

            for key in keys_to_delete:
                del self._cache[key]
                self._deletes += 1

    def clear_all(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()

    def _evict_oldest(self):
        """Evict the least recently used item (true LRU)."""
        if not self._cache:
            return

        # Key with the earliest last_access_time (tuple index 2)
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][2])
        del self._cache[oldest_key]

    def cleanup_expired(self):
        """Manually remove expired entries to free memory.

        Expired entries are also removed lazily during get().
        """
        current_time = time.time()

        with self._lock:
            expired_keys = [
                key for key, (value, expiry_time, _) in self._cache.items()
                if current_time > expiry_time
            ]

            for key in expired_keys:
                del self._cache[key]

    def get_stats(self) -> dict:
        """Return cache metrics (size, hits, misses, sets, deletes, hit_rate)."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'sets': self._sets,
                'deletes': self._deletes,
                'hit_rate': round(hit_rate, 2)
            }


_cache_instance: Optional[InMemoryCache] = None
_instance_lock = Lock()


class CacheService:
    """Factory for accessing the global cache instance."""

    @classmethod
    def get_instance(cls, max_size: int = 1000, default_ttl: int = 300) -> InMemoryCache:
        """Get or create the global cache instance (max_size/default_ttl apply on first call only)."""
        global _cache_instance

        if _cache_instance is None:
            with _instance_lock:
                if _cache_instance is None:
                    _cache_instance = InMemoryCache(max_size, default_ttl)

        return _cache_instance

    @classmethod
    def reset_instance(cls):
        """Reset the global cache instance (useful for testing)."""
        global _cache_instance
        with _instance_lock:
            _cache_instance = None


class CacheKeys:
    """Standard cache key formats for different entity types."""

    @staticmethod
    def folder_by_name(user_token: str, folder_name: str) -> str:
        return f"folder:{user_token}:{folder_name}"

    @staticmethod
    def folder_by_id(folder_id: int) -> str:
        return f"folder:id:{folder_id}"

    @staticmethod
    def files_in_folder(folder_id: int) -> str:
        return f"files:folder:{folder_id}"

    @staticmethod
    def file_by_id(file_id: str) -> str:
        return f"file:id:{file_id}"

    @staticmethod
    def rag_query(folder_id: int, query_hash: str) -> str:
        return f"query:{folder_id}:{query_hash}"

    @staticmethod
    def auth_token(token: str) -> str:
        return f"auth:{token}"

    @staticmethod
    def index_status(file_id: str) -> str:
        return f"index:file:{file_id}"


class CacheTTL:
    """Recommended TTL values in seconds."""
    FOLDER = 300
    FILE_LIST = 180
    FILE_DETAIL = 300
    QUERY_RESULT = 300
    AUTH_TOKEN = 3600
    INDEX_STATUS = 600
