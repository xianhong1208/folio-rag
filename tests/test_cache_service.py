
"""Unit tests for src/infrastructure/cache/cache_service.py.

Pure logic; TTL tests use a monkeypatched fake clock instead of sleeping.
"""

import pytest

import src.infrastructure.cache.cache_service as cache_mod
from src.infrastructure.cache.cache_service import (
    CacheKeys,
    CacheService,
    CacheTTL,
    InMemoryCache,
)


class _FakeTime:
    """Stand-in time module: provides only time(), can be advanced manually, no sleep"""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


@pytest.fixture
def fake_time(monkeypatch) -> _FakeTime:
    """Replace the time module inside cache_service with a controllable fake clock"""
    ft = _FakeTime()
    monkeypatch.setattr(cache_mod, "time", ft)
    return ft


@pytest.fixture
def clean_singleton():
    """Reset the global singleton before and after, to avoid polluting other tests"""
    CacheService.reset_instance()
    yield
    CacheService.reset_instance()


# InMemoryCache basic behavior

def test_set_get_roundtrip():
    """get after set returns the original value (TC-cache-01)"""
    cache = InMemoryCache()
    cache.set("k1", {"a": 1})
    assert cache.get("k1") == {"a": 1}


def test_get_missing_key_returns_none():
    """get on a missing key returns None and records a miss (TC-cache-02)"""
    cache = InMemoryCache()
    assert cache.get("nope") is None
    assert cache.get_stats()["misses"] == 1


def test_set_overwrites_existing_key():
    """set on an existing key overwrites the old value (TC-cache-03)"""
    cache = InMemoryCache()
    cache.set("k", "old")
    cache.set("k", "new")
    assert cache.get("k") == "new"


# TTL expiry

def test_get_after_ttl_expired_returns_none(fake_time):
    """After TTL, get returns None and the entry is removed (TC-cache-04)"""
    cache = InMemoryCache()
    cache.set("k", "v", ttl=10)
    fake_time.advance(11)
    assert cache.get("k") is None
    assert cache.get_stats()["size"] == 0


def test_get_at_exact_expiry_still_hits(fake_time):
    """Exactly at the expiry time still counts as valid (expiry check is strictly greater-than) (TC-cache-05)"""
    cache = InMemoryCache()
    cache.set("k", "v", ttl=10)
    fake_time.advance(10)  # now == expiry_time
    assert cache.get("k") == "v"


def test_set_without_ttl_uses_default_ttl(fake_time):
    """When ttl is unspecified, the constructor's default_ttl is used (TC-cache-06)"""
    cache = InMemoryCache(default_ttl=50)
    cache.set("k", "v")  # ttl=None -> default 50
    fake_time.advance(49)
    assert cache.get("k") == "v"
    fake_time.advance(2)  # 51 total > 50
    assert cache.get("k") is None


# delete / clear_pattern / clear_all

def test_delete_existing_key():
    """After deleting an existing key, get returns None and the deletes counter +1 (TC-cache-07)"""
    cache = InMemoryCache()
    cache.set("k", "v")
    cache.delete("k")
    assert cache.get("k") is None
    assert cache.get_stats()["deletes"] == 1


def test_delete_missing_key_is_noop():
    """delete on a missing key does not raise, deletes counter unchanged (TC-cache-08)"""
    cache = InMemoryCache()
    cache.delete("ghost")  # should not raise
    assert cache.get_stats()["deletes"] == 0


def test_clear_pattern_star_wildcard():
    """clear_pattern("folder:*") clears only keys with the folder: prefix (TC-cache-09)"""
    cache = InMemoryCache()
    cache.set("folder:t1:a", 1)
    cache.set("folder:t1:b", 2)
    cache.set("files:folder:9", 3)
    cache.clear_pattern("folder:*")
    assert cache.get("folder:t1:a") is None
    assert cache.get("folder:t1:b") is None
    assert cache.get("files:folder:9") == 3
    assert cache.get_stats()["deletes"] == 2


def test_clear_pattern_question_mark_single_char():
    """clear_pattern's ? matches only a single character (TC-cache-10)"""
    cache = InMemoryCache()
    cache.set("k1", 1)
    cache.set("k2", 2)
    cache.set("k10", 3)
    cache.clear_pattern("k?")
    assert cache.get("k1") is None
    assert cache.get("k2") is None
    assert cache.get("k10") == 3


def test_clear_all_empties_cache():
    """clear_all empties the entire cache (TC-cache-11)"""
    cache = InMemoryCache()
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear_all()
    assert cache.get_stats()["size"] == 0
    assert cache.get("a") is None


# LRU eviction

def test_lru_eviction_removes_least_recently_used(fake_time):
    """When the cache is full, adding a key evicts the least recently used entry (TC-cache-12)"""
    cache = InMemoryCache(max_size=2)
    cache.set("a", 1)
    fake_time.advance(1)
    cache.set("b", 2)
    fake_time.advance(1)
    cache.get("a")  # updates a's last_access -> b becomes the oldest
    fake_time.advance(1)
    cache.set("c", 3)  # triggers eviction of b
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_set_existing_key_when_full_does_not_evict(fake_time):
    """Overwriting an existing key when the cache is full does not trigger eviction (TC-cache-13)"""
    cache = InMemoryCache(max_size=2)
    cache.set("a", 1)
    fake_time.advance(1)
    cache.set("b", 2)
    fake_time.advance(1)
    cache.set("a", 99)  # key already exists -> no eviction
    assert cache.get("a") == 99
    assert cache.get("b") == 2


# cleanup_expired / get_stats

def test_cleanup_expired_removes_only_expired(fake_time):
    """cleanup_expired removes only expired entries, keeping live ones (TC-cache-14)"""
    cache = InMemoryCache()
    cache.set("short", 1, ttl=5)
    cache.set("long", 2, ttl=100)
    fake_time.advance(6)
    cache.cleanup_expired()
    assert cache.get_stats()["size"] == 1
    assert cache.get("long") == 2


def test_get_stats_counters_and_hit_rate():
    """get_stats correctly accumulates hits/misses/sets/deletes and hit_rate (TC-cache-15)"""
    cache = InMemoryCache(max_size=7)
    cache.get("miss")          # miss 1
    cache.set("k", "v")        # set 1
    cache.get("k")             # hit 1
    cache.delete("k")          # delete 1
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["sets"] == 1
    assert stats["deletes"] == 1
    assert stats["max_size"] == 7
    assert stats["hit_rate"] == 50.0


def test_get_stats_initial_hit_rate_zero():
    """With no requests, hit_rate is 0.0 rather than a division-by-zero error (TC-cache-16)"""
    stats = InMemoryCache().get_stats()
    assert stats["hit_rate"] == 0.0
    assert stats["size"] == 0


# CacheService singleton

def test_cache_service_returns_same_instance(clean_singleton):
    """Two get_instance calls return the same object (TC-cache-17)"""
    a = CacheService.get_instance()
    b = CacheService.get_instance()
    assert a is b


def test_reset_instance_creates_new_instance(clean_singleton):
    """After reset_instance, get_instance returns a brand-new instance (TC-cache-18)"""
    a = CacheService.get_instance()
    a.set("k", "v")
    CacheService.reset_instance()
    b = CacheService.get_instance()
    assert a is not b
    assert b.get("k") is None


# CacheKeys / CacheTTL

@pytest.mark.parametrize(
    "generated, expected",
    [
        (CacheKeys.folder_by_name("tok123", "月報"), "folder:tok123:月報"),
        (CacheKeys.folder_by_id(42), "folder:id:42"),
        (CacheKeys.files_in_folder(7), "files:folder:7"),
        (CacheKeys.file_by_id("uuid-1"), "file:id:uuid-1"),
        (CacheKeys.rag_query(3, "abcdef"), "query:3:abcdef"),
        (CacheKeys.auth_token("tok123"), "auth:tok123"),
        (CacheKeys.index_status("uuid-1"), "index:file:uuid-1"),
    ],
)
def test_cache_keys_formats(generated, expected):
    """Each CacheKeys staticmethod produces a fixed-format key (TC-cache-19)"""
    assert generated == expected


def test_cache_ttl_constants():
    """CacheTTL constant values match the design (seconds) (TC-cache-20)"""
    assert CacheTTL.FOLDER == 300
    assert CacheTTL.FILE_LIST == 180
    assert CacheTTL.FILE_DETAIL == 300
    assert CacheTTL.QUERY_RESULT == 300
    assert CacheTTL.AUTH_TOKEN == 3600
    assert CacheTTL.INDEX_STATUS == 600
