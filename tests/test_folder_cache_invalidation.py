"""A deleted or recreated folder never answers from the folder caches (issue #57).

``FolderAdapter`` creates and deletes folders with ``FolderDB`` directly, so it must drop
``CachedFolderDB``'s entries itself: the owner + name lookup, the id lookup, the owner's
list and the folder's cached query results. Otherwise a search by name keeps returning
the deleted folder's chunks, also from a new folder created with the same name, for up
to the cache TTL. The cache here is the real in-memory one; the database is mocked.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from db.cached_folderdb import CachedFolderDB
from src.adapter.folder import FolderAdapter
from src.infrastructure.cache.cache_service import CacheKeys

_F = "src.adapter.folder"
OWNER, OTHER = "sub:alice", "sub:bob"


def _folder(id_, owner=OWNER, name="secret"):
    return SimpleNamespace(id=id_, name=name, user_token=owner, vector_table_uuid=f"u{id_}", description=None,
                           file_count=0, total_size=0, created_at=datetime(2026, 11, 17),
                           updated_at=datetime(2026, 11, 17))


@pytest.fixture(autouse=True)
def cache():
    c = CachedFolderDB._cache
    c.clear_all()
    yield c
    c.clear_all()


def _warm(cache, folder):
    """What a search by name leaves behind: the lookups and a cached result."""
    cache.set(CacheKeys.folder_by_name(folder.user_token, folder.name), folder, ttl=300)
    cache.set(CacheKeys.folder_by_id(folder.id), folder, ttl=300)
    cache.set(f"folders:user:{folder.user_token}", [folder], ttl=300)
    cache.set(CacheKeys.rag_query(folder.id, "q1"), {"results": ["the deleted chunk"]}, ttl=300)


async def _delete(folder):
    with patch(f"{_F}.FolderDB") as db, patch(f"{_F}.FileDB") as filedb, \
         patch(f"{_F}.FileStorage"), patch(f"{_F}.VectorStoreManager"), patch(f"{_F}.notify_folder_deleted"):
        db.get.return_value = [folder]
        filedb.get.return_value = []
        assert await FolderAdapter.delete_folder(folder.id, user_token=folder.user_token) is True


async def test_dfc01_delete_drops_every_cached_lookup_and_result(cache):
    """DFC-01"""
    old = _folder(1)
    _warm(cache, old)
    await _delete(old)
    assert cache.get(CacheKeys.folder_by_name(OWNER, "secret")) is None
    assert cache.get(CacheKeys.folder_by_id(1)) is None
    assert cache.get(f"folders:user:{OWNER}") is None
    assert cache.get(CacheKeys.rag_query(1, "q1")) is None
    with patch("db.cached_folderdb.FolderDB") as db:
        db.get.return_value = []
        assert CachedFolderDB.get_by_name_and_token("secret", OWNER) is None  # from the DB: gone


async def test_dfc02_another_persons_folder_of_the_same_name_is_untouched(cache):
    """DFC-02"""
    mine, theirs = _folder(1), _folder(2, owner=OTHER)
    _warm(cache, mine)
    _warm(cache, theirs)
    await _delete(mine)
    assert cache.get(CacheKeys.folder_by_name(OTHER, "secret")) is theirs
    assert cache.get(CacheKeys.rag_query(2, "q1")) == {"results": ["the deleted chunk"]}


def test_dfc03_a_new_folder_replaces_a_stale_entry_of_the_same_name(cache):
    """DFC-03 -- a stale owner + name entry (a deleted folder's) must not survive the create."""
    _warm(cache, _folder(1))
    new = _folder(3)
    with patch(f"{_F}.FolderDB") as db, patch(f"{_F}.FileStorage"), patch(f"{_F}.notify_folder_created"):
        db.get.return_value = []
        db.create.return_value = new
        FolderAdapter.create_folder(name="secret", user_token=OWNER)
    with patch("db.cached_folderdb.FolderDB") as db:
        db.get.return_value = [new]
        assert CachedFolderDB.get_by_name_and_token("secret", OWNER).id == 3
    assert cache.get(f"folders:user:{OWNER}") is None


async def test_dfc04_a_search_racing_the_delete_is_dropped_too(cache):
    """DFC-04 -- a search that re-caches the folder while its row is being deleted (it still
    exists for that moment) is dropped by the invalidation that follows, once the row is gone."""
    old = _folder(1)
    with patch(f"{_F}.FolderDB") as db, patch(f"{_F}.FileDB") as filedb, \
         patch(f"{_F}.FileStorage"), patch(f"{_F}.VectorStoreManager"), patch(f"{_F}.notify_folder_deleted"):
        db.get.return_value = [old]
        filedb.get.return_value = []
        db.delete.side_effect = lambda id: _warm(cache, old)
        assert await FolderAdapter.delete_folder(1, user_token=OWNER) is True
    assert cache.get(CacheKeys.folder_by_name(OWNER, "secret")) is None
    assert cache.get(CacheKeys.rag_query(1, "q1")) is None
