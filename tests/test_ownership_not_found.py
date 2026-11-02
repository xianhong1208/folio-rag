"""A token that does not own a folder / file gets the same 404 as for one that
does not exist.

A distinct 403 told any authenticated caller which folder and file ids exist
(they are sequential integers / stable UUIDs), so ownership failures are now
reported exactly like a miss: same status, same body.
"""
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import auth
from src.domain.exceptions import FolderNotFoundError, RAGFileNotFoundError
from src.domain.rag.index_service import IndexingService
from src.middleware.error_handler import register_error_handlers

OWNER, OTHER = "owner-jti", "other-jti"
_FOLDER = SimpleNamespace(id=5, name="docs", user_token=OWNER)
_FILE_ID = uuid.uuid4()
_FILE = SimpleNamespace(id=_FILE_ID, folder_id=5)


def _folder_lookup(folder_id):
    return _FOLDER if folder_id == 5 else None


@pytest.fixture()
def db():
    with patch.object(auth.CachedFolderDB, "get_by_id", side_effect=_folder_lookup), \
         patch.object(auth.FileDB, "get",
                      side_effect=lambda id: [_FILE] if id == _FILE_ID else []):
        yield


class TestDependencies:
    async def test_owner_gets_folder(self, db):
        assert await auth.get_folder_by_id(5, token=OWNER) is _FOLDER

    async def test_non_owner_folder_is_not_found(self, db):
        with pytest.raises(FolderNotFoundError):
            await auth.get_folder_by_id(5, token=OTHER)

    async def test_ownerless_folder_is_not_found(self, db):
        orphan = SimpleNamespace(id=5, name="docs", user_token=None)
        with patch.object(auth.CachedFolderDB, "get_by_id", return_value=orphan):
            with pytest.raises(FolderNotFoundError):
                await auth.get_folder_by_id(5, token=OTHER)

    async def test_owner_gets_file(self, db):
        assert await auth.get_file_by_id(_FILE_ID, token=OWNER) is _FILE

    async def test_non_owner_file_is_not_found(self, db):
        with pytest.raises(RAGFileNotFoundError):
            await auth.get_file_by_id(_FILE_ID, token=OTHER)


class TestIndexService:
    def test_non_owner_folder_is_not_found(self):
        svc = IndexingService(cache_service=object())
        with patch("src.domain.rag.index_service.CachedFolderDB.get_by_id",
                   return_value=_FOLDER):
            assert svc.get_folder(5, OWNER) is _FOLDER
            with pytest.raises(FolderNotFoundError):
                svc.get_folder(5, OTHER)


@pytest.fixture()
def client(db):
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/folders/{folder_id}")
    async def _folder(folder=Depends(auth.get_folder_by_id)):
        return {"id": folder.id}

    @app.get("/files/{file_id}")
    async def _file(file=Depends(auth.get_file_by_id)):
        return {"id": str(file.id)}

    app.dependency_overrides[auth.extract_token] = lambda: OTHER
    return TestClient(app)


def _strip_volatile(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in ("timestamp", "path")}


def test_http_non_owner_folder_matches_missing_folder(client):
    """Same status and body as a folder id that does not exist (modulo the id)."""
    not_owned = client.get("/folders/5")
    assert not_owned.status_code == 404
    missing = client.get("/folders/6")
    assert missing.status_code == 404
    a, b = _strip_volatile(not_owned.json()), _strip_volatile(missing.json())
    assert a["error"] == b["error"]
    assert a["message"].replace("5", "N") == b["message"].replace("6", "N")


def test_http_non_owner_file_matches_missing_file(client):
    not_owned = client.get(f"/files/{_FILE_ID}")
    assert not_owned.status_code == 404
    other_id = uuid.uuid4()
    missing = client.get(f"/files/{other_id}")
    assert missing.status_code == 404
    a, b = _strip_volatile(not_owned.json()), _strip_volatile(missing.json())
    assert a["error"] == b["error"]
    assert a["message"].replace(str(_FILE_ID), "X") == b["message"].replace(str(other_id), "X")
