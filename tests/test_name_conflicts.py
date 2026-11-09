"""Folder names are unique per owner, and a folder holds one file per name.

A rename to a name the owner already used went through (create refused it): the
MCP tool registry, the storage directory and name-based queries are keyed on
(owner, name), so renaming either twin away removed the other's tool, and the
failed directory move still rewrote the files' stored paths into the other
folder's directory. A second upload of the same filename was stored next to the
first (files live under their own uuid), so search returned every passage twice.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from db.filedb import FileNameConflictError
from src.adapter.file import FileAdapter
from src.adapter.folder import FolderAdapter, FolderNameConflictError, FolderStorageMoveError


_NOW = datetime(2026, 11, 9)


def _folder(id_, name, token="tok"):
    return SimpleNamespace(id=id_, name=name, user_token=token, description=None,
                           file_count=0, total_size=0, created_at=_NOW, updated_at=_NOW)


class TestFolderNames:
    def test_create_refuses_a_name_the_owner_already_uses(self):
        with patch("src.adapter.folder.FolderDB") as db:
            db.get.return_value = [_folder(1, "docs")]
            with pytest.raises(FolderNameConflictError):
                FolderAdapter.create_folder(name="docs", user_token="tok")
            db.create.assert_not_called()

    async def test_rename_to_a_taken_name_changes_nothing(self):
        def get(**kw):
            return [_folder(2, "old")] if "id" in kw else [_folder(1, "docs")]

        with patch("src.adapter.folder.FolderDB") as db, \
                patch.object(FolderAdapter, "_move_storage") as move, \
                patch("src.adapter.folder.notify_folder_renamed") as notify:
            db.get.side_effect = get
            with pytest.raises(FolderNameConflictError):
                await FolderAdapter.update_folder(folder_id=2, user_token="tok", name="docs")
            move.assert_not_called()
            db.update.assert_not_called()
            notify.assert_not_called()

    async def test_rename_to_a_free_name_moves_and_updates(self):
        def get(**kw):
            return [_folder(2, "old")] if "id" in kw else []

        with patch("src.adapter.folder.FolderDB") as db, \
                patch.object(FolderAdapter, "_move_storage") as move, \
                patch("src.adapter.folder.notify_folder_renamed"):
            db.get.side_effect = get
            db.update.return_value = _folder(2, "new")
            await FolderAdapter.update_folder(folder_id=2, user_token="tok", name="new")
            move.assert_called_once()
            db.update.assert_called_once()

    async def test_keeping_the_same_name_is_not_a_conflict(self):
        with patch("src.adapter.folder.FolderDB") as db, \
                patch.object(FolderAdapter, "_move_storage") as move, \
                patch("src.adapter.folder.notify_folder_renamed"):
            db.get.return_value = [_folder(2, "docs")]
            db.update.return_value = _folder(2, "docs")
            await FolderAdapter.update_folder(folder_id=2, user_token="tok", name="docs", description="d")
            move.assert_not_called()
            db.update.assert_called_once()


class TestStorageMove:
    def test_a_failed_move_rewrites_no_paths(self, tmp_path, monkeypatch):
        from src.storage.file_storage import FileStorage
        monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path)
        (tmp_path / "tok" / "old").mkdir(parents=True)
        (tmp_path / "tok" / "taken").mkdir()  # an orphan directory under the new name
        engine = MagicMock()
        with patch("src.adapter.folder.get_engine", return_value=engine):
            with pytest.raises(FolderStorageMoveError):
                FolderAdapter._move_storage(2, "tok", "tok", "old", "taken")
        engine.connect.assert_not_called()
        assert (tmp_path / "tok" / "old").is_dir()

    def test_a_folder_without_a_directory_only_changes_its_name(self, tmp_path, monkeypatch):
        from src.storage.file_storage import FileStorage
        monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path)
        engine = MagicMock()
        with patch("src.adapter.folder.get_engine", return_value=engine):
            FolderAdapter._move_storage(2, "tok", "tok", "missing", "new")
        engine.connect.assert_called_once()

    def test_a_successful_move_rewrites_paths(self, tmp_path, monkeypatch):
        from src.storage.file_storage import FileStorage
        monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path)
        (tmp_path / "tok" / "old").mkdir(parents=True)
        engine = MagicMock()
        with patch("src.adapter.folder.get_engine", return_value=engine):
            FolderAdapter._move_storage(2, "tok", "tok", "old", "new")
        assert (tmp_path / "tok" / "new").is_dir()
        engine.connect.assert_called_once()


@pytest.fixture()
def folder_client():
    from src.api.dependencies.auth import extract_token
    from src.api.router import folder_api
    from src.auth.dependencies import authenticate_request
    app = FastAPI()
    app.include_router(folder_api.router)
    app.dependency_overrides[authenticate_request] = lambda: {}
    app.dependency_overrides[extract_token] = lambda: "tok"
    return TestClient(app)


class TestStatusCodes:
    def test_create_conflict_is_409(self, folder_client):
        with patch("src.adapter.folder.FolderAdapter.create_folder",
                   side_effect=FolderNameConflictError("A folder named 'docs' already exists.")):
            r = folder_client.post("/folders/", json={"name": "docs"})
        assert r.status_code == 409 and "already exists" in r.json()["detail"]

    @pytest.mark.parametrize("exc", [FolderNameConflictError("taken"), FolderStorageMoveError("disk")])
    def test_rename_conflicts_are_409(self, folder_client, exc):
        with patch("src.adapter.folder.FolderAdapter.update_folder", new=AsyncMock(side_effect=exc)):
            r = folder_client.patch("/folders/2", json={"name": "docs"})
        assert r.status_code == 409


class TestFileNames:
    def test_a_second_file_with_the_same_name_is_refused_before_it_is_written(self):
        folder = _folder(7, "docs")
        with patch.object(FileAdapter, "_name_taken", return_value=True), \
                patch("src.adapter.file.FileStorage") as storage, \
                patch("src.adapter.file.FileDB") as db:
            db._validate_file_content.return_value = {"file_size": 1, "mime_type": "text/plain"}
            with pytest.raises(FileNameConflictError, match="already in this folder"):
                FileAdapter._store_file(folder, b"x", "a.txt", None, None)
            storage.save_file.assert_not_called()
            db.create.assert_not_called()

    def test_upload_of_a_taken_name_is_409(self):
        from src.api.dependencies.auth import extract_token
        from src.api.router import file_api
        from src.auth.dependencies import authenticate_request
        app = FastAPI()
        app.include_router(file_api.router)
        app.dependency_overrides[authenticate_request] = lambda: {}
        app.dependency_overrides[extract_token] = lambda: "tok"
        with patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
                patch("src.api.router.file_api.FileAdapter.upload_file",
                      new=AsyncMock(side_effect=FileNameConflictError("A file named 'a.txt' is already in this folder."))):
            r = TestClient(app).post("/folders/7/file", files={"file": ("a.txt", b"x", "text/plain")})
        assert r.status_code == 409 and "already in this folder" in r.json()["detail"]

    async def test_batch_upload_reports_the_duplicate_and_keeps_going(self):
        folder = _folder(7, "docs")
        record = SimpleNamespace(id="f1", folder_id=7, file_name="b.txt", file_path="p", file_size=1,
                                 mime_type="text/plain", description=None, tags=None,
                                 upload_time=SimpleNamespace(isoformat=lambda: "t"), updated_time=None)

        def store(folder_, content, name, description, tags):
            if name == "a.txt":
                raise FileNameConflictError(f"A file named '{name}' is already in this folder.")
            return {"file_size": 1}, record

        with patch.object(FileAdapter, "_verify_folder_ownership", return_value=folder), \
                patch.object(FileAdapter, "_store_file", side_effect=store), \
                patch.object(FileAdapter, "_update_folder_stats"):
            out = await FileAdapter.upload_file(
                [{"file_content": b"x", "file_name": "a.txt"}, {"file_content": b"y", "file_name": "b.txt"}],
                folder_id=7, user_token="tok")
        assert out["summary"]["successful_count"] == 1 and out["summary"]["failed_count"] == 1
        assert "already in this folder" in str(out)
