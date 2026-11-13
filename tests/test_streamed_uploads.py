"""Uploads are copied to disk in chunks, with the size limit enforced while streaming.

Before 1.4.2 the upload endpoints called ``await file.read()``, so a file was held in
memory whole before the 100 MB check refused it, and an oversized request was read to
the end first. Now the spooled part is copied in 1 MiB pieces, the copy stops (and the
partial file is deleted) once it crosses the limit, and the single-file routes refuse a
request from its Content-Length, or as soon as its body crosses the limit.
"""

import hashlib
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from db.filedb import FileDB, FileTooLargeError, file_too_large_message
from src.adapter.file import FileAdapter
from src.middleware import upload_limit
from src.middleware.upload_limit import UploadSizeLimit
from src.storage.file_storage import FileStorage


@pytest.fixture
def storage_root(tmp_path, monkeypatch):
    monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path / "storage")
    monkeypatch.setattr(FileStorage, "STORAGE_FILE_ROOT", tmp_path)
    return tmp_path


class _Recorder(io.BytesIO):
    """A file object that records every read size (to prove nothing reads it whole)."""

    def __init__(self, data):
        super().__init__(data)
        self.sizes = []

    def read(self, size=-1):
        self.sizes.append(size)
        return super().read(size)


def _folder():
    return SimpleNamespace(id=7, name="docs", user_token="tok")


class TestSaveStream:
    def test_copies_in_chunks_and_returns_size_and_hash(self, storage_root, monkeypatch):
        monkeypatch.setattr(FileStorage, "STREAM_CHUNK", 4)
        data = b"0123456789abc"
        src = _Recorder(data)
        rel, size, digest = FileStorage.save_stream("tok", "docs", "f1", src, max_bytes=100)
        assert rel == "storage/tok/docs/f1"
        assert (storage_root / rel).read_bytes() == data
        assert size == len(data) and digest == hashlib.sha256(data).hexdigest()
        assert set(src.sizes) == {4}  # never read(-1)

    def test_a_file_at_the_limit_is_kept(self, storage_root):
        rel, size, _ = FileStorage.save_stream("tok", "docs", "f1", io.BytesIO(b"x" * 10), max_bytes=10)
        assert size == 10 and (storage_root / rel).exists()

    def test_crossing_the_limit_stops_and_deletes_the_partial_file(self, storage_root, monkeypatch):
        monkeypatch.setattr(FileStorage, "STREAM_CHUNK", 4)
        src = _Recorder(b"y" * 1000)
        with pytest.raises(FileTooLargeError, match=r"File size exceeds limit"):
            FileStorage.save_stream("tok", "docs", "f1", src, max_bytes=10)
        assert not (storage_root / "storage" / "tok" / "docs" / "f1").exists()
        assert len(src.sizes) == 3  # stopped at the first chunk past the limit, not at the end

    def test_a_failed_read_leaves_no_partial_file(self, storage_root):
        src = MagicMock()
        src.read.side_effect = [b"abc", OSError("disk")]
        with pytest.raises(OSError):
            FileStorage.save_stream("tok", "docs", "f1", src, max_bytes=100)
        assert not (storage_root / "storage" / "tok" / "docs" / "f1").exists()


class TestStoreFile:
    def test_a_stream_is_copied_not_read_whole(self, storage_root, monkeypatch):
        monkeypatch.setattr(FileStorage, "STREAM_CHUNK", 8)
        src = _Recorder(b"hello world, streamed")
        with patch.object(FileAdapter, "_name_taken", return_value=False), \
                patch("src.adapter.file.FileDB.create", side_effect=lambda **kw: SimpleNamespace(**kw)):
            validation, record = FileAdapter._store_file(_folder(), src, "a.txt", None, None)
        assert -1 not in src.sizes and None not in src.sizes
        assert validation["file_size"] == 21 and record.file_size == 21
        assert record.content_hash == hashlib.sha256(b"hello world, streamed").hexdigest()
        assert record.mime_type == "text/plain"

    def test_a_stream_with_a_bad_extension_is_refused_before_it_is_copied(self, storage_root):
        from db.filedb import UnsupportedFileTypeError
        src = _Recorder(b"MZ")
        with pytest.raises(UnsupportedFileTypeError):
            FileAdapter._store_file(_folder(), src, "tool.exe", None, None)
        assert src.sizes == []

    def test_an_oversized_stream_is_refused_and_nothing_is_recorded(self, storage_root, monkeypatch):
        monkeypatch.setattr(FileDB, "MAX_FILE_SIZE", 5)
        with patch.object(FileAdapter, "_name_taken", return_value=False), \
                patch("src.adapter.file.FileDB.create") as create:
            with pytest.raises(FileTooLargeError):
                FileAdapter._store_file(_folder(), io.BytesIO(b"123456"), "a.txt", None, None)
        create.assert_not_called()
        assert list((storage_root / "storage" / "tok" / "docs").iterdir()) == []

    def test_the_file_is_removed_when_the_record_cannot_be_written(self, storage_root):
        with patch.object(FileAdapter, "_name_taken", return_value=False), \
                patch("src.adapter.file.FileDB.create", side_effect=RuntimeError("db down")):
            with pytest.raises(RuntimeError):
                FileAdapter._store_file(_folder(), io.BytesIO(b"abc"), "a.txt", None, None)
        assert list((storage_root / "storage" / "tok" / "docs").iterdir()) == []


class TestDeclaredSize:
    async def test_single_upload_over_the_limit_is_refused_before_storing(self):
        with patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file") as store:
            with pytest.raises(FileTooLargeError, match=r"\(100\.0MB\)"):
                await FileAdapter.upload_file(
                    {"file_content": io.BytesIO(b""), "file_size": FileDB.MAX_FILE_SIZE + 1,
                     "file_name": "a.txt"}, folder_id=7, user_token="tok")
        store.assert_not_called()

    async def test_batch_reports_the_oversized_file_and_keeps_going(self):
        record = SimpleNamespace(id="f2", folder_id=7, file_name="b.txt", file_path="p", file_size=1,
                                 mime_type="text/plain", description=None, tags=None,
                                 upload_time=SimpleNamespace(isoformat=lambda: "t"), updated_time=None)
        with patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file", return_value=({"file_size": 1}, record)) as store, \
                patch.object(FileAdapter, "_update_folder_stats"):
            out = await FileAdapter.upload_file(
                [{"file_content": io.BytesIO(b""), "file_size": FileDB.MAX_FILE_SIZE + 1, "file_name": "a.txt"},
                 {"file_content": io.BytesIO(b"y"), "file_size": 1, "file_name": "b.txt"}],
                folder_id=7, user_token="tok")
        assert out["summary"]["successful_count"] == 1
        assert out["failed_uploads"] == [{"file_name": "a.txt", "error": "File size exceeds limit (100.0MB)"}]
        assert store.call_count == 1


def _file_app():
    from src.api.dependencies.auth import extract_token
    from src.api.router import file_api
    from src.auth.dependencies import authenticate_request
    app = FastAPI()
    app.include_router(file_api.router)
    app.dependency_overrides[authenticate_request] = lambda: {}
    app.dependency_overrides[extract_token] = lambda: "tok"
    return app


class TestEndpoints:
    def test_the_endpoint_passes_the_spooled_file_not_its_bytes(self):
        seen = {}

        async def upload(files_data, folder_id, user_token):
            seen.update(files_data)
            return [{"id": "00000000-0000-0000-0000-000000000001", "folder_id": 7, "file_name": "a.txt",
                     "file_path": "p", "file_size": 3}]

        with patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
                patch("src.api.router.file_api.FileAdapter.upload_file", new=AsyncMock(side_effect=upload)), \
                patch("starlette.datastructures.UploadFile.read", side_effect=AssertionError("read whole")):
            r = TestClient(_file_app()).post("/folders/7/file", data={"auto_index": "false"},
                                             files={"file": ("a.txt", b"abc", "text/plain")})
        assert r.status_code == 200, r.text
        assert not isinstance(seen["file_content"], bytes) and seen["file_size"] == 3

    def test_an_oversized_file_gets_the_same_413_as_before(self, monkeypatch):
        monkeypatch.setattr(FileDB, "MAX_FILE_SIZE", 2)
        with patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file") as store:
            r = TestClient(_file_app()).post("/folders/7/file", data={"auto_index": "false"},
                                             files={"file": ("a.txt", b"abc", "text/plain")})
        assert r.status_code == 413
        assert r.json() == {"detail": file_too_large_message(2)}
        store.assert_not_called()


def _guarded():
    calls = {"n": 0}
    inner = FastAPI()

    @inner.post("/api/folders/{fid}/file")
    async def single(fid: int):
        calls["n"] += 1
        return {"ok": True}

    @inner.post("/api/folders/{fid}/files")
    async def batch(fid: int):
        calls["n"] += 1
        return {"ok": True}

    from starlette.requests import Request

    @inner.post("/api/admin/manage/folders/{fid}/files")
    async def admin(fid: int, request: Request):
        calls["n"] += 1
        body = b""
        async for part in request.stream():
            body += part
        return {"len": len(body)}

    return UploadSizeLimit(inner), calls


class TestGuard:
    @pytest.fixture(autouse=True)
    def small_limit(self, monkeypatch):
        monkeypatch.setattr(FileDB, "MAX_FILE_SIZE", 10)
        monkeypatch.setattr(upload_limit, "FORM_ALLOWANCE", 5)

    def test_content_length_over_the_limit_is_refused_unread(self):
        app, calls = _guarded()
        r = TestClient(app).post("/api/folders/1/file", content=b"z" * 16)
        assert r.status_code == 413 and r.json() == {"detail": file_too_large_message(10)}
        assert calls["n"] == 0

    def test_console_route_is_guarded_too(self):
        app, calls = _guarded()
        r = TestClient(app).post("/api/admin/manage/folders/1/files", content=b"z" * 16)
        assert r.status_code == 413 and calls["n"] == 0

    def test_a_body_at_the_limit_passes(self):
        app, calls = _guarded()
        r = TestClient(app).post("/api/admin/manage/folders/1/files", content=b"z" * 15)
        assert r.status_code == 200 and r.json() == {"len": 15}

    def test_a_chunked_body_is_cut_off_once_it_crosses_the_limit(self):
        app, calls = _guarded()

        def gen():
            for i in range(100):
                yield b"z" * 4

        r = TestClient(app).post("/api/admin/manage/folders/1/files", content=gen())
        assert r.status_code == 413 and r.json() == {"detail": file_too_large_message(10)}

    def test_the_batch_route_and_other_methods_pass_through(self):
        app, calls = _guarded()
        client = TestClient(app)
        assert client.post("/api/folders/1/files", content=b"z" * 50).status_code == 200
        assert client.get("/api/folders/1/file").status_code == 405
        assert calls["n"] == 1

    async def test_a_cut_off_body_raises_inside_the_app(self):
        """Raw ASGI: the receive wrapper refuses the chunk that crosses the limit."""
        from fastapi import HTTPException
        seen = []

        async def app(scope, receive, send):
            while True:
                event = await receive()
                seen.append(len(event["body"]))
                if not event.get("more_body"):
                    break

        events = iter([{"type": "http.request", "body": b"z" * 8, "more_body": True}] * 5)

        async def receive():
            return next(events)

        scope = {"type": "http", "method": "POST", "path": "/api/folders/3/file", "headers": []}
        with pytest.raises(HTTPException) as exc:
            await UploadSizeLimit(app)(scope, receive, AsyncMock())
        assert exc.value.status_code == 413 and seen == [8]  # second chunk (16 > 15) refused
