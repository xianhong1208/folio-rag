"""ref / ref: uploads and names fail with 4xx, never a 500 carrying SQL details.

- Unsupported extension → 415, over the size limit → 413 (both ValueError
  subclasses, so older callers that catch ValueError keep working).
- Folder names are NFC-normalised (macOS sends "café" decomposed) and capped at
  the String(255) column; over-long or blank names are 422.
- Upload filenames lose client paths and control characters and are truncated
  keeping the extension; Content-Disposition cannot be broken by quotes.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from db.filedb import FileDB, FileTooLargeError, UnsupportedFileTypeError
from src.utils.names import MAX_NAME_LEN, content_disposition, normalize_name, sanitize_filename


class TestFileValidationErrors:
    def test_unsupported_type_is_its_own_error(self):
        with pytest.raises(UnsupportedFileTypeError, match=r"\.exe"):
            FileDB._validate_file_content(b"x", "tool.exe")
        with pytest.raises(UnsupportedFileTypeError, match="no extension"):
            FileDB._validate_file_content(b"x", "README")

    def test_too_large_is_its_own_error(self, monkeypatch):
        monkeypatch.setattr(FileDB, "MAX_FILE_SIZE", 3)
        with pytest.raises(FileTooLargeError):
            FileDB._validate_file_content(b"xxxx", "a.txt")

    def test_both_remain_value_errors(self):
        assert issubclass(UnsupportedFileTypeError, ValueError)
        assert issubclass(FileTooLargeError, ValueError)


@pytest.fixture()
def upload_client():
    from src.api.dependencies.auth import extract_token
    from src.api.router import file_api
    from src.auth.dependencies import authenticate_request
    app = FastAPI()
    app.include_router(file_api.router)
    app.dependency_overrides[authenticate_request] = lambda: {}
    app.dependency_overrides[extract_token] = lambda: "tok"
    with patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])):
        yield TestClient(app), file_api


def _upload_path(client):
    return "/folders/7/file"  # single-file upload (the console delegates here)


class TestUploadStatusCodes:
    @pytest.mark.parametrize("exc, code", [
        (UnsupportedFileTypeError("Unsupported file type: .exe"), 415),
        (FileTooLargeError("File size exceeds limit (100.0MB)"), 413),
        (ValueError("folder_name contains illegal characters"), 400),
    ])
    def test_rejections_map_to_4xx(self, upload_client, exc, code):
        client, file_api = upload_client
        with patch.object(file_api.FileAdapter, "upload_file", new=AsyncMock(side_effect=exc)):
            r = client.post(_upload_path(client), files={"file": ("x.exe", b"MZ")},
                            data={"auto_index": "false"})
        assert r.status_code == code, r.text
        assert r.json()["detail"] == str(exc)

    def test_filename_is_sanitised_before_storage(self, upload_client):
        client, file_api = upload_client
        info = {"id": "9e68c18e-3209-44d3-b7e9-2211b3660d66", "folder_id": 7, "file_name": "report.pdf",
                "file_path": "p", "file_size": 4}
        up = AsyncMock(return_value=[SimpleNamespace(model_dump=lambda: info, id=info["id"])])
        with patch.object(file_api.FileAdapter, "upload_file", new=up):
            r = client.post(_upload_path(client),
                            files={"file": ("C:\\Users\\me\\report.pdf", b"%PDF")},
                            data={"auto_index": "false"})
        assert r.status_code == 200, r.text
        assert up.await_args.kwargs["files_data"]["file_name"] == "report.pdf"


class TestNames:
    def test_nfd_folder_name_becomes_nfc(self):
        assert normalize_name(" cafe\u0301 ") == "caf\u00e9"

    def test_nfc_name_passes_the_safe_name_check(self):
        from src.storage.file_storage import _validate_safe_name
        _validate_safe_name(normalize_name("cafe\u0301"), "folder_name")  # no raise

    @pytest.mark.parametrize("raw, expected", [
        ("../../etc/passwd", "passwd"),
        ("dir\\sub\\a.pdf", "a.pdf"),
        ("a\x00b\x1f.txt", "ab.txt"),
        ("", "file"),
        (None, "file"),
        ("..", "file"),
        ("re\u0301sume\u0301.docx", "r\u00e9sum\u00e9.docx"),
    ])
    def test_sanitize_filename(self, raw, expected):
        assert sanitize_filename(raw) == expected

    def test_long_filename_keeps_extension(self):
        out = sanitize_filename("a" * 400 + ".pdf")
        assert len(out) == MAX_NAME_LEN and out.endswith(".pdf")

    def test_long_name_with_odd_suffix_is_cut(self):
        out = sanitize_filename("a" * 300 + "." + "b" * 40)
        assert len(out) == MAX_NAME_LEN

    def test_content_disposition_escapes_quotes(self):
        v = content_disposition('a"b\\c.pdf')
        assert 'filename="a_b_c.pdf"' in v
        assert "filename*=UTF-8''a%22b%5Cc.pdf" in v

    def test_content_disposition_non_ascii(self):
        v = content_disposition("報告.pdf")
        assert v.startswith('attachment; filename=".pdf"')
        assert "filename*=UTF-8''%E5%A0%B1%E5%91%8A.pdf" in v


class TestFolderRequests:
    @pytest.fixture()
    def client(self, as_identity, make_identity):
        from src.api.router.admin_settings import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(as_identity(app, make_identity("owner-tok", admin=False)))

    def test_over_long_name_is_422_not_db_error(self, client):
        with patch("src.api.router.folder_api.create_folder", new=AsyncMock()) as ep:
            r = client.post("/api/admin/manage/folders", json={"name": "x" * 256})
        assert r.status_code == 422 and not ep.await_count

    def test_non_string_name_is_422(self, client):
        r = client.post("/api/admin/manage/folders", json={"name": 42})
        assert r.status_code == 422

    def test_nfd_name_is_normalised_before_create(self, client):
        with patch("src.api.router.folder_api.create_folder",
                   new=AsyncMock(return_value={"id": 1})) as ep:
            r = client.post("/api/admin/manage/folders", json={"name": "cafe\u0301"})
        assert r.status_code == 200
        assert ep.await_args.args[0].name == "caf\u00e9"

    def test_rename_too_long_is_422(self, client):
        with patch("db.cached_folderdb.CachedFolderDB.get_by_id",
                   return_value=SimpleNamespace(id=7, name="f", user_token="owner-tok")):
            r = client.patch("/api/admin/manage/folders/7", json={"name": "y" * 300})
        assert r.status_code == 422
