"""Images are refused at upload while no vision model can read them (1.5).

Folio runs no OCR model any more: an image is read only by rag.parsing.vision. Without one
it could only fail at indexing, so the REST upload, the batch upload and the console upload
refuse it up front with 422 and a message saying what to configure. PDFs are always
accepted (the text layer reads them); other files are not affected.
"""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adapter.file import FileAdapter
from src.config.model import AsrConfig, ParsingConfig, VisionModelConfig
from src.domain.rag.cloud_parser import ImagesNeedVisionError
from tests.test_audio_upload_refusal import _file_app, _folder, _record

_CFG = "src.config.config_manager.Config.get_config_model"


def _live(vision=None):
    rag = SimpleNamespace(parsing=ParsingConfig(vision=vision), asr=AsrConfig())
    return SimpleNamespace(rag=rag)


NONE = _live()
SET = _live(VisionModelConfig(model="qwen3-vl-8b", base_url="https://fy.example/v1"))


class TestAdapter:
    async def test_single_image_is_refused_before_anything_is_stored(self):
        with patch(_CFG, return_value=NONE), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file") as store:
            with pytest.raises(ImagesNeedVisionError, match="rag.parsing.vision"):
                await FileAdapter.upload_file({"file_content": io.BytesIO(b"x"), "file_size": 1,
                                               "file_name": "scan.PNG"}, folder_id=7, user_token="tok")
        store.assert_not_called()

    async def test_image_is_stored_with_a_vision_model_and_pdfs_always(self):
        for live, name in ((SET, "photo.jpg"), (NONE, "report.pdf")):
            with patch(_CFG, return_value=live), \
                    patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                    patch.object(FileAdapter, "_store_file", return_value=({"file_size": 1}, _record(name))) as store, \
                    patch.object(FileAdapter, "_update_folder_stats"):
                await FileAdapter.upload_file({"file_content": io.BytesIO(b"x"), "file_size": 1,
                                               "file_name": name}, folder_id=7, user_token="tok")
            store.assert_called_once()

    async def test_batch_lists_the_image_as_failed_and_keeps_the_rest(self):
        with patch(_CFG, return_value=NONE), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file", return_value=({"file_size": 1}, _record("b.pdf"))) as store, \
                patch.object(FileAdapter, "_update_folder_stats"):
            out = await FileAdapter.upload_file(
                [{"file_content": io.BytesIO(b"x"), "file_size": 1, "file_name": "a.webp"},
                 {"file_content": io.BytesIO(b"y"), "file_size": 1, "file_name": "b.pdf"}],
                folder_id=7, user_token="tok")
        assert out["summary"]["successful_count"] == 1 and out["summary"]["failed_count"] == 1
        assert out["failed_uploads"][0]["file_name"] == "a.webp"
        assert "is an image" in out["failed_uploads"][0]["error"]
        assert store.call_count == 1


class TestEndpoints:
    def test_rest_upload_answers_422_with_what_to_set(self):
        with patch(_CFG, return_value=NONE), \
                patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file") as store:
            r = TestClient(_file_app()).post("/folders/7/file", data={"auto_index": "true"},
                                             files={"file": ("page.tiff", b"II*\x00", "image/tiff")})
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "page.tiff" in detail and "VISION_BASE_URL" in detail and "docs/migrate-1.5.md" in detail
        store.assert_not_called()

    def test_console_upload_answers_422(self, as_identity, make_identity):
        from src.api.router.admin_settings import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(as_identity(app, make_identity("sub:owner-tok", admin=False)))
        with patch(_CFG, return_value=NONE), \
                patch("db.cached_folderdb.CachedFolderDB.get_by_id",
                      return_value=SimpleNamespace(id=7, name="f", user_token="sub:owner-tok")), \
                patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file") as store:
            r = client.post("/api/admin/manage/folders/7/files", data={"auto_index": "true"},
                            files={"file": ("photo.png", b"\x89PNG", "image/png")})
        assert r.status_code == 422 and "is an image" in r.json()["detail"]
        store.assert_not_called()
