"""Audio is refused at upload while speech-to-text cannot transcribe it (1.5).

Folio has no in-process speech-to-text any more. Without ``rag.asr.base_url`` (or with
``rag.asr.enabled: false``) an audio file could only fail at indexing, or be indexed
empty, so the REST upload, the batch upload and the console upload refuse it up front
with 422 and a message saying what to configure. Other files are not affected.
"""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adapter.file import FileAdapter
from src.config.model import AsrConfig
from src.domain.rag.asr_provider import SpeechToTextUnavailableError

_LIVE = "src.domain.rag.asr_provider.current_asr_config"
UNSET = AsrConfig()
SET = AsrConfig(base_url="https://stt.example/v1", model="whisper-large-v3-turbo")


def _folder():
    return SimpleNamespace(id=7, name="docs", user_token="tok")


def _record(name):
    import uuid
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return SimpleNamespace(id=uuid.uuid4(), folder_id=7, file_name=name, file_path="p", file_size=1,
                           mime_type="audio/mpeg", description=None, tags=None,
                           upload_time=now, updated_time=now)


class TestAdapter:
    async def test_single_audio_is_refused_before_anything_is_stored(self):
        with patch(_LIVE, return_value=UNSET), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file") as store:
            with pytest.raises(SpeechToTextUnavailableError, match="rag.asr.base_url"):
                await FileAdapter.upload_file({"file_content": io.BytesIO(b"x"), "file_size": 1,
                                               "file_name": "talk.mp3"}, folder_id=7, user_token="tok")
        store.assert_not_called()

    async def test_disabled_speech_to_text_refuses_audio_too(self):
        with patch(_LIVE, return_value=AsrConfig(enabled=False, base_url="https://stt.example/v1")), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file") as store:
            with pytest.raises(SpeechToTextUnavailableError, match="disabled"):
                await FileAdapter.upload_file({"file_content": io.BytesIO(b"x"), "file_size": 1,
                                               "file_name": "talk.wav"}, folder_id=7, user_token="tok")
        store.assert_not_called()

    async def test_configured_audio_is_stored(self):
        with patch(_LIVE, return_value=SET), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file", return_value=({"file_size": 1}, _record("t.mp3"))) as store, \
                patch.object(FileAdapter, "_update_folder_stats"):
            await FileAdapter.upload_file({"file_content": io.BytesIO(b"x"), "file_size": 1,
                                           "file_name": "t.mp3"}, folder_id=7, user_token="tok")
        store.assert_called_once()

    async def test_batch_lists_the_audio_as_failed_and_keeps_the_rest(self):
        with patch(_LIVE, return_value=UNSET), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file", return_value=({"file_size": 1}, _record("b.pdf"))) as store, \
                patch.object(FileAdapter, "_update_folder_stats"):
            out = await FileAdapter.upload_file(
                [{"file_content": io.BytesIO(b"x"), "file_size": 1, "file_name": "a.m4a"},
                 {"file_content": io.BytesIO(b"y"), "file_size": 1, "file_name": "b.pdf"}],
                folder_id=7, user_token="tok")
        assert out["summary"]["successful_count"] == 1 and out["summary"]["failed_count"] == 1
        assert out["failed_uploads"][0]["file_name"] == "a.m4a"
        assert "not configured" in out["failed_uploads"][0]["error"]
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
    def test_rest_upload_answers_422_with_what_to_set(self):
        with patch(_LIVE, return_value=UNSET), \
                patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file") as store:
            r = TestClient(_file_app()).post("/folders/7/file", data={"auto_index": "true"},
                                             files={"file": ("meeting.mp3", b"ID3", "audio/mpeg")})
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "meeting.mp3" in detail and "ASR_BASE_URL" in detail and "docs/migrate-asr.md" in detail
        store.assert_not_called()

    def test_rest_upload_of_a_document_is_unaffected(self):
        with patch(_LIVE, return_value=UNSET), \
                patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file", return_value=({"file_size": 3}, _record("a.txt"))), \
                patch.object(FileAdapter, "_update_folder_stats"):
            r = TestClient(_file_app()).post("/folders/7/file", data={"auto_index": "false"},
                                             files={"file": ("a.txt", b"abc", "text/plain")})
        assert r.status_code == 200, r.text

    def test_console_upload_answers_422(self, as_identity, make_identity):
        from src.api.router.admin_settings import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(as_identity(app, make_identity("sub:owner-tok", admin=False)))
        with patch(_LIVE, return_value=UNSET), \
                patch("db.cached_folderdb.CachedFolderDB.get_by_id",
                      return_value=SimpleNamespace(id=7, name="f", user_token="sub:owner-tok")), \
                patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
                patch.object(FileAdapter, "_verify_folder_ownership", return_value=_folder()), \
                patch.object(FileAdapter, "_store_file") as store:
            r = client.post("/api/admin/manage/folders/7/files", data={"auto_index": "true"},
                            files={"file": ("voice.wav", b"RIFF", "audio/wav")})
        assert r.status_code == 422 and "not configured" in r.json()["detail"]
        store.assert_not_called()
