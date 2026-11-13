"""Advance notice (1.4.2): the built-in OCR, speech-to-text and /v1 media endpoints go in 1.5.

Nothing is removed yet. /v1 responses carry Deprecation + Link headers, and the
first /v1 call, the first in-process OCR and the first in-process transcription each
log a warning once per process; startup logs what the configuration uses.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.config.model import RAGConfig
from src.utils import deprecation as dep

_CFG = "src.config.config_manager.Config.get_config_model"


@pytest.fixture(autouse=True)
def fresh_notices():
    dep._reset_for_tests()
    yield
    dep._reset_for_tests()


def _rag(**over):
    data = {
        "embedding": {"provider": "openai", "model": "e", "dimension": 8},
        "chunking": {}, "retrieval": {"default_top_k": 1, "default_similarity_cutoff": 0,
                                      "hybrid_search": False},
        "vector_store": {"type": "pgvector", "table_prefix": "v"},
    }
    data.update(over)
    return RAGConfig.model_validate(data)


CLOUD = dict(profile="cloud", parsing={"vision": {"model": "v", "base_url": "http://x/v1"}},
             asr={"provider": "openai-compatible", "base_url": "http://x/v1", "model": "w"})


class TestWarnOnce:
    def test_each_notice_is_logged_once_per_process(self):
        with patch.object(dep, "logger") as log:
            assert dep.warn_once("ocr") is True
            assert dep.warn_once("ocr") is False
            assert dep.warn_once("asr") is True
        assert log.warning.call_count == 2
        first = log.warning.call_args_list[0].args[0]
        assert "removed in 1.5" in first and "vision model" in first and dep.NOTICE_URL in first

    def test_messages_name_the_replacements(self):
        assert "openai-compatible" in dep.MESSAGES["asr"] and "Forgeyard" in dep.MESSAGES["asr"]
        assert "rag.parsing.vision" in dep.MESSAGES["ocr"]
        assert "/v1/ocr/general" in dep.MESSAGES["media"]


def _app():
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"data": []}

    @app.post("/v1/ocr/general")
    def ocr():
        raise HTTPException(503, "OCR is disabled")

    @app.get("/health")
    def health():
        return {"ok": True}

    app.add_middleware(dep.MediaDeprecationMiddleware)
    return app


class TestHeaders:
    def test_v1_responses_are_marked_deprecated(self):
        r = TestClient(_app()).get("/v1/models")
        assert r.status_code == 200
        assert r.headers["deprecation"] == dep.DEPRECATION_HEADER
        assert r.headers["deprecation"].startswith("@") and int(r.headers["deprecation"][1:]) > 0
        assert 'rel="deprecation"' in r.headers["link"] and dep.NOTICE_URL in r.headers["link"]

    def test_errors_are_marked_too(self):
        client = TestClient(_app())
        assert client.post("/v1/ocr/general").headers.get("deprecation") == dep.DEPRECATION_HEADER
        assert client.get("/v1/nope").headers.get("deprecation") == dep.DEPRECATION_HEADER

    def test_other_paths_are_not(self):
        r = TestClient(_app()).get("/health")
        assert "deprecation" not in r.headers and "link" not in r.headers

    def test_v1_prefix_must_be_a_path_segment(self):
        app = FastAPI()

        @app.get("/v1x")
        def other():
            return {}

        app.add_middleware(dep.MediaDeprecationMiddleware)
        assert "deprecation" not in TestClient(app).get("/v1x").headers

    def test_first_call_logs_once(self):
        client = TestClient(_app())
        with patch.object(dep, "logger") as log:
            client.get("/v1/models")
            client.get("/v1/models")
            client.get("/health")
        assert log.warning.call_count == 1 and "/v1 media endpoints" in log.warning.call_args.args[0]

    def test_the_real_media_router(self):
        from src.api.router.media import router
        app = FastAPI()
        app.include_router(router)
        app.add_middleware(dep.MediaDeprecationMiddleware)
        r = TestClient(app).get("/v1/models")
        assert r.status_code == 200 and r.headers["deprecation"] == dep.DEPRECATION_HEADER


def _config(rag, modules=("media",)):
    return SimpleNamespace(rag=rag, modules=SimpleNamespace(enabled=list(modules)))


class TestStartup:
    def test_local_defaults_use_all_three(self):
        assert dep.deprecated_in_use(_config(_rag())) == ["ocr", "asr", "media"]

    def test_cloud_setup_without_the_media_module_uses_none(self):
        assert dep.deprecated_in_use(_config(_rag(**CLOUD), modules=("rag",))) == []

    def test_ocr_off_is_not_counted(self):
        rag = _rag(docling={"ocr_enabled": False},
                   asr={"provider": "openai-compatible", "base_url": "http://x/v1", "model": "w"})
        assert dep.deprecated_in_use(_config(rag, modules=())) == []

    def test_in_process_asr_is_counted(self):
        rag = _rag(**{**CLOUD, "profile": "hybrid", "asr": {"provider": "fireredasr"},
                      "parsing": {"pdf": "cloud", "images": "cloud", "vision": {"model": "v", "base_url": "http://x/v1"}}})
        assert dep.deprecated_in_use(_config(rag, modules=())) == ["asr"]

    def test_startup_logs_one_line_each(self):
        with patch.object(dep, "logger") as log:
            assert dep.log_startup_notice(_config(_rag())) == ["ocr", "asr", "media"]
        assert log.warning.call_count == 3

    def test_startup_check_never_raises(self):
        with patch.object(dep, "deprecated_in_use", side_effect=RuntimeError("bad config")):
            assert dep.log_startup_notice(object()) == []


def _cpu():
    from docling.datamodel.accelerator_options import AcceleratorOptions
    return AcceleratorOptions(device="cpu")


class TestFirstUse:
    def test_building_the_local_ocr_converter_warns(self, monkeypatch):
        from src.domain.rag import docling_loader as dl
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(asr={"provider": "fireredasr"})))
        monkeypatch.setattr(dl, "_converter", None)
        monkeypatch.setattr(dl, "_resolve_artifacts_path", lambda: None)
        monkeypatch.setattr(dl, "_get_accelerator_options", _cpu)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setattr(dl, "_apply_docling_env", lambda: None)
        monkeypatch.setattr("docling.document_converter.DocumentConverter", lambda **kw: MagicMock())
        with patch.object(dep, "warn_once") as warn:
            dl._build_converter()
        warn.assert_called_once_with("ocr")
        monkeypatch.setattr(dl, "_converter", None)

    def test_no_warning_when_ocr_is_off(self, monkeypatch):
        from src.domain.rag import docling_loader as dl
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(
            docling={"ocr_enabled": False}, asr={"provider": "fireredasr"})))
        monkeypatch.setattr(dl, "_converter", None)
        monkeypatch.setattr(dl, "_resolve_artifacts_path", lambda: None)
        monkeypatch.setattr(dl, "_get_accelerator_options", _cpu)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setattr(dl, "_apply_docling_env", lambda: None)
        monkeypatch.setattr("docling.document_converter.DocumentConverter", lambda **kw: MagicMock())
        with patch.object(dep, "warn_once") as warn:
            dl._build_converter()
        warn.assert_not_called()
        monkeypatch.setattr(dl, "_converter", None)

    def test_docling_whisper_transcription_warns(self):
        from src.domain.rag.asr_provider import DoclingWhisperAsrProvider
        with patch.object(dep, "warn_once") as warn, \
                patch("src.domain.rag.asr_provider.docling_convert_once", side_effect=RuntimeError("stop")):
            with pytest.raises(RuntimeError):
                DoclingWhisperAsrProvider().transcribe(audio_path="a.wav", file_name="a.wav", max_tokens=512)
        warn.assert_called_once_with("asr")

    def test_fireredasr_transcription_warns(self):
        from src.domain.rag.fireredasr_provider import FireRedAsrProvider
        with patch.object(dep, "warn_once") as warn, \
                patch("src.domain.rag.fireredasr_provider._resample_to_wav16k", side_effect=RuntimeError("stop")):
            with pytest.raises(RuntimeError):
                FireRedAsrProvider().transcribe(audio_path="a.wav", file_name="a.wav")
        warn.assert_called_once_with("asr")

    def test_cloud_transcription_does_not_warn(self):
        from src.domain.rag.asr_provider import OpenAICompatibleAsrProvider
        p = OpenAICompatibleAsrProvider(base_url="http://127.0.0.1:9/v1", api_key=None, model="w")
        with patch.object(dep, "warn_once") as warn:
            try:
                p.transcribe(audio_path="/nonexistent.wav", file_name="a.wav")
            except Exception:  # noqa: BLE001 - only whether it warned matters
                pass
        warn.assert_not_called()
