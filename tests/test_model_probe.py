"""Per-role probes (src/domain/rag/model_probe.py), the probe route, and the console's
per-role settings view: env-referenced secrets, key indicators, profile enforcement.

Every endpoint is the in-process fake server; no network, no keys.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import runtime_overrides as ro
from src.config.model import ConfigModel
from src.domain.rag.model_probe import _silent_wav, _tiny_png, probe_role
from tests.fake_model_server import FakeModelServer


def _config(base: str = "http://127.0.0.1:9", profile: str = "local", **rag_over) -> ConfigModel:
    rag = {
        "profile": profile,
        "embedding": {"provider": "openai-compatible", "model": "emb", "dimension": 8,
                      "base_url": base + "/v1", "api_key": "k"},
        "chunking": {}, "retrieval": {"default_top_k": 1, "default_similarity_cutoff": 0,
                                      "hybrid_search": False},
        "vector_store": {"type": "pgvector", "table_prefix": "v"},
        "llm": {"provider": "openai-compatible", "model": "chat", "base_url": base + "/v1", "api_key": "k"},
        "contextual_retrieval": {"enabled": True},
        "rerank": {"enabled": True, "api": "cohere", "model": "rr", "base_url": base, "api_key": "k"},
        "asr": {"provider": "openai-compatible", "model": "whisper-1", "base_url": base + "/v1", "api_key": "k"},
        "parsing": {"pdf": "cloud", "images": "cloud",
                    "vision": {"model": "vlm", "base_url": base + "/v1", "api_key": "k"}},
    }
    rag.update(rag_over)
    return ConfigModel.model_validate({
        "server": {"host": "0.0.0.0", "port": 1, "transport": "http"},
        "database": {"url": "postgresql://x/y", "echo": False, "pool_size": 1,
                     "max_overflow": 1, "pool_timeout": 1, "pool_recycle": 1},
        "auth": {"enabled": False}, "logging": {"level": "INFO", "format": "x", "file": None,
                                                "max_size": "1MB", "backup_count": 1},
        "app": {}, "modules": {"enabled": []}, "rag": rag,
    })


@pytest.fixture(autouse=True)
def _clean_secret_sources():
    ro._secret_source.clear()
    yield
    ro._secret_source.clear()


class TestProbes:
    def test_every_endpoint_role_ok(self):
        with FakeModelServer(dim=8) as srv:
            rag = _config(srv.url).rag
            results = {r: probe_role(rag, r) for r in
                       ("embedding", "contextual_llm", "answer_llm", "eval_llm", "rerank", "vision", "asr")}
            vision_req = [b for b in srv.posts("/chat/completions")
                          if isinstance(b["messages"][0]["content"], list)][0]
            asr_form = srv.posts("/audio/transcriptions")[0]
            auths = {r[2].get("Authorization") for r in srv.requests}
        assert {r: v["status"] for r, v in results.items()} == dict.fromkeys(results, "ok")
        assert results["embedding"]["detail"] == "8-dim vector"
        assert results["rerank"]["detail"].startswith("score ")
        assert results["vision"]["model"] == "vlm"
        assert vision_req["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png")
        assert asr_form["file"]["filename"] == "probe.wav" and asr_form["model"] == "whisper-1"
        assert auths == {"Bearer k"}
        assert all(isinstance(v["latency_ms"], int) for v in results.values())

    def test_dimension_mismatch_is_reported(self):
        with FakeModelServer(dim=4) as srv:
            r = probe_role(_config(srv.url).rag, "embedding")
        assert r["status"] == "down" and "4 dimensions but rag.embedding.dimension is 8" in r["detail"]

    def test_http_error_shows_only_the_status(self):
        with FakeModelServer() as srv:
            srv.fail_next = [(401, None)]
            r = probe_role(_config(srv.url).rag, "answer_llm")
        assert r == {**r, "status": "down", "detail": "HTTP 401"}

    def test_rerank_and_asr_errors(self):
        with FakeModelServer() as srv:
            srv.fail_next = [(503, None), (500, None)]
            rag = _config(srv.url).rag
            assert probe_role(rag, "rerank")["detail"] == "HTTP 503"
            assert probe_role(rag, "asr")["detail"] == "HTTP 500"

    def test_unreachable_shows_exception_class(self):
        r = probe_role(_config("http://127.0.0.1:9").rag, "contextual_llm")
        assert r["status"] == "down" and r["detail"] == "APIConnectionError"

    def test_disabled_and_local_roles_send_nothing(self):
        rag = _config(rerank={"enabled": False},
                      embedding={"provider": "huggingface", "model": "BAAI/bge-m3", "dimension": 8},
                      parsing=None, asr={"provider": "fireredasr"}).rag
        assert probe_role(rag, "rerank")["status"] == "disabled"
        assert probe_role(rag, "embedding")["status"] == "local"
        assert probe_role(rag, "vision")["status"] == "local"
        with patch("src.domain.rag.model_probe._asr_ready", return_value=False):
            assert probe_role(rag, "asr") == {**probe_role(rag, "asr"), "status": "down"}
        with patch("src.domain.rag.model_probe._asr_ready", return_value=True):
            assert probe_role(rag, "asr")["status"] == "local"

    def test_asr_without_endpoint_is_unset(self):
        rag = _config(asr={"provider": "openai-compatible", "base_url": None}).rag
        assert probe_role(rag, "asr")["status"] == "unset"

    def test_asr_ready_helper_is_fail_safe(self):
        from src.domain.rag.model_probe import _asr_ready
        assert _asr_ready(SimpleNamespace(asr=SimpleNamespace(provider="bogus"))) is False

    def test_default_llm_probe(self):
        with FakeModelServer() as srv:
            assert probe_role(_config(srv.url).rag, "llm")["status"] == "ok"
            srv.fail_next = [(500, None)]
            assert probe_role(_config(srv.url).rag, "llm")["detail"] == "HTTP 500"
        assert probe_role(_config(llm=None).rag, "llm")["status"] == "unset"

    def test_unknown_role(self):
        with pytest.raises(ValueError):
            probe_role(_config().rag, "summarizer")

    def test_payload_helpers(self):
        assert _tiny_png().startswith(b"\x89PNG")
        import io
        import wave
        with wave.open(io.BytesIO(_silent_wav(0.25)), "rb") as w:
            assert w.getnframes() == 4000 and w.getframerate() == 16000


class TestProbeRoute:
    @pytest.fixture()
    def client(self, as_identity, make_identity):
        from src.api.router.admin_settings import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(as_identity(app, make_identity("owner-tok", admin=True)))

    def test_probe_route(self, client):
        with FakeModelServer() as srv, \
                patch("src.config.config_manager.Config.get_config_model", return_value=_config(srv.url)):
            r = client.post("/api/admin/probe/rerank")
        assert r.status_code == 200 and r.json()["data"]["status"] == "ok"

    def test_unknown_role_404_and_no_rag_409(self, client):
        assert client.post("/api/admin/probe/nope").status_code == 404
        with patch("src.config.config_manager.Config.get_config_model", return_value=None):
            assert client.post("/api/admin/probe/embedding").status_code == 409

    def test_needs_admin(self, as_identity, make_identity):
        from src.api.router.admin_settings import router
        app = FastAPI()
        app.include_router(router)
        c = TestClient(as_identity(app, make_identity("owner-tok", admin=False)))
        assert c.post("/api/admin/probe/embedding").status_code == 403


class TestSecretsThroughTheConsole:
    def test_env_reference_is_resolved_and_persisted_as_reference(self, monkeypatch):
        monkeypatch.setenv("NEW_KEY", "sk-live-123")
        cfg = _config()
        applied, _ = ro.validate_and_apply(cfg, {"rag.rerank.api_key": "${NEW_KEY}"})
        assert cfg.rag.rerank.api_key == "sk-live-123"
        assert applied == {"rag.rerank.api_key": "${NEW_KEY}"}  # what the DB stores
        assert ro.secret_ref("rag.rerank.api_key") == "NEW_KEY"

    def test_unset_env_reference_is_rejected(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        with pytest.raises(ValueError, match="MISSING_KEY"):
            ro.validate_and_apply(_config(), {"rag.llm.api_key": "${MISSING_KEY}"})

    def test_mask_placeholder_is_rejected(self):
        with pytest.raises(ValueError, match="masked placeholder"):
            ro.validate_and_apply(_config(), {"rag.llm.api_key": "•••"})

    def test_literal_key_rejected_in_cloud(self, monkeypatch):
        monkeypatch.setattr("src.config.config_manager.Config._env_refs", {
            p: "K" for p, _ in __import__("src.config.profile", fromlist=["x"]).iter_secret_fields(_config().rag)})
        cfg = _config(profile="cloud")
        with pytest.raises(ValueError, match="literal value"):
            ro.validate_and_apply(cfg, {"rag.rerank.api_key": "sk-pasted"})
        assert cfg.rag.rerank.api_key == "k"  # nothing written
        monkeypatch.setenv("RR", "x")
        ro.validate_and_apply(cfg, {"rag.rerank.api_key": "${RR}"})
        assert cfg.rag.rerank.api_key == "x"

    def test_cloud_rejects_in_process_provider_change(self, monkeypatch):
        monkeypatch.setattr("src.config.config_manager.Config._env_refs", None, raising=False)
        monkeypatch.setattr("src.config.config_manager.Config.get_env_refs",
                            classmethod(lambda cls: {}))
        cfg = _config(profile="cloud")
        with pytest.raises(ValueError, match="profile 'cloud' rejects"):
            ro.validate_and_apply(cfg, {"rag.asr.provider": "fireredasr"})
        assert cfg.rag.asr.provider == "openai-compatible"

    def test_literal_clears_previous_reference(self, monkeypatch):
        monkeypatch.setenv("A", "1")
        cfg = _config()
        ro.validate_and_apply(cfg, {"rag.llm.api_key": "${A}"})
        ro.validate_and_apply(cfg, {"rag.llm.api_key": "plain"})
        assert ro.secret_ref("rag.llm.api_key") is None

    def test_new_paths_resolve_and_parsing_section_is_created(self):
        cfg = _config(parsing=None)
        ro.validate_and_apply(cfg, {"rag.parsing.max_pages_per_file": 10,
                                    "rag.answer_llm.model": "big"})
        assert cfg.rag.parsing.max_pages_per_file == 10 and cfg.rag.answer_llm.model == "big"
        assert ro.rebind_groups_for(["rag.contextual_llm.model", "rag.parsing.pdf"]) == ["rag.contextual_llm"]


class TestSettingsView:
    def test_view_has_profile_roles_and_key_indicators(self, monkeypatch):
        from src.adapter import runtime_settings_service as svc
        cfg = _config(profile="cloud")
        refs = {"rag.embedding.api_key": "EMB_KEY", "rag.llm.api_key": "LLM_KEY"}
        monkeypatch.setattr("src.config.config_manager.Config.get_env_refs", classmethod(lambda cls: refs))
        with patch(f"{svc.__name__}.Config.get_config_model", return_value=cfg), \
             patch(f"{svc.__name__}.RuntimeSettingsDB") as db:
            db.load_all.return_value = {}
            view = svc.get_settings_view()
        by_path = {s["path"]: s for s in view["settings"]}
        emb = by_path["rag.embedding.api_key"]
        assert emb["value"] == "•••" and emb["is_set"] is True and emb["env"] == "EMB_KEY"
        assert by_path["rag.answer_llm.api_key"]["is_set"] is False
        assert view["profile"] == "cloud" and isinstance(view["usage"], dict)
        roles = {r["role"]: r for r in view["roles"]}
        assert roles["answer_llm"]["key_env"] == "LLM_KEY"  # inherited from rag.llm
        assert roles["vision"]["key_set"] is True and roles["vision"]["key_env"] is None
        assert all(v != "k" for r in view["roles"] for v in r.values())  # no key values anywhere

    def test_view_without_rag(self):
        from src.adapter import runtime_settings_service as svc
        assert svc._profile_view(None, {}) == {"profile": None, "roles": [], "usage": {}}

    def test_audit_value_never_contains_a_key(self, monkeypatch):
        from src.adapter import runtime_settings_service as svc
        monkeypatch.setenv("Z", "secret-z")
        ro.validate_and_apply(_config(), {"rag.asr.api_key": "${Z}"})
        assert svc._audit_value("rag.asr.api_key", "secret-z") == "${Z}"
        assert svc._audit_value("rag.rerank.api_key", "lit") == "•••"
        assert svc._audit_value("rag.rerank.model", "m") == "m"


class TestHealthListsVision:
    def _client(self, as_identity, make_identity):
        from src.api.router.admin_settings import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(as_identity(app, make_identity("owner-tok", admin=False)))

    @pytest.mark.parametrize("parsing,listed", [
        ({"pdf": "cloud", "vision": {"model": "vlm", "base_url": "http://v/v1"}}, True),
        (None, False),
    ])
    def test_document_parsing_row(self, as_identity, make_identity, parsing, listed):
        async def ok(url, key):
            return "ok", "HTTP 200", 1
        with patch("src.config.config_manager.Config.get_config_model",
                   return_value=_config(parsing=parsing)), \
             patch("src.api.router.admin_overview._probe_db_health", return_value=("ok", "x", 1)), \
             patch("src.api.router.admin_overview._probe_endpoint_health", side_effect=ok):
            r = self._client(as_identity, make_identity).get("/api/admin/health")
        names = [c["name"] for c in r.json()["data"]["checks"]]
        assert ("Document Parsing" in names) is listed
