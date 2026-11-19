"""Model key references at runtime: only the config files' key variables, only for their servers.

Before 1.4.2 a settings change could reference any environment variable as a model
key (``${VAR}``) and point the role's base_url at any server, so the server would send
that variable's value there. Now a runtime ``${VAR}`` is accepted only for a variable
config/models.yaml uses as a model key, while the role points at a server (origin) that file
pairs it with; a key never follows its role to another server; and every refused
reference gets the same generic error.
"""

from unittest.mock import patch

from tests.config_helpers import config_dir_from_file
import pytest

from src.config import runtime_overrides as ro
from src.config.key_references import NOT_ALLOWED, origin_of, paired_origins, role_keys
from tests.test_model_probe import _config

HOME = "http://127.0.0.1:9"          # every role's server in _config()
EVIL = "http://attacker.example:8080/v1"
REFS = {"rag.llm.api_key": "LLM_KEY", "rag.rerank.api_key": "RR_KEY"}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "sk-llm")
    monkeypatch.setenv("RR_KEY", "sk-rr")
    monkeypatch.setenv("DB_PASSWORD", "hunter2")
    ro._secret_source.clear()
    yield
    ro._secret_source.clear()


def _cfg(**over):
    cfg = _config(**over)
    ro.capture_startup(cfg, refs=REFS)
    return cfg


class TestOrigins:
    @pytest.mark.parametrize("a,b", [
        ("http://Host:80/v1", "http://host/other"),
        ("https://h/v1", "https://h:443"),
    ])
    def test_same_origin(self, a, b):
        assert origin_of(a) == origin_of(b)

    @pytest.mark.parametrize("a,b", [
        ("http://h/v1", "https://h/v1"), ("http://h:8080", "https://h:8080"),
        ("http://h:1", "http://h:2"), ("http://h", "http://g"),
    ])
    def test_different_origin(self, a, b):
        assert origin_of(a) != origin_of(b)

    def test_no_base_url_is_the_provider_default(self):
        assert origin_of(None, "OpenAI") == "default:openai" != origin_of(None, "azure")

    def test_llm_roles_inherit_server_and_key(self):
        rk = role_keys(_config().rag)
        assert rk["contextual_llm"].origin == origin_of(HOME + "/v1")
        assert rk["contextual_llm"].key_path == "rag.llm.api_key"

    def test_pairs_come_from_the_startup_config(self):
        pairs = paired_origins(_config().rag, REFS)
        assert pairs == {"LLM_KEY": {origin_of(HOME)}, "RR_KEY": {origin_of(HOME)}}


class TestReferences:
    def test_a_config_variable_works_for_its_server(self):
        cfg = _cfg()
        applied, _ = ro.validate_and_apply(cfg, {"rag.embedding.api_key": "${LLM_KEY}"})
        assert cfg.rag.embedding.api_key == "sk-llm" and applied == {"rag.embedding.api_key": "${LLM_KEY}"}

    def test_cloud_profile_console_reference_still_works(self, monkeypatch):
        monkeypatch.setattr("src.config.config_manager.Config.get_env_refs", classmethod(
            lambda cls: {p: "LLM_KEY" for p, _ in __import__(
                "src.config.profile", fromlist=["x"]).iter_secret_fields(_config().rag)}))
        cfg = _config(profile="cloud")
        ro.validate_and_apply(cfg, {"rag.rerank.api_key": "${LLM_KEY}"})
        assert cfg.rag.rerank.api_key == "sk-llm"

    def test_a_config_variable_is_refused_for_another_server(self):
        cfg = _cfg()
        with pytest.raises(ValueError, match="not allowed"):
            ro.validate_and_apply(cfg, {"rag.embedding.base_url": EVIL,
                                        "rag.embedding.api_key": "${LLM_KEY}"})
        assert cfg.rag.embedding.base_url == HOME + "/v1" and cfg.rag.embedding.api_key == "k"

    def test_any_other_variable_is_never_read(self):
        cfg = _cfg()
        with patch("src.config.config_manager.expand_env_vars") as expand:
            with pytest.raises(ValueError, match="not allowed"):
                ro.validate_and_apply(cfg, {"rag.llm.api_key": "${DB_PASSWORD}"})
        expand.assert_not_called()
        assert cfg.rag.llm.api_key == "k"

    def test_one_message_whatever_the_variable(self, monkeypatch):
        monkeypatch.delenv("LLM_KEY")  # a config variable that is not set
        cfg = _cfg()
        msgs = set()
        for ref in ("${LLM_KEY}", "${DB_PASSWORD}", "${NOPE}"):
            with pytest.raises(ValueError) as exc:
                ro.validate_and_apply(cfg, {"rag.llm.api_key": ref})
            msgs.add(str(exc.value))
        assert msgs == {f"Settings validation failed: rag.llm.api_key: {NOT_ALLOWED}"}

    def test_a_reference_inside_a_longer_value_is_refused(self):
        with pytest.raises(ValueError, match="not allowed"):
            ro.validate_and_apply(_cfg(), {"rag.llm.api_key": "Bearer ${DB_PASSWORD}"})

    def test_two_step_move_then_reference_is_refused(self):
        cfg = _cfg()
        ro.validate_and_apply(cfg, {"rag.rerank.base_url": EVIL})
        with pytest.raises(ValueError, match="not allowed"):
            ro.validate_and_apply(cfg, {"rag.rerank.api_key": "${RR_KEY}"})
        assert cfg.rag.rerank.api_key is None


class TestKeyNeverFollowsToAnotherServer:
    def test_moving_a_role_clears_its_key(self):
        cfg = _cfg()
        applied, warnings = ro.validate_and_apply(cfg, {"rag.rerank.base_url": EVIL})
        assert cfg.rag.rerank.api_key is None
        assert applied == {"rag.rerank.base_url": EVIL, "rag.rerank.api_key": None}
        assert any("rag.rerank.api_key was cleared" in w and "Enter the key" in w for w in warnings)

    def test_same_server_keeps_the_key(self):
        cfg = _cfg()
        applied, warnings = ro.validate_and_apply(cfg, {"rag.rerank.base_url": HOME + "/other"})
        assert cfg.rag.rerank.api_key == "k" and "rag.rerank.api_key" not in applied and not warnings

    def test_a_new_key_in_the_same_change_is_kept(self):
        cfg = _cfg()
        applied, warnings = ro.validate_and_apply(
            cfg, {"rag.rerank.base_url": EVIL, "rag.rerank.api_key": "their-own-key"})
        assert cfg.rag.rerank.api_key == "their-own-key" and not warnings

    def test_an_inherited_key_makes_the_change_fail(self):
        cfg = _cfg()
        with pytest.raises(ValueError, match="inherits from rag.llm.api_key"):
            ro.validate_and_apply(cfg, {"rag.contextual_llm.base_url": EVIL})
        assert cfg.rag.contextual_llm is None or cfg.rag.contextual_llm.base_url is None
        ro.validate_and_apply(cfg, {"rag.contextual_llm.base_url": EVIL, "rag.contextual_llm.api_key": "own"})
        assert cfg.rag.contextual_llm.api_key == "own" and cfg.rag.llm.api_key == "k"

    def test_moving_rag_llm_clears_its_key_for_the_roles_that_inherit_it(self):
        cfg = _cfg()
        applied, _ = ro.validate_and_apply(cfg, {"rag.llm.base_url": EVIL})
        assert cfg.rag.llm.api_key is None and applied["rag.llm.api_key"] is None
        assert role_keys(cfg.rag)["contextual_llm"].key is None

    def test_startup_replay_never_clears(self):
        cfg = _cfg()
        ro.validate_and_apply(cfg, {"rag.rerank.base_url": EVIL}, carry_check=False)
        assert cfg.rag.rerank.api_key == "k"


_SVC = "src.adapter.runtime_settings_service"


class TestStartup:
    def _load(self, cfg, rows, refs=REFS):
        from src.adapter import runtime_settings_service as svc
        with patch(f"{_SVC}.Config.get_config_model", return_value=cfg), \
                patch("src.config.config_manager.Config.get_env_refs", classmethod(lambda cls: refs)), \
                patch("db.db.get_engine"), patch(f"{_SVC}.RuntimeSettingsDB") as db, \
                patch.object(svc, "logger") as log:
            db.load_all.return_value = rows
            n = svc.load_overrides_on_startup()
        return n, [c.args[0] for c in log.warning.call_args_list]

    @pytest.mark.parametrize("order", [0, 1])
    def test_a_stored_key_for_a_moved_role_survives_whatever_the_row_order(self, order):
        rows = [("rag.rerank.base_url", EVIL), ("rag.rerank.api_key", "their-own-key")]
        cfg = _config()
        n, warnings = self._load(cfg, dict(rows if order else rows[::-1]))
        assert n == 2 and cfg.rag.rerank.api_key == "their-own-key" and not warnings

    def test_a_stored_reference_for_another_server_is_skipped(self):
        cfg = _config()
        n, warnings = self._load(cfg, {"rag.embedding.base_url": EVIL,
                                       "rag.embedding.api_key": "${LLM_KEY}"})
        assert n == 1 and cfg.rag.embedding.api_key == "k"
        assert any("Skipping invalid override rag.embedding.api_key" in w for w in warnings)
        assert not any("sk-llm" in w for w in warnings)

    def test_a_config_reference_carried_to_another_server_is_cleared(self):
        cfg = _config()  # rerank's key comes from ${RR_KEY} in config/models.yaml
        n, warnings = self._load(cfg, {"rag.rerank.base_url": EVIL})
        assert n == 1 and cfg.rag.rerank.api_key is None
        assert any("rag.rerank.api_key cleared" in w for w in warnings)

    def test_a_stored_reference_is_checked_against_the_stored_server(self, monkeypatch):
        """config/models.yaml pairs EMB_KEY with a second server only; rerank was moved there and
        given ${EMB_KEY}. Whatever the row order, the key row is checked after the
        base_url row it was set with."""
        monkeypatch.setenv("EMB_KEY", "sk-emb")
        cfg = _config(embedding={"provider": "openai-compatible", "model": "emb", "dimension": 8,
                                 "base_url": "http://second:7/v1", "api_key": "k"})
        refs = {**REFS, "rag.embedding.api_key": "EMB_KEY"}
        n, warnings = self._load(cfg, {"rag.rerank.api_key": "${EMB_KEY}",
                                       "rag.rerank.base_url": "http://second:7"}, refs=refs)
        assert n == 2 and cfg.rag.rerank.api_key == "sk-emb", warnings

    def test_a_skipped_key_row_is_never_logged(self, monkeypatch):
        monkeypatch.setattr("src.config.config_manager.Config.get_env_refs", classmethod(
            lambda cls: {p: "LLM_KEY" for p, _ in __import__(
                "src.config.profile", fromlist=["x"]).iter_secret_fields(_config().rag)}))
        cfg = _config(profile="cloud")
        refs = {p: "LLM_KEY" for p in ro.SECRET_PATHS}
        n, warnings = self._load(cfg, {"rag.rerank.api_key": "sk-pasted-secret"}, refs=refs)
        assert n == 0 and warnings and not any("sk-pasted-secret" in w for w in warnings)

    def test_pairs_are_taken_before_the_overrides(self):
        cfg = _config()
        ro._startup_pairs = {"STALE": {"x"}}  # e.g. from an earlier config: replaced at startup
        self._load(cfg, {"rag.rerank.base_url": EVIL})
        assert ro._startup_pairs == {"LLM_KEY": {origin_of(HOME)}, "RR_KEY": {origin_of(HOME)}}


class TestService:
    def test_reset_stores_a_key_it_cleared(self):
        from src.adapter import runtime_settings_service as svc
        cfg = _cfg()
        ro.validate_and_apply(cfg, {"rag.rerank.base_url": EVIL, "rag.rerank.api_key": "their-own"})
        with patch(f"{_SVC}.Config.get_config_model", return_value=cfg), \
                patch(f"{_SVC}._yaml_value", return_value=HOME), \
                patch(f"{_SVC}.RuntimeSettingsDB") as db, patch(f"{_SVC}._rebind") as rb, \
                patch(f"{_SVC}._audit") as audit:
            db.delete.return_value = True
            assert svc.reset_setting("rag.rerank.base_url") is True
        assert cfg.rag.rerank.api_key is None  # their key does not go to the config server
        db.upsert.assert_called_once_with("rag.rerank.api_key", None, updated_by=None)
        rb.assert_called_once_with(["rag.rerank.base_url", "rag.rerank.api_key"])
        assert audit.call_count == 2

    def test_cleared_key_is_audited_as_cleared(self):
        from src.adapter import runtime_settings_service as svc
        assert svc._audit_value("rag.rerank.api_key", None) is None

    def test_settings_api_returns_the_warning_and_a_generic_422(self, as_identity, make_identity):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.api.router.admin_settings import router
        app = FastAPI()
        app.include_router(router)
        c = TestClient(as_identity(app, make_identity("sub:admin", admin=True)))
        cfg = _cfg()
        with patch(f"{_SVC}.Config.get_config_model", return_value=cfg), \
                patch(f"{_SVC}.RuntimeSettingsDB"), patch(f"{_SVC}._rebind"), patch(f"{_SVC}._audit"):
            r = c.put("/api/admin/settings", json={"settings": {"rag.rerank.base_url": EVIL}})
            assert r.status_code == 200, r.text
            assert any("Enter the key" in w for w in r.json()["data"]["warnings"])
            r = c.put("/api/admin/settings", json={"settings": {"rag.rerank.api_key": "${DB_PASSWORD}"}})
            assert r.status_code == 422 and "DB_PASSWORD" not in r.text and "hunter2" not in r.text


class TestResetKey:
    def test_reset_restores_the_config_reference_under_the_cloud_profile(self, tmp_path, monkeypatch):
        """Before 1.4.2 resetting a key sent the expanded key, which profile: cloud refused (422)."""
        from src.adapter import runtime_settings_service as svc
        from src.config.config_manager import Config
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text('rag:\n  rerank:\n    api_key: "${RR_KEY}"\n', encoding="utf-8")
        refs = {p: "RR_KEY" for p in ro.SECRET_PATHS}
        monkeypatch.setattr("src.config.config_manager.Config.get_env_refs", classmethod(lambda cls: refs))
        cfg = _config(profile="cloud")
        ro.capture_startup(cfg, refs=refs)
        ro.validate_and_apply(cfg, {"rag.rerank.api_key": "${RR_KEY}"})
        cfg.rag.rerank.api_key = "changed"
        with patch(f"{_SVC}.Config.get_config_model", return_value=cfg), \
                patch.object(Config, "_config_dir", config_dir_from_file(yaml_file)), \
                patch(f"{_SVC}.RuntimeSettingsDB") as db, patch(f"{_SVC}._rebind"), patch(f"{_SVC}._audit"):
            db.delete.return_value = True
            assert svc.reset_setting("rag.rerank.api_key") is True
        assert cfg.rag.rerank.api_key == "sk-rr" and ro.secret_ref("rag.rerank.api_key") == "RR_KEY"

    def test_only_whole_references_are_returned(self, tmp_path, monkeypatch):
        from src.adapter import runtime_settings_service as svc
        from src.config.config_manager import Config
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text('rag:\n  rerank:\n    api_key: "x-${RR_KEY}"\n    model: "${M}"\n',
                             encoding="utf-8")
        monkeypatch.setattr("src.config.config_manager.Config.get_env_refs", classmethod(
            lambda cls: {"rag.rerank.api_key": "RR_KEY", "rag.rerank.model": "M"}))
        with patch.object(Config, "_config_dir", config_dir_from_file(yaml_file)):
            assert svc._yaml_reference("rag.rerank.api_key") is None
            assert svc._yaml_reference("rag.rerank.model") is None  # not a key
