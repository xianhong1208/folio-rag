"""The Docker image and compose file (#6): the image's cloud config, auth.internal_url (MCP Gate reached at
another address than the issuer), FOLIO_CONFIG_DIR, FOLIO_LOG_DIR / NO_COLOR and FOLIO_SEGMENTER.

scripts/docker_smoke.sh runs the real image; these tests pin the pieces it relies on."""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from loguru import logger

from src.config import layout
from src.config.config_manager import Config, load_config_dir

REPO = Path(__file__).resolve().parents[1]
DOCKER_MODELS = REPO / "docker" / "config" / "models.yaml"
# The variables docker/config/models.yaml reads (comments left out)
IMAGE_VARS = sorted(set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)", "\n".join(
    line.split("#", 1)[0] for line in DOCKER_MODELS.read_text(encoding="utf-8").splitlines()))))
ALL_VARS = sorted({s.var for s in layout.ENV_SETTINGS.values()} | set(IMAGE_VARS))


@pytest.fixture(autouse=True)
def _reset_config_state():
    yield
    Config._config = {}
    Config._config_model = None
    Config._config_dir = None
    Config._env_refs = {}
    Config._startup_notices = []


@pytest.fixture()
def clean_env(monkeypatch):
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _image_config_dir(tmp_path: Path) -> Path:
    """config/ as the Dockerfile lays it out: the repository's six files, docker/config/models.yaml over models.yaml."""
    target = tmp_path / "config"
    shutil.copytree(REPO / "config", target, ignore=shutil.ignore_patterns("config.yaml", "_itest"))
    shutil.copy(DOCKER_MODELS, target / "models.yaml")
    return target


def _auth_config(**auth):
    base = dict(enabled=True, issuer="http://localhost:4568/", audience="http://localhost:5039/mcp",
                required_scopes=[])
    base.update(auth)
    return SimpleNamespace(auth=SimpleNamespace(**base), server=SimpleNamespace(host="0.0.0.0", port=5039))


# -- the image's configuration ------------------------------------------------------------------------------------

class TestImageConfig:
    def test_loads_as_profile_cloud_with_no_variable_set(self, tmp_path, clean_env):
        """The image starts from its own files alone: a valid cloud config, nothing in-process."""
        cfg_dir = _image_config_dir(tmp_path)
        clean_env.chdir(tmp_path)
        clean_env.setenv("FOLIO_SEGMENTER", "jieba")  # the Dockerfile's ENV
        Config.set_config("config")
        cfg = Config.get_config_model()
        assert cfg is not None, "the image's config does not validate"
        rag = cfg.rag
        assert rag.profile == "cloud"
        assert rag.retrieval.segmenter == "jieba"
        assert rag.embedding.provider == "openai-compatible" and rag.embedding.dimension == 1024
        assert rag.parsing.vision.model and rag.asr.enabled is False
        assert rag.rerank.enabled is True and rag.rerank.api == "generic"
        from src.config.roles import docling_warmup_needed, needs_local_docling_models
        assert not needs_local_docling_models(rag) and not docling_warmup_needed(rag)
        assert cfg_dir.is_dir()

    def test_endpoints_and_models_come_from_the_environment(self, tmp_path, clean_env):
        _image_config_dir(tmp_path)
        clean_env.chdir(tmp_path)
        for var, value in {"EMBEDDING_BASE_URL": "http://models:8000/v1", "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
                           "EMBEDDING_DIMENSION": "256", "LLM_MODEL": "m", "RERANK_ENABLED": "false",
                           "VISION_MODEL": "v", "ASR_ENABLED": "true", "ASR_BASE_URL": "http://a/v1",
                           "ASR_MODEL": "whisper", "FOLIO_SEGMENTER": "bigram"}.items():
            clean_env.setenv(var, value)
        Config.set_config("config")
        rag = Config.get_config_model().rag
        assert (rag.embedding.base_url, rag.embedding.model, rag.embedding.dimension) == (
            "http://models:8000/v1", "Qwen/Qwen3-Embedding-0.6B", 256)
        assert (rag.llm.model, rag.rerank.enabled, rag.parsing.vision.model) == ("m", False, "v")
        assert (rag.asr.enabled, rag.asr.base_url, rag.asr.model) == (True, "http://a/v1", "whisper")
        assert rag.retrieval.segmenter == "bigram"

    def test_every_key_is_an_env_reference(self, tmp_path, clean_env):
        """profile cloud refuses literal keys; every key in the image's file is a ${VAR} reference."""
        raw, _ = load_config_dir(_image_config_dir(tmp_path))
        for name in ("embedding", "llm", "rerank", "asr"):
            assert str(raw["rag"][name]["api_key"]).startswith("${"), name
        assert str(raw["rag"]["parsing"]["vision"]["api_key"]).startswith("${")

    def test_holds_only_models_yaml_sections(self):
        data = yaml.safe_load(DOCKER_MODELS.read_text(encoding="utf-8"))
        assert list(data) == ["rag"]
        for key in data["rag"]:
            assert layout.file_for(f"rag.{key}") == "models.yaml", key

    def test_env_example_lists_every_variable_of_the_image(self):
        text = (REPO / ".env.example").read_text(encoding="utf-8")
        assert "EMBEDDING_MODEL" in IMAGE_VARS and "VAR" not in IMAGE_VARS
        for var in IMAGE_VARS:
            assert re.search(rf"^#? ?{var}=", text, re.M), var

    def test_dockerfile_installs_the_cloud_profile_only(self):
        text = (REPO / "Dockerfile").read_text(encoding="utf-8")
        sync = next(line for line in text.splitlines() if line.startswith("RUN uv sync"))
        assert "--locked" in sync and "--no-dev" in sync and "--no-default-groups" in sync
        assert "--extra jieba" in sync and "--extra local" not in sync and "--group" not in sync
        assert "COPY docker/config/models.yaml config/models.yaml" in text
        assert re.search(r"^USER folio$", text, re.M) and 'VOLUME ["/app/storage"]' in text
        assert "HEALTHCHECK" in text and 'FOLIO_LOG_DIR=""' in text

    def test_compose_holds_no_secret(self):
        compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
        folio, db, gate = (compose["services"][n] for n in ("folio", "db", "mcp-gate"))
        assert "${POSTGRES_PASSWORD:?" in folio["environment"]["DATABASE_URL"]
        assert db["environment"]["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD:?")
        assert db["image"] == "pgvector/pgvector:pg16" and "healthcheck" in db
        assert gate["profiles"] == ["mcp-gate"] and "env_file" not in gate  # it never reads Folio's .env
        assert gate["environment"]["OAUTH_ISSUER"] == folio["environment"]["MCP_GATE_URL"]
        assert gate["environment"]["ALLOWED_HOSTS"] == "mcp-gate"
        for value in [*gate["environment"].values(), *folio["environment"].values()]:
            assert "${" in value or value == "mcp-gate", value


# -- auth.internal_url ----------------------------------------------------------------------------------------------

class TestInternalUrl:
    def test_unset_fetches_from_the_issuer(self):
        from src.auth.mcp_gate_auth import settings_from_config
        s = settings_from_config(_auth_config())
        assert s.issuer == "http://localhost:4568" and s.internal_url is None
        assert s.jwks_uri == "http://localhost:4568/.well-known/jwks.json"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_blank_means_unset(self, value):
        from src.auth.mcp_gate_auth import settings_from_config
        assert settings_from_config(_auth_config(internal_url=value)).fetch_base == "http://localhost:4568"

    def test_set_fetches_keys_there_and_keeps_the_issuer(self):
        from src.auth.mcp_gate_auth import settings_from_config
        s = settings_from_config(_auth_config(internal_url="http://mcp-gate:4568/"))
        assert s.jwks_uri == "http://mcp-gate:4568/.well-known/jwks.json"
        assert s.issuer == "http://localhost:4568"  # tokens are still checked against the issuer

    def test_the_mcp_transport_verifies_against_the_issuer_with_keys_from_internal_url(self):
        from src.auth.mcp_gate_auth import build_auth_provider
        provider = build_auth_provider(_auth_config(internal_url="http://mcp-gate:4568"))
        verifier = provider.token_verifier
        assert verifier.jwks_uri == "http://mcp-gate:4568/.well-known/jwks.json"
        assert verifier.issuer == "http://localhost:4568"
        assert [str(a) for a in provider.authorization_servers] == ["http://localhost:4568"]

    def test_the_revocation_feed_is_polled_at_internal_url(self, monkeypatch):
        from src.auth import init_auth, revocation as rv
        from src.config.model import RevocationConfig
        monkeypatch.setattr(Config, "get_env_refs", classmethod(
            lambda cls: {"auth.revocation.client_secret": "FOLIO_REVOCATION_CLIENT_SECRET"}))
        monkeypatch.setattr(init_auth, "McpGateTokenVerifier", lambda settings: object())
        monkeypatch.setattr(init_auth, "set_token_verifier", lambda v: None)
        monkeypatch.setattr(init_auth, "set_console_auth", lambda a: None)
        monkeypatch.setattr(init_auth, "build_console_auth", lambda *a, **k: None)
        cfg = _auth_config(internal_url="http://mcp-gate:4568",
                           revocation=RevocationConfig(enabled=True, client_id="c", client_secret="s"))
        try:
            init_auth.initialize_auth_system(cfg)
            assert rv.get_revocation_list().settings.feed_url == "http://mcp-gate:4568/oauth/revoked"
        finally:
            rv.set_revocation_list(None)

    def test_discovery_documents_are_fetched_at_internal_url(self):
        from src.api.router import oauth_discovery as od
        from src.auth.mcp_gate_auth import settings_from_config
        urls = []

        class Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                urls.append(url)
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"issuer": "http://localhost:4568"})

        import asyncio
        od.clear_cache()
        try:
            with patch.object(od.httpx, "AsyncClient", Client):
                doc = asyncio.run(od.fetch_issuer_document(
                    settings_from_config(_auth_config(internal_url="http://mcp-gate:4568")),
                    "oauth-authorization-server"))
        finally:
            od.clear_cache()
        assert urls == ["http://mcp-gate:4568/.well-known/oauth-authorization-server"]
        assert doc == {"issuer": "http://localhost:4568"}

    def test_config_model_reads_blank_as_unset(self):
        from src.config.model import AuthConfig
        assert AuthConfig(enabled=True, internal_url="").internal_url is None
        assert AuthConfig(enabled=True, internal_url="http://mcp-gate:4568").internal_url == "http://mcp-gate:4568"


# -- the public address OAuth clients are sent to ------------------------------------------------------------------

class TestAdvertisedAddress:
    """In a container (or behind a proxy) the bind address is not the public one: the RFC 9728 document and the 401
    challenge must name the audience's address, or OAuth clients look for the metadata at http://127.0.0.1:5039."""

    def _metadata(self, config):
        from starlette.applications import Starlette
        from starlette.testclient import TestClient

        from src.auth.mcp_gate_auth import build_auth_provider
        routes = [r for r in build_auth_provider(config).get_routes("/mcp") if "protected-resource" in r.path]
        assert [r.path for r in routes] == ["/.well-known/oauth-protected-resource/mcp"]
        return TestClient(Starlette(routes=routes)).get(routes[0].path).json()

    @pytest.mark.parametrize("audience", ["https://folio.example.com/mcp", "http://localhost:15040/mcp/"])
    def test_the_audience_is_the_advertised_resource(self, audience):
        from src.auth.mcp_gate_auth import settings_from_config
        config = _auth_config(audience=audience)
        assert settings_from_config(config).base_url == audience.rstrip("/")[: -len("/mcp")]
        assert self._metadata(config)["resource"] == audience.rstrip("/")

    def test_the_challenge_points_at_the_public_metadata(self):
        from src.auth.mcp_gate_auth import build_auth_provider
        provider = build_auth_provider(_auth_config(audience="https://folio.example.com/mcp"))
        assert str(provider._get_resource_url("/mcp")) == "https://folio.example.com/mcp"
        assert str(provider.base_url).rstrip("/") == "https://folio.example.com"

    def test_an_audience_without_the_mcp_path_keeps_the_bind_address(self):
        from src.auth.mcp_gate_auth import settings_from_config
        assert settings_from_config(_auth_config(audience="urn:folio")).base_url == "http://127.0.0.1:5039"
        assert settings_from_config(_auth_config(audience=None)).base_url == "http://127.0.0.1:5039"


# -- FOLIO_SEGMENTER ------------------------------------------------------------------------------------------------

_RETRIEVAL = dict(hybrid_search=True, default_top_k=10, default_similarity_cutoff=0.25)


class TestSegmenterVariable:
    @pytest.mark.parametrize("value, expected", [("", None), ("  ", None), ("jieba", "jieba"), ("bigram", "bigram")])
    def test_blank_is_the_profile_default(self, value, expected):
        from src.config.model import RAGRetrievalConfig
        assert RAGRetrievalConfig(**_RETRIEVAL, segmenter=value).segmenter == expected

    def test_unknown_segmenter_is_still_refused(self):
        from pydantic import ValidationError
        from src.config.model import RAGRetrievalConfig
        with pytest.raises(ValidationError):
            RAGRetrievalConfig(**_RETRIEVAL, segmenter="spaces")


# -- main.py: FOLIO_CONFIG_DIR and FOLIO_LOG_DIR --------------------------------------------------------------------

def _main():
    sys.path.insert(0, str(REPO))
    import main
    return main


class TestMainEnvironment:
    def test_config_dir_default(self, monkeypatch):
        monkeypatch.delenv("FOLIO_CONFIG_DIR", raising=False)
        assert _main().default_config_dir() == "config"
        monkeypatch.setenv("FOLIO_CONFIG_DIR", " ")
        assert _main().default_config_dir() == "config"

    def test_config_dir_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("FOLIO_CONFIG_DIR", "my-config")
        assert _main().default_config_dir() == "my-config"

    def test_the_cli_flag_defaults_to_it(self, monkeypatch):
        main = _main()
        monkeypatch.setenv("FOLIO_CONFIG_DIR", "from-env")
        seen = {}
        monkeypatch.setattr(main.Config, "set_config", lambda d: seen.setdefault("dir", d))
        monkeypatch.setattr(main.Config, "get_config_model", lambda: None)
        monkeypatch.setattr(sys, "argv", ["main.py"])
        with pytest.raises(SystemExit):
            main.main()
        assert seen["dir"] == "from-env"

    def test_main_hands_the_log_directory_to_the_logger(self, monkeypatch):
        main = _main()
        monkeypatch.setenv("FOLIO_LOG_DIR", "")
        monkeypatch.setattr(main.Config, "set_config", lambda d: None)
        monkeypatch.setattr(main.Config, "get_config_model", lambda: object())
        monkeypatch.setattr(main.Config, "get_logging_config",
                            lambda: SimpleNamespace(level="INFO", max_size="1 MB", backup_count=1))
        seen = {}

        def fake_setup_logger(**kwargs):
            seen.update(kwargs)
            raise SystemExit(3)

        monkeypatch.setattr(main, "setup_logger", fake_setup_logger)
        monkeypatch.setattr(sys, "argv", ["main.py"])
        with pytest.raises(SystemExit):
            main.main()
        assert seen["log_base_dir"] == ""

    @pytest.mark.parametrize("value, expected", [(None, "logs"), ("", ""), ("  ", ""), ("/var/log/folio", "/var/log/folio")])
    def test_log_directory(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv("FOLIO_LOG_DIR", raising=False)
        else:
            monkeypatch.setenv("FOLIO_LOG_DIR", value)
        assert _main().log_directory() == expected


class TestConsoleOnlyLogging:
    def _setup(self, monkeypatch, log_dir, no_color=None):
        from src import log
        added = []
        real_add = logger.add
        monkeypatch.setattr(log.logger, "add", lambda sink, **kw: added.append((sink, kw)) or real_add(sink, **kw))
        if no_color is None:
            monkeypatch.delenv("NO_COLOR", raising=False)
        else:
            monkeypatch.setenv("NO_COLOR", no_color)
        try:
            log.setup_logger(console_level="CRITICAL", log_base_dir=log_dir)
        finally:
            monkeypatch.undo()
            logger.remove()
            logger.add(sys.stderr, level="WARNING")
        return added

    def test_empty_log_dir_writes_no_files(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        added = self._setup(monkeypatch, "")
        assert [sink for sink, _ in added] == [sys.stderr]
        assert os.listdir(tmp_path) == []

    def test_a_log_dir_still_writes_files(self, monkeypatch, tmp_path):
        added = self._setup(monkeypatch, str(tmp_path / "logs"))
        assert sum(isinstance(sink, str) for sink, _ in added) >= 2
        assert (tmp_path / "logs" / "server").is_dir()

    def test_no_color_turns_colour_codes_off(self, monkeypatch):
        console = [kw for sink, kw in self._setup(monkeypatch, "", no_color="1") if sink is sys.stderr]
        assert console[0]["colorize"] is False

    def test_colour_by_default(self, monkeypatch):
        console = [kw for sink, kw in self._setup(monkeypatch, "") if sink is sys.stderr]
        assert console[0]["colorize"] is True
