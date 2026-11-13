"""Unit tests for src/config (model.py + config_manager.py).

Pure logic; YAML is written to tmp_path and monkeypatch.chdir sets the path
validation base directory.
"""

import pytest
from pydantic import ValidationError

from src.config.model import (
    ServerConfig,
    DatabaseConfig,
    DynamicToolsConfig,
    AuthConfig,
    AppConfig,
    ModulesConfig,
    RAGChunkingConfig,
    AutoMergingConfig,
    RAGRetrievalConfig,
    ContextualRetrievalConfig,
    RerankConfig,
    IndexingConfig,
    RAGConfig,
    ConfigModel,
)
from src.config.config_manager import Config, ConfigFileError, get_config
from tests.config_helpers import write_config_dir


# Shared fixtures / helpers

@pytest.fixture(autouse=True)
def _reset_config_state():
    """Reset Config class-level state before and after each test, to avoid cross-test pollution"""
    Config._config = {}
    Config._config_model = None
    Config._config_dir = None
    yield
    Config._config = {}
    Config._config_model = None
    Config._config_dir = None


VALID_YAML = """\
server:
  host: 127.0.0.1
  port: 8080
  transport: sse
database:
  url: "postgresql://u:p@localhost:5432/db"
  echo: false
  pool_size: 5
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 3600
auth:
  enabled: false
logging:
  level: INFO
  format: "{time} {message}"
  file: null
  max_size: 10MB
  backup_count: 3
app:
  name: TestApp
modules:
  enabled: []
"""


def _write_config(tmp_path, monkeypatch, text, name="config"):
    """Write the tree as the six files under tmp_path/<name> and chdir to tmp_path
    (the config dir must be inside the working directory)."""
    monkeypatch.chdir(tmp_path)
    write_config_dir(tmp_path / name, text)
    return name


# config/model.py -- representative Pydantic models

def test_server_config_required_and_coercion():
    """ServerConfig builds with all required fields; a string port is auto-coerced to int"""
    cfg = ServerConfig(host="0.0.0.0", port="8080", transport="sse")
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8080
    assert cfg.transport == "sse"


def test_server_config_missing_required():
    """ServerConfig missing any of the required host/port/transport -> ValidationError"""
    with pytest.raises(ValidationError) as exc_info:
        ServerConfig(host="0.0.0.0")
    missing = {e["loc"][0] for e in exc_info.value.errors()}
    assert missing == {"port", "transport"}


def test_database_config_optional_defaults():
    """DatabaseConfig host/port/database/user/password default to None"""
    cfg = DatabaseConfig(
        url="sqlite://", echo=False, pool_size=5,
        max_overflow=10, pool_timeout=30, pool_recycle=3600,
    )
    assert cfg.host is None
    assert cfg.port is None
    assert cfg.database is None
    assert cfg.user is None
    assert cfg.password is None


def test_auth_config_defaults():
    """AuthConfig with only enabled given; the rest of the connection settings use defaults"""
    cfg = AuthConfig(enabled=True)
    assert cfg.token_server_url is None
    assert cfg.cache_ttl == 60
    assert cfg.request_timeout == 5.0
    assert cfg.retry_count == 2
    assert cfg.dynamic_tools is None


def test_auth_config_missing_enabled():
    """AuthConfig missing required enabled -> ValidationError"""
    with pytest.raises(ValidationError):
        AuthConfig()


def test_dynamic_tools_config_default_disabled():
    """DynamicToolsConfig.enabled defaults to False"""
    assert DynamicToolsConfig().enabled is False


def test_app_config_all_defaults():
    """AppConfig built with no arguments uses all defaults"""
    cfg = AppConfig()
    assert cfg.name == "Folio"
    # The version field was removed from AppConfig -- the single source of the
    # version is pyproject.toml (src/version.py get_version()); config no longer
    # carries the version, to avoid dual-source drift
    assert not hasattr(cfg, "version")
    assert cfg.title == "Folio MCP Server"
    assert cfg.lifespan is None
    assert cfg.tool_prefix == "Folio"


def test_rag_chunking_config_defaults():
    """RAGChunkingConfig flat fields default to None; max_chunk_size=2000, chunk_overlap=50"""
    cfg = RAGChunkingConfig()
    assert cfg.default_chunk_size is None
    assert cfg.default_overlap is None
    assert cfg.max_chunk_size == 2000
    assert cfg.hierarchy_sizes is None
    assert cfg.chunk_overlap == 50


def test_auto_merging_config_defaults():
    """AutoMergingConfig defaults enabled=True, merge_threshold=0.5"""
    cfg = AutoMergingConfig()
    assert cfg.enabled is True
    assert cfg.merge_threshold == 0.5


def test_rag_retrieval_config_required_and_defaults():
    """RAGRetrievalConfig three required fields + the rest defaulted (sparse 12 / alpha 0.75 / simple, etc.)"""
    cfg = RAGRetrievalConfig(
        default_top_k=10, default_similarity_cutoff=0.25, hybrid_search=True,
    )
    assert cfg.default_sparse_top_k == 12
    assert cfg.default_hybrid_alpha == 0.75
    assert cfg.text_search_config == "simple"
    assert cfg.return_resource_files is False
    assert cfg.auto_merging is None
    assert cfg.expand_context_default is True
    assert cfg.expand_context_neighbors == 1
    assert cfg.mcp_response_token_budget == 6000
    assert cfg.resource_file_max_bytes == 1_000_000


def test_contextual_retrieval_config_defaults():
    """ContextualRetrievalConfig defaults enabled=False, apply_to='all'"""
    cfg = ContextualRetrievalConfig()
    assert cfg.enabled is False
    assert cfg.max_context_length == 150
    assert cfg.max_concurrent == 5
    assert cfg.apply_to == "all"


def test_rerank_config_defaults():
    """RerankConfig defaults disabled, bge-reranker model, threshold 0.0"""
    cfg = RerankConfig()
    assert cfg.enabled is False
    assert cfg.model == "BAAI/bge-reranker-v2-m3"
    assert cfg.base_url == "http://localhost:8787"
    assert cfg.top_n is None
    assert cfg.score_threshold == 0.0


def test_indexing_config_defaults():
    """IndexingConfig defaults: high concurrency / 10-day per-file / 10-day job cap

    The timeout defaults were relaxed to 10 days (for OCR of large files on slow
    machines; watchdog false positives are backstopped by a 60s keepalive
    heartbeat).
    """
    cfg = IndexingConfig()
    assert cfg.max_concurrent_jobs == 4
    assert cfg.per_file_timeout_seconds == 864000
    assert cfg.job_timeout_seconds == 864000


def test_rag_config_nested_assembly():
    """RAGConfig assembles sub-models from nested dicts; optional sub-items default to None"""
    cfg = RAGConfig(
        embedding={"provider": "vllm", "model": "bge-m3", "dimension": 1024},
        chunking={"hierarchy_sizes": [1024, 256]},
        retrieval={
            "default_top_k": 10,
            "default_similarity_cutoff": 0.25,
            "hybrid_search": True,
            "auto_merging": {"enabled": True, "merge_threshold": 0.6},
        },
        vector_store={"type": "pgvector", "table_prefix": "rag_"},
    )
    assert cfg.enabled is True
    assert cfg.embedding.provider == "vllm"
    assert cfg.chunking.hierarchy_sizes == [1024, 256]
    assert isinstance(cfg.retrieval.auto_merging, AutoMergingConfig)
    assert cfg.retrieval.auto_merging.merge_threshold == 0.6
    assert cfg.llm is None
    assert cfg.rerank is None
    assert cfg.docling is None
    assert cfg.indexing is None


def test_config_model_full_tree():
    """ConfigModel main tree assembles from nested dicts; rag defaults to None"""
    cfg = ConfigModel(
        server={"host": "h", "port": 1, "transport": "sse"},
        database={
            "url": "sqlite://", "echo": False, "pool_size": 1,
            "max_overflow": 1, "pool_timeout": 1, "pool_recycle": 1,
        },
        auth={"enabled": False},
        logging={
            "level": "INFO", "format": "f", "file": None,
            "max_size": "10MB", "backup_count": 1,
        },
        app={},
        modules={"enabled": ["rag", "admin"]},
    )
    assert isinstance(cfg.server, ServerConfig)
    assert cfg.modules.enabled == ["rag", "admin"]
    assert cfg.rag is None


def test_config_model_missing_section():
    """ConfigModel missing the server section -> ValidationError"""
    with pytest.raises(ValidationError) as exc_info:
        ConfigModel(
            database={
                "url": "sqlite://", "echo": False, "pool_size": 1,
                "max_overflow": 1, "pool_timeout": 1, "pool_recycle": 1,
            },
            auth={"enabled": False},
            logging={
                "level": "INFO", "format": "f", "file": None,
                "max_size": "10MB", "backup_count": 1,
            },
            app={},
            modules={"enabled": []},
        )
    assert any(e["loc"][0] == "server" for e in exc_info.value.errors())


def test_modules_config_rejects_wiring_and_unknown_modules():
    """modules holds only `enabled`: wiring lives in code, and an unknown name is refused"""
    assert ModulesConfig(enabled=["rag", "admin"]).enabled == ["rag", "admin"]
    with pytest.raises(ValidationError, match="Extra inputs"):
        ModulesConfig(enabled=["rag"], rag={"description": "x"})
    with pytest.raises(ValidationError, match="unknown module"):
        ModulesConfig(enabled=["demo"])


# config/config_manager.py -- _validate_config_dir

def test_validate_config_dir_allows_dirs_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Config._validate_config_dir("config").name == "config"
    assert Config._validate_config_dir("sub/dir").name == "dir"


def test_validate_config_dir_rejects_traversal(tmp_path, monkeypatch):
    """../ path traversal outside cwd -> ValueError"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside the allowed base directory"):
        Config._validate_config_dir("../evil")


def test_validate_config_dir_rejects_absolute_outside(tmp_path, monkeypatch):
    """An absolute path outside cwd -> ValueError"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside the allowed base directory"):
        Config._validate_config_dir("/etc")


# config/config_manager.py -- get_config / set_config lifecycle

def test_get_config_valid_yaml(tmp_path, monkeypatch):
    """Valid YAML -> returns a ConfigModel, getters can access each section"""
    name = _write_config(tmp_path, monkeypatch, VALID_YAML)
    model = get_config(name)
    assert isinstance(model, ConfigModel)
    assert model.server.port == 8080
    assert model.app.name == "TestApp"
    # class-level getters agree with the convenience functions
    assert Config.get_server_config().host == "127.0.0.1"
    assert Config.get_auth_config().enabled is False
    assert Config.get_auth_config().cache_ttl == 60  # AuthConfig default filled in
    assert Config.get_logging_config().level == "INFO"
    assert Config.get_modules_config().enabled == []
    assert Config.get_config()["server"]["port"] == 8080


def test_get_config_env_var_placeholder_expanded(tmp_path, monkeypatch):
    """`${VAR:-default}` is expanded from the environment at load time.

    config_manager.expand_env_vars runs over the parsed YAML, so the DB URL can
    come from the environment while the config files keep a safe default.
    """
    monkeypatch.setenv("TEST_DB_URL", "postgresql://real/db")
    yaml_text = VALID_YAML.replace(
        'url: "postgresql://u:p@localhost:5432/db"',
        'url: "${TEST_DB_URL:-postgresql://fallback/db}"',
    )
    name = _write_config(tmp_path, monkeypatch, yaml_text)
    model = get_config(name)
    assert model.database.url == "postgresql://real/db"


def test_get_config_env_var_placeholder_uses_default_when_unset(tmp_path, monkeypatch):
    """With the variable unset, the `:-default` part is used."""
    monkeypatch.delenv("TEST_DB_URL", raising=False)
    yaml_text = VALID_YAML.replace(
        'url: "postgresql://u:p@localhost:5432/db"',
        'url: "${TEST_DB_URL:-postgresql://fallback/db}"',
    )
    name = _write_config(tmp_path, monkeypatch, yaml_text)
    model = get_config(name)
    assert model.database.url == "postgresql://fallback/db"


def test_get_config_invalid_schema_returns_none(tmp_path, monkeypatch):
    """YAML missing a required section (Pydantic validation fails) -> get_config returns None"""
    name = _write_config(tmp_path, monkeypatch, "server:\n  host: only-host\n")
    assert get_config(name) is None
    assert Config.get_config_model() is None


def test_get_config_malformed_yaml_raises_naming_the_file(tmp_path, monkeypatch):
    """A YAML syntax error stops loading with the file's name"""
    name = _write_config(tmp_path, monkeypatch, VALID_YAML)
    (tmp_path / name / "models.yaml").write_text("rag: [unclosed\n  :::", encoding="utf-8")
    with pytest.raises(ConfigFileError, match="models.yaml is not valid YAML"):
        get_config(name)


def test_get_config_missing_dir_raises_listing_every_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigFileError) as e:
        get_config("no_such_dir")
    for f in ("database.yaml", "auth.yaml", "models.yaml", "mcp.yaml", "retrieval.yaml", "system.yaml"):
        assert f in str(e.value)


def test_get_module_model_comes_from_the_registry(tmp_path, monkeypatch):
    """An enabled module's wiring comes from src/config/modules.py; a disabled one is None"""
    yaml_text = VALID_YAML.replace("modules:\n  enabled: []\n", "modules:\n  enabled: [rag_query, rag]\n")
    name = _write_config(tmp_path, monkeypatch, yaml_text)
    Config.set_config(name)
    q = Config.get_module_model("rag_query")
    assert q.api_router.module == "src.api.router.rag_query" and q.api_router.prefix == "/api/rag"
    assert Config.get_module_model("rag").mcp_tools.function_name == "register_agentic_tools"
    assert Config.get_module_model("admin") is None  # known but not enabled
    assert Config.get_module_model("nope") is None


def test_every_registered_module_imports():
    """The registry's import paths and names exist (a typo would 404 a whole API)"""
    import importlib
    from src.config.modules import DEFAULT_ENABLED, MODULES
    assert set(DEFAULT_ENABLED) == set(MODULES)
    for name, m in MODULES.items():
        if m.api_router:
            assert hasattr(importlib.import_module(m.api_router.module), m.api_router.router_name), name
        if m.mcp_tools:
            assert hasattr(importlib.import_module(m.mcp_tools.module), m.mcp_tools.function_name), name


def test_getters_return_none_before_load():
    """Before config is loaded (model is None), all getters return None"""
    assert Config.get_config_model() is None
    assert Config.get_server_config() is None
    assert Config.get_database_config() is None
    assert Config.get_auth_config() is None
    assert Config.get_logging_config() is None
    assert Config.get_app_config_model() is None
    assert Config.get_modules_config() is None
    assert Config.get_module_model("any") is None


def test_config_dir_outside_cwd_rejected(tmp_path, monkeypatch):
    """Config dir safety check: a dir outside the working directory raises (path-traversal defense)."""
    write_config_dir(tmp_path / "outside", VALID_YAML)
    monkeypatch.chdir(tmp_path / "outside")
    with pytest.raises(ValueError, match="Security Error"):
        Config.set_config(str(tmp_path))


def test_asr_config_defaults():
    """AsrConfig minimal field set: enabled + provider + the cloud fields, nothing else

    Deliberately lean: every config field is a permanent maintenance surface;
    trimming/filtering is always on in code, and FireRedASR-specific fields are
    added only when the FireRedASR provider is implemented. The cloud profile added the four fields
    chunked cloud transcription needs (language hint, segment length, timeout, retries)."""
    from src.config.model import AsrConfig
    cfg = AsrConfig()
    assert cfg.enabled is True
    assert cfg.provider == "docling-whisper"
    assert cfg.base_url is None and cfg.api_key is None and cfg.model is None
    # Guard the lean set: the removed fields must not quietly grow back
    assert set(AsrConfig.model_fields) == {
        "enabled", "provider", "base_url", "api_key", "model",
        "language", "segment_seconds", "timeout", "max_retries"}


_RAG_REQUIRED = dict(
    embedding={"provider": "vllm", "model": "m", "dimension": 8},
    chunking={},
    retrieval={"default_top_k": 5, "default_similarity_cutoff": 0.2, "hybrid_search": True},
    vector_store={"type": "pgvector", "table_prefix": "rag_"},
)


def test_rag_config_asr_nested_assembly():
    """RAGConfig nested assembly of the asr section; defaults to None when omitted (backward compat)"""
    from src.config.model import AsrConfig
    cfg = RAGConfig(
        **_RAG_REQUIRED,
        asr={"enabled": False, "provider": "openai-compatible", "base_url": "http://x:1/v1"},
    )
    assert isinstance(cfg.asr, AsrConfig)
    assert cfg.asr.enabled is False
    assert cfg.asr.provider == "openai-compatible"
    assert cfg.asr.base_url == "http://x:1/v1"
    assert RAGConfig(**_RAG_REQUIRED).asr is None
