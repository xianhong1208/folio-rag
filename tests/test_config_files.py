"""The six config files (src/config/layout.py), the loader that merges them, .env, and the
1.4 -> 1.5 converter (src/config/split.py, scripts/split_config.py)."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

from src.config import layout
from src.config.config_manager import (Config, ConfigFileError, legacy_config_warning,
                                       load_config_dir)
from src.config.dotenv import load_dotenv, parse_dotenv, read_dotenv
from src.config.model import AsrConfig, DatabaseConfig, LLMRoleConfig, RAGEmbeddingConfig
from src.config.split import (check, diff_paths, legacy_config_dict, place_sections,
                              split_config_dict, split_legacy, write_outputs)
from tests.config_helpers import write_config_dir

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "tests" / "fixtures" / "legacy_config_1_4.yaml"  # config/config.yaml of 1.4.2

ALL_VARS = sorted({s.var for s in layout.ENV_SETTINGS.values()})


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
    """No per-machine variable leaks in from the developer's shell."""
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _legacy_raw():
    return yaml.safe_load(LEGACY.read_text(encoding="utf-8"))


def _env_without_config_vars():
    return {k: v for k, v in os.environ.items() if k not in ALL_VARS}


# -- layout ---------------------------------------------------------------------------

class TestLayout:
    def test_six_files(self):
        assert layout.CONFIG_FILES == ("database.yaml", "auth.yaml", "models.yaml", "mcp.yaml",
                                       "retrieval.yaml", "system.yaml")

    @pytest.mark.parametrize("path, expected", [
        ("database.url", "database.yaml"),
        ("auth.issuer", "auth.yaml"),
        ("auth.revocation.client_secret", "auth.yaml"),
        ("auth.dynamic_tools.enabled", "mcp.yaml"),       # longest prefix wins over "auth"
        ("app.tool_prefix", "mcp.yaml"),
        ("app.name", "system.yaml"),
        ("modules.enabled", "mcp.yaml"),
        ("rag.profile", "models.yaml"),
        ("rag.llm.base_url", "models.yaml"),
        ("rag.contextual_llm.model", "models.yaml"),
        ("rag.docling.device", "models.yaml"),
        ("rag.parsing.vision.api_key", "models.yaml"),
        ("rag.retrieval.segmenter", "retrieval.yaml"),
        ("rag.contextual_retrieval.enabled", "retrieval.yaml"),
        ("server.port", "system.yaml"),
        ("logging.level", "system.yaml"),
        ("nothing.here", None),
        ("rag.unknown", None),
    ])
    def test_file_for(self, path, expected):
        assert layout.file_for(path) == expected

    def test_every_env_setting_lives_in_a_file(self):
        for path in layout.ENV_SETTINGS:
            assert layout.file_for(path) is not None, path


# -- the committed files ------------------------------------------------------------

class TestCommittedFiles:
    def test_each_file_holds_only_its_own_sections(self):
        for name in layout.CONFIG_FILES:
            data = yaml.safe_load((REPO / "config" / name).read_text(encoding="utf-8")) or {}
            for path in _leaf_paths(data):
                assert layout.file_for(path) == name, f"{path} is in {name}"

    def test_every_per_machine_value_is_an_env_reference(self):
        """Secrets and per-machine values are ${VAR:-default} in the committed files, so
        every file under config/ is safe to commit."""
        raw, _ = load_config_dir(REPO / "config")
        refs = _refs(raw)
        for path, setting in layout.ENV_SETTINGS.items():
            if layout.file_for(path) and _has(raw, path):
                assert refs.get(path) == f"${{{setting.var}:-{setting.default}}}", path
        # the sections that are commented out in models.yaml use their documented variables
        text = (REPO / "config" / "models.yaml").read_text(encoding="utf-8")
        for var in ("CONTEXTUAL_LLM_BASE_URL", "CONTEXTUAL_LLM_API_KEY", "VISION_BASE_URL",
                    "VISION_API_KEY"):
            assert f"${{{var}:-}}" in text

    def test_no_literal_secret_in_committed_files(self):
        raw, _ = load_config_dir(REPO / "config")
        for path, value in _leaves(raw):
            if path.endswith(("api_key", "client_secret", "password")) or path == "database.url":
                assert isinstance(value, str) and value.startswith("${"), path

    def test_env_example_lists_every_variable(self):
        text = (REPO / ".env.example").read_text(encoding="utf-8")
        for var in ALL_VARS:
            assert f"{var}=" in text, var

    def test_committed_files_load(self, clean_env, monkeypatch):
        monkeypatch.chdir(REPO)
        Config.set_config("config")
        cfg = Config.get_config_model()
        assert cfg is not None
        assert cfg.database.port is None and cfg.rag.asr.base_url is None
        assert cfg.rag.llm.base_url == "http://localhost:5052/v1"
        assert cfg.auth.revocation.enabled is False and cfg.auth.dynamic_tools.enabled is True
        assert cfg.app.tool_prefix == "Folio" and cfg.app.name == "Folio"
        assert cfg.modules.enabled == list(__import__("src.config.modules", fromlist=["x"]).DEFAULT_ENABLED)
        assert not hasattr(cfg.rag, "answer_llm")
        assert "contextual_llm" in (REPO / "config" / "models.yaml").read_text(encoding="utf-8")

    def test_contextual_llm_comment_says_a_small_model_is_enough(self):
        text = (REPO / "config" / "models.yaml").read_text(encoding="utf-8")
        assert "one short sentence" in text and "small, fast model is enough" in text

    def test_committed_files_match_the_1_4_template(self, clean_env):
        """The six templates hold the same settings as 1.4's config.yaml, except the defaults
        that moved to .env (database port, devices) and the module wiring."""
        before = legacy_config_dict(_legacy_raw(), _env_without_config_vars())
        after = split_config_dict(REPO / "config", _env_without_config_vars())
        assert diff_paths(before, after) == ["database.port", "database.url"]


def _leaves(tree, prefix=""):
    for k, v in tree.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            yield from _leaves(v, p)
        else:
            yield p, v


def _leaf_paths(tree):
    return [p for p, _ in _leaves(tree)]


def _refs(raw):
    return {p: v for p, v in _leaves(raw) if isinstance(v, str) and "${" in v}


def _has(tree, path):
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


# -- the loader ---------------------------------------------------------------------

BASE = {
    "server": {"host": "127.0.0.1", "port": 8080, "transport": "http"},
    "database": {"url": "${T_DB:-postgresql://u:p@h/db}", "echo": False, "pool_size": 1,
                 "max_overflow": 1, "pool_timeout": 1, "pool_recycle": 1},
    "auth": {"enabled": False, "dynamic_tools": {"enabled": True}},
    "logging": {"level": "INFO", "format": "x", "file": None, "max_size": "1MB", "backup_count": 1},
    "app": {"name": "T", "tool_prefix": "Fo"},
    "modules": {"enabled": ["rag"]},
}


class TestLoader:
    def test_merges_the_six_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_config_dir(tmp_path / "config", BASE)
        assert yaml.safe_load((tmp_path / "config" / "mcp.yaml").read_text())["app"] == {"tool_prefix": "Fo"}
        assert yaml.safe_load((tmp_path / "config" / "system.yaml").read_text())["app"] == {"name": "T"}
        Config.set_config("config")
        cfg = Config.get_config_model()
        assert cfg.app.name == "T" and cfg.app.tool_prefix == "Fo"
        assert cfg.auth.dynamic_tools.enabled is True and cfg.auth.enabled is False
        _, origins = load_config_dir(tmp_path / "config")
        assert origins["app.name"] == "system.yaml" and origins["app.tool_prefix"] == "mcp.yaml"
        assert origins["auth.dynamic_tools.enabled"] == "mcp.yaml"

    def test_expands_env_references_and_records_them(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("T_DB", "postgresql://env/db")
        write_config_dir(tmp_path / "config", BASE)
        Config.set_config("config")
        assert Config.get_config_model().database.url == "postgresql://env/db"
        assert Config.get_env_refs() == {"database.url": "T_DB"}
        monkeypatch.delenv("T_DB")
        Config.set_config("config")
        assert Config.get_config_model().database.url == "postgresql://u:p@h/db"

    def test_unset_variable_without_default_stops_loading(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("T_NO_DEFAULT", raising=False)
        tree = {**BASE, "database": {**BASE["database"], "url": "${T_NO_DEFAULT}"}}
        write_config_dir(tmp_path / "config", tree)
        with pytest.raises(ValueError, match="T_NO_DEFAULT"):
            Config.set_config("config")

    @pytest.mark.parametrize("dup_file, text", [
        ("system.yaml", "server: {host: a, port: 1, transport: http}\nlogging: {level: INFO, format: x, file: null, max_size: 1MB, backup_count: 1}\napp: {name: T}\nmodules: {enabled: [rag]}\n"),
        ("models.yaml", "app:\n  tool_prefix: Other\n"),
    ])
    def test_a_setting_in_two_files_is_an_error_naming_both(self, tmp_path, monkeypatch, dup_file, text):
        monkeypatch.chdir(tmp_path)
        write_config_dir(tmp_path / "config", BASE)
        (tmp_path / "config" / dup_file).write_text(text, encoding="utf-8")
        with pytest.raises(ConfigFileError) as e:
            Config.set_config("config")
        msg = str(e.value)
        assert "config/mcp.yaml" in msg and f"config/{dup_file}" in msg
        assert ("'modules.enabled'" in msg) or ("'app.tool_prefix'" in msg)

    def test_a_section_against_a_value_is_a_duplicate_too(self, tmp_path):
        write_config_dir(tmp_path / "c", BASE)
        (tmp_path / "c" / "retrieval.yaml").write_text("app: just-a-string\n", encoding="utf-8")
        with pytest.raises(ConfigFileError, match="'app' is set in both config/mcp.yaml and config/retrieval.yaml"):
            load_config_dir(tmp_path / "c")

    def test_shared_parents_are_not_duplicates(self, tmp_path):
        """rag: (and auth:, app:) may open in several files; only settings must not repeat."""
        write_config_dir(tmp_path / "c", {**BASE, "rag": {"profile": "local", "chunking": {"chunk_overlap": 3}}})
        raw, origins = load_config_dir(tmp_path / "c")
        assert raw["rag"] == {"profile": "local", "chunking": {"chunk_overlap": 3}}
        assert origins["rag.profile"] == "models.yaml" and origins["rag.chunking.chunk_overlap"] == "retrieval.yaml"

    def test_missing_file_is_named(self, tmp_path):
        write_config_dir(tmp_path / "c", BASE)
        (tmp_path / "c" / "auth.yaml").unlink()
        with pytest.raises(ConfigFileError) as e:
            load_config_dir(tmp_path / "c")
        assert "auth.yaml" in str(e.value) and "models.yaml" not in str(e.value)
        assert "split_config" not in str(e.value)

    def test_missing_files_next_to_an_old_config_point_at_the_converter(self, tmp_path):
        (tmp_path / "c").mkdir()
        (tmp_path / "c" / "config.yaml").write_text("server: {}\n", encoding="utf-8")
        with pytest.raises(ConfigFileError, match="scripts/split_config.py"):
            load_config_dir(tmp_path / "c")

    def test_a_file_that_is_not_a_mapping_is_refused(self, tmp_path):
        write_config_dir(tmp_path / "c", BASE)
        (tmp_path / "c" / "models.yaml").write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigFileError, match="models.yaml must be a mapping"):
            load_config_dir(tmp_path / "c")

    def test_an_empty_file_is_fine(self, tmp_path):
        write_config_dir(tmp_path / "c", BASE)
        (tmp_path / "c" / "retrieval.yaml").write_text("# nothing yet\n", encoding="utf-8")
        assert load_config_dir(tmp_path / "c")[0]["server"]["port"] == 8080

    def test_old_config_yaml_is_reported_and_never_read(self, tmp_path, monkeypatch):
        """No backward compatibility: the old file is not read, but its presence is logged
        once, pointing at the upgrade notes; startup goes on with the six files."""
        monkeypatch.chdir(tmp_path)
        write_config_dir(tmp_path / "config", BASE)
        old = {**BASE, "app": {"name": "FromOldFile", "tool_prefix": "Old"}}
        (tmp_path / "config" / "config.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
        from src.config import config_manager
        with pytest.MonkeyPatch.context() as mp:
            errors = []
            mp.setattr(config_manager.logger, "error", lambda m, *a, **k: errors.append(m))
            Config.set_config("config")
        assert Config.get_config_model().app.name == "T"  # the six files, not the old one
        assert len(errors) == 1 and "IGNORED" in errors[0] and "split_config.py" in errors[0]
        assert "docs/configuration.md" in errors[0]
        assert legacy_config_warning(tmp_path) is None

    def test_removed_llm_roles_in_models_yaml_are_warned_about_and_ignored(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_config_dir(tmp_path / "config", BASE)
        (tmp_path / "config" / "models.yaml").write_text(
            "rag:\n  answer_llm: {model: big}\n  eval_llm: {model: judge}\n", encoding="utf-8")
        from src.config import config_manager
        warnings = []
        monkeypatch.setattr(config_manager.logger, "warning", lambda m, *a, **k: warnings.append(m))
        Config.set_config("config")
        assert [w.split(" ")[0] for w in warnings] == ["rag.answer_llm", "rag.eval_llm"]
        assert all("use rag.llm" in w for w in warnings)

    def test_module_wiring_in_mcp_yaml_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_config_dir(tmp_path / "config", BASE)
        (tmp_path / "config" / "mcp.yaml").write_text(
            "app: {tool_prefix: Fo}\nauth: {dynamic_tools: {enabled: true}}\n"
            "modules:\n  enabled: [rag]\n  rag: {description: x}\n", encoding="utf-8")
        Config.set_config("config")
        assert Config.get_config_model() is None

    def test_factory_raw_rereads_the_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_config_dir(tmp_path / "config", BASE)
        Config.set_config("config")
        (tmp_path / "config" / "system.yaml").write_text(
            "server: {host: x, port: 9, transport: http}\n"
            "logging: {level: INFO, format: x, file: null, max_size: 1MB, backup_count: 1}\n"
            "app: {name: Changed}\n", encoding="utf-8")
        assert Config.factory_raw()["app"]["name"] == "Changed"
        Config._config_dir = None
        assert Config.factory_raw() == {}


class TestBlankMeansUnset:
    def test_empty_endpoint_and_key_values_are_none(self):
        assert RAGEmbeddingConfig(provider="openai", model="m", dimension=8, base_url="",
                                  api_key="  ").base_url is None
        assert LLMRoleConfig(model="", api_key="").model is None
        assert AsrConfig(base_url="", api_key="", model="").model_dump()["base_url"] is None
        assert DatabaseConfig(url="u", echo=False, pool_size=1, max_overflow=1, pool_timeout=1,
                              pool_recycle=1, port="").port is None
        assert DatabaseConfig(url="u", echo=False, pool_size=1, max_overflow=1, pool_timeout=1,
                              pool_recycle=1, port="5444").port == 5444

    def test_non_blank_values_are_kept(self):
        assert LLMRoleConfig(model="m", api_key="k").api_key == "k"


# -- .env -----------------------------------------------------------------------------

class TestDotenv:
    def test_parse(self):
        text = "# c\n\nA=1\nexport B = two \nC=\"quoted value\"\nD='x'\nE=a=b\nnoequals\nF=\"\nG=\"x'\n"
        assert parse_dotenv(text) == {"A": "1", "B": "two", "C": "quoted value", "D": "x",
                                      "E": "a=b", "F": '"', "G": "\"x'"}

    def test_load_overrides_by_default(self, tmp_path, monkeypatch):
        f = tmp_path / ".env"
        f.write_text("T_DOTENV_A=from-file\nT_DOTENV_B=b\n", encoding="utf-8")
        monkeypatch.setenv("T_DOTENV_A", "from-shell")
        monkeypatch.delenv("T_DOTENV_B", raising=False)
        assert load_dotenv(f) == f
        assert os.environ["T_DOTENV_A"] == "from-file" and os.environ["T_DOTENV_B"] == "b"
        monkeypatch.setenv("T_DOTENV_A", "from-shell")
        load_dotenv(f, override=False)
        assert os.environ["T_DOTENV_A"] == "from-shell"

    def test_missing_file(self, tmp_path):
        assert read_dotenv(tmp_path / "none") == {}
        assert load_dotenv(tmp_path / "none") is None


# -- the converter --------------------------------------------------------------------

class TestConverterRoundTrip:
    def test_committed_1_4_config_converts_without_a_difference(self, tmp_path):
        """The 1.4 template, converted and loaded with its .env lines, equals the 1.4
        config field by field (the removed roles were never set there)."""
        env = _env_without_config_vars()
        old = _legacy_raw()
        result = split_legacy(old, env)
        write_outputs(result, tmp_path / "cfg", tmp_path / "env.lines")
        assert result.env_lines.keys() == {"DATABASE_URL", "DATABASE_PORT"}
        assert result.notes == []
        assert check(old, tmp_path / "cfg", env, result.env_lines) == []

    def test_every_kind_of_value_round_trips(self, tmp_path):
        """Literal keys, references with other defaults, references to other variables,
        partial references, nulls, booleans, removed roles and custom wiring."""
        env = {**_env_without_config_vars(), "MY_KEY": "sk-env", "OTHER": "o", "LLM_API_KEY": "shell"}
        old = _legacy_raw()
        rag = old["rag"]
        rag["embedding"]["api_key"] = "sk-literal"                         # literal secret
        rag["llm"]["api_key"] = "${MY_KEY}"                                # other variable, set
        rag["llm"]["base_url"] = "${LLM_BASE_URL:-http://gpu:9/v1}"        # same variable, other default
        rag["rerank"]["api_key"] = "${RR_UNSET:-rr-default}"              # other variable, unset
        rag["rerank"]["base_url"] = "http://${OTHER}:8787"                 # partial reference
        rag["asr"]["api_key"] = None                                       # null stays unset
        rag["contextual_llm"] = {"model": "small", "base_url": "http://small:1/v1", "api_key": "k2"}
        rag["answer_llm"] = {"model": "big"}
        rag["eval_llm"] = {"model": "judge"}
        old["auth"]["revocation"]["enabled"] = True                        # a literal boolean
        old["database"]["url"] = "${DATABASE_URL}"                         # no default, unset
        env["DATABASE_URL"] = "postgresql://x:y@db:5432/folio"
        old["modules"]["rag"]["mcp_tools"]["function_name"] = "custom"     # custom wiring
        old["modules"]["enabled"].append("demo")                           # unknown module
        result = split_legacy(old, env)
        write_outputs(result, tmp_path / "cfg", tmp_path / "env.lines")
        assert result.env_lines == {
            "EMBEDDING_API_KEY": "sk-literal",
            "LLM_BASE_URL": "http://gpu:9/v1",
            "RR_UNSET": "rr-default",
            "RERANK_BASE_URL": "http://o:8787",
            "CONTEXTUAL_LLM_BASE_URL": "http://small:1/v1",
            "CONTEXTUAL_LLM_API_KEY": "k2",
            "FOLIO_REVOCATION_ENABLED": "true",
            "DATABASE_PORT": "5444",
        }
        models = yaml.safe_load((tmp_path / "cfg" / "models.yaml").read_text())["rag"]
        assert models["llm"]["api_key"] == "${MY_KEY:-not-needed}"
        assert models["rerank"]["api_key"] == "${RR_UNSET:-not-needed}"
        assert models["embedding"]["api_key"] == "${EMBEDDING_API_KEY:-not-needed}"
        assert "answer_llm" not in models and "eval_llm" not in models
        assert any("rag.answer_llm was removed" in n for n in result.notes)
        assert any("modules.rag.mcp_tools differs" in n for n in result.notes)
        assert any("'demo' is not a built-in module" in n for n in result.notes)
        assert yaml.safe_load((tmp_path / "cfg" / "mcp.yaml").read_text())["modules"] == {
            "enabled": ["folder", "file", "rag_indexing", "rag_query", "rag", "media", "admin"]}
        assert check(old, tmp_path / "cfg", env, result.env_lines) == []
        text = "".join((tmp_path / "cfg" / n).read_text() for n in layout.CONFIG_FILES)
        for secret in ("sk-literal", "k2", "small:1", "gpu:9", "x:y@db"):
            assert secret not in text

    def test_a_variable_already_set_elsewhere_is_noted(self, tmp_path):
        env = {**_env_without_config_vars(), "LLM_API_KEY": "from-env"}
        old = _legacy_raw()
        old["rag"]["llm"]["api_key"] = "literal-key"
        result = split_legacy(old, env)
        assert result.env_lines["LLM_API_KEY"] == "literal-key"
        assert any(n.startswith("LLM_API_KEY is already set") for n in result.notes)
        assert all("literal-key" not in n and "from-env" not in n for n in result.notes)

    def test_null_with_the_variable_set_clears_it(self, tmp_path):
        env = {**_env_without_config_vars(), "ASR_BASE_URL": "http://set:1"}
        old = _legacy_raw()  # asr.base_url: null
        result = split_legacy(old, env)
        assert result.env_lines["ASR_BASE_URL"] == ""
        write_outputs(result, tmp_path / "cfg", tmp_path / "env.lines")
        assert check(old, tmp_path / "cfg", env, result.env_lines) == []

    def test_unset_reference_without_default_is_kept(self):
        old = _legacy_raw()
        old["rag"]["llm"]["api_key"] = "${NOPE_UNSET}"
        result = split_legacy(old, _env_without_config_vars())
        assert "LLM_API_KEY" not in result.env_lines and "NOPE_UNSET" not in result.env_lines

    def test_unknown_sections_are_kept_and_noted(self):
        files = place_sections({"extra": {"a": 1}, "rag": {"mystery": 2}, "app": {"odd": 3}}, notes := [])
        assert files["system.yaml"] == {"extra": {"a": 1}, "app": {"odd": 3}}
        assert files["retrieval.yaml"] == {"rag": {"mystery": 2}}
        assert len(notes) == 2  # app.odd is under app, which system.yaml holds

    def test_diff_paths(self):
        assert diff_paths({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"b": 1, "c": 3}, "e": 1}) == ["a.c", "d", "e"]
        assert diff_paths(1, 2) == ["<root>"]


class TestConverterOutputs:
    def test_env_file_is_mode_600_and_quotes_when_needed(self, tmp_path):
        result = split_legacy(_legacy_raw(), _env_without_config_vars())
        result.env_lines.update({"SPACEY": " x ", "HASH": "#h", "PLAIN": "a b"})
        write_outputs(result, tmp_path / "cfg", tmp_path / "env.lines")
        mode = stat.S_IMODE((tmp_path / "env.lines").stat().st_mode)
        assert mode == 0o600
        pairs = read_dotenv(tmp_path / "env.lines")
        assert pairs["SPACEY"] == " x " and pairs["HASH"] == "#h" and pairs["PLAIN"] == "a b"

    def test_refuses_to_overwrite_without_force(self, tmp_path):
        result = split_legacy(_legacy_raw(), _env_without_config_vars())
        write_outputs(result, tmp_path / "cfg", tmp_path / "env.lines")
        with pytest.raises(FileExistsError, match="--force"):
            write_outputs(result, tmp_path / "cfg", tmp_path / "other")
        (tmp_path / "cfg2").mkdir()
        with pytest.raises(FileExistsError, match="env.lines exists"):
            write_outputs(result, tmp_path / "cfg2", tmp_path / "env.lines")
        write_outputs(result, tmp_path / "cfg", tmp_path / "env.lines", force=True)

    def test_a_line_break_cannot_go_into_env(self, tmp_path):
        result = split_legacy(_legacy_raw(), _env_without_config_vars())
        result.env_lines["BAD"] = "a\nb"
        with pytest.raises(ValueError, match="line break"):
            write_outputs(result, tmp_path / "cfg", tmp_path / "env.lines")


class TestSplitConfigScript:
    def _main(self):
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            import split_config
        finally:
            sys.path.pop(0)
        return split_config.main

    def test_converts_checks_and_never_prints_values(self, tmp_path, capsys, clean_env):
        old = _legacy_raw()
        old["rag"]["llm"]["api_key"] = "sk-very-secret"
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "config.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
        main = self._main()
        rc = main(["--old", str(tmp_path / "config" / "config.yaml"), "--out-dir",
                   str(tmp_path / "config"), "--env-out", str(tmp_path / "env.lines"),
                   "--env-file", ""])
        out = capsys.readouterr().out
        assert rc == 0 and "Check passed" in out and "LLM_API_KEY" in out
        assert "sk-very-secret" not in out and "CHANGE_ME" not in out
        assert "no longer read" in out
        assert read_dotenv(tmp_path / "env.lines")["LLM_API_KEY"] == "sk-very-secret"

    def test_errors(self, tmp_path, capsys):
        main = self._main()
        assert main(["--old", str(tmp_path / "missing.yaml")]) == 2
        (tmp_path / "c.yaml").write_text(LEGACY.read_text(encoding="utf-8"), encoding="utf-8")
        args = ["--old", str(tmp_path / "c.yaml"), "--out-dir", str(tmp_path / "o"),
                "--env-out", str(tmp_path / "e"), "--env-file", str(tmp_path / "none")]
        assert main(args + ["--no-check"]) == 0
        assert main(args) == 2  # outputs exist, no --force
        assert "--force" in capsys.readouterr().err

    def test_a_failed_check_exits_1(self, tmp_path, capsys, monkeypatch):
        main = self._main()
        (tmp_path / "c.yaml").write_text(LEGACY.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr("split_config.check", lambda *a, **k: ["rag.llm.model"])
        rc = main(["--old", str(tmp_path / "c.yaml"), "--out-dir", str(tmp_path / "o"),
                   "--env-out", str(tmp_path / "e"), "--env-file", ""])
        assert rc == 1 and "rag.llm.model" in capsys.readouterr().out


# -- upgrade notices reach the log files ------------------------------------------------

def _config_with_upgrade_leftovers(root: Path) -> None:
    """The committed config plus both upgrade leftovers: an old config.yaml and a removed
    section (rag.answer_llm in models.yaml)."""
    import shutil
    shutil.copytree(REPO / "config", root / "config")
    models = root / "config" / "models.yaml"
    tree = yaml.safe_load(models.read_text(encoding="utf-8"))
    tree["rag"]["answer_llm"] = {"model": "big"}
    models.write_text(yaml.safe_dump(tree, sort_keys=False), encoding="utf-8")
    (root / "config" / "config.yaml").write_text(LEGACY.read_text(encoding="utf-8"),
                                                 encoding="utf-8")


def _server_log(root: Path) -> str:
    return "".join(f.read_text(encoding="utf-8")
                   for f in sorted((root / "logs" / "server").glob("*.log")))


@pytest.fixture()
def _restore_loguru():
    from loguru import logger
    yield
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


class TestStartupNotices:
    """set_config logs its upgrade notices before main.py sets up the log files from that
    config, so they used to reach the terminal only. They are logged again afterwards."""

    def test_notices_are_kept_and_logged_again_once(self, tmp_path, clean_env, monkeypatch, _restore_loguru):
        from src.log import get_server_logger, setup_logger
        monkeypatch.chdir(tmp_path)
        _config_with_upgrade_leftovers(tmp_path)
        Config.set_config("config")
        setup_logger(console_level="CRITICAL", log_base_dir=str(tmp_path / "logs"))
        logged = Config.log_startup_notices(get_server_logger())
        assert len(logged) == 2
        assert "IGNORED" in logged[0] and logged[1].startswith("rag.answer_llm ")
        assert Config.log_startup_notices(get_server_logger()) == []  # once only
        text = _server_log(tmp_path)
        for message in logged:
            assert text.count(message) == 1
        error_log = "".join(f.read_text(encoding="utf-8")
                            for f in (tmp_path / "logs" / "server").glob("error_*.log"))
        assert "IGNORED" in error_log and "| ERROR " in error_log  # the level is kept

    def test_a_new_load_starts_a_new_list(self, tmp_path, clean_env, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _config_with_upgrade_leftovers(tmp_path)
        Config.set_config("config")
        assert Config._startup_notices
        (tmp_path / "config" / "config.yaml").unlink()
        Config.set_config("config")
        assert [m.split(" ")[0] for _l, m in Config._startup_notices] == ["rag.answer_llm"]
        (tmp_path / "config" / "models.yaml").write_text(
            (REPO / "config" / "models.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        Config.set_config("config")
        calls = []

        class _Log:
            def __getattr__(self, level):
                return lambda message: calls.append((level, message))

        assert Config.log_startup_notices(_Log()) == []
        assert calls == []  # earlier loads' notices are not carried over

    def test_main_writes_the_notices_to_the_server_log(self, tmp_path, clean_env, monkeypatch,
                                                       _restore_loguru):
        """The real startup path: main.py sets up the files, then logs the notices."""
        import main
        monkeypatch.chdir(tmp_path)
        _config_with_upgrade_leftovers(tmp_path)
        monkeypatch.setattr(main, "run_migration_command", lambda *a, **k: True)
        monkeypatch.setattr(sys, "argv", ["main.py", "--config-dir", "config",
                                          "--migrate-only", "status", "--log-level", "critical"])
        with pytest.raises(SystemExit) as exit_info:
            main.main()
        assert exit_info.value.code == 0
        text = _server_log(tmp_path)
        assert text.count("config/config.yaml is no longer read") == 1
        assert text.count("rag.answer_llm was removed in Folio 1.5") == 1
