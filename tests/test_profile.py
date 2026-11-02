"""rag.profile + per-role model config (src/config/profile.py, src/config/roles.py).

Pure config logic: YAML is written to tmp_path; no model library is imported.
"""

import pytest

from src.config.config_manager import Config
from src.config.model import (
    AsrConfig,
    LLMRoleConfig,
    ParsingConfig,
    RAGConfig,
    RAGLLMConfig,
    VisionModelConfig,
)
from src.config.profile import (
    check_profile,
    env_refs,
    iter_secret_fields,
    literal_secret_paths,
)
from src.config.roles import (
    asr_runs_in_process,
    needs_local_docling_models,
    parsing_mode,
    resolve_llm,
    role_summary,
    startup_lines,
    uses_cloud_parsing,
)


def _rag(**over):
    data = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small", "dimension": 1536,
                      "base_url": "https://api.example.com/v1", "api_key": "k"},
        "chunking": {},
        "retrieval": {"default_top_k": 5, "default_similarity_cutoff": 0.2, "hybrid_search": True},
        "vector_store": {"type": "pgvector", "table_prefix": "vs"},
    }
    data.update(over)
    return RAGConfig.model_validate(data)


VISION = {"vision": {"provider": "openai-compatible", "model": "gpt-4o-mini",
                     "base_url": "https://api.example.com/v1", "api_key": "k"}}


class TestDefaults:
    def test_profile_defaults_to_local_and_keeps_old_behaviour(self):
        rag = _rag()
        assert rag.profile == "local"
        assert rag.asr is None  # no implicit section in local
        assert parsing_mode(rag, "pdf") == "local"
        assert parsing_mode(rag, "images") == "local"
        assert needs_local_docling_models(rag) and not uses_cloud_parsing(rag)
        assert check_profile(rag, refs={}).ok

    def test_cloud_profile_disables_asr_when_unset(self):
        rag = _rag(profile="cloud", parsing=VISION)
        assert rag.asr.enabled is False and rag.asr.provider == "openai-compatible"
        assert not asr_runs_in_process(rag)

    def test_cloud_profile_defaults_parsing_to_cloud(self):
        rag = _rag(profile="cloud", parsing=VISION)
        assert parsing_mode(rag, "pdf") == "cloud" and parsing_mode(rag, "images") == "cloud"
        assert not needs_local_docling_models(rag)

    def test_explicit_parsing_mode_wins(self):
        rag = _rag(parsing={"pdf": "cloud", **VISION})
        assert parsing_mode(rag, "pdf") == "cloud" and parsing_mode(rag, "images") == "local"

    def test_parsing_mode_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            parsing_mode(_rag(), "audio")

    def test_role_timeouts_have_sane_defaults(self):
        rag = _rag(llm={"provider": "openai", "model": "m"}, rerank={"enabled": True},
                   asr={"provider": "openai-compatible"})
        assert rag.embedding.timeout == 60 and rag.embedding.max_retries == 3
        assert rag.llm.timeout == 120 and rag.llm.max_retries == 2
        assert rag.rerank.timeout == 10 and rag.rerank.max_retries == 1 and rag.rerank.api == "score"
        assert rag.asr.segment_seconds == 600 and rag.asr.max_retries == 3 and rag.asr.timeout == 300

    def test_asr_segment_cap_respects_upload_limit(self):
        with pytest.raises(ValueError):
            AsrConfig(segment_seconds=900)  # 16k pcm16 over ~13 min exceeds a 25 MB upload


class TestAsrInProcess:
    def test_legacy_unset_asr_is_local_whisper(self):
        assert asr_runs_in_process(_rag())

    def test_cloud_provider_is_not_in_process(self):
        assert not asr_runs_in_process(_rag(asr={"provider": "openai-compatible"}))

    def test_local_provider_disabled_is_not_in_process(self):
        assert not asr_runs_in_process(_rag(asr={"provider": "fireredasr", "enabled": False}))


class TestResolveLlm:
    def test_inherits_base(self):
        rag = _rag(llm={"provider": "openai", "model": "base", "api_key": "k"})
        assert resolve_llm(rag, "answer_llm") is rag.llm

    def test_override_fields_win_and_rest_inherit(self):
        rag = _rag(llm={"provider": "openai", "model": "base", "base_url": "https://a/v1",
                        "api_key": "k", "timeout": 30},
                   contextual_llm={"model": "cheap", "max_retries": 5})
        got = resolve_llm(rag, "contextual_llm")
        assert got.model == "cheap" and got.max_retries == 5
        assert got.base_url == "https://a/v1" and got.timeout == 30 and got.api_key == "k"
        assert rag.llm.model == "base"  # base untouched

    def test_override_without_base(self):
        rag = _rag(eval_llm={"provider": "openai-compatible", "model": "judge",
                             "base_url": "https://j/v1"})
        got = resolve_llm(rag, "eval_llm")
        assert isinstance(got, RAGLLMConfig) and got.model == "judge"

    def test_none_when_nothing_configured(self):
        assert resolve_llm(_rag(), "answer_llm") is None

    def test_unknown_role(self):
        with pytest.raises(ValueError):
            resolve_llm(_rag(), "summary_llm")


class TestCheckProfile:
    def test_cloud_rejects_in_process_providers(self):
        rag = _rag(profile="cloud",
                   embedding={"provider": "huggingface", "model": "BAAI/bge-m3", "dimension": 1024},
                   asr={"provider": "fireredasr"},
                   parsing={"pdf": "local", **VISION})
        errs = "\n".join(check_profile(rag, refs={}).errors)
        assert "rag.embedding.provider 'huggingface'" in errs
        assert "rag.asr.provider 'fireredasr'" in errs
        assert "rag.parsing.pdf: local" in errs

    def test_cloud_allows_disabled_local_asr(self):
        rag = _rag(profile="cloud", asr={"provider": "fireredasr", "enabled": False}, parsing=VISION,
                   embedding={"provider": "openai", "model": "m", "dimension": 8,
                              "api_key": "${OPENAI_API_KEY}"})
        report = check_profile(rag, refs={"rag.embedding.api_key": "OPENAI_API_KEY",
                                          "rag.parsing.vision.api_key": "X"})
        assert report.ok, report.errors

    def test_cloud_parsing_requires_vision_model(self):
        rag = _rag(profile="cloud")
        errs = check_profile(rag, refs=None).errors
        assert any("rag.parsing.vision.model" in e for e in errs)

    def test_explicit_cloud_parsing_needs_vision_in_any_profile(self):
        rag = _rag(parsing={"images": "cloud"})
        errs = check_profile(rag, refs=None).errors
        assert errs and "images parsing is 'cloud'" in errs[0]

    def test_cloud_rejects_literal_keys(self):
        rag = _rag(profile="cloud", parsing=VISION)
        errs = check_profile(rag, refs={}).errors
        assert any(e.startswith("rag.embedding.api_key is a literal") for e in errs)
        assert any(e.startswith("rag.parsing.vision.api_key is a literal") for e in errs)

    def test_cloud_accepts_env_references(self):
        rag = _rag(profile="cloud", parsing=VISION)
        refs = {"rag.embedding.api_key": "E", "rag.parsing.vision.api_key": "V"}
        assert check_profile(rag, refs=refs).ok

    def test_hybrid_only_warns_about_literal_keys(self):
        report = check_profile(_rag(profile="hybrid"), refs={})
        assert report.ok and any("literal" in w for w in report.warnings)

    def test_local_is_silent_about_literal_keys(self):
        report = check_profile(_rag(), refs={})
        assert report.ok and not report.warnings

    def test_unknown_refs_skip_literal_rule(self):
        assert check_profile(_rag(profile="cloud", parsing=VISION), refs=None).ok

    def test_cloud_defaults_to_bigram_without_a_warning(self):
        from src.config.roles import effective_segmenter
        rag = _rag(profile="cloud", parsing=VISION)
        report = check_profile(rag, refs=None)
        assert report.ok and not any("segmenter" in w for w in report.warnings)
        assert effective_segmenter(rag) == "bigram"
        assert effective_segmenter(_rag()) == "ckip"           # local default unchanged
        assert effective_segmenter(_rag(profile="hybrid")) == "ckip"

    def test_cloud_rejects_explicit_ckip_with_the_way_out(self):
        rag = _rag(profile="cloud", parsing=VISION)
        rag.retrieval.segmenter = "ckip"
        report = check_profile(rag, refs=None)
        assert not report.ok
        msg = next(e for e in report.errors if "segmenter" in e)
        assert "bigram" in msg and "jieba" in msg and "--extra local" in msg

    def test_explicit_model_free_segmenters_are_fine_in_cloud(self):
        for name in ("bigram", "jieba"):
            rag = _rag(profile="cloud", parsing=VISION)
            rag.retrieval.segmenter = name
            assert check_profile(rag, refs=None).ok

    def test_explicit_ckip_is_fine_outside_cloud(self):
        for profile in ("local", "hybrid"):
            rag = _rag(profile=profile)
            rag.retrieval.segmenter = "ckip"
            assert not any("segmenter" in e for e in check_profile(rag, refs=None).errors)

    def test_none_config(self):
        assert check_profile(None).ok


class TestSecrets:
    def test_env_refs_walks_nested_values(self):
        raw = {"rag": {"embedding": {"api_key": "${OPENAI_API_KEY}", "model": "m"},
                       "llm": {"api_key": "Bearer ${X:-y}"}, "n": 3, "l": ["${Z}"]}}
        assert env_refs(raw) == {"rag.embedding.api_key": "OPENAI_API_KEY", "rag.llm.api_key": "X"}

    def test_iter_secret_fields_covers_every_role(self):
        rag = _rag(llm={"api_key": "a"}, answer_llm={"api_key": "b"}, rerank={"api_key": "c"},
                   asr={"api_key": "d"}, parsing=VISION)
        paths = dict(iter_secret_fields(rag))
        assert {"rag.embedding.api_key", "rag.llm.api_key", "rag.answer_llm.api_key",
                "rag.rerank.api_key", "rag.asr.api_key", "rag.parsing.vision.api_key"} <= set(paths)

    def test_literal_secret_paths(self):
        rag = _rag(llm={"api_key": "a"})
        assert literal_secret_paths(rag, {"rag.llm.api_key": "L"}) == ["rag.embedding.api_key"]

    def test_secrets_never_in_repr(self):
        assert "sekret" not in repr(VisionModelConfig(api_key="sekret"))
        assert "sekret" not in repr(LLMRoleConfig(api_key="sekret"))


class TestRoleSummary:
    def test_local_summary(self):
        rag = _rag(embedding={"provider": "huggingface", "model": "BAAI/bge-m3", "dimension": 1024},
                   llm={"provider": "vllm", "model": "gpt-oss", "base_url": "http://h/v1"},
                   contextual_retrieval={"enabled": True})
        roles = {e["role"]: e for e in role_summary(rag)}
        assert roles["embedding"]["runs"] == "in-process"
        assert roles["contextual_llm"]["runs"] == "endpoint" and roles["contextual_llm"]["inherits"] == "llm"
        assert roles["rerank"]["runs"] == "off"
        assert roles["vision"]["runs"] == "in-process" and roles["vision"]["provider"] == "docling"
        assert roles["asr"]["runs"] == "in-process"
        assert all("api_key" not in str(v) or k == "secret_path" for e in roles.values() for k, v in e.items())

    def test_cloud_summary_and_role_secret_path(self):
        rag = _rag(profile="cloud", parsing=VISION, llm={"provider": "openai", "model": "base"},
                   answer_llm={"model": "big", "api_key": "x"}, rerank={"enabled": True, "api": "cohere"},
                   asr={"provider": "openai-compatible", "model": "whisper-1"})
        roles = {e["role"]: e for e in role_summary(rag)}
        assert roles["vision"]["runs"] == "endpoint" and roles["vision"]["model"] == "gpt-4o-mini"
        assert roles["answer_llm"]["model"] == "big" and roles["answer_llm"]["inherits"] is None
        assert roles["answer_llm"]["secret_path"] == "rag.answer_llm.api_key"
        assert roles["eval_llm"]["secret_path"] == "rag.llm.api_key"
        assert roles["contextual_llm"]["runs"] == "off"  # contextual retrieval not enabled
        assert roles["rerank"]["provider"] == "cohere"
        assert roles["asr"]["runs"] == "endpoint"

    def test_no_llm_marks_llm_roles_off(self):
        roles = {e["role"]: e for e in role_summary(_rag())}
        assert roles["answer_llm"]["runs"] == "off"

    def test_cloud_unset_asr_is_off(self):
        roles = {e["role"]: e for e in role_summary(_rag(profile="cloud", parsing=VISION))}
        assert roles["asr"]["runs"] == "off"

    def test_startup_lines(self):
        lines = startup_lines(_rag(profile="cloud", parsing=VISION))
        text = "\n".join(lines)
        assert lines[0] == "Deployment profile: cloud"
        assert "Embedding" in text and "endpoint https://api.example.com/v1" in text
        assert "pdf=cloud, images=cloud" in text
        assert "Speech-to-text" in text and "off" in text
        assert "FTS segmenter" in text and "no model — bigram" in text  # cloud default

    def test_startup_lines_local_default_is_ckip(self):
        text = "\n".join(startup_lines(_rag()))
        assert "in-process model — ckip" in text

    def test_startup_lines_local_and_bigram(self):
        rag = _rag(embedding={"provider": "openai", "model": "m", "dimension": 8})
        rag.retrieval.segmenter = "bigram"
        text = "\n".join(startup_lines(rag))
        assert "in-process — docling" in text and "(provider default)" in text
        assert "no model — bigram" in text


# config.yaml integration: profile errors stop the model from loading.

_BASE_YAML = """\
server: {host: 127.0.0.1, port: 8080, transport: http}
database: {url: "postgresql://u:p@h/db", echo: false, pool_size: 1, max_overflow: 1, pool_timeout: 1, pool_recycle: 1}
auth: {enabled: false}
logging: {level: INFO, format: x, file: null, max_size: 1MB, backup_count: 1}
app: {name: T}
modules: {enabled: []}
rag:
  profile: cloud
  embedding: {provider: openai, model: m, dimension: 8, api_key: KEY}
  chunking: {}
  retrieval: {default_top_k: 5, default_similarity_cutoff: 0.2, hybrid_search: false}
  vector_store: {type: pgvector, table_prefix: vs}
  parsing:
    vision: {model: v, base_url: "https://x/v1", api_key: "${VISION_KEY}"}
"""


@pytest.fixture()
def _clean_config():
    yield
    Config._config = {}
    Config._config_model = None
    Config._module_configs = {}
    Config._env_refs = {}


def _load(tmp_path, monkeypatch, text):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.yaml").write_text(text, encoding="utf-8")
    Config.set_config("c.yaml")
    return Config.get_config_model()


def test_yaml_literal_key_in_cloud_blocks_startup(tmp_path, monkeypatch, _clean_config):
    monkeypatch.setenv("VISION_KEY", "v-secret")
    assert _load(tmp_path, monkeypatch, _BASE_YAML) is None
    assert Config.get_env_refs() == {"rag.parsing.vision.api_key": "VISION_KEY"}


def test_yaml_env_key_in_cloud_loads(tmp_path, monkeypatch, _clean_config):
    monkeypatch.setenv("VISION_KEY", "v-secret")
    monkeypatch.setenv("EMB_KEY", "e-secret")
    cfg = _load(tmp_path, monkeypatch, _BASE_YAML.replace("api_key: KEY", 'api_key: "${EMB_KEY}"'))
    assert cfg is not None and cfg.rag.embedding.api_key == "e-secret"
    assert cfg.rag.parsing.vision.api_key == "v-secret"


def test_yaml_without_rag_section_loads(tmp_path, monkeypatch, _clean_config):
    cfg = _load(tmp_path, monkeypatch, _BASE_YAML.split("rag:")[0])
    assert cfg is not None and cfg.rag is None


def test_yaml_hybrid_literal_key_warns_but_loads(tmp_path, monkeypatch, _clean_config):
    monkeypatch.setenv("VISION_KEY", "v")
    cfg = _load(tmp_path, monkeypatch, _BASE_YAML.replace("profile: cloud", "profile: hybrid"))
    assert cfg is not None and cfg.rag.profile == "hybrid"


def test_parsing_config_defaults():
    p = ParsingConfig()
    assert p.pdf is None and p.on_page_failure == "mark" and p.max_pages_per_file == 500


def test_yaml_explicit_ckip_in_cloud_blocks_startup(tmp_path, monkeypatch, _clean_config):
    monkeypatch.setenv("VISION_KEY", "v-secret")
    monkeypatch.setenv("EMB_KEY", "e-secret")
    base = _BASE_YAML.replace("api_key: KEY", 'api_key: "${EMB_KEY}"')
    ckip = base.replace("hybrid_search: false}", "hybrid_search: true, segmenter: ckip}")
    assert _load(tmp_path, monkeypatch, ckip) is None
    Config._config_model = None
    bigram = base.replace("hybrid_search: false}", "hybrid_search: true, segmenter: bigram}")
    assert _load(tmp_path, monkeypatch, bigram).rag.retrieval.segmenter == "bigram"
