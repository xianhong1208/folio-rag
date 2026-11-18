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
    images_readable,
    parsing_summary,
    pdf_parsing,
    resolve_llm,
    role_summary,
    startup_lines,
    vision_model,
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
        assert pdf_parsing(rag) == "text" and not images_readable(rag) and vision_model(rag) is None
        assert check_profile(rag, refs={}).ok

    def test_cloud_profile_disables_asr_when_unset(self):
        rag = _rag(profile="cloud", parsing=VISION)
        assert rag.asr.enabled is False and rag.asr.provider == "openai-compatible"

    def test_auto_follows_the_vision_model_in_every_profile(self):
        for profile in ("local", "hybrid", "cloud"):
            assert pdf_parsing(_rag(profile=profile, parsing=VISION)) == "vision"
            assert images_readable(_rag(profile=profile, parsing=VISION))
            assert pdf_parsing(_rag(profile=profile)) == "text"  # the cloud profile needs no model either

    def test_half_configured_vision_is_not_configured(self):
        for vision in ({"model": "m"}, {"base_url": "http://v/v1"}, {}):
            rag = _rag(parsing={"vision": vision})
            assert vision_model(rag) is None and pdf_parsing(rag) == "text"
        openai = _rag(parsing={"vision": {"provider": "openai", "model": "gpt-4o"}})
        assert vision_model(openai) is not None  # openai has a default endpoint

    def test_explicit_modes_win(self):
        assert pdf_parsing(_rag(parsing={"pdf": "text", **VISION})) == "text"
        assert images_readable(_rag(parsing={"pdf": "text", **VISION}))  # images still read
        assert pdf_parsing(_rag(parsing={"pdf": "vision", **VISION})) == "vision"

    def test_parsing_summary_names_the_mode(self):
        assert "vision model gpt-4o-mini" in parsing_summary(_rag(parsing=VISION))
        assert parsing_summary(_rag()).startswith("text layer only")
        assert "text layer only (rag.parsing.pdf: text)" in parsing_summary(_rag(parsing={"pdf": "text", **VISION}))

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


class TestNoInProcessAsr:
    """1.5: speech-to-text never runs in-process; the removed providers do not validate."""

    def test_removed_providers_fail_validation_in_every_profile(self):
        for profile in ("local", "hybrid", "cloud"):
            for name in ("fireredasr", "docling-whisper"):
                with pytest.raises(ValueError, match="migrate-1.5"):
                    _rag(profile=profile, asr={"provider": name}, parsing=VISION)

    def test_disabled_removed_provider_still_fails(self):
        """Even switched off, an old provider name stops the start: the file needs editing."""
        with pytest.raises(ValueError, match="removed in Folio 1.5"):
            _rag(asr={"provider": "fireredasr", "enabled": False})


class TestNoInProcessParsing:
    """1.5: PDFs and images are never parsed by in-process models; the old settings stop the start."""

    @pytest.mark.parametrize("value", ["local", "cloud", "LOCAL"])
    def test_old_pdf_modes_fail_validation(self, value):
        with pytest.raises(ValueError, match=r"rag.parsing.pdf: (local|cloud) was removed in Folio 1.5.*migrate-1.5.md"):
            _rag(parsing={"pdf": value, **VISION})

    def test_old_images_key_fails_validation_whatever_its_value(self):
        for value in ("local", "cloud", None):
            with pytest.raises(ValueError, match="rag.parsing.images"):
                _rag(parsing={"images": value, **VISION})

    def test_docling_section_fails_validation_naming_its_keys(self):
        with pytest.raises(ValueError, match=r"rag.docling \(device, ocr_enabled\).*HF_HUB_OFFLINE"):
            _rag(docling={"device": "cuda", "ocr_enabled": True})
        with pytest.raises(ValueError, match="rag.docling"):
            _rag(docling={})
        assert _rag(docling=None).parsing is None  # an explicit null is no setting at all

    def test_unknown_pdf_mode(self):
        with pytest.raises(ValueError, match="use auto, vision or text"):
            ParsingConfig(pdf="ocr")

    def test_forced_vision_without_a_model_is_an_error_in_every_profile(self):
        for profile in ("local", "hybrid", "cloud"):
            errs = check_profile(_rag(profile=profile, parsing={"pdf": "vision"}), refs=None).errors
            assert any("rag.parsing.pdf is 'vision' but rag.parsing.vision is not configured" in e for e in errs)
        assert check_profile(_rag(parsing={"pdf": "vision", **VISION}), refs=None).ok
        assert check_profile(_rag(parsing={"pdf": "text"}), refs=None).ok


class TestResolveLlm:
    def test_inherits_base(self):
        rag = _rag(llm={"provider": "openai", "model": "base", "api_key": "k"})
        assert resolve_llm(rag, "contextual_llm") is rag.llm
        assert resolve_llm(rag, "llm") is rag.llm  # answers and evaluation

    def test_override_fields_win_and_rest_inherit(self):
        rag = _rag(llm={"provider": "openai", "model": "base", "base_url": "https://a/v1",
                        "api_key": "k", "timeout": 30},
                   contextual_llm={"model": "cheap", "max_retries": 5})
        got = resolve_llm(rag, "contextual_llm")
        assert got.model == "cheap" and got.max_retries == 5
        assert got.base_url == "https://a/v1" and got.timeout == 30 and got.api_key == "k"
        assert rag.llm.model == "base"  # base untouched

    def test_override_without_base(self):
        rag = _rag(contextual_llm={"provider": "openai-compatible", "model": "small",
                                   "base_url": "https://j/v1"})
        got = resolve_llm(rag, "contextual_llm")
        assert isinstance(got, RAGLLMConfig) and got.model == "small"

    def test_none_when_nothing_configured(self):
        assert resolve_llm(_rag(), "contextual_llm") is None
        assert resolve_llm(_rag(), "llm") is None

    def test_unknown_role(self):
        with pytest.raises(ValueError):
            resolve_llm(_rag(), "summary_llm")

    def test_removed_roles_are_gone(self):
        """answer_llm / eval_llm were removed in 1.5: answers and evaluation use rag.llm"""
        for role in ("answer_llm", "eval_llm"):
            with pytest.raises(ValueError):
                resolve_llm(_rag(), role)
        rag = _rag(llm={"model": "base"}, answer_llm={"model": "big"}, eval_llm={"model": "judge"})
        assert not hasattr(rag, "answer_llm") and not hasattr(rag, "eval_llm")
        assert {e["role"] for e in role_summary(rag)}.isdisjoint({"answer_llm", "eval_llm"})


class TestCheckProfile:
    def test_cloud_rejects_in_process_providers(self):
        rag = _rag(profile="cloud",
                   embedding={"provider": "huggingface", "model": "BAAI/bge-m3", "dimension": 1024},
                   parsing=VISION)
        errs = "\n".join(check_profile(rag, refs={}).errors)
        assert "rag.embedding.provider 'huggingface'" in errs

    def test_cloud_allows_disabled_asr(self):
        rag = _rag(profile="cloud", asr={"enabled": False}, parsing=VISION,
                   embedding={"provider": "openai", "model": "m", "dimension": 8,
                              "api_key": "${OPENAI_API_KEY}"})
        report = check_profile(rag, refs={"rag.embedding.api_key": "OPENAI_API_KEY",
                                          "rag.parsing.vision.api_key": "X"})
        assert report.ok, report.errors

    def test_cloud_profile_needs_no_vision_model(self):
        """No vision model in the cloud profile: PDFs come from their text layer."""
        rag = _rag(profile="cloud")
        assert not [e for e in check_profile(rag, refs=None).errors if "parsing" in e]

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
        rag = _rag(llm={"api_key": "a"}, contextual_llm={"api_key": "b"}, rerank={"api_key": "c"},
                   asr={"api_key": "d"}, parsing=VISION)
        paths = dict(iter_secret_fields(rag))
        assert {"rag.embedding.api_key", "rag.llm.api_key", "rag.contextual_llm.api_key",
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
        assert roles["vision"]["runs"] == "off" and roles["vision"]["modes"] == {"pdf": "text", "images": "refused"}
        assert roles["vision"]["summary"].startswith("text layer only")
        assert roles["asr"]["runs"] == "endpoint" and roles["asr"]["provider"] == "openai-compatible"
        assert all("api_key" not in str(v) or k == "secret_path" for e in roles.values() for k, v in e.items())

    def test_cloud_summary_and_role_secret_path(self):
        rag = _rag(profile="cloud", parsing=VISION, llm={"provider": "openai", "model": "base"},
                   contextual_llm={"model": "small", "api_key": "x"}, rerank={"enabled": True, "api": "cohere"},
                   asr={"provider": "openai-compatible", "model": "whisper-1"})
        roles = {e["role"]: e for e in role_summary(rag)}
        assert roles["vision"]["runs"] == "endpoint" and roles["vision"]["model"] == "gpt-4o-mini"
        assert roles["vision"]["modes"] == {"pdf": "vision", "images": "vision"}
        assert roles["contextual_llm"]["model"] == "small" and roles["contextual_llm"]["inherits"] is None
        assert roles["contextual_llm"]["secret_path"] == "rag.contextual_llm.api_key"
        assert roles["llm"]["secret_path"] == "rag.llm.api_key" and roles["llm"]["model"] == "base"
        assert roles["llm"]["runs"] == "endpoint"
        assert roles["contextual_llm"]["runs"] == "off"  # contextual retrieval not enabled
        assert roles["rerank"]["provider"] == "cohere"
        assert roles["asr"]["runs"] == "endpoint"

    def test_no_llm_marks_llm_roles_off(self):
        roles = {e["role"]: e for e in role_summary(_rag())}
        assert roles["llm"]["runs"] == "off"

    def test_cloud_unset_asr_is_off(self):
        roles = {e["role"]: e for e in role_summary(_rag(profile="cloud", parsing=VISION))}
        assert roles["asr"]["runs"] == "off"

    def test_startup_lines(self):
        lines = startup_lines(_rag(profile="cloud", parsing=VISION))
        text = "\n".join(lines)
        assert lines[0] == "Deployment profile: cloud"
        assert "Embedding" in text and "endpoint https://api.example.com/v1" in text
        assert "pdf=vision, images=vision" in text
        assert "Speech-to-text" in text and "off" in text
        assert "FTS segmenter" in text and "no model — bigram" in text  # cloud default

    def test_startup_lines_local_default_is_ckip(self):
        text = "\n".join(startup_lines(_rag()))
        assert "in-process model — ckip" in text

    def test_startup_lines_local_and_bigram(self):
        rag = _rag(embedding={"provider": "openai", "model": "m", "dimension": 8})
        rag.retrieval.segmenter = "bigram"
        text = "\n".join(startup_lines(rag))
        assert "text layer only: PDFs from their text layer; images refused" in text
        assert "(provider default)" in text
        assert "no model — bigram" in text


# Config files integration: profile errors stop the model from loading.

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
    Config._config_dir = None
    Config._env_refs = {}


def _load(tmp_path, monkeypatch, text):
    from tests.config_helpers import write_config_dir
    monkeypatch.chdir(tmp_path)
    write_config_dir(tmp_path / "cfg", text)
    Config.set_config("cfg")
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
    assert p.pdf == "auto" and p.on_page_failure == "mark" and p.max_pages_per_file == 500
    assert p.vision is None and VisionModelConfig().ca_file is None


def test_yaml_explicit_ckip_in_cloud_blocks_startup(tmp_path, monkeypatch, _clean_config):
    monkeypatch.setenv("VISION_KEY", "v-secret")
    monkeypatch.setenv("EMB_KEY", "e-secret")
    base = _BASE_YAML.replace("api_key: KEY", 'api_key: "${EMB_KEY}"')
    ckip = base.replace("hybrid_search: false}", "hybrid_search: true, segmenter: ckip}")
    assert _load(tmp_path, monkeypatch, ckip) is None
    Config._config_model = None
    bigram = base.replace("hybrid_search: false}", "hybrid_search: true, segmenter: bigram}")
    assert _load(tmp_path, monkeypatch, bigram).rag.retrieval.segmenter == "bigram"
