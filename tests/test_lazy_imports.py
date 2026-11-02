"""Lazy imports for the cloud profile: torch-free token counting, warm-up gating, CKIP
skip, the lazy HuggingFace embedding import and the docling converter's model gating.

The end-to-end "nothing local is imported" proof is tests/test_cloud_profile_no_local_models.py.
"""

import importlib
from types import SimpleNamespace

import pytest

from src.config.model import RAGConfig
from src.config.roles import docling_warmup_needed
from src.domain.rag import token_counter as tc

_CFG = "src.config.config_manager.Config.get_config_model"


def _rag(**over):
    data = {
        "embedding": {"provider": "openai", "model": "e", "dimension": 8},
        "chunking": {}, "retrieval": {"default_top_k": 1, "default_similarity_cutoff": 0,
                                      "hybrid_search": False},
        "vector_store": {"type": "pgvector", "table_prefix": "v"},
    }
    data.update(over)
    return RAGConfig.model_validate(data)


CLOUD = dict(profile="cloud", parsing={"vision": {"model": "v", "base_url": "http://x/v1"}})


@pytest.fixture(autouse=True)
def _clear_tokenizer_cache():
    tc._cache.clear()
    yield
    tc._cache.clear()


def _tokenizer_json(dirpath):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tok = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    dirpath.mkdir(parents=True)
    tok.save(str(dirpath / "tokenizer.json"))


class TestTokenCounter:
    def test_bundled_exact_and_fallback(self, tmp_path):
        _tokenizer_json(tmp_path / "hf_tokenizers" / "BAAI--bge-m3")
        exact = tc.load_light_tokenizer("BAAI/bge-m3", tmp_path)
        assert isinstance(exact, tc.FastTokenizer) and exact.source == "bundled:BAAI--bge-m3"
        assert exact.encode("hello world again", add_special_tokens=False) == [1, 2, 0]
        other = tc.load_light_tokenizer("someone/else", tmp_path)
        assert other.source == "bundled:BAAI--bge-m3"  # any bundled tokenizer beats an estimate
        assert tc.load_light_tokenizer("BAAI/bge-m3", tmp_path) is exact  # cached

    def test_hub_download_when_online(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        from tokenizers import Tokenizer
        _tokenizer_json(tmp_path / "t")
        monkeypatch.setattr(Tokenizer, "from_pretrained",
                            staticmethod(lambda name: Tokenizer.from_file(str(tmp_path / "t" / "tokenizer.json"))))
        tok = tc.load_light_tokenizer("org/model", None)
        assert tok.source == "hub:org/model"

    def test_offline_or_unknown_falls_back_to_estimate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        tok = tc.load_light_tokenizer("org/model", tmp_path)
        assert isinstance(tok, tc.CharEstimateTokenizer)
        assert len(tok.encode("台北市 abcdefgh")) == 3 + 3  # 3 CJK + ceil(6/4)... incl. space

    def test_hub_failure_falls_back_to_estimate(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        from tokenizers import Tokenizer

        def boom(name):
            raise RuntimeError("404")
        monkeypatch.setattr(Tokenizer, "from_pretrained", staticmethod(boom))
        assert isinstance(tc.load_light_tokenizer("nope/nope", None), tc.CharEstimateTokenizer)

    def test_no_tokenizers_package(self, monkeypatch):
        import builtins
        real = builtins.__import__

        def fake(name, *a, **kw):
            if name == "tokenizers":
                raise ImportError("not installed")
            return real(name, *a, **kw)
        monkeypatch.setattr(builtins, "__import__", fake)
        assert isinstance(tc.load_light_tokenizer("x/y", None), tc.CharEstimateTokenizer)

    def test_bundled_dir_without_tokenizers(self, tmp_path):
        assert tc._bundled("x", tmp_path) is None
        (tmp_path / "hf_tokenizers").mkdir()
        assert tc._bundled("x", tmp_path) is None
        assert tc._bundled("x", None) is None

    def test_prefer_light(self, monkeypatch):
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(**CLOUD)))
        assert tc.prefer_light_tokenizer() is True
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag()))
        monkeypatch.setattr(tc.importlib.util, "find_spec", lambda name: object())
        assert tc.prefer_light_tokenizer() is False
        monkeypatch.setattr(tc.importlib.util, "find_spec", lambda name: None)
        assert tc.prefer_light_tokenizer() is True

    def test_prefer_light_config_error(self, monkeypatch):
        def boom():
            raise RuntimeError("no config")
        monkeypatch.setattr(_CFG, boom)
        monkeypatch.setattr(tc.importlib.util, "find_spec", lambda name: object())
        assert tc.prefer_light_tokenizer() is False

    def test_get_hf_tokenizer_uses_light_path_in_cloud(self, monkeypatch, tmp_path):
        from src.domain.rag import docling_loader as dl
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(**CLOUD)))
        monkeypatch.setattr(dl, "resolve_assets_dir", lambda: tmp_path)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert isinstance(dl.get_hf_tokenizer("org/model"), tc.CharEstimateTokenizer)

    def test_get_hf_tokenizer_none_name_does_not_crash_k56(self, monkeypatch, tmp_path):
        from src.domain.rag import docling_loader as dl
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag()))
        monkeypatch.setattr(dl, "resolve_assets_dir", lambda: tmp_path)
        monkeypatch.setattr(tc.importlib.util, "find_spec", lambda name: object())
        assert isinstance(dl.get_hf_tokenizer(None), tc.CharEstimateTokenizer)


class TestWarmupAndConverterGating:
    def test_docling_warmup_needed(self):
        assert docling_warmup_needed(None)
        assert docling_warmup_needed(_rag())  # local parsing
        assert not docling_warmup_needed(_rag(**CLOUD))
        hybrid = _rag(profile="hybrid", parsing={"pdf": "cloud", "images": "cloud",
                                                 "vision": {"model": "v"}},
                      asr={"provider": "fireredasr"})
        assert not docling_warmup_needed(hybrid)  # FireRedASR is not docling
        hybrid.asr.provider = "docling-whisper"
        assert docling_warmup_needed(hybrid)

    def test_local_pdf_models_wanted(self, monkeypatch):
        from src.domain.rag import docling_loader as dl
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(**CLOUD)))
        assert dl._local_pdf_models_wanted() is False
        monkeypatch.setattr(_CFG, lambda: None)
        assert dl._local_pdf_models_wanted() is True

        def boom():
            raise RuntimeError("x")
        monkeypatch.setattr(_CFG, boom)
        assert dl._local_pdf_models_wanted() is True

    def test_cloud_converter_configures_no_models(self, monkeypatch):
        """Under cloud parsing the shared converter gets no PDF/image/audio pipelines, never
        probes the accelerator (which imports torch for cuda) and logs no weights warning."""
        from src.domain.rag import docling_loader as dl
        rag = _rag(**CLOUD)
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=rag))
        monkeypatch.setattr(dl, "_converter", None)
        monkeypatch.setattr(dl, "_resolve_artifacts_path", lambda: pytest.fail("artifacts looked up"))
        monkeypatch.setattr(dl, "_get_accelerator_options", lambda: pytest.fail("accelerator probed"))
        captured = {}

        class FakeConverter:
            def __init__(self, **kw):
                captured.update(kw)
        monkeypatch.setattr("docling.document_converter.DocumentConverter", FakeConverter)
        dl._build_converter()
        assert captured == {}  # no format_options: Office/Markdown/HTML backends only
        monkeypatch.setattr(dl, "_converter", None)


class TestCkipUnderCloud:
    def test_cloud_profile_never_loads_ckip(self, monkeypatch):
        from src.domain.rag.ckip_segmenter import CkipSegmenter
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(**CLOUD)))
        seg = CkipSegmenter()
        assert seg.segment("台北市政府") == "台北市政府"
        assert seg.segment_batch(["一", ""]) == ["一", ""]
        assert seg._ws is None and seg._load_failed

    def test_profile_lookup_failure_is_not_cloud(self, monkeypatch):
        from src.domain.rag import ckip_segmenter as cs

        def boom():
            raise RuntimeError("x")
        monkeypatch.setattr(_CFG, boom)
        assert cs._cloud_profile() is False


class TestLazyEmbeddingImport:
    def test_huggingface_provider_is_resolved_on_access(self):
        pkg = importlib.import_module("src.domain.rag.embedding")
        assert "HuggingFaceEmbeddingProvider" in pkg.__all__
        with pytest.raises(AttributeError):
            pkg.NoSuchProvider  # noqa: B018

    def test_factory_module_does_not_import_huggingface(self):
        import ast
        import inspect
        from src.domain.rag.embedding import factory
        tree = ast.parse(inspect.getsource(factory))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert not any("huggingface" in (getattr(n, "module", "") or "") for n in top)
