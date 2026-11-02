"""Lazy imports for the cloud profile: torch-free chunk token counting, warm-up gating, CKIP
skip, the lazy HuggingFace embedding import and the docling converter's model gating.

The end-to-end "nothing local is imported" proof is tests/test_cloud_profile_no_local_models.py.
"""

import importlib
from types import SimpleNamespace

import pytest

from src.config.model import RAGConfig
from src.config.roles import docling_warmup_needed
from src.domain.rag import chunk_tokenizer as ct

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
    ct.clear_cache()
    yield
    ct.clear_cache()


def _tokenizer_json(dirpath):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tok = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    dirpath.mkdir(parents=True)
    tok.save(str(dirpath / "tokenizer.json"))


class TestChunkTokenizerInCloud:
    """Chunk token counting is torch-free in the cloud profile (chunk_tokenizer)."""

    def test_tokenizer_json_dir_is_used(self, tmp_path, monkeypatch):
        _tokenizer_json(tmp_path / "hf_tokenizers" / "BAAI--bge-m3")
        monkeypatch.setattr(ct, "_assets_tokenizers_dir", lambda: tmp_path / "hf_tokenizers")
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(**CLOUD)))
        tok = ct.get_chunk_tokenizer("BAAI/bge-m3")
        assert isinstance(tok, ct.FastTokenizerCounter) and tok.name == "BAAI--bge-m3"
        assert tok.encode("hello world again") == [1, 2, 0]

    def test_slow_only_dir_never_imports_transformers_in_cloud(self, tmp_path, monkeypatch):
        (tmp_path / "hf_tokenizers" / "org--slow").mkdir(parents=True)  # no tokenizer.json
        monkeypatch.setattr(ct, "_assets_tokenizers_dir", lambda: tmp_path / "hf_tokenizers")
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(**CLOUD)))
        import builtins
        real = builtins.__import__

        def guard(name, *a, **kw):
            if name.startswith("transformers"):
                raise AssertionError("transformers imported under profile: cloud")
            return real(name, *a, **kw)
        monkeypatch.setattr(builtins, "__import__", guard)
        tok = ct.get_chunk_tokenizer("org/slow")
        assert tok.name in (ct.BUNDLED_TOKENIZER_NAME, "char-estimate")

    def test_transformers_allowed(self, monkeypatch):
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(**CLOUD)))
        assert ct.transformers_allowed() is False
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag()))
        monkeypatch.setattr(ct.importlib.util, "find_spec", lambda name: object())
        assert ct.transformers_allowed() is True
        monkeypatch.setattr(ct.importlib.util, "find_spec", lambda name: None)
        assert ct.transformers_allowed() is False

    def test_transformers_allowed_config_error(self, monkeypatch):
        def boom():
            raise RuntimeError("no config")
        monkeypatch.setattr(_CFG, boom)
        monkeypatch.setattr(ct.importlib.util, "find_spec", lambda name: object())
        assert ct.transformers_allowed() is True

    def test_get_hf_tokenizer_delegates_and_never_downloads(self, monkeypatch, tmp_path):
        from src.domain.rag import docling_loader as dl
        monkeypatch.setattr(ct, "_assets_tokenizers_dir", lambda: None)
        monkeypatch.setattr(_CFG, lambda: SimpleNamespace(rag=_rag(**CLOUD)))
        tok = dl.get_hf_tokenizer(None)  # OpenAI / Azure embeddings: no HF name
        assert tok.name in (ct.BUNDLED_TOKENIZER_NAME, "char-estimate")


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
