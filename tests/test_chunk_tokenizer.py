"""Chunk token counting never downloads and never crashes (fresh clone, cloud embeddings)."""

import sys
from pathlib import Path

import pytest

from src.domain.rag import chunk_tokenizer as ct


@pytest.fixture(autouse=True)
def _fresh_cache():
    ct.clear_cache()
    yield
    ct.clear_cache()


@pytest.fixture
def no_assets(monkeypatch):
    monkeypatch.setattr(ct, "_assets_tokenizers_dir", lambda: None)


def test_bundled_tokenizer_is_shipped():
    p = ct.bundled_tokenizer_path()
    assert p is not None and p.name == "tokenizer.json.xz"


def test_none_model_name_uses_bundled_without_network(no_assets, monkeypatch):
    # A fresh clone with OpenAI/Azure embeddings: no assets, no HF model name.
    def _boom(*a, **k):
        raise AssertionError("must not download")

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    tok = ct.get_chunk_tokenizer(None)
    assert isinstance(tok, ct.FastTokenizerCounter)
    assert tok.name == ct.BUNDLED_TOKENIZER_NAME
    ids = tok.encode("The hardware warranty is 36 months. 保固期間為三十六個月。")
    assert 10 < len(ids) < 40


def test_bundled_matches_multilingual_e5_ids(no_assets):
    tok = ct.get_chunk_tokenizer("intfloat/multilingual-e5-large")
    # Known XLM-R ids: '▁Hello' = 35378, '▁world' = 8999
    assert tok.encode("Hello world") == [35378, 8999]


def test_bundled_does_not_import_transformers(no_assets, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # any import would raise
    tok = ct.get_chunk_tokenizer("text-embedding-3-small")
    assert tok.encode("abc")


def test_exact_assets_match_preferred(tmp_path, monkeypatch):
    base = tmp_path / "hf_tokenizers"
    (base / "aaa--first").mkdir(parents=True)
    exact = base / "org--model"
    exact.mkdir()
    loaded = []
    monkeypatch.setattr(ct, "_assets_tokenizers_dir", lambda: base)
    monkeypatch.setattr(ct, "_load_dir", lambda p: loaded.append(p.name) or ct.CharEstimateCounter())
    ct.get_chunk_tokenizer("org/model")
    assert loaded == ["org--model"]


def test_assets_fallback_to_any_local_dir(tmp_path, monkeypatch):
    base = tmp_path / "hf_tokenizers"
    (base / "zzz--other").mkdir(parents=True)
    loaded = []
    monkeypatch.setattr(ct, "_assets_tokenizers_dir", lambda: base)
    monkeypatch.setattr(ct, "_load_dir", lambda p: loaded.append(p.name) or ct.CharEstimateCounter())
    ct.get_chunk_tokenizer("org/missing")
    assert loaded == ["zzz--other"]


def test_load_dir_with_tokenizer_json(tmp_path):
    import lzma

    src = ct.bundled_tokenizer_path()
    d = tmp_path / "org--model"
    d.mkdir()
    (d / "tokenizer.json").write_bytes(lzma.decompress(src.read_bytes()))
    tok = ct._load_dir(d)
    assert tok.encode("Hello world") == [35378, 8999]


def test_broken_assets_dir_falls_back_to_bundled(tmp_path, monkeypatch):
    base = tmp_path / "hf_tokenizers"
    (base / "org--model").mkdir(parents=True)
    (base / "org--model" / "tokenizer.json").write_text("not json")
    monkeypatch.setattr(ct, "_assets_tokenizers_dir", lambda: base)
    tok = ct.get_chunk_tokenizer("org/model")
    assert tok.name == ct.BUNDLED_TOKENIZER_NAME


def test_missing_bundle_degrades_to_char_estimate(no_assets, monkeypatch):
    monkeypatch.setattr(ct, "bundled_tokenizer_path", lambda: None)
    tok = ct.get_chunk_tokenizer(None)
    assert isinstance(tok, ct.CharEstimateCounter)
    assert len(tok.encode("保固期間")) == 4
    assert len(tok.encode("abcdefgh")) == 2


def test_unreadable_bundle_degrades_to_char_estimate(no_assets, monkeypatch, tmp_path):
    bad = tmp_path / "tokenizer.json.xz"
    bad.write_bytes(b"not xz")
    monkeypatch.setattr(ct, "bundled_tokenizer_path", lambda: bad)
    assert isinstance(ct.get_chunk_tokenizer(None), ct.CharEstimateCounter)


def test_result_is_cached(no_assets):
    assert ct.get_chunk_tokenizer(None) is ct.get_chunk_tokenizer("other/name")


def test_assets_dir_resolution(tmp_path, monkeypatch):
    (tmp_path / "hf_tokenizers").mkdir()
    import src.utils.runtime_paths as rp

    monkeypatch.setattr(rp, "resolve_external_dir", lambda name, dev_root: tmp_path)
    assert ct._assets_tokenizers_dir() == tmp_path / "hf_tokenizers"
    monkeypatch.setattr(rp, "resolve_external_dir", lambda name, dev_root: None)
    assert ct._assets_tokenizers_dir() is None


def test_docling_loader_get_hf_tokenizer_never_crashes_on_none(no_assets):
    # Regression: openai/azure embeddings pass None; this used to call
    # AutoTokenizer.from_pretrained(None) and fail every docling-parsed file.
    from src.domain.rag.docling_loader import get_hf_tokenizer

    tok = get_hf_tokenizer(None)
    assert tok.encode("hello", add_special_tokens=False)
