"""CKIP runs on the CPU (the only rag.retrieval.ckip_device since 1.5.1) without progress
bars, and a segmenter that falls back to raw text flags the folder for Rebuild FTS (M7)."""

import sys
import types
from types import SimpleNamespace

import pytest

from pydantic import ValidationError

from src.config.model import RAGRetrievalConfig
from src.domain.rag import ckip_segmenter as cs
from src.domain.rag import text_segmenter as ts


def _retrieval(ckip_device):
    return RAGRetrievalConfig(default_top_k=5, default_similarity_cutoff=0.0, hybrid_search=True,
                              ckip_device=ckip_device)


@pytest.mark.parametrize("value", ["cpu", "CPU", " cpu ", "", None])
def test_ckip_device_accepts_cpu(value):
    assert _retrieval(value).ckip_device == "cpu"


@pytest.mark.parametrize("value", ["cuda", "CUDA", "cuda:0", "cuda:4", "auto"])
def test_ckip_device_refuses_gpu_values_with_the_upgrade_guide(value):
    """1.5.1 installs CPU torch only: a GPU device stops the start (no silent CPU fallback)
    and says what to change."""
    with pytest.raises(ValidationError) as e:
        _retrieval(value)
    msg = str(e.value)
    assert "removed in Folio 1.5.1" in msg and "CKIP_DEVICE" in msg and "docs/migrate-1.5.md" in msg


@pytest.mark.parametrize("value", ["tpu", "mps", "gpu"])
def test_ckip_device_refuses_unknown_values(value):
    with pytest.raises(ValidationError, match="the only value is cpu"):
        _retrieval(value)


class _FakeWS:
    instances = []

    def __init__(self, model, device=-1):
        self.model, self.device, self.calls = model, device, []
        _FakeWS.instances.append(self)

    def __call__(self, texts, show_progress=True):
        self.calls.append(show_progress)
        if any("boom" in t for t in texts):
            raise RuntimeError("CUDA OOM")
        return [list(t) for t in texts]


@pytest.fixture
def fake_ckip(monkeypatch):
    _FakeWS.instances.clear()
    pkg = types.ModuleType("ckip_transformers")
    nlp = types.ModuleType("ckip_transformers.nlp")
    nlp.CkipWordSegmenter = _FakeWS
    pkg.nlp = nlp
    monkeypatch.setitem(sys.modules, "ckip_transformers", pkg)
    monkeypatch.setitem(sys.modules, "ckip_transformers.nlp", nlp)
    monkeypatch.setattr(cs, "_cloud_profile", lambda: False)
    return _FakeWS


def test_runs_on_the_cpu_without_progress_bar(fake_ckip):
    seg = cs.CkipSegmenter()
    assert seg.segment_batch(["台北", ""]) == ["台 北", ""]
    assert seg.segment("市政府") == "市 政 府"
    ws = fake_ckip.instances[0]
    assert ws.device == -1  # ckip-transformers' CPU
    assert ws.calls == [False, False]  # tqdm off on both paths


def test_batch_report_flags_runtime_fallback(fake_ckip):
    seg = cs.CkipSegmenter()
    assert seg.segment_batch_report(["台北"]) == (["台 北"], False)
    assert seg.segment_batch_report(["boom 台北"]) == (["boom 台北"], True)
    assert seg.segment_batch(["boom"]) == ["boom"]
    assert seg.segment_batch_report([]) == ([], False)


def test_batch_report_flags_unavailable_model(monkeypatch):
    seg = cs.CkipSegmenter()
    seg._load_failed = True
    assert seg.segment_batch_report(["台北"]) == (["台北"], True)


def test_jieba_report(monkeypatch):
    seg = ts.JiebaSegmenter()
    seg._failed = True
    assert seg.segment_batch_report(["台北", ""]) == (["台北", ""], True)
    seg2 = ts.JiebaSegmenter()
    seg2._jieba = SimpleNamespace(lcut=lambda t: list(t))
    assert seg2.segment_batch_report(["台北"]) == (["台 北"], False)
    seg2._jieba = SimpleNamespace(lcut=lambda t: (_ for _ in ()).throw(ValueError("x")))
    assert seg2.segment_batch_report(["台北"]) == (["台北"], True)
