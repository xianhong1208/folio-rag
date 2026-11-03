"""CKIP runs on the configured device without progress bars, and a segmenter that
falls back to raw text flags the folder for Rebuild FTS (M7)."""

import sys
import types
from types import SimpleNamespace

import pytest

from src.domain.rag import ckip_segmenter as cs
from src.domain.rag import text_segmenter as ts


@pytest.mark.parametrize("spec, expected", [
    ("cpu", -1), (None, -1), ("CUDA", 0), ("cuda:1", 1), ("cuda:x", -1), ("tpu", -1)])
def test_resolve_device(spec, expected):
    assert cs.resolve_device(spec) == expected


def test_resolve_auto_follows_torch(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert cs.resolve_device("auto") == 0
    fake_torch.cuda.is_available = lambda: False
    assert cs.resolve_device("auto") == -1


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


def test_device_from_config_and_no_progress_bar(fake_ckip, monkeypatch):
    monkeypatch.setattr(cs, "_configured_device", lambda: "cuda:2")
    seg = cs.CkipSegmenter()
    assert seg.segment_batch(["台北", ""]) == ["台 北", ""]
    assert seg.segment("市政府") == "市 政 府"
    ws = fake_ckip.instances[0]
    assert ws.device == 2
    assert ws.calls == [False, False]  # tqdm off on both paths


def test_configured_device_reads_retrieval_key(monkeypatch):
    monkeypatch.setattr("src.config.config_manager.Config.get_config_model",
                        lambda: SimpleNamespace(rag=SimpleNamespace(
                            retrieval=SimpleNamespace(ckip_device="cuda:0"))))
    assert cs._configured_device() == "cuda:0"
    monkeypatch.setattr("src.config.config_manager.Config.get_config_model",
                        lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert cs._configured_device() == "cpu"


def test_batch_report_flags_runtime_fallback(fake_ckip):
    seg = cs.CkipSegmenter(device="cpu")
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
