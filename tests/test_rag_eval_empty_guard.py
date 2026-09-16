"""run_eval must never score/persist an empty eval set.

If question generation yields no items (e.g. the LLM endpoint is down), evaluating
would produce all-zero metrics (n = len(items) or 1) and _append_history would bake
that fake data point into the baseline forever. run_eval must raise before
evaluate()/_append_history instead.

The module is a script (importlib-loaded in prod); collaborators are patched on the
loaded module so no DB / LLM is touched.
"""
import importlib.util
import os
from types import SimpleNamespace

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "rag_eval.py")


def _load():
    spec = importlib.util.spec_from_file_location("rag_eval_empty_guard", _PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_run_eval_raises_on_empty_set_and_writes_no_history(monkeypatch):
    m = _load()
    called = {"history": False, "evaluate": False}

    monkeypatch.setattr(
        m, "_resolve_folder",
        lambda ref: SimpleNamespace(name="t", id=1, vector_table_uuid="u"))
    # regenerate=True skips the cached-file branch; generation yields nothing.
    monkeypatch.setattr(m, "generate_eval_set", lambda folder, n, path: [])
    monkeypatch.setattr(m, "_read_history", lambda slug: [])
    monkeypatch.setattr(
        m, "_append_history",
        lambda slug, rep: called.__setitem__("history", True))

    async def _fake_evaluate(*a, **k):
        called["evaluate"] = True
        return {}
    monkeypatch.setattr(m, "evaluate", _fake_evaluate)

    with pytest.raises(RuntimeError, match="evaluation aborted"):
        m.run_eval("t", n=5, k=10, regenerate=True)

    assert called["evaluate"] is False, "must not evaluate an empty set"
    assert called["history"] is False, "must not write an empty run to history"
