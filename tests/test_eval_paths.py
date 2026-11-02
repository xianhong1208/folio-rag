"""Eval artifacts are keyed by folder id, so same-named folders of different owners never share them."""

from types import SimpleNamespace

import pytest

from src.domain.rag import eval_paths as ep


@pytest.fixture(autouse=True)
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, "REPORTS_DIR", str(tmp_path))
    return tmp_path


def _f(fid, name="docs"):
    return SimpleNamespace(id=fid, name=name)


def _write_legacy(reports, slug="docs"):
    for name in (f"rag_eval_{slug}.json", f"rag_eval_history_{slug}.jsonl", f"eval_set_{slug}.jsonl"):
        (reports / name).write_text("legacy")


def test_same_name_different_folders_get_different_paths():
    a = ep.report_slug(_f(1), name_is_unique=lambda n: False)
    b = ep.report_slug(_f(2), name_is_unique=lambda n: False)
    assert a == "1_docs" and b == "2_docs"
    assert ep.report_path(a) != ep.report_path(b)
    assert ep.history_path(a).endswith("rag_eval_history_1_docs.jsonl")
    assert ep.eval_set_path(a).endswith("eval_set_1_docs.jsonl")


def test_slashes_are_flattened():
    assert ep.report_slug(_f(3, "a/b"), name_is_unique=lambda n: True) == "3_a_b"


def test_legacy_files_adopted_when_name_is_unique(reports):
    _write_legacy(reports)
    slug = ep.report_slug(_f(1), name_is_unique=lambda n: True)
    assert (reports / "rag_eval_1_docs.json").read_text() == "legacy"
    assert (reports / "rag_eval_history_1_docs.jsonl").exists()
    assert (reports / "eval_set_1_docs.jsonl").exists()
    assert not (reports / "rag_eval_docs.json").exists()
    assert slug == "1_docs"


def test_legacy_files_left_alone_when_name_is_shared(reports):
    _write_legacy(reports)
    ep.report_slug(_f(1), name_is_unique=lambda n: False)
    assert (reports / "rag_eval_docs.json").exists()
    assert not (reports / "rag_eval_1_docs.json").exists()


def test_new_files_win_over_legacy(reports):
    _write_legacy(reports)
    (reports / "rag_eval_1_docs.json").write_text("new")
    ep.report_slug(_f(1), name_is_unique=lambda n: True)
    assert (reports / "rag_eval_1_docs.json").read_text() == "new"
    assert (reports / "rag_eval_docs.json").exists()


def test_default_uniqueness_check_queries_folders(reports, monkeypatch):
    from unittest.mock import patch
    _write_legacy(reports)
    with patch("db.folderdb.FolderDB.get", return_value=[_f(1), _f(2)]) as g:
        ep.report_slug(_f(1))
    g.assert_called_once_with(name="docs")
    assert (reports / "rag_eval_docs.json").exists()
    with patch("db.folderdb.FolderDB.get", return_value=[_f(1)]):
        ep.report_slug(_f(1))
    assert (reports / "rag_eval_1_docs.json").exists()
