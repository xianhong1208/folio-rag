"""Renamed environment variables: the FOLIO_* name wins, the old name still works."""
import pytest

from src.utils import env_compat


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for new, old in env_compat.RENAMED_ENV_VARS.items():
        monkeypatch.delenv(new, raising=False)
        monkeypatch.delenv(old, raising=False)
    monkeypatch.setattr(env_compat, "_warned", set())


def test_new_name_is_read(monkeypatch):
    monkeypatch.setenv("FOLIO_JOB_TIMEOUT_SECONDS", "30")
    assert env_compat.getenv("FOLIO_JOB_TIMEOUT_SECONDS") == "30"


def test_new_name_wins_over_old(monkeypatch):
    monkeypatch.setenv("FOLIO_JOB_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("RAG_JOB_TIMEOUT_SECONDS", "99")
    assert env_compat.getenv("FOLIO_JOB_TIMEOUT_SECONDS") == "30"


def test_old_name_is_a_fallback_with_one_warning(monkeypatch):
    warnings = []
    monkeypatch.setattr(env_compat.logger, "warning", warnings.append)
    monkeypatch.setenv("AGENTIC_RAG_CORE_DUMPS", "1")
    assert env_compat.getenv("FOLIO_CORE_DUMPS") == "1"
    assert env_compat.getenv("FOLIO_CORE_DUMPS") == "1"
    assert len(warnings) == 1
    assert "AGENTIC_RAG_CORE_DUMPS" in warnings[0] and "FOLIO_CORE_DUMPS" in warnings[0]


def test_unset_returns_default():
    assert env_compat.getenv("FOLIO_INDEXING_CONCURRENCY") is None
    assert env_compat.getenv("FOLIO_INDEXING_CONCURRENCY", "4") == "4"


def test_unrenamed_variable_has_no_fallback(monkeypatch):
    monkeypatch.setenv("FOLIO_STORAGE_ROOT", "/x")
    assert env_compat.getenv("FOLIO_STORAGE_ROOT") == "/x"
    assert env_compat.getenv("SOMETHING_ELSE") is None


def test_every_old_name_maps_to_a_folio_name():
    for new, old in env_compat.RENAMED_ENV_VARS.items():
        assert new.startswith("FOLIO_") and not old.startswith("FOLIO_")


def test_job_timeout_reads_the_old_name(monkeypatch):
    from src.domain.rag.index_job_manager import IndexingJobManager
    monkeypatch.setattr(env_compat.logger, "warning", lambda *_a, **_k: None)
    monkeypatch.setenv("RAG_JOB_TIMEOUT_SECONDS", "123")
    assert IndexingJobManager._resolve_job_timeout() == 123
    monkeypatch.setenv("FOLIO_JOB_TIMEOUT_SECONDS", "45")
    assert IndexingJobManager._resolve_job_timeout() == 45
