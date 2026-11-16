"""The server's backup schedule and its /health/detailed view."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.backup import archive as A
from src.backup import scheduler as SCH
from src.backup import service as S
from src.config.model import BackupConfig


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(SCH, "_state", {"running": False, "last_attempt_at": None, "last_result": None,
                                        "last_error": None, "next_run_at": None, "waiting_since": None})


def _conf(tmp_path, **kw):
    return BackupConfig(dir=str(tmp_path), **kw)


def _archive(tmp_path: Path, age: timedelta):
    p = tmp_path / A.archive_name(datetime.now(timezone.utc) - age)
    p.write_bytes(b"x" * 10)
    return p


class TestStatus:
    def test_disabled_still_shows_the_newest_archive(self, tmp_path):
        _archive(tmp_path, timedelta(hours=3))
        v = SCH.status(_conf(tmp_path))
        assert v["status"] == "disabled" and v["enabled"] is False
        assert v["latest"]["size_bytes"] == 10 and 2.9 < v["latest"]["age_hours"] < 3.1
        assert "last_error" not in v

    def test_enabled_and_fresh(self, tmp_path):
        _archive(tmp_path, timedelta(hours=1))
        v = SCH.status(_conf(tmp_path, enabled=True))
        assert v["status"] == "healthy" and v["last_result"] is None

    def test_enabled_without_any_archive_yet(self, tmp_path):
        assert SCH.status(_conf(tmp_path, enabled=True))["status"] == "healthy"

    def test_overdue_is_degraded(self, tmp_path):
        _archive(tmp_path, timedelta(hours=50))
        v = SCH.status(_conf(tmp_path, enabled=True, interval_hours=24))
        assert v["status"] == "degraded" and "older than two intervals" in v["message"]

    def test_a_failed_round_is_degraded(self, tmp_path):
        SCH._state["last_result"] = "failed"
        assert SCH.status(_conf(tmp_path, enabled=True))["status"] == "degraded"

    def test_waiting_is_not_degraded(self, tmp_path):
        SCH._state["last_result"] = "waiting"
        assert SCH.status(_conf(tmp_path, enabled=True))["status"] == "healthy"

    def test_an_unreadable_directory_does_not_raise(self, tmp_path, monkeypatch):
        def boom(conf=None):
            raise OSError("gone")
        monkeypatch.setattr(SCH, "latest_archive", boom)
        assert SCH.status(_conf(tmp_path))["latest"] is None


class TestDueAt:
    def test_first_run_shortly_after_start(self, tmp_path):
        now = datetime(2026, 11, 14, tzinfo=timezone.utc)
        assert SCH.due_at(_conf(tmp_path), now) == now + SCH.FIRST_RUN_DELAY

    def test_one_interval_after_the_newest_archive(self, tmp_path):
        p = _archive(tmp_path, timedelta(hours=1))
        made = A.archive_time(p.name)
        assert SCH.due_at(_conf(tmp_path, interval_hours=24)) == made + timedelta(hours=24)

    def test_an_overdue_backup_runs_soon(self, tmp_path):
        _archive(tmp_path, timedelta(hours=100))
        due = SCH.due_at(_conf(tmp_path, interval_hours=24))
        assert due - datetime.now(timezone.utc) <= SCH.FIRST_RUN_DELAY + timedelta(seconds=5)


class _Stop(Exception):
    pass


def _run_loop(conf, monkeypatch, outcomes, rounds):
    """Run backup_loop for ``rounds`` rounds; outcomes[i] is a result dict or an exception."""
    calls = []

    def fake_run_once(c, allow_running_jobs=False):
        calls.append(allow_running_jobs)
        out = outcomes[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        return out

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(calls) >= rounds:
            raise _Stop()

    monkeypatch.setattr(SCH, "run_once", fake_run_once)
    monkeypatch.setattr(SCH.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(SCH, "due_at", lambda c, now=None: datetime.now(timezone.utc) - timedelta(seconds=1))
    with pytest.raises(_Stop):
        asyncio.run(SCH.backup_loop(conf))
    return calls, sleeps


class TestLoop:
    def test_success_then_next_interval(self, tmp_path, monkeypatch):
        ok = {"path": tmp_path / "a.tar", "size": 1, "removed": 2, "warnings": ["w"]}
        calls, sleeps = _run_loop(_conf(tmp_path, enabled=True, interval_hours=6), monkeypatch, [ok], 1)
        assert calls == [False] and SCH._state["last_result"] == "ok"
        assert 6 * 3600 - 5 < sleeps[-1] <= 6 * 3600

    def test_waits_for_index_jobs_then_forces_after_the_max_wait(self, tmp_path, monkeypatch):
        ok = {"path": tmp_path / "a.tar", "size": 1, "removed": 0, "warnings": []}
        monkeypatch.setattr(SCH, "BUSY_MAX_WAIT", timedelta(0))
        calls, sleeps = _run_loop(_conf(tmp_path, enabled=True), monkeypatch,
                                  [S.JobsRunning("busy"), ok], 2)
        assert calls == [False, True]
        assert SCH._state["last_result"] == "ok" and SCH._state["waiting_since"] is None

    def test_waiting_is_reported(self, tmp_path, monkeypatch):
        calls, sleeps = _run_loop(_conf(tmp_path, enabled=True), monkeypatch, [S.JobsRunning("busy")], 1)
        assert SCH._state["last_result"] == "waiting" and SCH._state["waiting_since"] is not None
        assert abs(sleeps[-1] - SCH.BUSY_RETRY.total_seconds()) < 5

    def test_a_failure_is_reported_without_details(self, tmp_path, monkeypatch):
        calls, sleeps = _run_loop(_conf(tmp_path, enabled=True), monkeypatch,
                                  [RuntimeError("password authentication failed for user x")], 1)
        assert SCH._state["last_result"] == "failed"
        assert SCH._state["last_error"] == "RuntimeError (see the server log)"
        assert abs(sleeps[-1] - SCH.FAILURE_RETRY.total_seconds()) < 5


class TestRunOnceAndStart:
    def test_run_once_backs_up_then_prunes(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        from src.config.config_manager import Config
        seen = {}
        monkeypatch.setattr(Config, "get_database_config",
                            classmethod(lambda cls: SimpleNamespace(url="postgresql://u@h/db", port=None)))
        monkeypatch.setattr(Config, "get_config_model", classmethod(lambda cls: SimpleNamespace(rag="R")))

        def fake_create(**kw):
            seen.update(kw)
            return S.BackupResult(path=tmp_path / "a.tar", size=3, manifest={}, warnings=[])
        monkeypatch.setattr(S, "create_backup", fake_create)
        monkeypatch.setattr(S, "prune", lambda d, keep: [d / "old.tar"] if keep == 3 else [])
        out = SCH.run_once(_conf(tmp_path, keep=3), allow_running_jobs=True)
        assert out["removed"] == 1 and seen["allow_running_jobs"] is True and seen["rag_conf"] == "R"
        assert seen["out_dir"] == tmp_path

    def test_start_only_when_enabled(self, tmp_path, monkeypatch):
        assert SCH.start(_conf(tmp_path)) is None

        async def main():
            async def never(conf):
                await asyncio.Event().wait()
            monkeypatch.setattr(SCH, "backup_loop", never)
            task = SCH.start(_conf(tmp_path, enabled=True))
            assert task is not None and task.get_name() == "backup-schedule"
            task.cancel()
        asyncio.run(main())
