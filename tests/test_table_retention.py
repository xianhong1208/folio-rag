"""Retention purges for QueryLogs / SettingsAudit / IndexJobs and their scheduler."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from db import retention_db
from src.config.model import DatabaseConfig, RetentionConfig


def _engine(rowcounts):
    """Engine whose begin() yields a conn returning the given rowcounts in turn."""
    conn = MagicMock()
    conn.execute.side_effect = [MagicMock(rowcount=n) for n in rowcounts]
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    return engine, conn


class TestPurgeOlderThan:
    def test_deletes_in_batches_until_a_short_batch(self):
        engine, conn = _engine([100, 100, 7])
        n = retention_db.purge_older_than("query_logs", 90, batch_size=100, engine=engine)
        assert n == 207
        assert engine.begin.call_count == 3          # one short transaction per batch
        sql = str(conn.execute.call_args_list[0][0][0])
        assert 'DELETE FROM "QueryLogs" WHERE id IN (SELECT id FROM "QueryLogs"' in sql
        assert "created_at < :cutoff" in sql and "LIMIT :n" in sql
        params = conn.execute.call_args_list[0][0][1]
        assert params["n"] == 100 and isinstance(params["cutoff"], datetime)

    def test_cutoff_is_days_before_now(self):
        engine, conn = _engine([0])
        now = datetime(2026, 11, 3, tzinfo=timezone.utc)
        retention_db.purge_older_than("settings_audit", 365, engine=engine, now=now)
        params = conn.execute.call_args[0][1]
        assert params["cutoff"] == datetime(2025, 11, 3, tzinfo=timezone.utc)
        assert '"SettingsAudit"' in str(conn.execute.call_args[0][0])

    def test_index_jobs_uses_iso_string_and_terminal_statuses_only(self):
        engine, conn = _engine([0])
        now = datetime(2026, 11, 3, tzinfo=timezone.utc)
        retention_db.purge_older_than("index_jobs", 30, engine=engine, now=now)
        sql = str(conn.execute.call_args[0][0])
        assert "job_id IN (SELECT job_id" in sql
        assert "status IN ('succeeded', 'partial_success', 'failed', 'cancelled', 'interrupted')" in sql
        assert "running" not in sql and "pending" not in sql
        assert conn.execute.call_args[0][1]["cutoff"] == "2026-10-04T00:00:00+00:00"

    @pytest.mark.parametrize("days", [0, -1, None])
    def test_non_positive_days_keeps_everything(self, days):
        engine, conn = _engine([])
        assert retention_db.purge_older_than("query_logs", days, engine=engine) == 0
        engine.begin.assert_not_called()

    def test_uses_shared_engine_by_default(self):
        engine, _ = _engine([0])
        with patch("db.db.get_engine", return_value=engine):
            retention_db.purge_older_than("query_logs", 1)
        engine.begin.assert_called_once()


class TestRunRetention:
    def test_applies_each_window_and_isolates_failures(self):
        calls = []

        def fake(name, days, batch, engine=None):
            calls.append((name, days, batch))
            if name == "query_logs":
                raise RuntimeError("table missing")
            return 3

        conf = RetentionConfig(query_logs_days=10, settings_audit_days=20, index_jobs_days=5,
                               batch_size=500)
        with patch.object(retention_db, "purge_older_than", side_effect=fake):
            result = retention_db.run_retention(conf)
        assert result == {"index_jobs": 3, "query_logs": None, "settings_audit": 3}
        assert ("settings_audit", 20, 500) in calls and ("index_jobs", 5, 500) in calls

    def test_defaults(self):
        with patch.object(retention_db, "purge_older_than", return_value=0) as p:
            retention_db.run_retention()
        windows = {c.args[0]: c.args[1] for c in p.call_args_list}
        assert windows == {"index_jobs": 30, "query_logs": 90, "settings_audit": 365}


class TestRetentionConfig:
    def test_database_config_gets_default_retention(self):
        db = DatabaseConfig(url="postgresql://x/y", echo=False, pool_size=1, max_overflow=0,
                            pool_timeout=1, pool_recycle=1)
        assert db.retention.query_logs_days == 90 and db.retention.interval_hours == 24

    @pytest.mark.parametrize("kwargs", [{"query_logs_days": -1}, {"batch_size": 10},
                                        {"interval_hours": 0}])
    def test_rejects_bad_values(self, kwargs):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RetentionConfig(**kwargs)


class TestScheduler:
    async def test_run_once_offloads_and_returns_result(self):
        from src.utils import retention_scheduler as rs
        with patch("db.retention_db.run_retention", return_value={"query_logs": 4}) as rr:
            out = await rs.run_once(RetentionConfig())
        assert out == {"query_logs": 4}
        rr.assert_called_once()

    async def test_loop_runs_at_start_then_each_interval_and_survives_errors(self):
        from src.utils import retention_scheduler as rs
        results = [RuntimeError("db down"), {"query_logs": 0}, {"query_logs": 1}]

        def fake(conf):
            r = results.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with patch("db.retention_db.run_retention", side_effect=fake):
            task = asyncio.create_task(rs.retention_loop(RetentionConfig(), interval_seconds=0.01))
            for _ in range(200):
                if not results:
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert results == []   # first round failed, later rounds still ran

    async def test_start_uses_configured_retention(self):
        from src.utils import retention_scheduler as rs
        conf = RetentionConfig(interval_hours=1)
        db_conf = MagicMock(retention=conf)
        with patch("src.config.config_manager.Config.get_database_config", return_value=db_conf), \
             patch("db.retention_db.run_retention", return_value={}) as rr:
            task = rs.start()
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert rr.call_args[0][0] is conf

    def test_conf_falls_back_to_defaults(self):
        from src.utils import retention_scheduler as rs
        with patch("src.config.config_manager.Config.get_database_config", side_effect=RuntimeError):
            assert rs._retention_conf() == RetentionConfig()
        with patch("src.config.config_manager.Config.get_database_config",
                   return_value=MagicMock(retention="nope")):
            assert rs._retention_conf() == RetentionConfig()
