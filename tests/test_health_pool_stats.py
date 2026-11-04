"""Connection-pool block of /health/detailed, against real SQLAlchemy pools.

The block once called ``checked_in_connections()`` / ``checked_out_connections()``,
which do not exist on ``QueuePool``; the AttributeError was swallowed and the pool
was reported "degraded" on every call. These tests use real pools, not mocks, so a
wrong method name fails here.
"""
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool, QueuePool

from src.api.router import health
from src.api.router.health import _pool_stats


def _queue_pool(size=2, overflow=3):
    return QueuePool(lambda: MagicMock(), pool_size=size, max_overflow=overflow)


def test_idle_pool_reports_zero_overflow():
    stats = _pool_stats(_queue_pool(size=2))
    assert stats == {"size": 2, "checked_in": 0, "checked_out": 0, "overflow": 0}


def test_counts_follow_checkouts_and_overflow():
    pool = _queue_pool(size=2, overflow=3)
    conns = [pool.connect() for _ in range(3)]  # one past pool_size
    stats = _pool_stats(pool)
    assert stats["checked_out"] == 3
    assert stats["overflow"] == 1

    for c in conns:
        c.close()
    stats = _pool_stats(pool)
    assert stats["checked_out"] == 0
    assert stats["checked_in"] == 2  # the overflow connection is discarded, not pooled


def test_pool_without_counters_raises():
    with pytest.raises(AttributeError):
        _pool_stats(NullPool(lambda: MagicMock()))


async def test_detailed_health_reports_pool_healthy(monkeypatch):
    engine = create_engine("sqlite://", poolclass=QueuePool, pool_size=3, max_overflow=1)
    monkeypatch.setattr(health, "get_engine", lambda: engine)

    body = await health.detailed_health_check()

    pool = body["checks"]["connection_pool"]
    assert pool["status"] == "healthy", pool
    assert pool["size"] == 3
    assert pool["checked_out"] == 0
    assert pool["overflow"] == 0
