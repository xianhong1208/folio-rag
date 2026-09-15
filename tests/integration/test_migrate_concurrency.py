
"""Integration tests -- migration concurrency protection via pg_advisory_lock.

Two independent DatabaseMigrator instances upgrade concurrently (simulating a
rolling deploy); the advisory lock serializes them so both return success and
the version converges to head, rather than one crashing on concurrent DDL.
"""

import threading

from db.migrate import DatabaseMigrator


def test_concurrent_upgrade_serialized_by_advisory_lock(itest_db):
    """After a one-step downgrade, two threads upgrade('head') concurrently: both succeed, version converges."""
    m0 = DatabaseMigrator()
    head = m0._get_head_revision()
    assert m0.downgrade("-1"), "前置:先退一步製造『有 migration 要套』的狀態"

    results = {}

    def _run(tag: str):
        try:
            # Each with its own migrator (independent engine/connection) -- simulating two processes
            results[tag] = DatabaseMigrator().upgrade("head")
        except Exception as e:  # must not blow each other up
            results[tag] = e

    t1 = threading.Thread(target=_run, args=("a",))
    t2 = threading.Thread(target=_run, args=("b",))
    t1.start(); t2.start()
    t1.join(timeout=60); t2.join(timeout=60)

    assert results.get("a") is True and results.get("b") is True, f"並發 upgrade 有一方失敗:{results}"
    assert m0._get_current_revision() == head, "版本必須收斂到 head"


def test_real_chain_passes_new_guards(itest_db):
    """The real versions/ chain still passes under the new triple guard (parseable / single head / no orphans)."""
    chain = DatabaseMigrator._discover_revisions(DatabaseMigrator().versions_dir)
    assert len(chain) >= 3
    assert chain[-1][0] == "20260819cafe01"
