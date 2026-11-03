"""Integration test -- retention purges real rows in batches, keeps recent and live ones."""

import uuid

from sqlalchemy import text


def test_retention_purges_only_expired_rows(itest_db):
    from db.db import get_engine
    from db.retention_db import run_retention
    from src.config.model import RetentionConfig

    eng = get_engine()
    tag = uuid.uuid4().hex[:8]
    with eng.begin() as c:
        for i in range(250):   # old -> 3 batches of 100
            c.execute(text('INSERT INTO "QueryLogs"(query, created_at) '
                           "VALUES (:q, now() - interval '100 days')"), {"q": f"old-{tag}-{i}"})
        c.execute(text('INSERT INTO "QueryLogs"(query, created_at) VALUES (:q, now())'),
                  {"q": f"new-{tag}"})
        c.execute(text('INSERT INTO "SettingsAudit"(key, action, changed_at) '
                       "VALUES (:k, 'set', now() - interval '400 days'), (:k2, 'set', now())"),
                  {"k": f"old-{tag}", "k2": f"new-{tag}"})
        for status, age in (("succeeded", "40 days"), ("running", "40 days"), ("failed", "1 day")):
            c.execute(text(
                'INSERT INTO "IndexJobs"(job_id, folder_id, status, last_updated_at) '
                "VALUES (:j, 987654, :s, to_char((now() - CAST(:age AS interval)) AT TIME ZONE 'UTC', "
                "'YYYY-MM-DD\"T\"HH24:MI:SS') || '+00:00')"),
                {"j": str(uuid.uuid4()), "s": status, "age": age})

    out = run_retention(RetentionConfig(batch_size=100))
    assert out["query_logs"] >= 250 and out["settings_audit"] >= 1 and out["index_jobs"] >= 1

    with eng.connect() as c:
        q = c.execute(text('SELECT query FROM "QueryLogs" WHERE query LIKE :p'),
                      {"p": f"%-{tag}%"}).scalars().all()
        a = c.execute(text('SELECT key FROM "SettingsAudit" WHERE key LIKE :p'),
                      {"p": f"%-{tag}"}).scalars().all()
        jobs = c.execute(text('SELECT status FROM "IndexJobs" WHERE folder_id = 987654')).scalars().all()
    assert q == [f"new-{tag}"]
    assert a == [f"new-{tag}"]
    assert sorted(jobs) == ["failed", "running"]   # in-flight and recent rows kept
