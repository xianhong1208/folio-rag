
"""QueryLog persistence and analytics aggregation — data source for the
Control Center's query-analytics view.

record is best-effort (failures are logged only, never blocking a query);
analytics computes, in one pass, the KPIs (query volume / average latency /
zero-result rate), top queries, zero-result queries, and the daily trend.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func

from .db import QueryLog, Session
from .baseDB import BaseDB
from src.log import get_db_logger

log = get_db_logger()


class QueryLogDB(BaseDB):
    @classmethod
    def get_orm_class(cls) -> type:
        return QueryLog

    @classmethod
    def get_log(cls):
        return log

    @classmethod
    def ensure_table(cls) -> None:
        try:
            from db.db import get_engine
            cls.get_orm_class().__table__.create(get_engine(), checkfirst=True)
        except Exception as e:
            log.warning(f"QueryLog ensure_table failed: {e}")

    @classmethod
    def record(cls, query: str, folder_id: Optional[int], result_count: int,
               latency_ms: Optional[int], token_prefix: Optional[str]) -> None:
        """Record one query (best-effort — telemetry failure does not affect the query itself)."""
        try:
            q = (query or "").strip()[:2000]
            if not q:
                return
            with Session() as s:
                s.add(QueryLog(query=q, folder_id=folder_id,
                               result_count=result_count, latency_ms=latency_ms,
                               token_prefix=token_prefix))
                s.commit()
        except Exception as e:
            log.warning(f"QueryLog.record failed: {e}")

    @classmethod
    def analytics(cls, folder_id: Optional[int] = None, days: int = 30,
                  top: int = 10) -> Dict[str, Any]:
        """Query-analytics aggregation. folder_id=None means all folders; days
        is the statistics window in days.

        Returns {total, zero_result, zero_rate, avg_latency_ms, top_queries[],
        zero_queries[], by_day[]}. If the DB is down or the table is missing,
        returns an all-zero structure (does not raise).
        """
        empty = {"total": 0, "zero_result": 0, "zero_rate": 0.0,
                 "avg_latency_ms": None, "top_queries": [], "zero_queries": [],
                 "by_day": [], "days": days, "folder_id": folder_id}
        try:
            since = datetime.now(timezone.utc) - timedelta(days=days)
            with Session() as s:
                def scoped(q):
                    q = q.filter(QueryLog.created_at >= since)
                    if folder_id is not None:
                        q = q.filter(QueryLog.folder_id == folder_id)
                    return q

                total = scoped(s.query(func.count(QueryLog.id))).scalar() or 0
                if not total:
                    return empty
                zero = scoped(
                    s.query(func.count(QueryLog.id)).filter(QueryLog.result_count == 0)
                ).scalar() or 0
                avg_lat = scoped(
                    s.query(func.avg(QueryLog.latency_ms)).filter(QueryLog.latency_ms.isnot(None))
                ).scalar()

                top_rows = scoped(
                    s.query(QueryLog.query, func.count(QueryLog.id).label("n"),
                            func.avg(QueryLog.result_count).label("avg_hits"))
                    .group_by(QueryLog.query)
                ).order_by(func.count(QueryLog.id).desc()).limit(top).all()

                zero_rows = scoped(
                    s.query(QueryLog.query, func.count(QueryLog.id).label("n"))
                    .filter(QueryLog.result_count == 0)
                    .group_by(QueryLog.query)
                ).order_by(func.count(QueryLog.id).desc()).limit(top).all()

                day = func.date(QueryLog.created_at)
                day_rows = scoped(
                    s.query(day.label("d"), func.count(QueryLog.id).label("n"))
                    .group_by(day)
                ).order_by(day).all()

                return {
                    "total": int(total),
                    "zero_result": int(zero),
                    "zero_rate": round(zero / total, 4) if total else 0.0,
                    "avg_latency_ms": round(float(avg_lat)) if avg_lat is not None else None,
                    "top_queries": [
                        {"query": r[0], "count": int(r[1]),
                         "avg_hits": round(float(r[2]), 1) if r[2] is not None else 0}
                        for r in top_rows],
                    "zero_queries": [{"query": r[0], "count": int(r[1])} for r in zero_rows],
                    "by_day": [{"date": str(r[0]), "count": int(r[1])} for r in day_rows],
                    "days": days, "folder_id": folder_id,
                }
        except Exception as e:
            log.warning(f"QueryLog.analytics failed: {e}")
            return empty
