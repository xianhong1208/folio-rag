
"""QueryLog persistence and analytics aggregation — data source for the
Control Center's query-analytics view.

record is best-effort (failures are logged only, never blocking a query);
analytics computes, in one pass, the KPIs (query volume / average latency /
zero-result rate), top queries, zero-result queries, and the daily trend.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import and_, func, or_

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
               latency_ms: Optional[int], token_prefix: Optional[str],
               top_score: Optional[float] = None) -> None:
        """Record one query (best-effort — telemetry failure does not affect the query itself).

        top_score is the reranker's best score for the query before its threshold
        (None without a reranker or when reranking failed).
        """
        try:
            q = (query or "").strip()[:2000]
            if not q:
                return
            with Session() as s:
                s.add(QueryLog(query=q, folder_id=folder_id,
                               result_count=result_count, latency_ms=latency_ms,
                               token_prefix=token_prefix,
                               top_score=float(top_score) if top_score is not None else None))
                s.commit()
        except Exception as e:
            log.warning(f"QueryLog.record failed: {e}")

    @classmethod
    def analytics(cls, folder_id: Optional[int] = None, days: int = 30,
                  top: int = 10, folder_ids: Optional[Iterable[int]] = None,
                  gap_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Query-analytics aggregation. folder_id=None means all folders; days
        is the statistics window in days. ``folder_ids`` (when not None) further
        limits the aggregation to those folders — the console's per-owner scope.

        Knowledge gaps are judged on the **latest** search of each question per
        folder: a gap when it returned nothing, or its top rerank score was below
        ``gap_threshold`` (default ``rag.rerank.gap_threshold``). A question that
        was a gap and now finds a good match drops off the list.

        Returns {total, zero_result, zero_rate, avg_latency_ms, top_queries[],
        zero_queries[], gap_queries[], gap_count, gap_threshold, by_day[]}. If the
        DB is down or the table is missing, returns an all-zero structure.
        """
        if gap_threshold is None:
            gap_threshold = _configured_gap_threshold()
        empty = {"total": 0, "zero_result": 0, "zero_rate": 0.0,
                 "avg_latency_ms": None, "top_queries": [], "zero_queries": [],
                 "gap_queries": [], "gap_count": 0, "gap_threshold": gap_threshold,
                 "by_day": [], "days": days, "folder_id": folder_id}
        try:
            since = datetime.now(timezone.utc) - timedelta(days=days)
            scope_ids = list(folder_ids) if folder_ids is not None else None
            with Session() as s:
                def scoped(q):
                    q = q.filter(QueryLog.created_at >= since)
                    if folder_id is not None:
                        q = q.filter(QueryLog.folder_id == folder_id)
                    if scope_ids is not None:
                        q = q.filter(QueryLog.folder_id.in_(scope_ids))
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

                gap_rows, gap_count = cls._gaps(s, scoped, gap_threshold, top)

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
                    "gap_queries": gap_rows, "gap_count": gap_count,
                    "gap_threshold": gap_threshold,
                    "by_day": [{"date": str(r[0]), "count": int(r[1])} for r in day_rows],
                    "days": days, "folder_id": folder_id,
                }
        except Exception as e:
            log.warning(f"QueryLog.analytics failed: {e}")
            return empty

    @staticmethod
    def _gaps(s, scoped, gap_threshold: float, top: int):
        """(gap rows, distinct gap count) from the latest record per (folder, question)."""
        part = (QueryLog.folder_id, QueryLog.query)
        latest = scoped(s.query(
            QueryLog.folder_id, QueryLog.query, QueryLog.result_count,
            QueryLog.top_score, QueryLog.created_at,
            func.row_number().over(partition_by=part, order_by=(
                QueryLog.created_at.desc(), QueryLog.id.desc())).label("rn"),
            func.count(QueryLog.id).over(partition_by=part).label("n"),
        )).subquery()
        is_gap = and_(latest.c.rn == 1, or_(
            latest.c.result_count == 0,
            and_(latest.c.top_score.isnot(None), latest.c.top_score < gap_threshold)))
        count = s.query(func.count()).select_from(latest).filter(is_gap).scalar() or 0
        rows = (s.query(latest.c.query, latest.c.folder_id, latest.c.n, latest.c.result_count,
                        latest.c.top_score, latest.c.created_at)
                .filter(is_gap)
                .order_by(latest.c.n.desc(), latest.c.created_at.desc())
                .limit(top).all())
        return [{"query": r[0], "folder_id": r[1], "count": int(r[2]),
                 "result_count": int(r[3] or 0),
                 "top_score": round(float(r[4]), 4) if r[4] is not None else None,
                 "last_at": r[5].isoformat() if r[5] is not None else None}
                for r in rows], int(count)


def _configured_gap_threshold() -> float:
    try:
        from src.config.config_manager import Config
        rr = getattr(getattr(Config.get_config_model(), "rag", None), "rerank", None)
        v = getattr(rr, "gap_threshold", None)
        return float(v) if v is not None else DEFAULT_GAP_THRESHOLD
    except Exception:
        return DEFAULT_GAP_THRESHOLD


DEFAULT_GAP_THRESHOLD = 0.5
