
"""Read-only aggregate queries for the admin console (cross-table reads; never writes).

Backs the Control Center data views: overview KPIs, per-folder index status,
file-level status, and the index job list. Each view is a single aggregate/join
query to avoid N+1.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import func

from db.db import File, FileIndex, Folder, IndexJob, Session
from src.log import get_db_logger

log = get_db_logger()

# Any non-terminal status counts as "in progress" (matches IndexingJobManager's state machine)
_JOB_ACTIVE = ("pending", "running", "queued")


def _ensure_bound() -> None:
    """Ensure the Session is bound — the admin API may be hit before any other DB call."""
    from db.db import get_engine
    get_engine()


def overview() -> Dict[str, Any]:
    """Overview KPIs: global folder/file/index/job statistics."""
    _ensure_bound()
    with Session() as s:
        folders = s.query(func.count(Folder.id)).scalar() or 0
        files, total_size = s.query(
            func.count(File.id), func.coalesce(func.sum(File.file_size), 0)
        ).one()
        idx_rows = s.query(
            FileIndex.status, func.count(FileIndex.id), func.coalesce(func.sum(FileIndex.num_chunks), 0)
        ).group_by(FileIndex.status).all()
        by_status = {r[0]: {"files": r[1], "chunks": int(r[2])} for r in idx_rows}
        job_rows = s.query(IndexJob.status, func.count(IndexJob.job_id)) \
            .group_by(IndexJob.status).all()
        jobs_by_status = {r[0]: r[1] for r in job_rows}
        running = s.query(IndexJob).filter(IndexJob.status.in_(_JOB_ACTIVE)) \
            .order_by(IndexJob.last_updated_at.desc()).limit(5).all()
    indexed = by_status.get("indexed", {"files": 0, "chunks": 0})
    failed = by_status.get("failed", {"files": 0, "chunks": 0})
    return {
        "folders": folders,
        "files": files,
        "total_size_bytes": int(total_size or 0),
        "index": {
            "indexed_files": indexed["files"],
            "failed_files": failed["files"],
            "unindexed_files": max(files - sum(v["files"] for v in by_status.values()), 0),
            "total_chunks": sum(v["chunks"] for v in by_status.values()),
        },
        "jobs": {
            "by_status": jobs_by_status,
            "active": [_job_dict(j) for j in running],
        },
    }


def folders_with_index_stats() -> List[Dict[str, Any]]:
    """All folders plus their index summaries (single join query, no N+1)."""
    _ensure_bound()
    with Session() as s:
        idx = s.query(
            FileIndex.folder_id.label("fid"),
            func.count(FileIndex.id).filter(FileIndex.status == "indexed").label("indexed"),
            func.count(FileIndex.id).filter(FileIndex.status == "failed").label("failed"),
            func.coalesce(func.sum(FileIndex.num_chunks), 0).label("chunks"),
            func.max(FileIndex.indexed_at).label("last_indexed"),
        ).group_by(FileIndex.folder_id).subquery()
        rows = s.query(Folder, idx.c.indexed, idx.c.failed, idx.c.chunks, idx.c.last_indexed) \
            .outerjoin(idx, Folder.id == idx.c.fid) \
            .order_by(Folder.updated_at.desc()).all()
        out = []
        for f, indexed, failed_n, chunks, last_indexed in rows:
            out.append({
                "id": f.id,
                "name": f.name,
                "description": f.description,
                "file_count": f.file_count or 0,
                "total_size_bytes": int(f.total_size or 0),
                "indexed_files": indexed or 0,
                "failed_files": failed_n or 0,
                "chunks": int(chunks or 0),
                "vector_table": f"data_{f.id}_{f.vector_table_uuid}",
                "token_prefix": (f.user_token or "")[:8] + "…" if f.user_token else None,
                "created_at": _iso(f.created_at),
                "last_indexed_at": _iso(last_indexed),
            })
        return out


def files_with_index_status(folder_id: int) -> List[Dict[str, Any]]:
    """File list for a single folder plus index status (left join FileIndex)."""
    _ensure_bound()
    with Session() as s:
        rows = s.query(File, FileIndex).outerjoin(
            FileIndex, File.id == FileIndex.file_id
        ).filter(File.folder_id == folder_id) \
         .order_by(File.upload_time.desc()).all()
        out = []
        for f, ix in rows:
            err = (ix.error_message or "") if ix else ""
            out.append({
                "id": str(f.id),
                "name": f.file_name,
                "size_bytes": int(f.file_size or 0),
                "mime": f.mime_type,
                "uploaded_at": _iso(f.upload_time),
                "status": (ix.status if ix else "unindexed"),
                "chunks": (ix.num_chunks or 0) if ix else 0,
                "embedding_model": ix.embedding_model if ix else None,
                "indexed_at": _iso(ix.indexed_at) if ix else None,
                # Truncate the error message to 300 chars (full message stays in the DB; keep the list view lean)
                "error": (err[:300] + ("…" if len(err) > 300 else "")) or None,
            })
        return out


def recent_jobs(limit: int = 20) -> List[Dict[str, Any]]:
    """Recent index jobs (including folder name), ordered by last update."""
    _ensure_bound()
    with Session() as s:
        rows = s.query(IndexJob, Folder.name).outerjoin(
            Folder, IndexJob.folder_id == Folder.id
        ).order_by(IndexJob.last_updated_at.desc()).limit(limit).all()
        return [{**_job_dict(j), "folder_name": name} for j, name in rows]


def indexing_trend(days: int = 7) -> List[Dict[str, Any]]:
    """Daily count of files indexed over the last N days (for the sparkline; by FileIndex.indexed_at)."""
    _ensure_bound()
    from sqlalchemy import func as _f
    from datetime import datetime, timedelta, timezone
    since = datetime.now(timezone.utc) - timedelta(days=days - 1)
    with Session() as s:
        rows = s.query(
            _f.date(FileIndex.indexed_at).label("d"), _f.count(FileIndex.id)
        ).filter(FileIndex.status == "indexed", FileIndex.indexed_at >= since) \
         .group_by("d").all()
    by_day = {str(r[0]): r[1] for r in rows}
    out = []
    for i in range(days):
        day = (since + timedelta(days=i)).date().isoformat()
        out.append({"date": day, "count": by_day.get(day, 0)})
    return out


def chunks_for_file(folder_id: int, file_id: str, limit: int = 500) -> Dict[str, Any]:
    """List the leaf chunks of a file (ordered by chunk_index; lets users inspect what the index looks like).

    Reads the folder's physical pgvector table directly. file_id is passed as a bind
    param to prevent injection; the table name is composed from folder_id and the
    server-side vector_table_uuid (not user input). A missing table (not indexed)
    returns an empty list rather than raising.
    """
    _ensure_bound()
    from sqlalchemy import text
    from db.db import get_engine
    with Session() as s:
        folder = s.get(Folder, folder_id)
        if folder is None:
            return {"chunks": [], "total": 0, "file_name": None}
        table = f"data_{folder_id}_{folder.vector_table_uuid}"
        fidx = s.query(File.file_name).filter(File.id == file_id).scalar()
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(text(
                f'SELECT text, metadata_ FROM "{table}" '
                "WHERE metadata_->>'file_id' = :fid "
                "AND COALESCE(metadata_->>'node_role', 'leaf') = 'leaf' "
                "ORDER BY (metadata_->>'chunk_index')::int NULLS LAST "
                "LIMIT :lim"),
                {"fid": str(file_id), "lim": limit},
            ).fetchall()
    except Exception as e:
        log.warning(f"chunks_for_file: table {table} unavailable: {e}")
        return {"chunks": [], "total": 0, "file_name": fidx}
    chunks = []
    for txt, meta in rows:
        meta = meta or {}
        chunks.append({
            "index": meta.get("chunk_index"),
            "text": txt or "",
            "chars": len(txt or ""),
            "headings": meta.get("headings"),
            "page": meta.get("page_no"),
            "content_type": meta.get("content_type"),
        })
    return {"chunks": chunks, "total": len(chunks), "file_name": fidx}


def system_resources() -> Dict[str, Any]:
    """DB size / total vector-table size / disk usage (overview resource card). All best-effort."""
    _ensure_bound()
    from sqlalchemy import text
    from db.db import get_engine
    out = {"db_size_bytes": None, "vector_bytes": None, "vector_tables": 0,
           "disk_used_bytes": None, "disk_total_bytes": None}
    try:
        with get_engine().connect() as conn:
            out["db_size_bytes"] = conn.execute(
                text("SELECT pg_database_size(current_database())")).scalar()
            rows = conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE tablename LIKE 'data\\_%'")).fetchall()
            out["vector_tables"] = len(rows)
            total = 0
            for (tbl,) in rows:
                try:
                    total += conn.execute(text("SELECT pg_total_relation_size(:t)"),
                                          {"t": tbl}).scalar() or 0
                except Exception:
                    pass
            out["vector_bytes"] = total
    except Exception as e:
        log.warning(f"system_resources db size failed: {e}")
    try:
        import shutil
        du = shutil.disk_usage(".")
        out["disk_used_bytes"] = du.used
        out["disk_total_bytes"] = du.total
    except Exception:
        pass
    return out


def _job_dict(j: IndexJob) -> Dict[str, Any]:
    return {
        "job_id": str(j.job_id),
        "folder_id": j.folder_id,
        "status": j.status,
        "total_files": j.total_files or 0,
        "processed_files": j.processed_files or 0,
        "current_file_name": j.current_file_name,
        "last_file_status": j.last_file_status,
        "message": j.message,
        "error": (j.error or "")[:300] or None,
        "started_at": j.started_at,
        "completed_at": j.completed_at,
        "last_updated_at": j.last_updated_at,
    }


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None
