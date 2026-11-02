"""Rebuild-FTS as resumable, observable work: re-segment a folder's chunks file by file.

The console's Rebuild FTS used to run ``rebuild_text_search_tsv`` over a whole folder
inside the HTTP request; a 5.6k-chunk folder took ~10 minutes, well past any proxy
timeout. This module splits the same work per file so the job manager can report
progress (files done / total), heartbeat between files, and stop between files when
the job is cancelled. Each per-file call is the scoped variant the indexer already
uses after adding a file, so the stored result is identical to a whole-table rebuild.

Blocking (sync SQLAlchemy + CKIP inference): run it in a worker thread.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from sqlalchemy import text

from src.log import get_api_logger

logger = get_api_logger()


def list_chunk_file_ids(engine, table_name: str) -> List[str]:
    """Distinct ``metadata_->>'file_id'`` values in a folder's vector table (sorted).

    A table that does not exist yet (folder never indexed) has no chunks: returns [].
    ``table_name`` is server-generated (folder id + UUID), never user input.
    """
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT to_regclass(:t)"), {"t": f'"{table_name}"'}).scalar()
        if not exists:
            return []
        rows = conn.execute(text(
            f"SELECT DISTINCT metadata_->>'file_id' AS fid FROM \"{table_name}\" "
            "WHERE metadata_->>'file_id' IS NOT NULL ORDER BY fid"
        )).fetchall()
    return [r[0] for r in rows]


def rebuild_fts_by_file(
    engine,
    table_name: str,
    segmenter,
    progress: Callable[..., None],
    should_stop: Callable[[], bool],
    rebuild_one: Optional[Callable[..., int]] = None,
) -> Dict[str, object]:
    """Re-segment every file's chunks, one file at a time.

    Args:
        engine: sync SQLAlchemy engine.
        table_name: the folder's physical vector table.
        segmenter: CKIP segmenter (``segment_batch``).
        progress: ``progress(done, total, label=None)`` — called before the first file
            and after each one.
        should_stop: polled before each file; True stops the rebuild early (cancel).
        rebuild_one: the per-file rebuild (defaults to
            ``vector_store_manager.rebuild_text_search_tsv``); injectable for tests.

    Returns:
        ``{"files", "files_done", "rows_updated", "stopped", "message"}``.
    """
    if rebuild_one is None:
        from src.domain.rag.vector_store_manager import rebuild_text_search_tsv as rebuild_one

    file_ids = list_chunk_file_ids(engine, table_name)
    total = len(file_ids)
    progress(0, total)
    rows = 0
    done = 0
    stopped = False
    for fid in file_ids:
        if should_stop():
            stopped = True
            break
        rows += rebuild_one(engine, table_name, segmenter, file_id=fid) or 0
        done += 1
        progress(done, total, fid)

    if stopped:
        message = f"Stopped after {done}/{total} file(s); {rows} chunk(s) rebuilt"
    else:
        message = f"Rebuilt full-text search for {rows} chunk(s) in {total} file(s)"
    logger.info(f"[FTS_REBUILD] {table_name}: {message}")
    return {"files": total, "files_done": done, "rows_updated": rows,
            "stopped": stopped, "message": message}
