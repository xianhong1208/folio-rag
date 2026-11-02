"""Write one file's chunks into a folder's pgvector table in a single transaction.

The previous write path was three separate steps: delete the file's old rows,
``PGVectorStore.add`` the new ones, then re-segment them with CKIP to fill the
full-text column. Between the first two the file had no chunks at all, and
between the last two its new rows had a NULL search vector (and stayed that way
if CKIP failed, which was only logged). Re-indexing a file therefore made it
briefly — or permanently — invisible to search.

``replace_file_chunks`` computes everything first (embeddings are passed in,
segmentation runs before the transaction) and then, in ONE transaction, deletes
the file's old rows and inserts the new ones with their ``text_search_tsv``
already set. Readers see either the old version or the new one.

The search vector of a leaf is built from its retrieval text (prefix +
breadcrumb + original text, see chunk_identity), not from the displayed text.
Parents get no search vector: they exist only so auto-merge can fetch them, and
a parent's text is a superset of its leaves, so keyword hits on it would only
crowd out the leaves.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from sqlalchemy import text

from src.log import get_api_logger

logger = get_api_logger()

TSV_COLUMN = "text_search_tsv"


def detach_generated_tsv(conn, table: str) -> bool:
    """Turn llama-index's GENERATED ``text_search_tsv`` into a plain column we write.

    Postgres forbids writing a STORED GENERATED column, and its expression only sees
    the displayed text. ``DROP EXPRESSION`` takes an exclusive lock, so it runs only
    when the column is still generated (checked first). Returns True when it detached.
    """
    generated = conn.execute(
        text(
            "SELECT a.attgenerated FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.relname = :t AND a.attname = :col AND NOT a.attisdropped"
        ),
        {"t": table, "col": TSV_COLUMN},
    ).scalar()
    if generated in ("s", b"s"):
        conn.execute(text(f'ALTER TABLE "{table}" ALTER COLUMN {TSV_COLUMN} DROP EXPRESSION IF EXISTS'))
        return True
    return False


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def build_rows(
    vector_store: Any,
    nodes: Sequence[Any],
    search_texts: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Rows for INSERT. Metadata is serialized exactly like PGVectorStore.add does, so the
    query side rebuilds the nodes unchanged. ``search_texts`` maps leaf node_id ->
    text for the search vector (already segmented when CKIP is on); others get NULL."""
    from llama_index.core.schema import MetadataMode
    from llama_index.core.vector_stores.utils import node_to_metadata_dict

    flat = getattr(vector_store, "flat_metadata", False)
    rows = []
    for node in nodes:
        emb = node.get_embedding()
        rows.append({
            "node_id": node.node_id,
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
            "meta": json.dumps(node_to_metadata_dict(node, remove_text=True, flat_metadata=flat)),
            "emb": _vector_literal(emb),
            "tsv": search_texts.get(node.node_id),
        })
    return rows


def replace_file_chunks(
    vector_store: Any,
    file_id: str,
    nodes: Sequence[Any],
    search_texts: Dict[str, str],
    text_search_config: str = "simple",
    segment: Optional[Callable[[List[str]], List[str]]] = None,
    engine: Any = None,
    table: Optional[str] = None,
) -> int:
    """Atomically replace every row of ``file_id`` with ``nodes``. Returns rows written.

    ``segment`` (CKIP mode) maps search texts to space-joined tokens before the
    transaction; Postgres then tokenizes them with ``text_search_config``.
    Blocking: run it off the event loop.
    """
    from src.domain.rag.chunk_lookup import ChunkLookup

    init = getattr(vector_store, "_initialize", None)
    if callable(init):
        init()  # creates the table (and extension) on a folder's first write
    engine = engine or ChunkLookup._get_engine(vector_store)
    table = table or ChunkLookup._get_table_name(vector_store)

    if segment is not None and search_texts:
        ids = list(search_texts)
        segmented = segment([search_texts[i] for i in ids])
        search_texts = dict(zip(ids, segmented))
    rows = build_rows(vector_store, nodes, search_texts)

    meta_type = "jsonb" if getattr(vector_store, "use_jsonb", False) else "json"
    insert_sql = text(
        f'INSERT INTO "{table}" (text, metadata_, node_id, embedding, {TSV_COLUMN}) '
        f"VALUES (:text, CAST(:meta AS {meta_type}), :node_id, CAST(:emb AS vector), "
        f"to_tsvector(CAST(:cfg AS regconfig), CAST(:tsv AS text)))"
    )
    for r in rows:
        r["cfg"] = text_search_config

    with engine.connect() as conn:
        if detach_generated_tsv(conn, table):
            conn.commit()
            logger.info(f"[CHUNK_STORE] {table}.{TSV_COLUMN} is now written by the indexer")
    with engine.begin() as conn:  # one transaction: old rows out, new rows in
        removed = conn.execute(
            text(f"DELETE FROM \"{table}\" WHERE metadata_->>'file_id' = :fid"),
            {"fid": str(file_id)},
        ).rowcount or 0
        if rows:
            conn.execute(insert_sql, rows)
    logger.info(
        f"[CHUNK_STORE] file {file_id}: replaced {removed} row(s) with {len(rows)} in {table}"
    )
    return len(rows)
