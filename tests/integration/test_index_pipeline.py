
"""Integration tests -- indexing pipeline (chunk -> hierarchy -> embed ->
pgvector write -> read back).

Verifies orchestration and write correctness for
HierarchicalIndexer.index_document: leaf/parent rows written with correct
file_id metadata, leaves carrying real vectors and parents zero vectors,
ChunkLookup.fetch_file_full reassembling text in chunk order, real-DB
delete_chunks_by_file, and abort-before-embed leaving no residue. Embedding
uses an injected deterministic FakeEmbed, so this tests writes, not embedding
quality.
"""

import hashlib

import pytest
from llama_index.core import Document

from src.domain.rag.hierarchical_indexer import HierarchicalIndexer, IndexingAbortedError
from src.domain.rag.chunk_lookup import ChunkLookup
from src.domain.rag.vector_store_manager import VectorStoreManager

_DIM = 1024


class FakeEmbed:
    """Deterministic embedding: hash(text) expanded into a 1024-dim unit vector. Zero external dependencies."""

    def get_text_embedding_batch(self, texts, **kwargs):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [(h[i % 32] - 128) / 128.0 for i in range(_DIM)]
            out.append(vec)
        return out


_DOC_TEXT = "\n\n".join(
    f"## 章節 {i}\n這是第 {i} 段測試內容。混合中英文 content for chunking,"
    f"確保切出多個 leaf。" + "填充句。" * 30
    for i in range(1, 7)
)


@pytest.fixture()
def vsm(itest_db):
    return VectorStoreManager(embed_dim=_DIM)


@pytest.fixture()
def indexer(itest_db):
    return HierarchicalIndexer(
        leaf_chunk_size=128,
        parent_target_tokens=512,
        chunk_overlap=20,
        context_generator=None,   # LLM step disabled -- out of scope for this net
        embed_model=FakeEmbed(),  # injection point
    )


def _doc(file_id: str) -> Document:
    return Document(
        text=_DOC_TEXT,
        metadata={"file_id": file_id, "file_name": "pipeline.md"},
    )


def _table_rows(vsm, folder, where: str = "", params=None):
    from sqlalchemy import text
    from db.db import get_engine
    table = vsm.physical_table_name(folder.id, folder.vector_table_uuid)
    with get_engine().connect() as conn:
        return conn.execute(
            text(f'SELECT metadata_ FROM "{table}" {where}'), params or {}
        ).fetchall()


async def test_index_document_writes_leaves_and_parents(vsm, indexer, folder, file_row):
    store = vsm.get_or_create(folder_id=folder.id, vector_table_uuid=str(folder.vector_table_uuid))
    fid = str(file_row.id)

    result = await indexer.index_document(_doc(fid), store)

    assert result["leaves"] > 1, f"expected several leaves, got {result}"
    assert result["parents"] >= 1
    rows = _table_rows(vsm, folder)
    assert len(rows) == result["leaves"] + result["parents"], "table rows should equal leaves + parents"
    # file_id metadata all correct (file-level deletion / read mode both rely on it)
    assert all(r[0].get("file_id") == fid for r in rows)
    roles = {r[0].get("node_role") for r in rows}
    assert roles == {"leaf", "parent"}


async def test_leaves_have_real_vectors_parents_zero(vsm, indexer, folder, file_row):
    store = vsm.get_or_create(folder_id=folder.id, vector_table_uuid=str(folder.vector_table_uuid))
    fid = str(file_row.id)
    await indexer.index_document(_doc(fid), store)

    from sqlalchemy import text
    from db.db import get_engine
    table = vsm.physical_table_name(folder.id, folder.vector_table_uuid)
    with get_engine().connect() as conn:
        # pgvector has no element-wise aggregation, so compare the "L2 distance to the zero vector" directly
        zero = "[" + ",".join(["0"] * _DIM) + "]"
        leaf_zero = conn.execute(text(
            f'SELECT count(*) FROM "{table}" '
            f"WHERE metadata_->>'node_role' = 'leaf' AND embedding <-> :z = 0"
        ), {"z": zero}).scalar()
        parent_nonzero = conn.execute(text(
            f'SELECT count(*) FROM "{table}" '
            f"WHERE metadata_->>'node_role' = 'parent' AND embedding <-> :z > 0"
        ), {"z": zero}).scalar()
    assert leaf_zero == 0, "a leaf must not be a zero vector (retrieval would break entirely)"
    assert parent_nonzero == 0, "a parent should be a zero vector (queries filter on node_role)"


async def test_fetch_file_full_reassembles_content(vsm, indexer, folder, file_row):
    store = vsm.get_or_create(folder_id=folder.id, vector_table_uuid=str(folder.vector_table_uuid))
    fid = str(file_row.id)
    await indexer.index_document(_doc(fid), store)

    full = ChunkLookup.fetch_file_full(vector_store=store, file_id=fid)
    assert full["chunk_count"] > 1
    assert not full["truncated"]
    # First and last sections both present, in the correct order (chunks concatenated by index order)
    assert "章節 1" in full["text"] and "章節 6" in full["text"]
    assert full["text"].index("章節 1") < full["text"].index("章節 6")


async def test_delete_chunks_by_file_real_db(vsm, indexer, folder, file_row):
    """delete_chunks_by_file: the mock version pinned the contract; here we verify real-DB behavior."""
    store = vsm.get_or_create(folder_id=folder.id, vector_table_uuid=str(folder.vector_table_uuid))
    fid = str(file_row.id)
    result = await indexer.index_document(_doc(fid), store)
    total = result["leaves"] + result["parents"]

    table = vsm.physical_table_name(folder.id, folder.vector_table_uuid)
    deleted = VectorStoreManager.delete_chunks_by_file(table, fid)
    assert deleted == total, f"expected all {total} nodes of the file deleted, deleted {deleted}"
    assert _table_rows(vsm, folder) == []


async def test_abort_before_embed_leaves_no_residue(vsm, indexer, folder, file_row):
    """should_abort fires before embed -> IndexingAbortedError, no residual writes in the table."""
    store = vsm.get_or_create(folder_id=folder.id, vector_table_uuid=str(folder.vector_table_uuid))
    fid = str(file_row.id)

    with pytest.raises(IndexingAbortedError):
        await indexer.index_document(_doc(fid), store, should_abort=lambda: True)

    # abort precedes vector_store.add -> the physical table that PGVectorStore
    # lazily creates never appears at all.
    # "table does not exist" = the strongest proof of no residue; if the table
    # does exist (created by another path), also verify its contents are empty.
    from sqlalchemy.exc import ProgrammingError
    try:
        rows = _table_rows(vsm, folder, "WHERE metadata_->>'file_id' = :fid", {"fid": fid})
        assert rows == [], "aborted before the write: no chunk may remain"
    except ProgrammingError:
        pass  # UndefinedTable -- the table was never even created, zero residue


# Citation provenance end to end: real file -> docling chunking with provenance ->
# stored -> every record is reverse-lookupable.

async def test_e2e_traceability_real_md_to_store_and_back(
    vsm, indexer, folder, file_row, tmp_path
):
    """Full chain: real md -> docling chunk records with headings -> leaf_splitter
    metadata -> index_document storage -> three reverse-lookup directions
    (node_id, parent children_node_ids, file_id -> full text with sentinel).
    """
    from docling.document_converter import DocumentConverter
    from docling.chunking import HierarchicalChunker
    from llama_index.core import Document as LIDocument

    from src.domain.rag.docling_loader import _refine_chunks_for_token_budget
    from src.domain.rag.leaf_splitter import LeafSplitter

    # -- 1. Real file -> docling -> chunk records (same chain as production docling_convert_once)
    md = tmp_path / "trace.md"
    md.write_text(
        "# 追溯章\n\n哨兵內容SENTINEL 第一段。" + "內文填充。" * 40
        + "\n\n## 次章\n\n次章內容。" + "更多填充。" * 40,
        encoding="utf-8",
    )

    class _LenTok:
        def encode(self, text, add_special_tokens=False):
            return list(text)

    doc = DocumentConverter().convert(str(md)).document
    records = _refine_chunks_for_token_budget(
        list(HierarchicalChunker().chunk(doc)), _LenTok(), 200)
    assert any(r["headings"] for r in records), "docling records should carry headings"

    # -- 2. Production split chain: leaf_splitter (_docling_chunks) -> index_document
    file_id = str(file_row.id)
    li_doc = LIDocument(
        text=doc.export_to_markdown(),
        metadata={"file_id": file_id, "file_name": "trace.md",
                  "_docling_chunks": records},
    )
    leaves = LeafSplitter(leaf_chunk_size=128, chunk_overlap=20)\
        .split_into_leaf_documents(li_doc)
    assert any("headings" in d.metadata for d in leaves)
    # NOTE: split_into_leaf_documents pops _docling_chunks (production semantics:
    # a Document is split only once) -- the interim assertion above consumed it,
    # so restore it before feeding the indexer
    li_doc.metadata["_docling_chunks"] = records

    vector_store = vsm.get_or_create(folder.id, folder.vector_table_uuid)
    stats = await indexer.index_document(li_doc, vector_store)
    assert stats["leaves"] >= 2 and stats["parents"] >= 1

    # -- 3. Per-record assertions on the stored data
    rows = _table_rows(vsm, folder)
    metas = [r[0] for r in rows]
    leaf_metas = [m for m in metas if m.get("node_role") == "leaf"]
    parent_metas = [m for m in metas if m.get("node_role") == "parent"]
    assert leaf_metas and parent_metas
    for m in leaf_metas:
        assert m.get("file_id") == file_id
        assert isinstance(m.get("chunk_index"), int)
        assert m.get("parent_node_id")  # every leaf attaches to a parent
    assert any(m.get("headings") for m in leaf_metas), "provenance headings should be stored"

    # -- 4a. node_id reverse-lookup of a single chunk
    from src.domain.rag.chunk_lookup import ChunkLookup
    from sqlalchemy import text as sql_text
    from db.db import get_engine
    table = vsm.physical_table_name(folder.id, folder.vector_table_uuid)
    with get_engine().connect() as conn:
        node_ids = [r[0] for r in conn.execute(sql_text(
            f'SELECT node_id FROM "{table}"')).fetchall()]
    for nid in node_ids:
        node = ChunkLookup.fetch_node(vector_store, nid)
        assert node is not None, f"node_id={nid} not found"

    # -- 4b. parent <-> children bidirectional links are complete
    leaf_ids = {r[1] for r in _id_role_rows(vsm, folder) if r[0] == "leaf"}
    for pm in parent_metas:
        for child in pm.get("children_node_ids", []):
            assert child in leaf_ids, "a parent points at a leaf that does not exist"

    # -- 4c. file_id reassembles the full text
    full = ChunkLookup.fetch_file_full(vector_store, file_id)
    assert "哨兵內容SENTINEL" in full["text"]


def _id_role_rows(vsm, folder):
    from sqlalchemy import text as sql_text
    from db.db import get_engine
    table = vsm.physical_table_name(folder.id, folder.vector_table_uuid)
    with get_engine().connect() as conn:
        return conn.execute(sql_text(
            f"SELECT metadata_->>'node_role', node_id FROM \"{table}\"")).fetchall()


async def test_reindex_same_file_purges_stale_chunks(vsm, indexer, folder, file_row):
    """Regression: re-running index_document on the same file must not double
    the total row count (idempotent purge of the old data before writing).

    Scenario: last write succeeded but finalization failed (timeout cancel /
    record failure / gate 3) -> file marked failed -> user re-runs. Old
    behavior: node ids regenerated with uuid4 each time + PGVector does not
    dedup -> a second full set of vectors stacks into the same table.
    """
    store = vsm.get_or_create(folder_id=folder.id, vector_table_uuid=str(folder.vector_table_uuid))
    fid = str(file_row.id)

    r1 = await indexer.index_document(_doc(fid), store)
    n1 = len(_table_rows(vsm, folder))
    assert n1 == r1["leaves"] + r1["parents"]

    r2 = await indexer.index_document(_doc(fid), store)  # simulate a re-run
    n2 = len(_table_rows(vsm, folder))
    assert n2 == r2["leaves"] + r2["parents"], f"after a re-run the count should equal one copy ({n2} vs {r2})"
    assert n2 == n1  # no doubling


async def test_reindex_reuses_unchanged_chunks_and_writes_tsv(vsm, indexer, folder, file_row):
    """Incremental re-index on the real table: identical content embeds nothing; leaves
    get a search vector in the same INSERT, parents none."""
    store = vsm.get_or_create(folder_id=folder.id, vector_table_uuid=str(folder.vector_table_uuid))
    fid = str(file_row.id)
    first = await indexer.index_document(_doc(fid), store)
    assert first["embedded"] == first["leaves"]
    again = await indexer.index_document(_doc(fid), store)
    assert again["embedded"] == 0 and again["leaves"] == first["leaves"]

    from sqlalchemy import text
    from db.db import get_engine
    table = vsm.physical_table_name(folder.id, folder.vector_table_uuid)
    with get_engine().connect() as conn:
        rows = conn.execute(text(
            f"SELECT metadata_->>'node_role', text_search_tsv IS NULL FROM \"{table}\" "
            "WHERE metadata_->>'file_id' = :f"), {"f": fid}).fetchall()
    assert rows and all(is_null == (role == "parent") for role, is_null in rows)


async def test_failed_replace_keeps_the_previous_version(vsm, indexer, folder, file_row):
    """The delete and the insert share one transaction: if the insert fails, the old
    rows are still there (no window where the file has no chunks)."""
    from llama_index.core.schema import TextNode
    from src.domain.rag.chunk_store import replace_file_chunks

    store = vsm.get_or_create(folder_id=folder.id, vector_table_uuid=str(folder.vector_table_uuid))
    fid = str(file_row.id)
    await indexer.index_document(_doc(fid), store)
    before = len(_table_rows(vsm, folder))
    bad = TextNode(id_="bad", text="x", metadata={"file_id": fid, "node_role": "leaf"})
    bad.embedding = [0.1, 0.2]  # wrong dimension -> INSERT fails
    with pytest.raises(Exception):
        replace_file_chunks(store, fid, [bad], {"bad": "x"}, "simple")
    assert len(_table_rows(vsm, folder)) == before
