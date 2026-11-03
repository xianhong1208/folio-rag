"""CSV files are indexed as a table: every chunk carries the header row (M6)."""

from unittest.mock import patch

from llama_index.core import Document

from src.domain.rag.document_loader import DocumentLoader, csv_to_markdown
from src.domain.rag.leaf_splitter import LeafSplitter

_HEADER = "employee_id,name,department,annual_leave_days"


def _csv(rows=120):
    lines = [_HEADER] + [f"E{i:04d},Person {i},Dept {i % 7},{10 + i % 15}" for i in range(rows)]
    return "\n".join(lines) + "\n"


def _load(tmp_path, content, name="staff.csv", contextual=True):
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    loader = DocumentLoader(leaf_chunk_size=256, embedding_model_name=None,
                            has_context_generator=contextual)
    with patch("src.domain.rag.document_loader.FileStorage.resolve_path", return_value=f):
        return loader.load(file_path=str(f), file_id="c1", file_name=name)


def test_every_chunk_repeats_the_header(tmp_path):
    doc = _load(tmp_path, _csv())
    chunks = doc.metadata["_docling_chunks"]
    assert len(chunks) > 1
    for c in chunks:
        first_line = c["text"].split("\n", 1)[0]
        assert first_line == "| employee_id | name | department | annual_leave_days |"
        assert c["content_type"] == "table"
    # every data row appears exactly once
    body = [line for c in chunks for line in c["text"].split("\n")[2:]]
    assert len(body) == 120 and len(set(body)) == 120


def test_leaves_keep_header_and_table_badge(tmp_path):
    doc = _load(tmp_path, _csv())
    leaves = LeafSplitter(leaf_chunk_size=256, chunk_overlap=50).split_into_leaf_documents(doc)
    assert len(leaves) > 1
    assert all(ld.text.startswith("| employee_id |") for ld in leaves)
    assert all(ld.metadata["content_type"] == "table" for ld in leaves)


def test_pieces_share_one_prefix_group(tmp_path):
    chunks = _load(tmp_path, _csv()).metadata["_docling_chunks"]
    assert [c["table_part"] for c in chunks] == list(range(len(chunks)))


def test_semicolon_csv_and_quoted_cells(tmp_path):
    md = csv_to_markdown('a;b\n"x; y";"1|2"\n')
    assert md == "| a | b |\n| --- | --- |\n| x; y | 1\\|2 |"


def test_not_a_table_stays_plain_text(tmp_path):
    doc = _load(tmp_path, "just one line of text\n", name="notes.csv")
    assert "_docling_chunks" not in doc.metadata
    assert doc.text.strip() == "just one line of text"
    assert csv_to_markdown("") is None
    assert csv_to_markdown("single\ncolumn\n") is None


def test_ragged_rows_are_padded():
    assert csv_to_markdown("a,b,c\n1,2\n") == "| a | b | c |\n| --- | --- | --- |\n| 1 | 2 |  |"


def test_plain_text_path_lost_the_header_for_contrast():
    """The old route (plain text through the fallback splitter) leaves later chunks headerless."""
    leaves = LeafSplitter(leaf_chunk_size=256, chunk_overlap=50).split_into_leaf_documents(
        Document(text=_csv(), metadata={"file_id": "c1"}))
    assert len(leaves) > 1
    assert not leaves[1].text.startswith(_HEADER)
