"""Garbage chunks (empty code fences, structure-only) and in-document duplicates are dropped."""

from llama_index.core import Document

from src.domain.rag.leaf_splitter import LeafSplitter, _is_low_value_chunk


def test_empty_code_fence_placeholder_is_low_value():
    assert _is_low_value_chunk("[code block]\n\n```")
    assert _is_low_value_chunk("```python\n```")


def test_structure_only_and_zero_width_are_low_value():
    assert _is_low_value_chunk("| --- | --- |\n> ** __")
    assert _is_low_value_chunk("​​  ")
    assert _is_low_value_chunk("")


def test_real_content_is_kept():
    assert not _is_low_value_chunk("Short line.")
    long_body = "A paragraph long enough to carry real retrieval value here.\n\n```"
    assert not _is_low_value_chunk(long_body)


def test_splitter_drops_placeholders_and_exact_duplicates():
    splitter = LeafSplitter(leaf_chunk_size=256, chunk_overlap=0)
    doc = Document(text="x", metadata={"file_id": "f", "_docling_chunks": [
        {"text": "Warranty is 36 months.", "headings": ["A"]},
        {"text": "[image]\n\n```", "headings": ["A"]},
        {"text": "Warranty is 36 months.", "headings": ["A"]},
        {"text": "Pricing starts at 100.", "headings": ["B"]},
    ]})
    leaves = splitter.split_into_leaf_documents(doc)
    assert [d.text for d in leaves] == ["Warranty is 36 months.", "Pricing starts at 100."]
    assert leaves[0].metadata["headings"] == ["A"]
    assert "_docling_chunks" not in leaves[0].metadata


def test_same_sentence_in_two_sections_is_not_a_duplicate():
    splitter = LeafSplitter(leaf_chunk_size=256, chunk_overlap=0)
    doc = Document(text="x", metadata={"file_id": "f", "_docling_chunks": [
        {"text": "Warranty: 24 months.", "headings": ["Product A"]},
        {"text": "Warranty: 24 months.", "headings": ["Product B"]},
    ]})
    leaves = splitter.split_into_leaf_documents(doc)
    assert [d.metadata["headings"] for d in leaves] == [["Product A"], ["Product B"]]


def test_clean_text_removes_zero_width_and_odd_spaces():
    from src.domain.rag.document_loader import clean_text
    assert clean_text("ROC\u200bm\u00a0driver\x00\x07") == "ROCm driver"
