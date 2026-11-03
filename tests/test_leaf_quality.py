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


# --- fallback path (txt / transcripts): no overlap, sized in embedding tokens ----------

_CLAUSES = ["員工出差應事先填寫申請單並經主管核准", "住宿費每日以新臺幣二千元為上限",
            "膳雜費依出差日數按日計算", "返回後五個工作日內應提出出差報告"]


def _transcript(lines=12, per_line=6):
    """ASR-style text: no punctuation, one line per audio segment, unique markers."""
    out = []
    for p in range(lines):
        out.append("".join(f"第{p * per_line + k}項{_CLAUSES[k % 4]}其適用範圍以編號{1000 + k}為限"
                           for k in range(per_line)))
    return "\n".join(out)


def _markers(text):
    import re
    return re.findall(r"第\d+項", text)


def _fallback_leaves(text, contextual=True):
    splitter = LeafSplitter(leaf_chunk_size=256, chunk_overlap=50,
                            embedding_model_name="intfloat/multilingual-e5-large",
                            has_context_generator=contextual)
    return splitter.split_into_leaf_documents(Document(text=text, metadata={"file_id": "f"}))


def test_fallback_transcript_has_no_duplicated_text():
    text = _transcript()
    leaves = _fallback_leaves(text)
    joined = [m for d in leaves for m in _markers(d.text)]
    assert joined == _markers(text)  # every sentence exactly once, in order
    assert sum(len(d.text.replace("\n", "")) for d in leaves) == len(text.replace("\n", ""))


def test_fallback_sizes_are_embedding_tokens():
    from src.domain.rag.chunk_tokenizer import get_chunk_tokenizer

    tok = get_chunk_tokenizer("intfloat/multilingual-e5-large")
    text = "\n\n".join(_transcript(lines=1, per_line=40).split("\n"))  # one huge paragraph
    leaves = _fallback_leaves(text)
    sizes = [len(tok.encode(d.text)) for d in leaves]
    assert len(sizes) > 1
    assert max(sizes) <= 256
    # The old tiktoken-sized splitter gave ~70-110 e5 tokens of Chinese per leaf.
    assert min(sizes) >= 150, sizes


def test_fallback_small_paragraphs_merge_without_overlap():
    paras = [f"Paragraph {i} says something specific about topic {i}." for i in range(30)]
    leaves = _fallback_leaves("\n\n".join(paras))
    assert len(leaves) < 30  # merged
    seen = [p for d in leaves for p in paras if p in d.text]
    assert seen == paras  # each paragraph exactly once


def test_fallback_keeps_tables_whole():
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    leaves = _fallback_leaves(f"Intro text.\n\n[TABLE_START]\n{table}\n[TABLE_END]\n\nOutro.")
    assert any(table in d.text for d in leaves)
    assert not any("[TABLE_START]" in d.text for d in leaves)


def test_fallback_degrades_without_tokenizer(monkeypatch):
    import src.domain.rag.docling_loader as dl

    def boom(_name):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(dl, "get_hf_tokenizer", boom)
    text = _transcript(lines=3)
    leaves = _fallback_leaves(text)
    assert leaves
    joined = [m for d in leaves for m in _markers(d.text)]
    assert joined == _markers(text)  # degraded, but still overlap-free
