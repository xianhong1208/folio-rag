
"""Unit tests for chunk records (citation provenance).

_refine_chunks_for_token_budget returns List[dict] of {text, headings, page_no, content_type}.
Contract: an oversized-chunk split has each segment inherit the source meta; small
chunks merge only within one section, and the merged chunk carries the headings the
pieces share; bare-string input yields None meta; leaf_splitter injects
headings/page_no into leaf metadata.
The fake tokenizer is 1 char = 1 token, never touching HuggingFace.
"""

from types import SimpleNamespace

from src.domain.rag.docling_loader import (
    _chunk_provenance,
    _refine_chunks_for_token_budget,
)


class LenTokenizer:
    """1 char = 1 token (_count_tokens uses encode(add_special_tokens=False))."""

    def encode(self, text, add_special_tokens=False):
        return list(text)


def _mk_chunk(text, headings=None, page_no=None):
    """Mimic a docling HierarchicalChunker chunk object (meta.headings + doc_items.prov)."""
    prov = [SimpleNamespace(page_no=page_no)] if page_no is not None else []
    doc_items = [SimpleNamespace(prov=prov)]
    return SimpleNamespace(
        text=text,
        meta=SimpleNamespace(headings=headings, doc_items=doc_items),
    )


TOK = LenTokenizer()


class TestChunkProvenance:
    def test_bare_string_no_meta(self):
        assert _chunk_provenance("純字串") == {"headings": None, "page_no": None, "content_type": None}

    def test_full_extraction(self):
        c = _mk_chunk("x", headings=["第一章", "1.1 節"], page_no=3)
        assert _chunk_provenance(c) == {"headings": ["第一章", "1.1 節"], "page_no": 3, "content_type": None}

    def test_missing_fields_defensive(self):
        # meta present but headings empty, prov empty -> all None, no crash
        c = _mk_chunk("x", headings=[], page_no=None)
        assert _chunk_provenance(c) == {"headings": None, "page_no": None, "content_type": None}

    def test_page_no_from_first_prov_only(self):
        # Multiple doc_items: take the first one with a page_no
        c = SimpleNamespace(
            text="x",
            meta=SimpleNamespace(
                headings=None,
                doc_items=[
                    SimpleNamespace(prov=[]),
                    SimpleNamespace(prov=[SimpleNamespace(page_no=7),
                                          SimpleNamespace(page_no=8)]),
                ],
            ),
        )
        assert _chunk_provenance(c)["page_no"] == 7


class TestRefineRecords:
    def test_bare_strings_yield_none_meta(self):
        # The fallback split path feeds bare strings -> record meta is None
        out = _refine_chunks_for_token_budget(["a" * 15], TOK, max_tokens=20)
        assert out == [{"text": "a" * 15, "headings": None, "page_no": None, "content_type": None}]

    def test_passthrough_keeps_meta(self):
        c = _mk_chunk("b" * 15, headings=["H"], page_no=2)
        out = _refine_chunks_for_token_budget([c], TOK, max_tokens=20)
        assert out == [{"text": "b" * 15, "headings": ["H"], "page_no": 2, "content_type": None}]

    def test_oversized_split_inherits_meta(self):
        # 3 lines x 10 chars, budget 10 -> split into several segments, each inheriting the source chunk's meta
        text = "\n".join(["r" * 10] * 3)
        c = _mk_chunk(text, headings=["表格章"], page_no=5)
        out = _refine_chunks_for_token_budget([c], TOK, max_tokens=10)
        assert len(out) > 1
        for rec in out:
            assert rec["headings"] == ["表格章"]
            assert rec["page_no"] == 5
            assert len(rec["text"]) <= 10

    def test_small_chunks_never_merge_across_sections(self):
        # Two small chunks of different sections used to merge, and the merged
        # chunk was labelled with the first one's heading (the second one's text then
        # carried the wrong section).
        c1 = _mk_chunk("aaa", headings=["首章"], page_no=1)
        c2 = _mk_chunk("bbb", headings=["次章"], page_no=9)
        out = _refine_chunks_for_token_budget([c1, c2], TOK, max_tokens=20)
        assert out == [
            {"text": "aaa", "headings": ["首章"], "page_no": 1, "content_type": None},
            {"text": "bbb", "headings": ["次章"], "page_no": 9, "content_type": None},
        ]

    def test_merge_within_section_takes_common_headings(self):
        # Same section (first two levels), different subsections -> merged, labelled
        # with the headings both share; page from the first piece.
        c1 = _mk_chunk("aaa", headings=["Manual", "5. Warranty"], page_no=4)
        c2 = _mk_chunk("bbb", headings=["Manual", "5. Warranty", "5.1 Hardware"], page_no=5)
        out = _refine_chunks_for_token_budget([c1, c2], TOK, max_tokens=20)
        assert out == [{"text": "aaa\nbbb", "headings": ["Manual", "5. Warranty"],
                        "page_no": 4, "content_type": None}]

    def test_merged_content_type_table_wins(self):
        table = SimpleNamespace(prov=[], label="table")
        c1 = _mk_chunk("intro", headings=["A"])
        c2 = SimpleNamespace(text="r, c = 1", meta=SimpleNamespace(headings=["A"], doc_items=[table]))
        out = _refine_chunks_for_token_budget([c1, c2], TOK, max_tokens=40)
        assert len(out) == 1 and out[0]["content_type"] == "table"

    def test_table_may_use_larger_budget_than_prose(self):
        table = SimpleNamespace(prov=[], label="table")
        body = "A, x = 1. B, x = 2. C, x = 3"  # 28 tokens
        c = SimpleNamespace(text=body, meta=SimpleNamespace(headings=["T"], doc_items=[table]))
        whole = _refine_chunks_for_token_budget([c], TOK, max_tokens=20, table_max_tokens=40)
        assert [r["text"] for r in whole] == [body]

    def test_oversized_triplet_table_splits_at_row_boundaries(self):
        table = SimpleNamespace(prov=[], label="table")
        body = "Basic, Fee = 15. Basic, Seats = 5. Pro, Fee = 35. Pro, Seats = 20. Max, Fee = 90. Max, Seats = 99"
        c = SimpleNamespace(text=body, meta=SimpleNamespace(headings=["Plans"], doc_items=[table]))
        out = _refine_chunks_for_token_budget([c], TOK, max_tokens=40)
        texts = [r["text"] for r in out]
        assert len(texts) > 1
        assert all(len(t) <= 40 for t in texts)
        # every row stays whole (both cells of a row in the same piece)
        for row in ("Basic", "Pro", "Max"):
            holders = [t for t in texts if f"{row}, " in t]
            assert len(holders) == 1, (row, texts)
        assert all(r["content_type"] == "table" and r["headings"] == ["Plans"] for r in out)

    def test_empty_text_skipped(self):
        out = _refine_chunks_for_token_budget(["   ", ""], TOK, max_tokens=20)
        assert out == []


class TestLeafSplitterProvenance:
    def _split(self, chunks):
        from llama_index.core import Document
        from src.domain.rag.leaf_splitter import LeafSplitter

        splitter = LeafSplitter(leaf_chunk_size=450, chunk_overlap=50)
        doc = Document(
            text="全文",
            metadata={"file_id": "f1", "file_name": "a.pdf",
                      "_docling_chunks": chunks},
        )
        return splitter.split_into_leaf_documents(doc)

    def test_dict_records_inject_provenance(self):
        leaves = self._split([
            {"text": "第一段內容", "headings": ["第一章"], "page_no": 2},
        ])
        assert len(leaves) == 1
        md = leaves[0].metadata
        assert md["headings"] == ["第一章"]
        assert md["page_no"] == 2
        assert md["file_id"] == "f1"  # document-level metadata is still present

    def test_none_provenance_not_injected(self):
        # meta is None (audio/bare string) -> do not insert the key, do not pollute node metadata
        leaves = self._split([
            {"text": "音檔轉錄段", "headings": None, "page_no": None},
            "裸字串段",
        ])
        assert len(leaves) == 2
        for leaf in leaves:
            assert "headings" not in leaf.metadata
            assert "page_no" not in leaf.metadata

    def test_empty_text_record_filtered(self):
        leaves = self._split([{"text": "  ", "headings": ["H"], "page_no": 1}])
        assert leaves == []
