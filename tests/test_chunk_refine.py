
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

    def test_content_type_table_wins_over_picture(self):
        # A chunk mixing a picture and a table is badged "table"; text-only stays None
        c = SimpleNamespace(
            text="x",
            meta=SimpleNamespace(
                headings=None,
                doc_items=[
                    SimpleNamespace(label="picture", prov=[]),
                    SimpleNamespace(label=SimpleNamespace(value="table"),
                                    prov=[SimpleNamespace(page_no=4)]),
                ],
            ),
        )
        assert _chunk_provenance(c) == {"headings": None, "page_no": 4, "content_type": "table"}

    def test_content_type_picture_only(self):
        c = SimpleNamespace(
            text="x",
            meta=SimpleNamespace(headings=None,
                                 doc_items=[SimpleNamespace(label="picture", prov=[])]),
        )
        assert _chunk_provenance(c)["content_type"] == "picture"


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


class TestSentenceSplit:
    """H1: CJK sentence ends need no whitespace, and pieces are balanced."""

    ZH = ("第一條員工出差應事先申請並經主管核准。未經核准者費用不予核銷！"
          "住宿費每日以二千元為上限；超過部分自行負擔。膳雜費每日五百元？"
          "當日往返者減半支給。出差報告應於返回後五日內提出。")

    def test_cjk_paragraph_splits_at_sentence_ends(self):
        from src.domain.rag.docling_loader import _split_by_sentence

        pieces = _split_by_sentence(self.ZH, TOK, 40)
        assert len(pieces) > 1
        assert "".join(pieces) == self.ZH  # nothing lost, nothing duplicated
        for p in pieces:
            assert len(p) <= 40
            assert p[-1] in "。！？；", p  # never cut mid-sentence

    def test_cjk_paragraph_through_refine_is_not_char_cut(self):
        recs = _refine_chunks_for_token_budget([self.ZH], TOK, 40)
        assert len(recs) > 1
        assert all(r["text"][-1] in "。！？；" for r in recs)
        assert "".join(r["text"] for r in recs) == self.ZH

    def test_latin_needs_whitespace_after_terminator(self):
        from src.domain.rag.docling_loader import _SENTENCE_END_RE, _split_keep_ends

        text = "Version 3.5 costs $2.50 today. Next sentence! Last?"
        parts = _split_keep_ends(text, _SENTENCE_END_RE)
        assert parts == ["Version 3.5 costs $2.50 today. ", "Next sentence! ", "Last?"]

    def test_closing_quote_stays_with_sentence(self):
        from src.domain.rag.docling_loader import _SENTENCE_END_RE, _split_keep_ends

        parts = _split_keep_ends("他說：「好。」然後離開。", _SENTENCE_END_RE)
        assert parts == ["他說：「好。」", "然後離開。"]

    def test_last_piece_is_not_a_runt(self):
        from src.domain.rag.docling_loader import _split_by_sentence

        # 12 sentences x 20 chars = 240 tokens, budget 200: greedy gives 200 + 40.
        text = "".join(f"Sentence number {i:02d}. " for i in range(12)).strip()
        pieces = _split_by_sentence(text, TOK, 200)
        sizes = [len(p) for p in pieces]
        assert len(pieces) == 2
        assert min(sizes) >= 100, sizes
        assert " ".join(pieces) == text

    def test_oversized_sentence_cut_at_clauses_before_chars(self):
        from src.domain.rag.docling_loader import _split_by_sentence

        sent = "甲方應於簽約後付款，乙方應於收款後交貨，丙方應於交貨後驗收，丁方應於驗收後開立發票。"
        pieces = _split_by_sentence(sent, TOK, 25)
        assert "".join(pieces) == sent
        for p in pieces:
            assert len(p) <= 25
            assert p[-1] in "，。", p

    def test_real_tokenizer_zh_pieces_fill_the_budget(self):
        from src.domain.rag.chunk_tokenizer import get_chunk_tokenizer
        from src.domain.rag.docling_loader import _count_tokens

        tok = get_chunk_tokenizer("intfloat/multilingual-e5-large")
        text = self.ZH * 12
        recs = _refine_chunks_for_token_budget([text], tok, 256)
        sizes = [_count_tokens(tok, r["text"]) for r in recs]
        assert all(s <= 256 for s in sizes)
        # Balanced: no piece below 60% of the largest one.
        assert min(sizes) >= 0.6 * max(sizes), sizes
        assert all(r["text"][-1] in "。！？；" for r in recs)


class TestPackBalanced:
    def test_respects_budget(self):
        from src.domain.rag.docling_loader import _pack_balanced

        groups = _pack_balanced([30] * 10, 100)
        assert all(sum(30 for _ in g) <= 100 for g in groups)
        assert [i for g in groups for i in g] == list(range(10))

    def test_balances_tail(self):
        from src.domain.rag.docling_loader import _pack_balanced

        # greedy: [100] [20]; balanced: two pieces of 60
        groups = _pack_balanced([20] * 6, 100)
        assert [len(g) for g in groups] == [3, 3]

    def test_oversized_unit_alone(self):
        from src.domain.rag.docling_loader import _pack_balanced

        assert _pack_balanced([10, 500, 10], 100) == [[0], [1], [2]]

    def test_empty(self):
        from src.domain.rag.docling_loader import _pack_balanced

        assert _pack_balanced([], 100) == []


class TestSplitByChars:
    def test_pieces_fit_cover_text_and_are_balanced(self):
        from src.domain.rag.docling_loader import _split_by_chars

        text = "字" * 250
        pieces = _split_by_chars(text, TOK, 100)
        assert "".join(pieces) == text
        assert [len(p) for p in pieces] == [83, 83, 84]

    def test_space_delimited_text_not_cut_mid_word(self):
        from src.domain.rag.docling_loader import _split_by_chars

        text = " ".join(f"word{i:03d}" for i in range(60))  # 479 chars, no punctuation
        pieces = _split_by_chars(text, TOK, 100)
        assert "".join(pieces) == text
        assert all(len(p) <= 100 for p in pieces)
        for p in pieces[:-1]:
            assert p.endswith(" "), p

    def test_single_oversized_character_still_progresses(self):
        from src.domain.rag.docling_loader import _split_by_chars

        class Heavy:
            def encode(self, text, add_special_tokens=False):
                return [0] * (len(text) * 10)

        assert _split_by_chars("abc", Heavy(), 5) == ["a", "b", "c"]


def test_split_table_pieces_are_numbered_and_prose_is_not():
    rows = "\n".join(f"| r{i} | {'x' * 20} |" for i in range(12))
    table = {"text": f"| a | b |\n| --- | --- |\n{rows}", "content_type": "table"}
    chunk = SimpleNamespace(text=table["text"], meta=SimpleNamespace(
        headings=["H"], doc_items=[SimpleNamespace(label="table", prov=[])]))
    recs = _refine_chunks_for_token_budget([chunk], TOK, 100)
    assert len(recs) > 1
    assert [r["table_part"] for r in recs] == list(range(len(recs)))
    prose = _refine_chunks_for_token_budget(["x. " * 80], TOK, 100)
    assert len(prose) > 1 and all("table_part" not in r for r in prose)
