"""PDFs read from their text layer (src/domain/rag/pdf_text_layer.py): paragraphs, page
numbers, a visible marker for pages without text, and no empty index for a scanned PDF.

PDFs are built in the test (tests/pdf_fixtures.py); no model, no network."""

from types import SimpleNamespace

import pytest

from src.domain.rag import pdf_text_layer as tl
from src.domain.rag.pdf_text_layer import _Line
from tests.pdf_fixtures import (encrypt_pdf, join_pdfs, lines_pdf_bytes, make_image_pdf,
                                make_pdf)


class _WordTok:
    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.fixture(autouse=True)
def _word_tokenizer(monkeypatch):
    monkeypatch.setattr("src.domain.rag.docling_loader.get_hf_tokenizer", lambda name: _WordTok())


def _lines_pdf(path, pages):
    path.write_bytes(lines_pdf_bytes(pages))
    return str(path)


class TestParagraphs:
    def test_a_line_reaching_the_margin_continues_on_the_next(self):
        lines = [_Line("The quick brown fox jumps over", 72, 500), _Line("the lazy dog.", 72, 160),
                 _Line("A new paragraph.", 72, 180)]
        assert tl.lines_to_paragraphs(lines) == ["The quick brown fox jumps over the lazy dog.",
                                                 "A new paragraph."]

    def test_a_sentence_end_at_the_margin_ends_the_paragraph(self):
        lines = [_Line("It ends here.", 72, 500), _Line("Next one.", 72, 200)]
        assert tl.lines_to_paragraphs(lines) == ["It ends here.", "Next one."]

    def test_cjk_lines_join_without_a_space(self):
        lines = [_Line("本文件為虛構公", 72, 500), _Line("司資料。", 72, 150)]
        assert tl.lines_to_paragraphs(lines) == ["本文件為虛構公司資料。"]

    def test_pdfium_hyphen_marker_keeps_the_hyphen(self):
        lines = [_Line("Kettle (non￾", 72, 300), _Line("connected), 1.5 L", 72, 200)]
        assert tl.lines_to_paragraphs(lines) == ["Kettle (non-connected), 1.5 L"]
        assert tl.lines_to_paragraphs([_Line("a￾b", 72, 100)]) == ["a-b"]

    def test_table_rows_are_never_joined(self):
        lines = [_Line("SKU   Price", 72, 500, gappy=True), _Line("AK2   3,980", 72, 500, gappy=True),
                 _Line("Body text", 72, 120)]
        assert tl.lines_to_paragraphs(lines) == ["SKU   Price", "AK2   3,980", "Body text"]

    def test_without_geometry_every_line_is_a_paragraph(self):
        assert tl.lines_to_paragraphs([_Line("one"), _Line(" "), _Line("two")]) == ["one", "two"]

    def test_real_pdf_lines_and_page_numbers(self, tmp_path):
        path = _lines_pdf(tmp_path / "a.pdf", [["Refund policy", "Returns within 7 days."],
                                                ["Shipping", "Taipei 100 NT$."]])
        pages = tl.read_pages(path, "a.pdf")
        assert [p.page_no for p in pages] == [1, 2] and not any(p.marker for p in pages)
        assert pages[0].paragraphs == ["Refund policy", "Returns within 7 days."]
        assert pages[1].paragraphs == ["Shipping", "Taipei 100 NT$."]


class TestClassifyPage:
    def test_scanned_page_gets_a_marker(self):
        p = tl.classify_page(3, "", [], has_image=True, image_share=1.0)
        assert not p.has_text and p.marker.startswith("[Page 3: no text layer")
        assert "rag.parsing.vision" in p.marker

    def test_a_page_number_on_a_scan_is_still_a_scan(self):
        p = tl.classify_page(2, "2", ["2"], has_image=True, image_share=0.9)
        assert not p.has_text and p.marker

    def test_blank_page_contributes_nothing(self):
        p = tl.classify_page(4, "", [], has_image=False, image_share=0.0)
        assert not p.has_text and p.marker is None

    def test_short_text_without_image_is_kept(self):
        p = tl.classify_page(1, "Notes", ["Notes"], has_image=False, image_share=0.0)
        assert p.paragraphs == ["Notes"] and p.marker is None

    def test_broken_font_encoding_is_a_marker(self):
        text = "�" * 40 + "ab"
        p = tl.classify_page(1, text, [text], has_image=False, image_share=0.0)
        assert not p.has_text and "unreadable" in p.marker

    def test_mostly_picture_page_keeps_its_text_and_adds_a_marker(self):
        text = "Figure 1: the kettle base, front view"
        p = tl.classify_page(5, text, [text], has_image=True, image_share=0.8)
        assert p.paragraphs == [text] and "most of this page is an image" in p.marker

    def test_text_page_with_a_small_logo_has_no_marker(self):
        text = "x" * 300
        assert tl.classify_page(1, text, [text], has_image=True, image_share=0.05).marker is None

    def test_garbled_ratio(self):
        assert tl.garbled_ratio("") == 0.0 and tl.garbled_ratio("ab�") == 0.5


class TestParse:
    def test_text_pdf_records_carry_pages(self, tmp_path):
        path = make_pdf(tmp_path / "a.pdf", ["Alpha page text here.", "Beta page text here."])
        warnings = {}
        full, records = tl.parse(path, "a.pdf", max_tokens=100, warnings=warnings)
        assert "Alpha page text here." in full and "Beta page text here." in full
        assert {r["page_no"] for r in records} == {1}  # small pieces merge from page 1 on
        assert warnings == {}

    def test_each_page_starts_where_it_is(self, tmp_path):
        long = " ".join(f"word{i}" for i in range(60))
        path = make_pdf(tmp_path / "a.pdf", [long, long.replace("word", "term")])
        _, records = tl.parse(path, "a.pdf", max_tokens=60)
        pages = [r["page_no"] for r in records]
        assert pages[0] == 1 and 2 in pages

    def test_image_only_pdf_fails_with_what_to_set(self, tmp_path):
        path = make_image_pdf(tmp_path / "scan.pdf", pages=2)
        with pytest.raises(tl.NoTextLayerError, match=r"'scan.pdf' has no usable text layer.*rag.parsing.vision"):
            tl.parse(path, "scan.pdf")

    def test_mixed_pdf_marks_the_scanned_page_and_warns(self, tmp_path):
        text = make_pdf(tmp_path / "t.pdf", ["Opening hours are nine to five on weekdays."])
        scan = make_image_pdf(tmp_path / "s.pdf")
        path = join_pdfs(tmp_path / "mixed.pdf", text, scan, text)
        warnings = {}
        full, records = tl.parse(path, "mixed.pdf", max_tokens=200, warnings=warnings)
        assert "[Page 2: no text layer (a scanned or image-only page); set rag.parsing.vision" in full
        assert warnings["missing_pages"] == [2]
        assert warnings["warning"].startswith("Page 2 has no usable text layer and is indexed as a marker only")
        assert full.count("Opening hours") == 2

    def test_progress_is_reported_per_page(self, tmp_path):
        path = make_pdf(tmp_path / "a.pdf", ["one one one", "two two two", "three three"])
        seen = []
        tl.parse(path, "a.pdf", progress_cb=lambda *a: seen.append(a))
        assert seen == [("loading", 1, 3), ("loading", 2, 3), ("loading", 3, 3)]

    def test_bad_progress_callback_is_ignored(self, tmp_path):
        path = make_pdf(tmp_path / "a.pdf", ["fine text on the page"])
        tl.parse(path, "a.pdf", progress_cb=lambda *a: 1 / 0)

    def test_encrypted_and_corrupt(self, tmp_path):
        plain = make_pdf(tmp_path / "p.pdf", ["secret"])
        locked = encrypt_pdf(plain, tmp_path / "l.pdf")
        with pytest.raises(ValueError, match="password-protected"):
            tl.parse(locked, "l.pdf")
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"%PDF-1.4 nonsense")
        with pytest.raises(ValueError, match="corrupted or unreadable"):
            tl.parse(str(bad), "bad.pdf")

    def test_marker_warning_wording(self):
        assert tl.marker_warning([]) is None
        assert tl.marker_warning([2, 5]).startswith("Pages 2, 5 have no usable text layer and are indexed")


class TestGoldenPdfs:
    """The public eval PDFs (evals/rag/documents): Traditional Chinese and English prose."""

    def test_chinese_paragraph_rejoined_across_lines(self):
        pages = tl.read_pages("evals/rag/documents/zh-harborlight-2026-budget-and-travel.pdf", "zh.pdf")
        text = "\n".join(p for page in pages for p in page.paragraphs)
        assert "本文件為 Folio 檢索評測集撰寫之虛構公司資料（港光科技股份有限公司為虛構公司）。授權：MIT。" in text
        assert "研發部 48,200 56,500 +17.2%" in text  # a table row stays on one line
        assert not any(p.marker for p in pages)

    def test_english_paragraph_and_hyphen(self):
        pages = tl.read_pages("evals/rag/documents/en-harborlight-2026-price-list.pdf", "en.pdf")
        text = "\n".join(p for page in pages for p in page.paragraphs)
        assert ("Effective 1 January 2026. All prices are in New Taiwan dollars (NT$) and include 5% "
                "business tax unless stated otherwise.") in text
        assert "Kettle (non-connected)," in text


def test_document_is_built_with_page_provenance():
    pages = [tl.PageText(1, ["one"]), tl.PageText(2, [], marker="[Page 2: x]"), tl.PageText(3)]
    doc = tl.build_document(pages, "f.pdf")
    items = [(it.text, it.prov[0].page_no) for it, _ in doc.iterate_items()]
    assert items == [("one", 1), ("[Page 2: x]", 2)]
    assert sorted(doc.pages) == [1, 2]


def test_image_coverage_survives_a_broken_object_tree():
    class _Page:
        def get_size(self):
            return (100, 100)

        def get_objects(self, **kw):
            raise RuntimeError("boom")
    assert tl._image_coverage(_Page()) == (False, 0.0)


def test_image_coverage_without_position():
    class _Obj:
        def get_pos(self):
            raise RuntimeError("no pos")

    page = SimpleNamespace(get_size=lambda: (100, 100), get_objects=lambda **kw: [_Obj()])
    assert tl._image_coverage(page) == (True, 0.0)
