"""Document parsing: partial docling conversions reported, and how the loader routes PDFs and
images (vision model, text layer, refusal) under the live config."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.domain.rag import docling_loader as dl

_DL = "src.domain.rag.docling_loader"


def _real_md_result(tmp_path):
    from docling.document_converter import DocumentConverter

    md = tmp_path / "a.md"
    md.write_text("# Title\n\n## One\n\nFirst section text.\n\n## Two\n\nSecond section text.\n")
    return DocumentConverter().convert(str(md))


class TestConvertOnceReportsPartialPages:
    def test_partial_result_retried_and_reported(self, tmp_path, monkeypatch):
        base = _real_md_result(tmp_path)
        err = SimpleNamespace(page_no=2, error_message="FontBBox", module_name="preprocess")
        partial = SimpleNamespace(
            status=SimpleNamespace(value="partial_success"), errors=[err], document=base.document)
        calls = []

        class _Conv:
            def convert(self, source):
                calls.append(source)
                return partial

        monkeypatch.setattr(dl, "get_converter", lambda: _Conv())
        report = {}
        full, chunks = dl.docling_convert_once(
            str(tmp_path / "a.md"), "a.md", max_tokens=64, report=report)
        assert len(calls) == 2                      # one retry for a partial result
        assert report["missing_pages"] == [2]
        assert "page 2 could not be parsed" in report["warning"]
        assert "First section text." in full and chunks

    def test_complete_result_has_no_warning(self, tmp_path, monkeypatch):
        base = _real_md_result(tmp_path)

        class _Conv:
            def convert(self, source):
                return base

        monkeypatch.setattr(dl, "get_converter", lambda: _Conv())
        report = {}
        dl.docling_convert_once(str(tmp_path / "a.md"), "a.md", max_tokens=64, report=report)
        assert report == {"warning": None, "missing_pages": []}

    def test_extensionless_storage_path_gets_fresh_stream_per_attempt(self, tmp_path, monkeypatch):
        base = _real_md_result(tmp_path)
        stored = tmp_path / "uuid-no-ext"
        stored.write_text("# T\n\ntext\n")
        err = SimpleNamespace(page_no=1, error_message="x", module_name="m")
        partial = SimpleNamespace(status=SimpleNamespace(value="partial_success"), errors=[err],
                                  document=base.document)
        seen = []

        class _Conv:
            def convert(self, source):
                seen.append(source)
                return partial

        monkeypatch.setattr(dl, "get_converter", lambda: _Conv())
        dl.docling_convert_once(str(stored), "doc.md", max_tokens=64)
        assert len(seen) == 2 and seen[0] is not seen[1]
        assert all(s.name == "doc.md" for s in seen)


class TestDocumentLoaderParseWarning:
    def _loader(self):
        from src.domain.rag.document_loader import DocumentLoader
        return DocumentLoader(leaf_chunk_size=256, embedding_model_name=None)

    def test_warning_travels_in_private_metadata(self, tmp_path):
        f = tmp_path / "x.docx"
        f.write_bytes(b"PK")

        def _fake_convert(**kw):
            kw["report"]["warning"] = "Partial parse: page 3 ..."
            return "some text", [{"text": "some text"}]

        with patch("src.domain.rag.document_loader.FileStorage.resolve_path", return_value=f), \
             patch("src.domain.rag.document_loader.docling_convert_once", side_effect=_fake_convert):
            doc = self._loader().load(file_path="s/x", file_id="1", file_name="x.docx")
        assert doc.metadata["_parse_warning"].startswith("Partial parse: page 3")

    def test_empty_content_keeps_generic_error(self, tmp_path):
        f = tmp_path / "e.docx"
        f.write_bytes(b"x")
        with patch("src.domain.rag.document_loader.FileStorage.resolve_path", return_value=f), \
             patch("src.domain.rag.document_loader.docling_convert_once", return_value=("", [])):
            with pytest.raises(ValueError, match="Empty content"):
                self._loader().load(file_path="s/x", file_id="1", file_name="e.docx")


_VISION = SimpleNamespace(provider="openai-compatible", model="vlm", azure_deployment=None,
                          base_url="http://v/v1", api_key=None, configured=lambda: True)


def _live(pdf="auto", vision=None):
    return SimpleNamespace(rag=SimpleNamespace(parsing=SimpleNamespace(pdf=pdf, vision=vision)))


class TestPageParserSelection:
    """auto → the vision model when configured, else the text layer; images need the model."""

    def _pick(self, monkeypatch, name, **live):
        from src.domain.rag.document_loader import DocumentLoader
        monkeypatch.setattr("src.config.config_manager.Config.get_config_model", lambda: _live(**live))
        monkeypatch.setattr("src.domain.rag.cloud_parser.CloudDocumentParser.from_config",
                            classmethod(lambda cls, rag: SimpleNamespace(parse="VISION")))
        return DocumentLoader._page_parser_for(name)

    def test_auto_without_vision_reads_the_text_layer(self, monkeypatch):
        from src.domain.rag import pdf_text_layer
        assert self._pick(monkeypatch, "a.PDF") is pdf_text_layer.parse

    def test_auto_with_vision_sends_pdfs_and_images_to_the_model(self, monkeypatch):
        assert self._pick(monkeypatch, "a.pdf", vision=_VISION) == "VISION"
        assert self._pick(monkeypatch, "b.png", vision=_VISION) == "VISION"

    def test_text_mode_keeps_pdfs_off_the_model_but_images_still_use_it(self, monkeypatch):
        from src.domain.rag import pdf_text_layer
        assert self._pick(monkeypatch, "a.pdf", pdf="text", vision=_VISION) is pdf_text_layer.parse
        assert self._pick(monkeypatch, "b.jpg", pdf="text", vision=_VISION) == "VISION"

    def test_image_without_vision_fails_with_what_to_set(self, monkeypatch):
        from src.domain.rag.cloud_parser import ImagesNeedVisionError
        with pytest.raises(ImagesNeedVisionError, match="rag.parsing.vision"):
            self._pick(monkeypatch, "scan.tiff")

    def test_forced_vision_without_a_model_never_falls_back(self, monkeypatch):
        with pytest.raises(ValueError, match="rag.parsing.pdf is 'vision'"):
            self._pick(monkeypatch, "a.pdf", pdf="vision")

    def test_other_formats_get_no_page_parser(self, monkeypatch):
        assert self._pick(monkeypatch, "a.docx", vision=_VISION) is None
        assert self._pick(monkeypatch, "a.md") is None


class TestLoaderUsesThePageParser:
    def _loader(self):
        from src.domain.rag.document_loader import DocumentLoader
        return DocumentLoader(leaf_chunk_size=256, embedding_model_name=None)

    def test_marker_warning_and_records_reach_the_document(self, tmp_path, monkeypatch):
        from src.domain.rag.document_loader import DocumentLoader
        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF")

        def parse(path, name, **kw):
            kw["warnings"]["warning"] = "Page 2 has no usable text layer ..."
            return "page one text", [{"text": "page one text", "page_no": 1}]

        monkeypatch.setattr(DocumentLoader, "_page_parser_for", staticmethod(lambda name: parse))
        with patch("src.domain.rag.document_loader.FileStorage.resolve_path", return_value=f):
            doc = self._loader().load(file_path="s/x", file_id="1", file_name="x.pdf")
        assert doc.text == "page one text" and doc.metadata["_docling_chunks"][0]["page_no"] == 1
        assert doc.metadata["_parse_warning"].startswith("Page 2 has no usable text layer")

    def test_empty_page_output_is_never_indexed(self, tmp_path, monkeypatch):
        from src.domain.rag.document_loader import DocumentLoader
        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF")
        monkeypatch.setattr(DocumentLoader, "_page_parser_for",
                            staticmethod(lambda name: lambda *a, **k: ("  ", [])))
        with patch("src.domain.rag.document_loader.FileStorage.resolve_path", return_value=f):
            with pytest.raises(ValueError, match="Empty content"):
                self._loader().load(file_path="s/x", file_id="1", file_name="x.pdf")
