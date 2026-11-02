"""Local parsing: OCR switch honoured, partial conversions reported, cache released."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.domain.rag import docling_loader as dl

_DL = "src.domain.rag.docling_loader"


def _cfg(ocr=True):
    return SimpleNamespace(rag=SimpleNamespace(docling=SimpleNamespace(ocr_enabled=ocr)))


class TestOcrSwitch:
    @pytest.mark.parametrize("on", [True, False])
    def test_ocr_enabled_reads_config(self, on):
        with patch("src.config.config_manager.Config.get_config_model", return_value=_cfg(on)):
            assert dl.ocr_enabled() is on

    def test_ocr_enabled_defaults_on_when_config_unreadable(self):
        with patch("src.config.config_manager.Config.get_config_model", side_effect=RuntimeError):
            assert dl.ocr_enabled() is True

    @pytest.mark.parametrize("on", [True, False])
    def test_converter_gets_do_ocr(self, on, monkeypatch):
        # The switch used to only set an env var docling never reads.
        captured = {}

        class _FakeConverter:
            def __init__(self, format_options=None):
                captured["fo"] = format_options

        monkeypatch.setattr(dl, "_converter", None)
        monkeypatch.setattr(dl, "_apply_docling_env", lambda: None)
        monkeypatch.setattr(dl, "_resolve_artifacts_path", lambda: None)
        monkeypatch.setattr(dl, "_resolve_whisper_path", lambda: None)
        monkeypatch.setattr(dl, "asr_wants_local_whisper", lambda: False)
        monkeypatch.setattr(dl, "ocr_enabled", lambda: on)
        monkeypatch.setattr("docling.document_converter.DocumentConverter", _FakeConverter)
        try:
            dl._build_converter()
        finally:
            monkeypatch.setattr(dl, "_converter", None)
        from docling.datamodel.base_models import InputFormat
        opts = captured["fo"][InputFormat.PDF].pipeline_options
        assert opts.do_ocr is on
        assert captured["fo"][InputFormat.IMAGE].pipeline_options.do_ocr is on


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

        released = []
        monkeypatch.setattr(dl, "get_converter", lambda: _Conv())
        monkeypatch.setattr(dl, "release_accelerator_cache", lambda: released.append(1))
        monkeypatch.setattr("src.domain.rag.docling_guard.install_batch_hook", lambda: True)
        report = {}
        full, chunks = dl.docling_convert_once(
            str(tmp_path / "a.md"), "a.md", max_tokens=64, report=report)
        assert len(calls) == 2                      # one retry for a non-OOM partial result
        assert report["missing_pages"] == [2]
        assert "page 2 could not be parsed" in report["warning"]
        assert "First section text." in full and chunks
        assert released                             # cache released after the file

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
        monkeypatch.setattr("src.domain.rag.docling_guard.install_batch_hook", lambda: True)
        dl.docling_convert_once(str(stored), "doc.md", max_tokens=64)
        assert len(seen) == 2 and seen[0] is not seen[1]
        assert all(s.name == "doc.md" for s in seen)


class TestDocumentLoaderParseWarning:
    def _loader(self):
        from src.domain.rag.document_loader import DocumentLoader
        return DocumentLoader(leaf_chunk_size=256, embedding_model_name=None)

    def test_warning_travels_in_private_metadata(self, tmp_path):
        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF")

        def _fake_convert(**kw):
            kw["report"]["warning"] = "Partial parse: page 3 ..."
            return "some text", [{"text": "some text"}]

        with patch("src.domain.rag.document_loader.FileStorage.resolve_path", return_value=f), \
             patch("src.domain.rag.document_loader.is_docling_supported", return_value=True), \
             patch("src.domain.rag.document_loader.docling_convert_once", side_effect=_fake_convert):
            doc = self._loader().load(file_path="s/x", file_id="1", file_name="x.pdf")
        assert doc.metadata["_parse_warning"].startswith("Partial parse: page 3")

    def test_empty_scan_with_ocr_off_explains_why(self, tmp_path):
        f = tmp_path / "scan.pdf"
        f.write_bytes(b"%PDF")
        with patch("src.domain.rag.document_loader.FileStorage.resolve_path", return_value=f), \
             patch("src.domain.rag.document_loader.is_docling_supported", return_value=True), \
             patch("src.domain.rag.document_loader.ocr_enabled", return_value=False), \
             patch("src.domain.rag.document_loader.docling_convert_once", return_value=("  ", [])):
            with pytest.raises(ValueError, match="OCR is disabled"):
                self._loader().load(file_path="s/x", file_id="1", file_name="scan.pdf")

    def test_empty_content_with_ocr_on_keeps_generic_error(self, tmp_path):
        f = tmp_path / "e.docx"
        f.write_bytes(b"x")
        with patch("src.domain.rag.document_loader.FileStorage.resolve_path", return_value=f), \
             patch("src.domain.rag.document_loader.is_docling_supported", return_value=True), \
             patch("src.domain.rag.document_loader.docling_convert_once", return_value=("", [])):
            with pytest.raises(ValueError, match="Empty content"):
                self._loader().load(file_path="s/x", file_id="1", file_name="e.docx")


class TestMediaSwitches:
    def test_asr_disabled_helper(self):
        from src.api.router.media import _asr_disabled
        cfg = SimpleNamespace(rag=SimpleNamespace(asr=SimpleNamespace(enabled=False)))
        with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
            assert _asr_disabled() is True
        cfg.rag.asr.enabled = True
        with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
            assert _asr_disabled() is False
        with patch("src.config.config_manager.Config.get_config_model", side_effect=RuntimeError):
            assert _asr_disabled() is False
