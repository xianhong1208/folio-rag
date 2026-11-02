"""Cloud PDF / image parsing (src/domain/rag/cloud_parser.py).

PDFs and images are generated locally (PyMuPDF / Pillow); the vision model is a scripted
stub client, plus one round trip through the real OpenAI client against the in-process
fake server. No network, no keys.
"""

import re
import threading
from datetime import date
from types import SimpleNamespace

import pytest

from src.config.model import ParsingConfig, RAGConfig, VisionModelConfig
from src.domain.rag import cloud_parser as cp
from src.domain.rag.cloud_parser import (
    CONTINUE_TOKEN,
    CloudDocumentParser,
    CloudParseError,
    PageResult,
    assign_page_provenance,
    clean_page_output,
    is_cloud_parsed,
    stitch_pages,
)
from src.domain.rag.usage_budget import BudgetExceeded, DailyBudget
from tests.fake_model_server import FakeModelServer


class _WordTok:
    """Whitespace token counter standing in for the HF tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.fixture(autouse=True)
def _word_tokenizer(monkeypatch):
    monkeypatch.setattr("src.domain.rag.docling_loader.get_hf_tokenizer", lambda name: _WordTok())


def _pdf(path, pages):
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return str(path)


class _Status(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        import httpx
        headers = {"Retry-After": retry_after} if retry_after else {}
        self.response = httpx.Response(status, headers=headers)


class _StubVision:
    """chat.completions.create → scripted output per page ("Page N of" in the user text)."""

    def __init__(self, pages, finish="stop"):
        self.pages = pages  # page_no -> str | Exception | list[str|Exception] (per attempt)
        self.calls = []
        self.finish = finish
        self.closed = False
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        user = kw["messages"][1]["content"][0]["text"]
        no = int(re.search(r"Page (\d+) of", user).group(1))
        with self._lock:
            self.calls.append((no, kw))
            spec = self.pages.get(no, f"Text of page {no}.")
            if isinstance(spec, list):
                spec = spec.pop(0) if len(spec) > 1 else spec[0]
        if isinstance(spec, Exception):
            raise spec
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=spec), finish_reason=self.finish)])

    def close(self):
        self.closed = True


def _cfg(**kw):
    vision = VisionModelConfig(model="vlm", base_url="http://v/v1", max_retries=kw.pop("max_retries", 2))
    return ParsingConfig(vision=vision, **kw)


def _parser(stub, budget=None, **kw):
    return CloudDocumentParser(_cfg(**kw), client_factory=lambda: stub, sleep=lambda s: None,
                               budget=budget or DailyBudget())


class TestCleanAndStitch:
    def test_clean_page_output(self):
        assert clean_page_output("```markdown\n# T\n\nx\n```") == ("# T\n\nx", False)
        assert clean_page_output(f"{CONTINUE_TOKEN}\nrest of it") == ("rest of it", True)
        assert clean_page_output(None) == ("", False)

    def test_table_continuation_drops_repeated_header(self):
        p1 = PageResult(1, "## Prices\n\n| Item | Cost |\n|---|---|\n| a | 1 |")
        p2 = PageResult(2, "| Item | Cost |\n|---|---|\n| b | 2 |\n\nAfter table.", continues=True)
        md = stitch_pages([p1, p2])
        assert "| a | 1 |\n| b | 2 |" in md
        assert md.count("| Item | Cost |") == 1
        # the page-2 marker comes after the continued table, before page 2's own text
        assert md.index("| b | 2 |") < md.index("⁣PAGE 2⁣") < md.index("After table.")

    def test_headerless_table_joins_without_model_flag(self):
        p1 = PageResult(1, "| A | B |\n|---|---|\n| 1 | 2 |")
        p2 = PageResult(2, "| 3 | 4 |\n\nNext.")  # block boundary: no context, no flag
        assert "| 1 | 2 |\n| 3 | 4 |" in stitch_pages([p1, p2])

    def test_different_table_is_not_joined(self):
        p1 = PageResult(1, "| A | B |\n|---|---|\n| 1 | 2 |")
        p2 = PageResult(2, "| X | Y | Z |\n|---|---|---|\n| 1 | 2 | 3 |")
        md = stitch_pages([p1, p2])
        assert md.index("⁣PAGE 2⁣") < md.index("| X | Y | Z |")

    @pytest.mark.parametrize("tail,head,joined", [
        ("The quick brown", "fox jumps.", "The quick brown fox jumps."),
        ("本條規定建築物", "起造人應申請。", "本條規定建築物起造人應申請。"),
        ("inter-", "national law.", "inter-national law."),
        ("1. first", "2. second", "1. first\n2. second"),
    ])
    def test_paragraph_and_list_glue(self, tail, head, joined):
        md = stitch_pages([PageResult(1, tail), PageResult(2, head, continues=True)])
        assert joined in md

    def test_failed_page_is_marked_and_never_glued(self):
        pages = [PageResult(1, "One."), PageResult(2, failed=True, error="HTTP 500 after 3 attempt(s)"),
                 PageResult(3, "three continues", continues=True)]
        md = stitch_pages(pages)
        assert "[Page 2: not transcribed — HTTP 500 after 3 attempt(s)]" in md
        assert "]\n\n⁣PAGE 3⁣\n\nthree continues" in md

    def test_continuation_on_first_page_is_ignored_and_empty_pages_keep_marker(self):
        md = stitch_pages([PageResult(1, "starts mid-way", continues=True), PageResult(2, "")])
        assert md.startswith("⁣PAGE 1⁣\n\nstarts mid-way") and "⁣PAGE 2⁣" in md

    def test_heading_path(self):
        path = cp._heading_path("# A\n\ntext\n\n## B\n\n### C", [])
        assert path == ["# A", "## B", "### C"]
        assert cp._heading_path("## D", path) == ["# A", "## D"]


class TestProvenance:
    def test_markers_become_page_numbers_and_disappear(self):
        md = stitch_pages([
            PageResult(1, "# Report\n\n## Scope\n\nIntro text.\n\n| a | b |\n|---|---|\n| 1 | 2 |"),
            PageResult(2, "| 3 | 4 |\n\nSecond page text.\n\n## Results\n\nFinal words."),
        ])
        doc = cp._markdown_to_docling(md, "r.pdf")
        assert assign_page_provenance(doc) == 2
        assert "\u2063" not in doc.export_to_markdown()
        from docling.chunking import HierarchicalChunker
        from src.domain.rag.docling_loader import _chunk_provenance
        raw = [(c.text, _chunk_provenance(c)) for c in HierarchicalChunker().chunk(dl_doc=doc)]
        second = next(m for t, m in raw if "Second page text" in t)
        assert second["page_no"] == 2
        assert second["headings"] == ["Report", "Scope"]  # breadcrumb carried across the break
        tables = [(t, m) for t, m in raw if m["content_type"] == "table"]
        assert len(tables) == 1 and tables[0][1]["page_no"] == 1  # one table spanning pages
        assert "3, b = 4" in tables[0][0]
        final = next(m for t, m in raw if "Final words" in t)
        assert final["headings"] == ["Report", "Results"] and final["page_no"] == 2

    def test_items_with_existing_provenance_are_kept(self):
        doc = cp._markdown_to_docling("Before any marker.\n\n\u2063PAGE 1\u2063\n\nAfter.", "x.pdf")
        assign_page_provenance(doc)
        items = [i for i, _ in doc.iterate_items()]
        assert items[0].prov == [] and items[1].prov[0].page_no == 1


class TestParse:
    def test_pdf_end_to_end_with_context_and_provenance(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["one", "two", "three"])
        stub = _StubVision({
            1: "# Handbook\n\n## Travel\n\nRules start here and continue",
            2: f"{CONTINUE_TOKEN}\non the next page.\n\n## Meals\n\nMeal policy.",
            3: "Closing notes.",
        })
        progress = []
        full, records = _parser(stub, max_concurrency=1).parse(
            path, "handbook.pdf", max_tokens=4, progress_cb=lambda *a: progress.append(a))
        assert "Rules start here and continue on the next page." in full
        rec = {r["text"]: r for r in records}
        meal = next(r for t, r in rec.items() if "Meal policy" in t)
        assert meal["page_no"] == 2 and meal["headings"] == ["Handbook", "Meals"]
        assert next(r for t, r in rec.items() if "Closing" in t)["page_no"] == 3
        # page 2 was sent page 1's tail and the heading path
        user2 = next(kw for no, kw in stub.calls if no == 2)["messages"][1]["content"][0]["text"]
        assert "continue" in user2 and "# Handbook > ## Travel" in user2
        img = next(kw for no, kw in stub.calls if no == 1)["messages"][1]["content"][1]
        assert img["image_url"]["url"].startswith("data:image/png;base64,")
        assert progress[-1] == ("loading", 3, 3) and stub.closed

    def test_blocks_run_in_parallel_and_block_starts_have_no_context(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", [str(i) for i in range(4)])
        stub = _StubVision({})
        md, report = _parser(stub, max_concurrency=2).parse_to_markdown(path, "a.pdf")
        assert report.pages == 4 and not report.failed_pages
        texts = {no: kw["messages"][1]["content"][0]["text"] for no, kw in stub.calls}
        assert "previous page is not shown" in texts[3]  # first page of block 2
        assert "Text of page 1." in texts[2]  # same block → context
        assert [m for m in re.findall(r"⁣PAGE (\d+)⁣", md)] == ["1", "2", "3", "4"]

    def test_retry_then_success_honours_retry_after(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["x"])
        slept = []
        stub = _StubVision({1: [_Status(429, "7"), "", "Recovered."]})
        p = CloudDocumentParser(_cfg(), client_factory=lambda: stub, sleep=slept.append, budget=DailyBudget())
        md, _ = p.parse_to_markdown(path, "a.pdf")
        assert "Recovered." in md and len(stub.calls) == 3
        assert slept[0] == 7.0  # Retry-After honoured; the empty answer was retried too

    def test_transport_error_is_retried(self, tmp_path):
        import httpx
        path = _pdf(tmp_path / "a.pdf", ["x"])
        stub = _StubVision({1: [httpx.ConnectError("reset"), "Fine."]})
        assert "Fine." in _parser(stub).parse_to_markdown(path, "a.pdf")[0]

    def test_failed_page_is_marked_not_fatal(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a", "b", "c"])
        stub = _StubVision({2: _Status(500)})
        md, report = _parser(stub, max_concurrency=1, max_retries=1).parse_to_markdown(path, "a.pdf")
        assert report.failed_pages == [2]
        assert "[Page 2: not transcribed — HTTP 500 after 2 attempt(s)]" in md

    def test_non_retryable_error_fails_page_without_retry(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a", "b"])
        stub = _StubVision({2: ValueError("bad request shape")})
        md, report = _parser(stub, max_concurrency=1).parse_to_markdown(path, "a.pdf")
        assert report.failed_pages == [2] and sum(1 for no, _ in stub.calls if no == 2) == 1

    def test_fatal_auth_error_fails_the_file(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a"])
        with pytest.raises(CloudParseError, match="HTTP 401"):
            _parser(_StubVision({1: _Status(401)})).parse_to_markdown(path, "a.pdf")

    def test_too_many_failures_fail_the_file(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a", "b", "c"])
        stub = _StubVision({1: _Status(503), 2: _Status(503)})
        with pytest.raises(CloudParseError, match="only 1/3 pages"):
            _parser(stub, max_retries=0).parse_to_markdown(path, "a.pdf")

    def test_strict_mode_fails_on_first_bad_page(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a", "b"])
        with pytest.raises(CloudParseError, match="page 1"):
            _parser(_StubVision({1: _Status(503)}), max_retries=0,
                    on_page_failure="fail").parse_to_markdown(path, "a.pdf")

    def test_page_cap_refuses_before_any_call(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a", "b", "c"])
        stub = _StubVision({})
        with pytest.raises(CloudParseError, match="max_pages_per_file is 2"):
            _parser(stub, max_pages_per_file=2).parse_to_markdown(path, "a.pdf")
        assert stub.calls == []

    def test_daily_budget(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a", "b"])
        budget = DailyBudget()
        p = _parser(_StubVision({}), budget=budget, max_pages_per_day=3)
        p.parse_to_markdown(path, "a.pdf")
        assert budget.used("vision_pages") == 2
        with pytest.raises(BudgetExceeded, match="2/3 used"):
            p.parse_to_markdown(path, "a.pdf")

    def test_truncated_output_is_kept(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a"])
        stub = _StubVision({1: "Partial page"}, finish="length")
        assert "Partial page" in _parser(stub).parse_to_markdown(path, "a.pdf")[0]

    def test_render_failure_marks_page(self, tmp_path, monkeypatch):
        path = _pdf(tmp_path / "a.pdf", ["a", "b"])

        def boom(self, no):
            if no == 2:
                raise RuntimeError("broken page")
            return b"png"
        monkeypatch.setattr(cp._PageSource, "render", boom)
        md, report = _parser(_StubVision({})).parse_to_markdown(path, "a.pdf")
        assert report.failed_pages == [2] and "render failed (RuntimeError)" in md

    def test_client_close_errors_are_ignored(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a"])
        stub = _StubVision({})

        def bad_close():
            raise RuntimeError("already closed")
        stub.close = bad_close
        assert "Text of page 1." in _parser(stub).parse_to_markdown(path, "a.pdf")[0]

    def test_bad_progress_callback_is_ignored(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf", ["a"])

        def boom(*a):
            raise RuntimeError("ui")
        assert _parser(_StubVision({})).parse_to_markdown(path, "a.pdf", progress_cb=boom)[0]

    def test_unreadable_and_encrypted_files(self, tmp_path):
        bad = tmp_path / "x.pdf"
        bad.write_bytes(b"not a pdf")
        with pytest.raises(CloudParseError, match="cannot open"):
            _parser(_StubVision({})).parse_to_markdown(str(bad), "x.pdf")
        import fitz
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "secret")
        enc = tmp_path / "e.pdf"
        doc.save(str(enc), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
        with pytest.raises(CloudParseError, match="password-protected"):
            _parser(_StubVision({})).parse_to_markdown(str(enc), "e.pdf")

    def test_empty_pdf(self, tmp_path, monkeypatch):
        path = _pdf(tmp_path / "a.pdf", ["a"])
        monkeypatch.setattr(cp._PageSource, "__init__",
                            lambda self, p, k, d: setattr(self, "count", 0))
        with pytest.raises(CloudParseError, match="no pages"):
            _parser(_StubVision({})).parse_to_markdown(path, "a.pdf")


class TestImages:
    def test_png_and_large_image_downscaled(self, tmp_path):
        from PIL import Image
        path = tmp_path / "scan"  # extensionless, like FileStorage
        Image.new("RGBA", (3000, 1000), "white").save(path, format="PNG")
        stub = _StubVision({1: "Receipt total 42"})
        md, report = _parser(stub).parse_to_markdown(str(path), "receipt.png")
        assert report.pages == 1 and "Receipt total 42" in md
        import base64
        import io
        uri = stub.calls[0][1]["messages"][1]["content"][1]["image_url"]["url"]
        sent = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        assert max(sent.size) == cp.MAX_IMAGE_SIDE and sent.mode == "RGB"

    def test_multi_frame_tiff_is_one_page_per_frame(self, tmp_path):
        from PIL import Image
        path = tmp_path / "fax.tif"
        frames = [Image.new("L", (50, 50), c) for c in (0, 128)]
        frames[0].save(path, save_all=True, append_images=frames[1:])
        md, report = _parser(_StubVision({})).parse_to_markdown(str(path), "fax.tiff")
        assert report.pages == 2 and "Text of page 2." in md


def test_real_openai_client_against_fake_server(tmp_path):
    path = _pdf(tmp_path / "a.pdf", ["hello"])
    with FakeModelServer() as srv:
        srv.chat_reply = lambda body: "# Title\n\nBody from the vision model."
        cfg = ParsingConfig(vision=VisionModelConfig(model="gpt-4o-mini", base_url=srv.url + "/v1",
                                                     api_key="k", max_retries=1))
        full, records = CloudDocumentParser(cfg, sleep=lambda s: None, budget=DailyBudget()).parse(
            path, "a.pdf", max_tokens=100)
        body = srv.posts("/chat/completions")[0]
    assert "Body from the vision model." in full and records[0]["page_no"] == 1
    assert body["model"] == "gpt-4o-mini" and body["temperature"] == 0
    assert body["messages"][0]["role"] == "system" and "original language" in body["messages"][0]["content"]
    assert body["messages"][1]["content"][1]["type"] == "image_url"


def test_requires_vision_model():
    with pytest.raises(CloudParseError):
        CloudDocumentParser(ParsingConfig())


def test_custom_prompt_and_from_config():
    rag = RAGConfig.model_validate({
        "profile": "cloud",
        "embedding": {"provider": "openai", "model": "e", "dimension": 8},
        "chunking": {}, "retrieval": {"default_top_k": 1, "default_similarity_cutoff": 0,
                                      "hybrid_search": False},
        "vector_store": {"type": "pgvector", "table_prefix": "v"},
        "parsing": {"prompt": "Custom.", "vision": {"model": "m", "base_url": "http://x/v1"}},
    })
    p = CloudDocumentParser.from_config(rag)
    assert p.prompt == "Custom."
    assert is_cloud_parsed(rag, "a.PDF") and is_cloud_parsed(rag, "b.jpeg")
    assert not is_cloud_parsed(rag, "c.docx")
    rag.parsing.images = "local"
    assert not is_cloud_parsed(rag, "b.jpeg")


def test_global_slots_resize():
    a = cp._global_slots(3)
    assert cp._global_slots(3) is a and cp._global_slots(5) is not a


class TestDailyBudget:
    def test_unlimited_and_rollover(self):
        today = {"d": date(2026, 1, 1)}
        b = DailyBudget(today=lambda: today["d"])
        assert b.try_consume("k", 10, 0) and b.used("k") == 10
        assert b.try_consume("k", 5, 15) and not b.try_consume("k", 1, 15)
        assert b.snapshot() == {"k": 15}
        today["d"] = date(2026, 1, 2)
        assert b.used("k") == 0 and b.try_consume("k", 15, 15)


class TestUnauthenticatedOcrEndpoint:
    """/v1/ocr/general is unauthenticated: under cloud parsing it must not spend the
    vision budget on anonymous uploads (and must not load local OCR models)."""

    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.api.router.media import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_refused_when_cloud_parsing(self, monkeypatch):
        rag = SimpleNamespace(profile="cloud", parsing=None)
        monkeypatch.setattr("src.config.config_manager.Config.get_config_model",
                            lambda: SimpleNamespace(rag=rag))
        r = self._client().post("/v1/ocr/general", files={"file": ("scan.png", b"x")})
        assert r.status_code == 503 and "cloud vision model" in r.json()["detail"]

    def test_selection_helper_is_fail_safe(self, monkeypatch):
        from src.api.router.media import _cloud_parsing_selected

        def boom():
            raise RuntimeError("no config")
        monkeypatch.setattr("src.config.config_manager.Config.get_config_model", boom)
        assert _cloud_parsing_selected("a.pdf") is False
