"""PDF page counts and rendering through PDFium (src/domain/rag/pdf_pages.py), and the two
callers: the progress-bar total (docling_loader._pdf_page_count) and the cloud parser's
page source (cloud_parser._PageSource)."""

import io
import threading

import pytest
from PIL import Image

from src.domain.rag import pdf_pages
from src.domain.rag.cloud_parser import CloudParseError, _PageSource
from src.domain.rag.docling_loader import _pdf_page_count
from tests.pdf_fixtures import encrypt_pdf, make_pdf


@pytest.fixture
def three_pages(tmp_path):
    return make_pdf(tmp_path / "three.pdf", ["one", "two", "three"])


@pytest.fixture
def encrypted(tmp_path):
    return encrypt_pdf(make_pdf(tmp_path / "plain.pdf", ["secret"]), tmp_path / "enc.pdf")


def _png_image(data: bytes) -> Image.Image:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return Image.open(io.BytesIO(data))


# ---------------------------------------------------------------- pdf_pages

def test_pdf_pages_counts_multi_page_pdf(three_pages):
    assert pdf_pages.page_count(three_pages) == 3


def test_pdf_pages_encrypted_pdf_raises_password_error(encrypted):
    with pytest.raises(pdf_pages.PdfPasswordError):
        pdf_pages.page_count(encrypted)


def test_pdf_pages_owner_password_only_opens(tmp_path):
    # An empty user password (permissions only) opens without asking, as with any viewer.
    path = encrypt_pdf(make_pdf(tmp_path / "p.pdf", ["a", "b"]), tmp_path / "o.pdf", user_pw="")
    assert pdf_pages.page_count(path) == 2


def test_pdf_pages_corrupt_pdf_raises_pdfium_error_not_password(tmp_path):
    import pypdfium2 as pdfium
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\n garbage")
    with pytest.raises(pdfium.PdfiumError) as e:
        pdf_pages.page_count(str(bad))
    assert not isinstance(e.value, pdf_pages.PdfPasswordError)


def test_pdf_pages_or_none_for_non_pdf_corrupt_missing_encrypted(tmp_path, encrypted):
    txt = tmp_path / "a.txt"
    txt.write_text("# not a pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\n garbage")
    assert pdf_pages.page_count_or_none(str(txt)) is None
    assert pdf_pages.page_count_or_none(str(bad)) is None
    assert pdf_pages.page_count_or_none(str(tmp_path / "missing.pdf")) is None
    assert pdf_pages.page_count_or_none(encrypted) is None


def test_pdf_pages_render_png_size_follows_dpi(three_pages):
    # A4 = 595 x 842 pt; scale = dpi / 72, partial pixels rounded up (the sizes PyMuPDF gave)
    at72 = _png_image(pdf_pages.render_png(three_pages, 1, 72))
    at150 = _png_image(pdf_pages.render_png(three_pages, 1, 150))
    assert at72.size == (595, 842)
    assert at150.size == (1240, 1755)
    assert at150.mode == "RGB"


def test_pdf_pages_render_exact_sizes_stay_exact(tmp_path):
    # US Letter at 150 dpi is exactly 1275 x 1650; a plain float scale makes pdfium's ceil()
    # add a row (1651). PyMuPDF gave 1650.
    path = make_pdf(tmp_path / "letter.pdf", ["x"], size=(612, 792))
    assert _png_image(pdf_pages.render_png(path, 1, 150)).size == (1275, 1650)
    assert _png_image(pdf_pages.render_png(path, 1, 300)).size == (2550, 3300)


def test_pdf_pages_render_picks_the_requested_page(tmp_path):
    # page 2 carries much more ink than page 1, so the renders must differ
    path = make_pdf(tmp_path / "p.pdf", [".", "W" * 60])
    one = _png_image(pdf_pages.render_png(path, 1, 72)).convert("L")
    two = _png_image(pdf_pages.render_png(path, 2, 72)).convert("L")
    dark = lambda im: sum(1 for v in im.tobytes() if v < 128)  # noqa: E731
    assert dark(two) > dark(one) > 0


def test_pdf_pages_render_background_is_white(three_pages):
    im = _png_image(pdf_pages.render_png(three_pages, 1, 72))
    assert im.getpixel((5, 5)) == (255, 255, 255)


def test_pdf_pages_render_out_of_range_raises(three_pages):
    with pytest.raises(Exception):
        pdf_pages.render_png(three_pages, 4, 72)
    with pytest.raises(IndexError):
        pdf_pages.render_png(three_pages, 0, 72)


def test_pdf_pages_render_encrypted_raises_password_error(encrypted):
    with pytest.raises(pdf_pages.PdfPasswordError):
        pdf_pages.render_png(encrypted, 1, 72)


def test_pdf_pages_holds_the_shared_docling_lock(three_pages, monkeypatch):
    # PDFium is not thread-safe: our calls must take the same lock docling's backend takes.
    from docling.utils.locks import pypdfium2_lock
    assert pdf_pages._PDFIUM_LOCK is pypdfium2_lock
    seen = []
    real_open = pdf_pages._open

    def spy(path):
        seen.append(pypdfium2_lock.locked())
        return real_open(path)
    monkeypatch.setattr(pdf_pages, "_open", spy)
    pdf_pages.page_count(three_pages)
    pdf_pages.render_png(three_pages, 1, 72)
    assert seen == [True, True]
    assert not pypdfium2_lock.locked()


def test_pdf_pages_lock_released_after_errors(encrypted, tmp_path):
    with pytest.raises(pdf_pages.PdfPasswordError):
        pdf_pages.render_png(encrypted, 1, 72)
    assert pdf_pages.page_count_or_none(str(tmp_path / "missing.pdf")) is None
    assert not pdf_pages._PDFIUM_LOCK.locked()


def test_pdf_pages_parallel_renders(three_pages):
    out, errors = [], []

    def work(n):
        try:
            out.append(len(pdf_pages.render_png(three_pages, n % 3 + 1, 100)))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(out) == 12


# ---------------------------------------------------------------- docling_loader._pdf_page_count

def test_progress_page_count_pdf(three_pages):
    assert _pdf_page_count(three_pages) == 3


def test_progress_page_count_images_count_frames(tmp_path):
    png = tmp_path / "scan"  # extensionless, like FileStorage
    Image.new("RGB", (20, 20), "white").save(png, format="PNG")
    tif = tmp_path / "multi"
    Image.new("RGB", (20, 20)).save(tif, format="TIFF", save_all=True,
                                    append_images=[Image.new("RGB", (20, 20))] * 2)
    assert _pdf_page_count(str(png)) == 1
    assert _pdf_page_count(str(tif)) == 3


def test_progress_page_count_none_for_text_corrupt_encrypted(tmp_path, encrypted):
    txt = tmp_path / "doc"
    txt.write_text("# heading\n\nbody")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\n garbage")
    assert _pdf_page_count(str(txt)) is None
    assert _pdf_page_count(str(bad)) is None
    assert _pdf_page_count(encrypted) is None
    assert _pdf_page_count(str(tmp_path / "missing")) is None


# ---------------------------------------------------------------- cloud_parser._PageSource

def test_page_source_pdf_count_and_render_at_dpi(three_pages):
    src = _PageSource(three_pages, "pdf", 150)
    assert src.count == 3
    im = _png_image(src.render(3))
    assert im.size == (1240, 1755)


def test_page_source_encrypted_pdf_message(encrypted):
    with pytest.raises(CloudParseError, match="^PDF is password-protected — decrypt it before uploading$"):
        _PageSource(encrypted, "pdf", 150)


def test_page_source_corrupt_pdf_is_not_reported_as_password(tmp_path):
    import pypdfium2 as pdfium
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(pdfium.PdfiumError):
        _PageSource(str(bad), "pdf", 150)
