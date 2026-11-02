"""UTF-8 HTML without a charset declaration must not be decoded as a legacy codepage."""

from docling.document_converter import DocumentConverter

from src.domain.rag import docling_loader as dl

_SENT = "哨兵內容SENTINEL"


def test_short_cjk_html_without_meta_charset_parses_as_utf8(tmp_path, monkeypatch):
    # BeautifulSoup guesses the encoding of undeclared bytes; for short CJK text it
    # picked cp874 (Thai) and the text came out garbled.
    page = tmp_path / "t.html"
    page.write_text(f"<html><body><p>{_SENT}</p></body></html>", encoding="utf-8")
    monkeypatch.setattr(dl, "get_converter", lambda: DocumentConverter())
    full, chunks = dl.docling_convert_once(str(page), "t.html", max_tokens=64)
    assert _SENT in full


def test_extensionless_storage_path_html(tmp_path, monkeypatch):
    stored = tmp_path / "uuid-no-ext"
    stored.write_text(f"<html><body><p>{_SENT}</p></body></html>", encoding="utf-8")
    monkeypatch.setattr(dl, "get_converter", lambda: DocumentConverter())
    full, _ = dl.docling_convert_once(str(stored), "page.htm", max_tokens=64)
    assert _SENT in full


def test_declare_utf8_only_when_needed():
    body = f"<p>{_SENT}</p>".encode()
    assert dl._declare_html_utf8(body).startswith(b"\xef\xbb\xbf")
    declared = b'<meta charset="big5"><p>x</p>'
    assert dl._declare_html_utf8(declared) == declared            # author's declaration wins
    xml = '<?xml version="1.0" encoding="utf-8"?><p>é</p>'.encode()
    assert dl._declare_html_utf8(xml) == xml
    ascii_only = b"<p>plain</p>"
    assert dl._declare_html_utf8(ascii_only) == ascii_only
    big5 = "保固".encode("big5")
    assert dl._declare_html_utf8(big5) == big5                     # not UTF-8: leave to detection
    bom = b"\xef\xbb\xbf<p>x</p>"
    assert dl._declare_html_utf8(bom) == bom
