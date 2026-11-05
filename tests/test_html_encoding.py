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


# ---- MIME web archives (.mht / .mhtml) ------------------------------------------------

def _mhtml(html: bytes, *, charset="utf-8", cte="quoted-printable") -> bytes:
    import base64
    import quopri
    body = quopri.encodestring(html) if cte == "quoted-printable" else base64.encodebytes(html)
    ctype = "text/html" + (f"; charset={charset}" if charset else "")
    return (b'MIME-Version: 1.0\r\nContent-Type: multipart/related; boundary="B"; type="text/html"\r\n\r\n'
            b"--B\r\nContent-Type: " + ctype.encode() + b"\r\nContent-Transfer-Encoding: " + cte.encode()
            + b"\r\nContent-Location: http://example.invalid/p.html\r\n\r\n" + body
            + b"\r\n--B\r\nContent-Type: image/png\r\nContent-Transfer-Encoding: base64\r\n"
            b"Content-ID: <i1>\r\n\r\niVBORw0KGgo=\r\n--B--\r\n")


def test_short_cjk_mhtml_without_meta_charset_parses_as_utf8(tmp_path, monkeypatch):
    # docling's MHTML backend ignored the part's charset and BeautifulSoup guessed cp874.
    page = tmp_path / "uuid-no-ext"
    page.write_bytes(_mhtml(f"<html><body><p>{_SENT}</p></body></html>".encode()))
    monkeypatch.setattr(dl, "get_converter", lambda: DocumentConverter())
    full, _ = dl.docling_convert_once(str(page), "saved page.mhtml", max_tokens=64)
    assert _SENT in full


def test_mhtml_to_html_uses_the_part_charset():
    big5 = f"<p>{_SENT}</p>".encode("big5", errors="ignore")
    out = dl._mhtml_to_html(_mhtml(big5, charset="big5", cte="base64"))
    assert out.startswith(b"\xef\xbb\xbf")
    assert out[3:].decode("utf-8") == big5.decode("big5")       # re-encoded as UTF-8
    undeclared = b'<meta charset="big5"><p>x</p>'
    assert dl._mhtml_to_html(_mhtml(undeclared, charset=None)) == undeclared  # page's own declaration
    no_html = b'MIME-Version: 1.0\r\nContent-Type: text/plain\r\n\r\nhello\r\n'
    assert dl._mhtml_to_html(no_html) is None                   # left to docling as-is
