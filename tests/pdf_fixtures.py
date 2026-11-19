"""Small PDFs for tests, written by hand (no PDF library needed to create them).

``make_pdf`` writes one A4 page per string, each with that string as real text in
Helvetica, so text-layer parsers (docling) find it. ``encrypt_pdf`` adds an AES-256 user
password with pypdf (a core dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

A4 = (595, 842)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_bytes(pages: Iterable[str], size=A4) -> bytes:
    pages = list(pages)
    w, h = size
    n = len(pages)
    # 1 catalog, 2 pages tree, 3 font, then (page, content) per page
    objs = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = []
    for i, text in enumerate(pages):
        page_id, content_id = 4 + 2 * i, 5 + 2 * i
        kids.append(f"{page_id} 0 R")
        stream = f"BT /F1 12 Tf 72 {h - 72} Td ({_escape(text)}) Tj ET"
        objs[page_id] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
                         f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>")
        objs[content_id] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    objs[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {n} >>"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n{objs[num]}\nendobj\n".encode("latin-1")
    xref = len(out)
    size_ = max(objs) + 1
    out += f"xref\n0 {size_}\n0000000000 65535 f \n".encode()
    for num in range(1, size_):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size_} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def make_pdf(path, pages: Iterable[str], size=A4) -> str:
    Path(path).write_bytes(pdf_bytes(pages, size))
    return str(path)


def encrypt_pdf(src, dst, user_pw: str = "u", owner_pw: str = "o") -> str:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(str(src)))
    writer.encrypt(user_password=user_pw, owner_password=owner_pw, algorithm="AES-256")
    with open(dst, "wb") as f:
        writer.write(f)
    return str(dst)


def lines_pdf_bytes(pages: Iterable[Iterable[str]], size=A4) -> bytes:
    """Like ``pdf_bytes``, with several lines per page (12 pt Helvetica, 14 pt apart)."""
    w, h = size
    pages = [list(p) for p in pages]
    objs = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = []
    for i, lines in enumerate(pages):
        page_id, content_id = 4 + 2 * i, 5 + 2 * i
        kids.append(f"{page_id} 0 R")
        body = " ".join(f"({_escape(t)}) Tj 0 -14 Td" for t in lines)
        stream = f"BT /F1 12 Tf 72 {h - 72} Td {body} ET"
        objs[page_id] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
                         f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>")
        objs[content_id] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    objs[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n{objs[num]}\nendobj\n".encode("latin-1")
    xref = len(out)
    size_ = max(objs) + 1
    out += f"xref\n0 {size_}\n0000000000 65535 f \n".encode()
    for num in range(1, size_):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size_} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def make_image_pdf(path, pages: int = 1) -> str:
    """An image-only PDF (a "scan"): one grey picture per page, no text layer."""
    from PIL import Image

    imgs = [Image.new("RGB", (300, 400), (200, 200, 200)) for _ in range(pages)]
    imgs[0].save(str(path), save_all=True, append_images=imgs[1:], resolution=72)
    return str(path)


def join_pdfs(dst, *sources) -> str:
    """Concatenate PDFs page by page (pypdf), e.g. a text page followed by a scanned one."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for src in sources:
        for page in PdfReader(str(src)).pages:
            writer.add_page(page)
    with open(dst, "wb") as f:
        writer.write(f)
    return str(dst)
