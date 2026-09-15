
"""Unit tests for src/utils/text_io.py — encoding-tolerant reading of uploaded text files.

A UTF-8-only reader raised UnicodeDecodeError when a user uploaded a .txt saved
as "Unicode" (= UTF-16 LE + BOM) by Windows Notepad. read_text_robust handles
the common encodings instead.
"""

import pytest

from src.utils.text_io import read_text_robust

SAMPLE = "新興科技教育遠距示範服務計畫,Budget: 1,000 元。\n第二行內容。"


def _write(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_plain_utf8(tmp_path):
    """A plain UTF-8 file is read back unchanged (TC-text-io-01)."""
    p = _write(tmp_path, "a.txt", SAMPLE.encode("utf-8"))
    assert read_text_robust(p) == SAMPLE


def test_utf8_with_bom_stripped(tmp_path):
    """A UTF-8 BOM (common in Excel exports) must be stripped, leaving no \\ufeff (TC-text-io-02)."""
    p = _write(tmp_path, "a.csv", b"\xef\xbb\xbf" + SAMPLE.encode("utf-8"))
    out = read_text_robust(p)
    assert out == SAMPLE
    assert not out.startswith("﻿")


def test_utf16_le_bom(tmp_path):
    """UTF-16 LE + BOM (Windows Notepad "Unicode") — reproduces the reported incident (TC-text-io-03)."""
    p = _write(tmp_path, "notepad.txt", SAMPLE.encode("utf-16"))  # includes LE BOM
    assert read_text_robust(p) == SAMPLE


def test_utf16_be_bom(tmp_path):
    """UTF-16 BE + BOM is equally readable (TC-text-io-04)."""
    p = _write(tmp_path, "be.txt", codecs_encode_utf16_be(SAMPLE))
    assert read_text_robust(p) == SAMPLE


def codecs_encode_utf16_be(s: str) -> bytes:
    import codecs
    return codecs.BOM_UTF16_BE + s.encode("utf-16-be")


def test_utf32_le_bom_not_mistaken_for_utf16(tmp_path):
    """The first two bytes of a UTF-32 LE BOM equal a UTF-16 LE BOM — an order-sensitive regression test (TC-text-io-05)."""
    p = _write(tmp_path, "u32.txt", SAMPLE.encode("utf-32"))  # includes LE BOM
    assert read_text_robust(p) == SAMPLE


def test_big5_detected(tmp_path):
    """Big5/CP950 (legacy Taiwan txt/csv) is read via charset detection (TC-text-io-06)."""
    p = _write(tmp_path, "big5.csv", text.encode("cp950"))
    assert read_text_robust(p) == text


def test_undecodable_binary_raises_actionable_error(tmp_path):
    """Unrecognizable random binary → ValueError with actionable guidance, never garbled text (TC-text-io-07)."""
    p = _write(tmp_path, "junk.txt", bytes(range(256)) * 8)
    with pytest.raises(ValueError, match="另存為 UTF-8"):
        read_text_robust(p)


def test_empty_file_returns_empty_string(tmp_path):
    """An empty file returns an empty string without raising (empty-content checks are handled upstream) (TC-text-io-08)."""
    p = _write(tmp_path, "empty.txt", b"")
    assert read_text_robust(p) == ""
