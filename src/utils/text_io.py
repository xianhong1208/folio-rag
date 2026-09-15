
"""Encoding-tolerant reading of uploaded text files (.txt/.json/.csv).

Uploaded files are not always UTF-8: Windows Notepad "Unicode" is UTF-16 LE,
and legacy Taiwanese txt/csv are often Big5/CP950. Strategy, strict to
lenient, never silently emitting mojibake: BOM sniffing, strict UTF-8,
charset_normalizer detection, then raise with actionable guidance.
"""

from __future__ import annotations

import codecs
from pathlib import Path

from src.log import get_api_logger

logger = get_api_logger()

# Order-sensitive: a UTF-32 LE BOM (FF FE 00 00) starts with the UTF-16 LE
# BOM (FF FE), so match the longer one first, otherwise a UTF-32 file would
# be misdetected as UTF-16
_BOM_TABLE = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def read_text_robust(path: str | Path, file_name: str = "") -> str:
    """Read a text file, auto-handling common encodings (UTF-8/16/32, Big5).

    Raises ValueError with remediation guidance (stored in
    FileIndex.error_message) when the encoding cannot be identified.
    """
    label = file_name or Path(path).name
    data = Path(path).read_bytes()

    # 1. BOM sniffing -- a BOM is an explicit declaration, trust it directly
    for bom, enc in _BOM_TABLE:
        if data.startswith(bom):
            text = data.decode(enc)
            if enc != "utf-8-sig":
                logger.info(f"[TEXT_IO] '{label}' decoded as {enc} (BOM detected)")
            return text

    # 2. Strict UTF-8 (the vast majority of files return here, at no extra cost)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # 3. charset_normalizer detection (Big5/CP950, BOM-less UTF-16, etc.)
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[TEXT_IO] charset detection failed for '{label}': {e}")
        best = None

    if best is not None:
        logger.info(
            f"[TEXT_IO] '{label}' is not UTF-8; decoded as detected "
            f"encoding '{best.encoding}'"
        )
        return str(best)

    # 4. Fail loudly -- do not feed garbled text into the vector store
    raise ValueError(
        f"Could not detect the text encoding of '{label}' (not UTF-8/UTF-16/Big5 or another common encoding). "
        f"Please save the file as UTF-8 and upload it again."
    )
