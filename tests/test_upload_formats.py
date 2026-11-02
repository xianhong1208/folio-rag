
"""Regression tests for the upload-format policy — consistency between the whitelist and parser routing.

Format support v2 (docling 2.119): legacy Office (.doc/.xls/.ppt, via
LibreOffice) + Outlook .msg + the plain-text family (yaml/xml/conf/log). Core
invariant: **every extension in the whitelist must have an explicit parse path**
(docling or direct plain-text read); none may fall into the SimpleDirectoryReader
fallback that force-reads binary as text (the root cause of the .doc garbled-text
incident, see a64641d).
"""

from db.filedb import FileDB
from src.domain.rag.docling_loader import DOCLING_EXTENSIONS

# Kept in sync with text_extensions in hierarchical_indexer.load_document_from_file
_TEXT_EXTENSIONS = {".txt", ".text", ".json", ".csv", ".yaml", ".yml", ".xml", ".conf", ".log"}


def test_legacy_office_and_msg_allowed():
    """Format support v2: doc/xls/ppt/msg are in the whitelist."""
    for ext in ("doc", "xls", "ppt", "msg"):
        assert ext in FileDB.ALLOWED_EXTENSIONS, f"{ext} should be on the allow-list"


def test_text_family_allowed():
    """The plain-text family (yaml/yml/xml/conf/log) is in the whitelist."""
    for ext in ("yaml", "yml", "xml", "conf", "log"):
        assert ext in FileDB.ALLOWED_EXTENSIONS, f"{ext} should be on the allow-list"


def test_every_allowed_extension_has_a_parser_route():
    """Invariant: whitelist ∖ (docling ∪ plain-text) = empty set.

    Any extension that is only in the whitelist with no parse path falls into the
    SimpleDirectoryReader fallback — for a binary format that means silently
    indexing garbled text.
    """
    routed = {e.lstrip(".") for e in DOCLING_EXTENSIONS} | {
        e.lstrip(".") for e in _TEXT_EXTENSIONS
    }
    orphans = FileDB.ALLOWED_EXTENSIONS - routed
    assert not orphans, f"these extensions have no parse path and would index garbage: {orphans}"


def test_unsupported_formats_stay_rejected():
    """Still-rejected formats are not accidentally let through.

    Video (webm/mkv/mp4) is pending a docling VIDEO pipeline; executables and
    archives are always rejected.
    """
    for ext in ("webm", "mkv", "mp4", "exe", "zip", "pages"):
        assert ext not in FileDB.ALLOWED_EXTENSIONS, f"{ext} should not be on the allow-list"


def test_validate_rejects_disallowed_and_accepts_new(tmp_path):
    """_validate_file_content: new formats pass, unsupported formats are rejected."""
    import pytest
    ok = FileDB._validate_file_content(b"dummy", "letter.doc")
    assert ok["extension"] == "doc"
    ok = FileDB._validate_file_content(b"k: v", "settings.yaml")
    assert ok["extension"] == "yaml"
    with pytest.raises(ValueError):
        FileDB._validate_file_content(b"dummy", "movie.mp4")


def test_format_v3_official_parity():
    """Format v3 (parity with docling 2.124's site):
    audio aligned to the official 6; adds OpenDocument / epub / latex / eml / xlsm.
    (Each new format's parse routing is guarded automatically by the whitelist-has-a-parse-path invariant above.)
    """
    for ext in ("m4a", "aac", "ogg", "flac"):          # the official 6 audio types (wav/mp3 already present)
        assert ext in FileDB.ALLOWED_EXTENSIONS, f"audio {ext} should be on the allow-list"
    for ext in ("odt", "ods", "odp", "epub", "tex", "eml", "xlsm"):
        assert ext in FileDB.ALLOWED_EXTENSIONS, f"{ext} should be on the allow-list"


def test_official_docling_format_parity():
    """Programmatic parity with the official site: every
    extension of every non-excluded format in docling's FormatToExtensions must
    have a parse route (docling ∪ direct plain-text read). If docling adds a
    format later → this test turns red automatically, so parity never goes stale.
    An exclusion must carry a reason in _EXCLUDED_FORMATS — no silent absences.
    """
    from docling.datamodel.base_models import InputFormat, FormatToExtensions

    _EXCLUDED_FORMATS = {
        "video": "video: needs a VIDEO pipeline (keyframes + ffmpeg wiring), not just an extension",
        "xml_uspto": "ambiguous .xml family: generic .xml is already read as plain text; specialised XML variants are not auto-routed",
        "xml_jats": "same as xml_uspto",
        "xml_xbrl": "same as xml_uspto",
        "xml_doclang": "docling-internal format",
        "dclx": "docling-internal format",
        "json_docling": ".json is already read as plain text",
        "mets_gbs": "a tar.gz archive; archives are not accepted",
        "boxnote": "proprietary Box collaboration format; no use case",
        "ebcdic": "mainframe encoding; no use case",
        "iwork_pages": "needs the docling[format-iwork] extra, which is not installed",
    }
    # ext-level exclusions: individual extensions the official mapping lists but that **fail in practice** (the format itself is accepted)
    _EXCLUDED_EXTS = {
        "ott": "tested 2026-09-02: docling's odfdo backend cannot open a text template (a real LibreOffice file fails too)",
        "ots": "same as ott (spreadsheet template)",
        "otp": "same as ott (presentation template)",
        "potx": "tested 2026-09-02: docling's MsPowerpointDocument cannot open a potx template (pptx/ppsx/pptm/potm/ppsm all work)",
    }
    routed = {e.lstrip(".") for e in DOCLING_EXTENSIONS} | {
        e.lstrip(".") for e in _TEXT_EXTENSIONS
    }
    missing = {}
    for fmt in InputFormat:
        if fmt.value in _EXCLUDED_FORMATS:
            continue
        gap = {e.lower() for e in FormatToExtensions.get(fmt, [])} - routed - set(_EXCLUDED_EXTS)
        if gap:
            missing[fmt.value] = sorted(gap)
    assert not missing, (
        f"official docling formats not covered (add them to the allow-list, or to _EXCLUDED_FORMATS/_EXCLUDED_EXTS with a reason): {missing}"
    )
