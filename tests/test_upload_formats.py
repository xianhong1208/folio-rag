
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
    """Format support v2: doc/xls/ppt/msg are in the whitelist (TC-upload-fmt-01)."""
    for ext in ("doc", "xls", "ppt", "msg"):
        assert ext in FileDB.ALLOWED_EXTENSIONS, f"{ext} 應在白名單"


def test_text_family_allowed():
    """The plain-text family (yaml/yml/xml/conf/log) is in the whitelist (TC-upload-fmt-02)."""
    for ext in ("yaml", "yml", "xml", "conf", "log"):
        assert ext in FileDB.ALLOWED_EXTENSIONS, f"{ext} 應在白名單"


def test_every_allowed_extension_has_a_parser_route():
    """Invariant: whitelist ∖ (docling ∪ plain-text) = empty set (TC-upload-fmt-03).

    Any extension that is only in the whitelist with no parse path falls into the
    SimpleDirectoryReader fallback — for a binary format that means silently
    indexing garbled text.
    """
    routed = {e.lstrip(".") for e in DOCLING_EXTENSIONS} | {
        e.lstrip(".") for e in _TEXT_EXTENSIONS
    }
    orphans = FileDB.ALLOWED_EXTENSIONS - routed
    assert not orphans, f"這些副檔名沒有解析路徑,會索引亂碼: {orphans}"


def test_unsupported_formats_stay_rejected():
    """Still-rejected formats are not accidentally let through (TC-upload-fmt-04).

    Video (webm/mkv/mp4) is pending a docling VIDEO pipeline; executables and
    archives are always rejected.
    """
    for ext in ("webm", "mkv", "mp4", "exe", "zip", "pages"):
        assert ext not in FileDB.ALLOWED_EXTENSIONS, f"{ext} 不應在白名單"


def test_validate_rejects_disallowed_and_accepts_new(tmp_path):
    """_validate_file_content: new formats pass, unsupported formats are rejected (TC-upload-fmt-05)."""
    import pytest
    ok = FileDB._validate_file_content(b"dummy", "letter.doc")
    assert ok["extension"] == "doc"
    ok = FileDB._validate_file_content(b"k: v", "settings.yaml")
    assert ok["extension"] == "yaml"
    with pytest.raises(ValueError):
        FileDB._validate_file_content(b"dummy", "movie.mp4")


def test_format_v3_official_parity():
    """Format v3 (parity with docling 2.124's site, TC-upload-fmt-06):
    audio aligned to the official 6; adds OpenDocument / epub / latex / eml / xlsm.
    (Each new format's parse routing is guarded automatically by the ref invariant.)
    """
    for ext in ("m4a", "aac", "ogg", "flac"):          # the official 6 audio types (wav/mp3 already present)
        assert ext in FileDB.ALLOWED_EXTENSIONS, f"audio {ext} 應在白名單"
    for ext in ("odt", "ods", "odp", "epub", "tex", "eml", "xlsm"):
        assert ext in FileDB.ALLOWED_EXTENSIONS, f"{ext} 應在白名單"


def test_official_docling_format_parity():
    """Programmatic parity with the official site (TC-upload-fmt-07): every
    extension of every non-excluded format in docling's FormatToExtensions must
    have a parse route (docling ∪ direct plain-text read). If docling adds a
    format later → this test turns red automatically, so parity never goes stale.
    An exclusion must carry a reason in _EXCLUDED_FORMATS — no silent absences.
    """
    from docling.datamodel.base_models import InputFormat, FormatToExtensions

    _EXCLUDED_FORMATS = {
        "video": "video:需掛 VIDEO pipeline(keyframe + ffmpeg 接線),非僅加副檔名",
        "xml_uspto": ".xml 歧義家族:泛用 .xml 已走純文字直讀,專業 XML 變體不自動路由",
        "xml_jats": "同 xml_uspto",
        "xml_xbrl": "同 xml_uspto",
        "xml_doclang": "docling 內部格式",
        "dclx": "docling 內部格式",
        "json_docling": ".json 已走純文字直讀",
        "mets_gbs": "tar.gz 壓縮包,不收壓縮檔",
        "boxnote": "Box 專有協作格式,無使用場景",
        "ebcdic": "大型主機編碼,無使用場景",
        "iwork_pages": "需 docling[format-iwork] extra 套件,未安裝",
    }
    # ext-level exclusions: individual extensions the official mapping lists but that **fail in practice** (the format itself is accepted)
    _EXCLUDED_EXTS = {
        "ott": "2026-09-02 實測:docling odfdo backend 打不開 text-template(LibreOffice 真檔亦 fail)",
        "ots": "同 ott(spreadsheet-template)",
        "otp": "同 ott(presentation-template)",
        "potx": "2026-09-02 實測:docling MsPowerpointDocument 打不開 potx 範本(pptx/ppsx/pptm/potm/ppsm 皆 OK)",
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
        f"docling 官方格式未對標(加進白名單,或進 _EXCLUDED_FORMATS/_EXCLUDED_EXTS 附理由):{missing}"
    )
