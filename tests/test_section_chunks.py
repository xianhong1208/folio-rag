"""Real docling parses: section-aligned chunks and xlsx sheet names in the heading path."""

import openpyxl
import pytest
from docling.chunking import HierarchicalChunker
from docling.document_converter import DocumentConverter

from src.domain.rag.chunk_tokenizer import get_chunk_tokenizer
from src.domain.rag.docling_loader import _refine_chunks_for_token_budget


@pytest.fixture(scope="module")
def tok():
    return get_chunk_tokenizer(None)


def _records(path, tok, max_tokens=256):
    doc = DocumentConverter().convert(str(path)).document
    raw = list(HierarchicalChunker().chunk(dl_doc=doc))
    return _refine_chunks_for_token_budget(raw, tok, max_tokens, doc=doc)


def test_small_sections_stay_separate_with_their_own_headings(tmp_path, tok):
    md = tmp_path / "manual.md"
    md.write_text(
        "# Aurora Manual\n\n## 3. Voice\n\nVoice calls use the SIP trunk.\n\n"
        "## 4. Warranty\n\nThe warranty lasts 36 months.\n", encoding="utf-8")
    recs = _records(md, tok)
    warranty = [r for r in recs if "36 months" in r["text"]]
    assert len(warranty) == 1
    assert "SIP trunk" not in warranty[0]["text"]
    assert warranty[0]["headings"] == ["Aurora Manual", "4. Warranty"]


def test_xlsx_sheet_name_is_part_of_the_heading_path(tmp_path, tok):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Aurora plan"
    ws.append(["Tier", "Fee"])
    ws.append(["Pro", "USD 35"])
    ws2 = wb.create_sheet("Borealis plan")
    ws2.append(["Tier", "Fee"])
    ws2.append(["Pro", "USD 29"])
    path = tmp_path / "plans.xlsx"
    wb.save(path)
    recs = _records(path, tok)
    by_fee = {("35" in r["text"]): r for r in recs}
    assert by_fee[True]["headings"] == ["Aurora plan"]
    assert by_fee[False]["headings"] == ["Borealis plan"]
    assert all(r["content_type"] == "table" for r in recs)
    assert len(recs) == 2  # the two sheets are different sections: not merged
