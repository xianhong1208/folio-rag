"""Sections, breadcrumbs and retrieval text (section-aligned chunks, original-text display)."""

from types import SimpleNamespace

from llama_index.core import Document

from src.domain.rag import chunk_identity as ci
from src.domain.rag.hierarchy import build_hierarchy


def test_section_key_is_first_two_levels():
    assert ci.section_key(["Manual", "5. Warranty", "5.1 Parts"]) == ("Manual", "5. Warranty")
    assert ci.section_key(None) == ()
    assert ci.section_key([" ", "A"]) == ("A",)


def test_common_headings_prefix():
    assert ci.common_headings([["M", "5", "5.1"], ["M", "5"], ["M", "5", "5.2"]]) == ["M", "5"]
    assert ci.common_headings([["A"], ["B"]]) is None
    assert ci.common_headings([]) is None


def test_merge_content_type_priority():
    assert ci.merge_content_type([None, "picture", "table"]) == "table"
    assert ci.merge_content_type([None, "picture"]) == "picture"
    assert ci.merge_content_type([None, None]) is None


def test_breadcrumb_uses_file_stem_and_dedupes_title():
    assert ci.breadcrumb("Aurora_X2.docx", ["Aurora_X2", "5. Warranty"]) == "Aurora_X2 › 5. Warranty"
    assert ci.breadcrumb("plans.xlsx", ["Borealis plan"]) == "plans › Borealis plan"
    assert ci.breadcrumb(None, None) == ""


def test_retrieval_text_composes_prefix_breadcrumb_text():
    meta = {"context_prefix": "Warranty terms of the Aurora router.", "breadcrumb": "Aurora › 5. Warranty"}
    assert ci.retrieval_text("36-month warranty.", meta) == (
        "Warranty terms of the Aurora router.\nAurora › 5. Warranty\n36-month warranty.")
    assert ci.retrieval_text("t", {"breadcrumb": "B"}) == "B\nt"


def test_retrieval_text_leaves_old_rows_untouched():
    # rows indexed before this layout carry the prefix inside text and no breadcrumb
    assert ci.retrieval_text("old prefix\n\nold text", {"file_id": "x"}) == "old prefix\n\nold text"
    assert ci.retrieval_text("x", None) == "x"


def test_node_retrieval_text_unwraps_node_with_score():
    node = SimpleNamespace(text="body", metadata={"breadcrumb": "F › H"})
    assert ci.node_retrieval_text(SimpleNamespace(node=node, score=0.3)) == "F › H\nbody"
    assert ci.node_retrieval_text(node) == "F › H\nbody"


def test_search_text_sql_matches_python_composition():
    # The Postgres twin must use the same keys and separator as retrieval_text.
    assert "context_prefix" in ci.SEARCH_TEXT_SQL and "breadcrumb" in ci.SEARCH_TEXT_SQL
    assert ci.SEARCH_TEXT_SQL.startswith("concat_ws(E'\\n'")


def test_token_budgets():
    assert ci.token_budgets(256, contextual=True) == (256, 330)
    assert ci.token_budgets(512, contextual=True) == (330, 330)
    assert ci.token_budgets(512, contextual=False) == (450, 450)


def test_parents_never_span_sections_and_carry_shared_headings():
    leaves = [
        Document(text="a", metadata={"headings": ["M", "1. Intro"]}),
        Document(text="b", metadata={"headings": ["M", "1. Intro", "1.1 Ports"]}),
        Document(text="c", metadata={"headings": ["M", "2. Warranty"]}),
    ]
    leaf_nodes, parents = build_hierarchy(leaves, parent_target_tokens=10_000, base_metadata={})
    assert [p.metadata["children_index_range"] for p in parents] == [[0, 1], [2, 2]]
    assert parents[0].metadata["headings"] == ["M", "1. Intro"]
    assert parents[0].text == "a\n\nb"  # original text only, no prefixes
    assert leaf_nodes[2].metadata["parent_node_id"] == parents[1].id_
