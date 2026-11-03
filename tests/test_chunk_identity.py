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


def test_reindex_recommended_for_files_from_older_pipelines():
    assert ci.reindex_recommended("indexed", None) is True           # before versions were recorded
    assert ci.reindex_recommended("indexed", "1") is True
    assert ci.reindex_recommended("indexed", ci.CHUNK_PIPELINE_VERSION) is False
    assert ci.reindex_recommended("failed", None) is False


def test_status_models_expose_pipeline_fields():
    from src.api.router.response import FileIndexStatusResponse, IndexedFileEntry

    st = FileIndexStatusResponse(file_id="f", file_name="a", status="indexed")
    assert st.reindex_recommended is False and st.pipeline_version is None and st.warning is None
    entry = IndexedFileEntry(file_id="f", file_name="a", num_chunks=1, indexed_at="t", status="indexed",
                             embedding_model=None, chunk_size=None, chunk_overlap=None,
                             reindex_recommended=True)
    assert entry.reindex_recommended is True


# --- embedding window: breadcrumb cap and fit (L1) ----------------------------------

def _chars(text):
    return len(text)


def test_cap_breadcrumb_keeps_short_ones():
    from src.domain.rag.chunk_identity import cap_breadcrumb
    assert cap_breadcrumb("Manual › 5. Warranty", _chars, 48) == "Manual › 5. Warranty"


def test_cap_breadcrumb_drops_middle_headings_first():
    from src.domain.rag.chunk_identity import cap_breadcrumb
    crumb = " › ".join(["Manual", "Part One", "Chapter Two", "Section Three", "Warranty"])
    out = cap_breadcrumb(crumb, _chars, 40)
    assert out == "Manual › … › Section Three › Warranty"
    assert len(out) <= 40


def test_cap_breadcrumb_cuts_a_long_heading():
    from src.domain.rag.chunk_identity import cap_breadcrumb
    out = cap_breadcrumb("Manual › " + "x" * 100, _chars, 20)
    assert len(out) <= 20 and out.startswith("Manual") and out.endswith("…")
    assert cap_breadcrumb("anything", _chars, 0) == ""


def test_cap_breadcrumb_real_tokenizer_default_cap():
    from src.domain.rag.chunk_identity import BREADCRUMB_MAX_TOKENS, cap_breadcrumb
    from src.domain.rag.chunk_tokenizer import get_chunk_tokenizer
    tok = get_chunk_tokenizer(None)
    count = lambda t: len(tok.encode(t))  # noqa: E731
    crumb = " › ".join(["員工差旅管理辦法"] + [f"第{i}章 出差申請與核銷之相關規定" for i in range(6)])
    assert count(crumb) > BREADCRUMB_MAX_TOKENS
    assert count(cap_breadcrumb(crumb, count)) <= BREADCRUMB_MAX_TOKENS


def test_fit_retrieval_text_shortens_crumb_then_prefix_never_text():
    from src.domain.rag.chunk_identity import compose_retrieval_text, fit_retrieval_text
    text = "t" * 60
    prefix, crumb = "p" * 30, "File › " + "h" * 30
    # fits: unchanged
    assert fit_retrieval_text(prefix, crumb, text, _chars, 200) == (prefix, crumb)
    # crumb shortened, prefix kept
    p, c = fit_retrieval_text(prefix, crumb, text, _chars, 110)
    assert p == prefix and 0 < len(c) < len(crumb)
    assert len(compose_retrieval_text(p, c, text)) <= 110
    # no room for a crumb: prefix trimmed, text intact
    p, c = fit_retrieval_text(prefix, crumb, text, _chars, 80)
    assert c == "" and len(p) < 30 and len(compose_retrieval_text(p, c, text)) <= 80
    # text alone over the window: neither is kept, the text is not cut
    assert fit_retrieval_text(prefix, crumb, text, _chars, 50) == ("", "")


def test_files_from_pipeline_2_are_flagged_for_reindex():
    import src.domain.rag.chunk_identity as ci
    assert ci.CHUNK_PIPELINE_VERSION == "3"
    assert ci.reindex_recommended("indexed", "2") is True
