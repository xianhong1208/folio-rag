"""Unit tests for hierarchy.build_hierarchy

No dependency on the DB / vLLM / any external service — pure logic tests.
Related docs: docs/testing/specs/SPEC-hierarchy.md, docs/testing/test-cases/TC-hierarchy.md
"""

from llama_index.core import Document

from src.domain.rag.hierarchy import build_hierarchy, estimate_tokens


def test_estimate_tokens_chinese():
    """CJK characters are estimated at 1.5 token/char (TC-hierarchy-01)."""
    assert estimate_tokens("公司財報") == int(4 * 1.5)


def test_estimate_tokens_english():
    """English is estimated at 0.3 token/char (rough) (TC-hierarchy-02)."""
    assert estimate_tokens("hello") == int(5 * 0.3)


def test_estimate_tokens_empty():
    """An empty string returns 0 (TC-hierarchy-03)."""
    assert estimate_tokens("") == 0


def test_build_hierarchy_single_small_doc():
    """One small leaf → one parent wrapping one leaf (TC-hierarchy-04)."""
    leaves = [Document(text="短短的內容", metadata={"file_id": "f1", "file_name": "a.txt"})]

    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=1024,
        base_metadata={"file_id": "f1", "file_name": "a.txt"},
    )

    assert len(leaf_nodes) == 1
    assert len(parent_nodes) == 1

    leaf = leaf_nodes[0]
    parent = parent_nodes[0]

    assert leaf.metadata["node_role"] == "leaf"
    assert leaf.metadata["parent_node_id"] == parent.id_
    assert leaf.metadata["chunk_index"] == 0
    assert leaf.metadata["total_chunks"] == 1

    assert parent.metadata["node_role"] == "parent"
    assert parent.metadata["children_node_ids"] == [leaf.id_]
    assert parent.metadata["chunk_count"] == 1
    assert parent.metadata["children_index_range"] == [0, 0]


def test_build_hierarchy_multiple_leaves_in_one_parent():
    """Three leaves, total tokens < parent_target_tokens → all under one parent (TC-hierarchy-05)."""
    leaves = [
        Document(text="leaf one short text"),
        Document(text="leaf two short text"),
        Document(text="leaf three short text"),
    ]

    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=1024,
        base_metadata={"file_id": "f1"},
    )

    assert len(leaf_nodes) == 3
    assert len(parent_nodes) == 1

    parent = parent_nodes[0]
    assert parent.metadata["chunk_count"] == 3
    assert parent.metadata["children_index_range"] == [0, 2]

    # All leaves point to the same parent
    parent_ids = {ln.metadata["parent_node_id"] for ln in leaf_nodes}
    assert parent_ids == {parent.id_}


def test_build_hierarchy_splits_when_token_budget_exceeded():
    """Total leaf tokens exceed parent_target → split into multiple parents (TC-hierarchy-06)."""
    # Each Chinese segment is ~30 chars → ~45 tokens; with budget=100 → ~2 leaves per group
    leaves = [
        Document(text="這是測試段落數字一" * 5),  # ~45 chars * 1.5 = ~67 tokens
        Document(text="這是測試段落數字二" * 5),
        Document(text="這是測試段落數字三" * 5),
        Document(text="這是測試段落數字四" * 5),
    ]

    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=100,
        base_metadata={},
    )

    assert len(leaf_nodes) == 4
    assert len(parent_nodes) >= 2  # at least 2 groups

    # Each parent's children_index_range should be contiguous and non-overlapping
    ranges = [p.metadata["children_index_range"] for p in parent_nodes]
    ranges.sort()
    for i in range(len(ranges) - 1):
        assert ranges[i][1] < ranges[i + 1][0], "Parent ranges should not overlap"


def test_build_hierarchy_parent_text_concatenation():
    """parent.text should be the ordered join of its children leaves (TC-hierarchy-07)."""
    leaves = [
        Document(text="第一段"),
        Document(text="第二段"),
        Document(text="第三段"),
    ]

    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=1024,
        base_metadata={},
    )

    parent = parent_nodes[0]
    assert "第一段" in parent.text
    assert "第二段" in parent.text
    assert "第三段" in parent.text
    assert parent.text.index("第一段") < parent.text.index("第二段") < parent.text.index("第三段")


def test_build_hierarchy_metadata_propagation():
    """base_metadata should be copied onto every leaf and parent (TC-hierarchy-08)."""
    leaves = [Document(text="x", metadata={"existing_key": "preserve_me"})]

    base = {"file_id": "f1", "file_name": "report.pdf", "folder_name": "公司"}

    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=1024,
        base_metadata=base,
    )

    for k, v in base.items():
        assert leaf_nodes[0].metadata[k] == v
        assert parent_nodes[0].metadata[k] == v

    # The leaf's own metadata is also preserved
    assert leaf_nodes[0].metadata["existing_key"] == "preserve_me"


def test_build_hierarchy_empty_input():
    """Empty input returns ([], []) (TC-hierarchy-09)."""
    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=[],
        parent_target_tokens=1024,
        base_metadata={},
    )
    assert leaf_nodes == []
    assert parent_nodes == []


def test_estimate_tokens_mixed_chinese_english():
    """Mixed Chinese/English is summed by each coefficient, then truncated to int (TC-hierarchy-10)."""
    # "公司ABC" = 2 Chinese * 1.5 + 3 English * 0.3
    assert estimate_tokens("公司ABC") == int(2 * 1.5 + 3 * 0.3)


def test_estimate_tokens_non_cjk_range_counts_as_other():
    """Digits and Japanese kana fall outside the CJK range ('一'~'鿿') and are counted at 0.3/char (TC-hierarchy-11)."""
    assert estimate_tokens("12345") == int(5 * 0.3)
    # Hiragana (from U+3042) < U+4E00, not counted as Chinese → 3 * 0.3 = 0.9 → 0
    assert estimate_tokens("あいう") == 0


def test_build_hierarchy_oversized_leaf_gets_own_parent():
    """Every leaf exceeds the token budget → one leaf per parent, none dropped (TC-hierarchy-12)."""
    leaves = [
        Document(text="這是超過預算的長段落之一" * 3),
        Document(text="這是超過預算的長段落之二" * 3),
        Document(text="這是超過預算的長段落之三" * 3),
    ]

    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=10,  # far smaller than any single leaf
        base_metadata={},
    )

    assert len(leaf_nodes) == 3
    assert len(parent_nodes) == 3
    for parent in parent_nodes:
        assert parent.metadata["chunk_count"] == 1


def test_build_hierarchy_filters_none_metadata_values():
    """Keys with a None value in the leaf's existing metadata are not copied into the leaf node (TC-hierarchy-13)."""
    leaves = [Document(text="內容", metadata={"keep_me": "v", "drop_me": None})]

    leaf_nodes, _ = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=1024,
        base_metadata={},
    )

    assert leaf_nodes[0].metadata["keep_me"] == "v"
    assert "drop_me" not in leaf_nodes[0].metadata


def test_build_hierarchy_chunk_index_and_total_chunks_across_parents():
    """When split across parents, chunk_index stays globally contiguous and total_chunks is the total leaf count (TC-hierarchy-14)."""
    leaves = [Document(text="這是測試段落內容編號" * 5) for _ in range(4)]

    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=100,
        base_metadata={},
    )

    assert len(parent_nodes) >= 2  # confirm the split actually happened
    assert [ln.metadata["chunk_index"] for ln in leaf_nodes] == [0, 1, 2, 3]
    assert all(ln.metadata["total_chunks"] == 4 for ln in leaf_nodes)


def test_build_hierarchy_default_base_metadata():
    """Building works fine without base_metadata (None) (TC-hierarchy-15)."""
    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=[Document(text="內容")],
        parent_target_tokens=1024,
    )

    assert len(leaf_nodes) == 1
    assert len(parent_nodes) == 1
    assert leaf_nodes[0].metadata["node_role"] == "leaf"


def test_build_hierarchy_leaf_metadata_overrides_base():
    """On a key shared between the leaf's own metadata and base_metadata, the leaf wins; the parent uses base (TC-hierarchy-16)."""
    leaves = [Document(text="內容", metadata={"file_name": "leaf-level.txt"})]

    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=1024,
        base_metadata={"file_name": "base-level.txt"},
    )

    assert leaf_nodes[0].metadata["file_name"] == "leaf-level.txt"
    assert parent_nodes[0].metadata["file_name"] == "base-level.txt"


def test_build_hierarchy_empty_input():
    """Empty leaf list: returns empty leaves/parents without crashing (flush_group early-returns on an empty group)."""
    from src.domain.rag.hierarchy import build_hierarchy
    leaves, parents = build_hierarchy(
        leaf_documents=[], parent_target_tokens=512, base_metadata={})
    assert leaves == [] and parents == []
