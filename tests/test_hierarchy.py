"""Unit tests for hierarchy.build_hierarchy

No dependency on the DB / vLLM / any external service — pure logic tests.
"""

from llama_index.core import Document

from src.domain.rag.hierarchy import build_hierarchy, estimate_tokens


def _e5(text):
    from src.domain.rag.chunk_tokenizer import get_chunk_tokenizer
    return len(get_chunk_tokenizer("intfloat/multilingual-e5-large").encode(text))


def test_estimate_tokens_uses_the_chunk_tokenizer():
    """Counted with the real tokenizer: CJK is ~0.7 token/char, not 1.5."""
    text = "第十二條員工因公出差者，應於出發前三日內檢具出差申請單，經直屬主管核准後始得辦理。" * 4
    assert estimate_tokens(text) == _e5(text)
    assert estimate_tokens(text) < len(text)


def test_estimate_tokens_english():
    assert estimate_tokens("hello world") == _e5("hello world")


def test_estimate_tokens_empty():
    """An empty string returns 0."""
    assert estimate_tokens("") == 0


def test_parents_fill_their_target_in_real_tokens():
    """A long Chinese section: parents reach ~the target, not half of it (M5)."""
    sent = "第十二條員工因公出差者，應於出發前三日內檢具出差申請單，經直屬主管核准後始得辦理。"
    leaves = [Document(text=sent * 6, metadata={"headings": ["Rules"]}) for _ in range(40)]
    _, parents = build_hierarchy(leaves, parent_target_tokens=1024)
    sizes = [_e5(p.text) for p in parents[:-1]]
    assert all(800 <= s <= 1024 + 10 for s in sizes), sizes


def test_custom_counter_is_used():
    leaves = [Document(text="a"), Document(text="b"), Document(text="c")]
    _, parents = build_hierarchy(leaves, parent_target_tokens=2, count_tokens=lambda t: 1)
    assert [p.metadata["chunk_count"] for p in parents] == [2, 1]


def test_build_hierarchy_single_small_doc():
    """One small leaf → one parent wrapping one leaf."""
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
    """Three leaves, total tokens < parent_target_tokens → all under one parent."""
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
    """Total leaf tokens exceed parent_target → split into multiple parents."""
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
    """parent.text should be the ordered join of its children leaves."""
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
    """base_metadata should be copied onto every leaf and parent."""
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


def test_build_hierarchy_oversized_leaf_gets_own_parent():
    """Every leaf exceeds the token budget → one leaf per parent, none dropped."""
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
    """Keys with a None value in the leaf's existing metadata are not copied into the leaf node."""
    leaves = [Document(text="內容", metadata={"keep_me": "v", "drop_me": None})]

    leaf_nodes, _ = build_hierarchy(
        leaf_documents=leaves,
        parent_target_tokens=1024,
        base_metadata={},
    )

    assert leaf_nodes[0].metadata["keep_me"] == "v"
    assert "drop_me" not in leaf_nodes[0].metadata


def test_build_hierarchy_chunk_index_and_total_chunks_across_parents():
    """When split across parents, chunk_index stays globally contiguous and total_chunks is the total leaf count."""
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
    """Building works fine without base_metadata (None)."""
    leaf_nodes, parent_nodes = build_hierarchy(
        leaf_documents=[Document(text="內容")],
        parent_target_tokens=1024,
    )

    assert len(leaf_nodes) == 1
    assert len(parent_nodes) == 1
    assert leaf_nodes[0].metadata["node_role"] == "leaf"


def test_build_hierarchy_leaf_metadata_overrides_base():
    """On a key shared between the leaf's own metadata and base_metadata, the leaf wins; the parent uses base."""
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
