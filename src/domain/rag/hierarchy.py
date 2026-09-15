
"""Hierarchical chunking — aggregate leaf chunks into parents

Design trade-offs:
- LlamaIndex's built-in HierarchicalNodeParser works well on plain text, but it bypasses
  the structured Docling chunks. Docling already produces good chunks (tables preserved /
  headings aligned), which would be wasteful to discard.
- So we use a post-synthesis strategy: Docling produces the leaves first, then a token
  budget aggregates leaves into parents. A parent is not re-split — it is just an ordered
  join of its leaves plus metadata linkage.

Metadata schema (written to LlamaIndex Node.metadata; the PGVector schema is untouched):
    leaf:
        node_role:        "leaf"
        chunk_index:      sequential index (0-based within the file)
        parent_node_id:   node_id of the owning parent
        total_chunks:     total leaf count for this file

    parent:
        node_role:        "parent"
        children_node_ids: list[str]
        children_index_range: [start, end]   # inclusive
        chunk_count:      len(children_node_ids)
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from uuid import uuid4

from llama_index.core import Document
from llama_index.core.schema import TextNode

from src.log import get_api_logger

logger = get_api_logger()


def estimate_tokens(text: str) -> int:
    """Rough token estimate for hierarchy budget decisions (use tiktoken for accuracy)."""
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars * 0.3)


def build_hierarchy(
    leaf_documents: List[Document],
    parent_target_tokens: int = 1024,
    base_metadata: Optional[dict] = None,
) -> Tuple[List[TextNode], List[TextNode]]:
    """Aggregate leaf documents into two levels: (leaf_nodes, parent_nodes)

    Args:
        leaf_documents: leaf chunks produced by Docling/SentenceSplitter (already structured)
        parent_target_tokens: cap each parent once it accumulates this many tokens
            (default aligns with config.hierarchy_sizes[0])
        base_metadata: shared metadata (file_id, file_name, etc.) merged into every node

    Returns:
        (leaf_nodes, parent_nodes) — two lists
        - only leaf_nodes are embedded and stored in the vector store
        - parent_nodes are also stored (no embedding, plain-text cache) for lookup during auto-merge

    Why parents are stored too: when auto-merge triggers, the parent's full text must be
    fetched quickly. Storing them in the same PGVector table (distinguished by
    metadata.node_role) is simpler than a separate table and faster to query.
    """
    if not leaf_documents:
        return [], []

    base_metadata = base_metadata or {}
    leaf_nodes: List[TextNode] = []
    parent_nodes: List[TextNode] = []

    current_group: List[Tuple[int, Document]] = []  # [(leaf_index, doc), ...]
    current_tokens = 0

    def flush_group():
        """Turn the accumulated leaf group into one parent plus its leaf nodes."""
        if not current_group:
            return

        parent_id = str(uuid4())
        leaf_ids = []
        leaf_texts = []
        index_start = current_group[0][0]
        index_end = current_group[-1][0]

        for leaf_idx, leaf_doc in current_group:
            leaf_id = str(uuid4())
            leaf_meta = {
                **base_metadata,
                **{k: v for k, v in leaf_doc.metadata.items() if v is not None},
                "node_role": "leaf",
                "chunk_index": leaf_idx,
                "parent_node_id": parent_id,
                "total_chunks": len(leaf_documents),
            }
            leaf_node = TextNode(
                id_=leaf_id,
                text=leaf_doc.text,
                metadata=leaf_meta,
            )
            leaf_nodes.append(leaf_node)
            leaf_ids.append(leaf_id)
            leaf_texts.append(leaf_doc.text)

        # Create parent node — text is the children joined in order,
        # separated by "\n\n" to preserve structural readability
        parent_text = "\n\n".join(leaf_texts)
        parent_meta = {
            **base_metadata,
            "node_role": "parent",
            "children_node_ids": leaf_ids,
            "children_index_range": [index_start, index_end],
            "chunk_count": len(leaf_ids),
        }
        parent_node = TextNode(
            id_=parent_id,
            text=parent_text,
            metadata=parent_meta,
        )
        parent_nodes.append(parent_node)

    for i, leaf_doc in enumerate(leaf_documents):
        tokens = estimate_tokens(leaf_doc.text)

        if current_tokens + tokens > parent_target_tokens and current_group:
            flush_group()
            current_group = []
            current_tokens = 0

        current_group.append((i, leaf_doc))
        current_tokens += tokens

    flush_group()

    logger.info(
        f"Hierarchy built: {len(leaf_documents)} leaves → "
        f"{len(parent_nodes)} parents (avg {len(leaf_documents)/max(len(parent_nodes),1):.1f} leaves/parent)"
    )

    return leaf_nodes, parent_nodes
