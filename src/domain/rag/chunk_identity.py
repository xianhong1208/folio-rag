"""What a chunk is: its section, its breadcrumb, and the text it is retrieved by.

A leaf chunk is stored with its *original* document text in the ``text``
column: that is what every read path shows (search results, MCP, read mode,
neighbour expansion, the console's chunk viewer). What the chunk is *found*
by is richer:

    {contextual prefix}\\n{breadcrumb}\\n{original text}

- the contextual prefix is the LLM-written situating sentence
  (``metadata.context_prefix``, empty when contextual retrieval is off);
- the breadcrumb is the file name plus the heading path
  (``metadata.breadcrumb``, e.g. ``Aurora_X2_Manual › 5. Warranty``). It lets
  two documents with the same structure (two product manuals, two plan sheets)
  be told apart even when the chunk itself never names the product.

That retrieval text is what gets embedded, written into the full-text search
vector, and given to the reranker. ``retrieval_text`` (Python) and
``SEARCH_TEXT_SQL`` (Postgres) build it identically, so the console's
Rebuild-FTS reproduces exactly what indexing wrote.

Rows written before this layout have no ``breadcrumb``/``context_prefix``
metadata (their prefix is inside ``text``); both helpers then return ``text``
unchanged, so old folders keep working until they are re-indexed.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

# Small chunks merge only inside the same section: the first SECTION_DEPTH heading
# levels. Two levels = document title + chapter for most documents (Markdown "# title"
# + "## chapter", docx Title + Heading 1); finer subsections of one chapter may still
# merge, which avoids a flood of tiny chunks (each one an LLM prefix call).
SECTION_DEPTH = 2

BREADCRUMB_SEP = " › "

# Version of the chunking + retrieval-text + prefix pipeline. Bump it whenever chunk
# boundaries, retrieval text or the prefix prompt change in a way that makes existing
# indexes stale. Recorded per file (FileIndices.pipeline_version); files indexed by an
# older pipeline (or before versions were recorded: NULL) are flagged "re-index
# recommended" in the console and the REST status endpoints. They keep working as-is.
# 2: section-aligned chunks, breadcrumbs, original-text display, section-scoped prefixes.
CHUNK_PIPELINE_VERSION = "2"


def reindex_recommended(status: Optional[str], pipeline_version: Optional[str]) -> bool:
    """True for an indexed file whose chunks were made by an older pipeline."""
    return status == "indexed" and (pipeline_version or "1") != CHUNK_PIPELINE_VERSION


# Postgres twin of retrieval_text(): concat_ws skips NULLs, NULLIF drops empty strings.
SEARCH_TEXT_SQL = (
    "concat_ws(E'\\n', NULLIF(metadata_->>'context_prefix', ''), "
    "NULLIF(metadata_->>'breadcrumb', ''), text)"
)


# Hard ceilings on a chunk's original text so that prefix + breadcrumb + text stays
# inside a 512-token embedding window (multilingual-e5, bge-m3). With contextual
# retrieval the prefix (up to ~150 tokens of CJK) and the breadcrumb (~30) eat into it.
_CAP_WITH_PREFIX = 330
_CAP_WITHOUT_PREFIX = 450


def token_budgets(leaf_chunk_size: int, contextual: bool) -> Tuple[int, int]:
    """(prose budget, table budget) in tokens for one chunk's original text.

    Prose follows the configured leaf size; a table may use the whole safe window
    so that it stays in one piece whenever it fits."""
    cap = _CAP_WITH_PREFIX if contextual else _CAP_WITHOUT_PREFIX
    return min(leaf_chunk_size, cap), cap


def _clean_headings(headings: Optional[Sequence[Any]]) -> List[str]:
    return [str(h).strip() for h in (headings or []) if h is not None and str(h).strip()]


def section_key(headings: Optional[Sequence[Any]]) -> Tuple[str, ...]:
    """The section a chunk belongs to, for merge and parent-grouping decisions."""
    return tuple(_clean_headings(headings)[:SECTION_DEPTH])


def common_headings(items: Iterable[Optional[Sequence[Any]]]) -> Optional[List[str]]:
    """Heading path shared by every chunk merged into one (their common prefix).

    Labelling a merged chunk with its first piece's headings mislabelled the rest
    (a warranty paragraph tagged as the previous subsection)."""
    lists = [_clean_headings(h) for h in items]
    if not lists:
        return None
    out: List[str] = []
    for level in zip(*lists):
        if all(x == level[0] for x in level):
            out.append(level[0])
        else:
            break
    return out or None


_CONTENT_RANK = {"table": 2, "picture": 1}


def merge_content_type(types: Iterable[Optional[str]]) -> Optional[str]:
    """Content type of a merged chunk: table wins over picture, which wins over text."""
    best: Optional[str] = None
    for t in types:
        if _CONTENT_RANK.get(t or "", 0) > _CONTENT_RANK.get(best or "", 0):
            best = t
    return best


def breadcrumb(file_name: Optional[str], headings: Optional[Sequence[Any]]) -> str:
    """``file stem › heading › subheading`` (headings already include xlsx sheet names)."""
    stem = Path(file_name or "").stem.strip()
    parts = ([stem] if stem else []) + _clean_headings(headings)
    # Docling often repeats the document title as the first heading; don't say it twice.
    deduped: List[str] = []
    for p in parts:
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    return BREADCRUMB_SEP.join(deduped)


def compose_retrieval_text(prefix: Optional[str], crumb: Optional[str], raw: str) -> str:
    return "\n".join(p for p in ((prefix or "").strip(), (crumb or "").strip(), raw or "") if p)


def retrieval_text(text: str, metadata: Optional[Mapping[str, Any]]) -> str:
    """What a stored chunk is embedded / keyword-indexed / reranked by.

    For rows without breadcrumb metadata (indexed before this layout) the stored
    text already carries its prefix, so it is returned unchanged."""
    meta = metadata or {}
    if not meta.get("breadcrumb") and not meta.get("context_prefix"):
        return text or ""
    return compose_retrieval_text(meta.get("context_prefix"), meta.get("breadcrumb"), text or "")


def node_retrieval_text(node: Any) -> str:
    """retrieval_text for a llama-index node or NodeWithScore."""
    inner = getattr(node, "node", node)
    return retrieval_text(getattr(inner, "text", "") or "", getattr(inner, "metadata", None))


# --- identity for incremental re-indexing ------------------------------------------
# A leaf is identified by what determines its retrieval text apart from the prefix:
# its breadcrumb and original text. When a file is re-indexed, a leaf whose hash is
# unchanged keeps its contextual prefix (an LLM call) and its embedding, as long as
# the prefix generator and the embedding model are the same as when they were made.

# Fixed namespace for deterministic node ids (changing it re-keys every chunk).
_NODE_NAMESPACE = uuid.UUID("2f6b1c84-5d0e-4c3a-9a57-8e41d0b7c6f2")


def chunk_hash(raw_text: str, crumb: str) -> str:
    """Content hash of a leaf: breadcrumb + original text (moving a paragraph to
    another section changes its breadcrumb, hence its hash)."""
    return hashlib.sha256(f"{crumb}\n{raw_text}".encode("utf-8")).hexdigest()[:32]


def leaf_node_id(file_id: str, chash: str, occurrence: int = 0) -> str:
    """Stable leaf id: the same chunk of the same file keeps its id across re-indexes."""
    return str(uuid.uuid5(_NODE_NAMESPACE, f"{file_id}|leaf|{chash}|{occurrence}"))


def parent_node_id(file_id: str, child_hashes: Sequence[str]) -> str:
    """Stable parent id, derived from its children."""
    joined = hashlib.sha256("|".join(child_hashes).encode("utf-8")).hexdigest()[:32]
    return str(uuid.uuid5(_NODE_NAMESPACE, f"{file_id}|parent|{joined}"))


def embedding_key(embed_model: Any) -> str:
    """Identity of the embedding space: model class, model name and passage prefix
    (e5's "passage: " changes every vector). Reuse a stored vector only on a match."""
    inner = getattr(embed_model, "_inner", None) or embed_model
    name = getattr(embed_model, "model_name", None) or getattr(inner, "model_name", None) or ""
    passage = getattr(embed_model, "_passage_prefix", "") or ""
    return f"{type(inner).__name__}|{name}|passage={passage}"
