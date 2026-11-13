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
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

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
# 3: CJK sentence splitting with balanced pieces, overlap-free fallback split (txt /
#    transcripts), CSV as a table, parents sized in real tokens, capped breadcrumbs,
#    timed transcript blocks, no cached empty / chain-of-thought prefixes.
CHUNK_PIPELINE_VERSION = "3"


# Why an indexed file should be re-indexed (``reindex_reason``):
REINDEX_EMBEDDING = "embedding"   # its vectors came from another embedding model / dimension
REINDEX_PIPELINE = "pipeline"     # its chunks came from an older chunking pipeline


def configured_embedding() -> Tuple[Optional[str], Optional[int]]:
    """(model, dimension) of the embedding model configured now (runtime settings
    included); (None, None) when the config cannot be read, which flags nothing."""
    try:
        from src.config.config_manager import Config
        emb = Config.get_config_model().rag.embedding
        dim = getattr(emb, "dimension", None)
        return (getattr(emb, "model", None) or None), (int(dim) if dim else None)
    except Exception:  # noqa: BLE001 - advisory only
        return None, None


def reindex_reason(status: Optional[str], pipeline_version: Optional[str],
                   embedding_model: Optional[str] = None,
                   embedding_dimension: Optional[int] = None,
                   current: Optional[Tuple[Optional[str], Optional[int]]] = None) -> Optional[str]:
    """Why an indexed file should be re-indexed, or None.

    ``embedding``: it was indexed with an embedding model (or dimension) other than
    the configured one, so its vectors and the query's are not comparable. Only a
    recorded value is compared: an unknown (NULL) model or dimension flags nothing.
    ``pipeline``: its chunks were made by an older chunking pipeline (NULL = before
    versions were recorded). ``current`` is (model, dimension); None reads the config.
    """
    if status != "indexed":
        return None
    model, dim = configured_embedding() if current is None else current
    if (embedding_model and model and embedding_model != model) or (
            embedding_dimension and dim and int(embedding_dimension) != int(dim)):
        return REINDEX_EMBEDDING
    if (pipeline_version or "1") != CHUNK_PIPELINE_VERSION:
        return REINDEX_PIPELINE
    return None


def reindex_recommended(status: Optional[str], pipeline_version: Optional[str],
                        embedding_model: Optional[str] = None,
                        embedding_dimension: Optional[int] = None,
                        current: Optional[Tuple[Optional[str], Optional[int]]] = None) -> bool:
    """True for an indexed file that an older pipeline or another embedding model produced."""
    return reindex_reason(status, pipeline_version, embedding_model, embedding_dimension,
                          current) is not None


# Postgres twin of retrieval_text(): concat_ws skips NULLs, NULLIF drops empty strings.
SEARCH_TEXT_SQL = (
    "concat_ws(E'\\n', NULLIF(metadata_->>'context_prefix', ''), "
    "NULLIF(metadata_->>'breadcrumb', ''), text)"
)


# Rows that are keyword-searchable: leaves (and legacy rows without a role). Parents
# carry no search vector (see chunk_store); BM25 and Rebuild-FTS share this predicate.
LEAF_ONLY_SQL = "(metadata_->>'node_role' IS DISTINCT FROM 'parent')"


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


# A breadcrumb is identity, not content: past this many tokens a deep heading path
# (or a title-length heading) eats into the chunk's share of the embedding window.
BREADCRUMB_MAX_TOKENS = 48
_ELLIPSIS = "…"


def cap_breadcrumb(crumb: str, count: Callable[[str], int],
                   max_tokens: int = BREADCRUMB_MAX_TOKENS) -> str:
    """Shorten ``crumb`` to at most ``max_tokens``: drop middle headings first (keep
    the file stem and the innermost heading, with an ellipsis between), then cut the
    end. ``max_tokens <= 0`` gives ""."""
    if not crumb or max_tokens <= 0:
        return ""
    if count(crumb) <= max_tokens:
        return crumb
    parts = crumb.split(BREADCRUMB_SEP)
    if len(parts) > 2:
        # Keep the stem and as many innermost headings as fit.
        for keep in range(len(parts) - 2, 0, -1):
            cand = BREADCRUMB_SEP.join([parts[0], _ELLIPSIS] + parts[-keep:])
            if count(cand) <= max_tokens:
                return cand
        crumb = BREADCRUMB_SEP.join([parts[0], _ELLIPSIS, parts[-1]])
    lo, hi = 0, len(crumb)  # longest prefix (plus ellipsis) that fits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(crumb[:mid].rstrip() + _ELLIPSIS) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return crumb[:lo].rstrip() + _ELLIPSIS if lo else ""


def fit_retrieval_text(prefix: Optional[str], crumb: Optional[str], raw: str,
                       count: Callable[[str], int], max_tokens: int) -> Tuple[str, str]:
    """(prefix, crumb) shortened so ``count(compose_retrieval_text(...)) <= max_tokens``.

    The breadcrumb gives way first, then the prefix; the chunk text is never cut
    (a text that alone exceeds the window keeps neither and is returned as is)."""
    prefix, crumb = (prefix or "").strip(), (crumb or "").strip()
    if count(compose_retrieval_text(prefix, crumb, raw)) <= max_tokens:
        return prefix, crumb
    base = count(compose_retrieval_text(prefix, "", raw))
    if crumb and base < max_tokens:
        room = max_tokens - base - 1  # the joining newline
        short = cap_breadcrumb(crumb, count, room)
        if short and count(compose_retrieval_text(prefix, short, raw)) <= max_tokens:
            return prefix, short
    crumb = ""
    if not prefix or count(raw) >= max_tokens:
        return "", ""
    lo, hi = 0, len(prefix)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(compose_retrieval_text(prefix[:mid], "", raw)) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return prefix[:lo].strip(), ""


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
