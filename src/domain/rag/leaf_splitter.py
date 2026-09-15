
"""Leaf Splitter — Document → leaf-sized chunks (extracted from HierarchicalIndexer).

Two paths:
- Docling pre-parsed (metadata._docling_chunks): used directly
- fallback: structure-aware split (preserving [TABLE_START] tables) + the same token-budget refine
"""

from __future__ import annotations

import re
from typing import List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from src.log import get_api_logger
from src.domain.rag.document_loader import clean_text

logger = get_api_logger()

# Detect trailing "empty code fence" residual chunks: for code blocks / images whose
# content Docling cannot extract, it emits chunks like "[description]\n\n```" that have
# only a heading/placeholder description and an empty body. Such chunks only pollute
# retrieval if indexed (rerank may even score them high on a literal heading match).
_TRAILING_EMPTY_FENCE = re.compile(r"\n\s*```[a-zA-Z0-9_+-]*\s*$")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)
_STRUCT_CHARS = re.compile(r"[\s`#|>*_~\-→▪●·•]+")


def _is_low_value_chunk(text: str) -> bool:
    """Decide whether a chunk has no retrieval value (empty code-fence placeholder / structure-only).

    Rules (conservative; prefer under-killing):
    - After stripping zero-width chars, if it "ends with an empty code fence" and has
      "fewer than 40 substantive chars after removing that fence" → treat as a Docling
      empty/placeholder residual chunk.
    - If no substantive char remains after removing all structural symbols → empty chunk.
    Normal content (has a body, or is short but does not end with an empty fence) is always kept.
    """
    t = (text or "").translate(_ZERO_WIDTH).strip()
    if not t:
        return True
    ended_empty_fence = bool(_TRAILING_EMPTY_FENCE.search(t))
    stripped = _TRAILING_EMPTY_FENCE.sub("", t).strip()
    body = _STRUCT_CHARS.sub("", stripped)
    if len(body) == 0:
        return True
    if ended_empty_fence and len(body) < 40:
        return True
    return False


class LeafSplitter:
    """Splitter (composed by HierarchicalIndexer; logic moved from its
    _structure_aware_split / _split_into_leaf_documents)."""

    def __init__(
        self,
        leaf_chunk_size: int,
        chunk_overlap: int,
        embedding_model_name: Optional[str] = None,
        has_context_generator: bool = False,
    ):
        self._leaf_chunk_size = leaf_chunk_size
        self._embedding_model_name = embedding_model_name
        self._has_context_generator = has_context_generator
        self._text_splitter = SentenceSplitter(
            chunk_size=leaf_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def structure_aware_split(self, text: str) -> List[str]:
        """Same as the previous flat-RAG version — tables preserved, text goes through SentenceSplitter."""
        table_pattern = re.compile(
            r'\[TABLE_START\]\s*(.*?)\s*\[TABLE_END\]',
            re.DOTALL,
        )
        chunks = []
        last_end = 0
        for match in table_pattern.finditer(text):
            text_before = text[last_end:match.start()].strip()
            if text_before:
                chunks.extend(self._text_splitter.split_text(text_before))
            table_md = match.group(1).strip()
            if table_md:
                chunks.append(table_md)
            last_end = match.end()
        text_after = text[last_end:].strip()
        if text_after:
            chunks.extend(self._text_splitter.split_text(text_after))
        if not chunks:
            chunks = self._text_splitter.split_text(text)
        return [c for c in chunks if c.strip()]

    def split_into_leaf_documents(self, document: Document) -> List[Document]:
        """Produce a leaf-sized Document list (Docling parsing preferred, SentenceSplitter fallback).

        Docling chunks may contain control chars such as NUL (\\x00) (from PDF font
        tables / ICC profiles) that PostgreSQL rejects, so we force a final cleanup.

        After the fallback path's structure-aware split (.txt/.csv/SimpleDirectoryReader),
        it runs the same `_refine_chunks_for_token_budget` as the Docling path: measuring
        the budget with the same embedding tokenizer, token-bounded splitting of large
        [TABLE_START] table blocks with header repetition, and merging small chunks —
        keeping both paths consistent.

        Args:
            document: the fully loaded Document (its metadata may hide `_docling_chunks`).

        Returns:
            Leaf Document list, with empty chunks filtered and control chars cleaned.
        """
        pre_parsed = document.metadata.pop("_docling_chunks", None)
        if pre_parsed:
            chunk_texts = pre_parsed
        else:
            chunk_texts = self.structure_aware_split(clean_text(document.text))
            # Uniform token-budget guarantee (same formula as DocumentLoader's docling path)
            try:
                from src.domain.rag.docling_loader import (
                    get_hf_tokenizer,
                    _refine_chunks_for_token_budget,
                )
                hf_tok = get_hf_tokenizer(self._embedding_model_name)
                budget = min(
                    self._leaf_chunk_size, 380 if self._has_context_generator else 450
                )
                chunk_texts = _refine_chunks_for_token_budget(chunk_texts, hf_tok, budget)
            except Exception as e:
                # When the tokenizer is unavailable, degrade to the structure-only split (legacy behavior) without blocking indexing
                logger.warning(
                    f"Token-budget refine unavailable for fallback split "
                    f"(degrading to structure-only): {e}"
                )

        leaf_docs: List[Document] = []
        seen_texts: set = set()  # dedupe within the same document (placeholder residual chunks often repeat 10+ times)
        dropped_lowvalue = 0
        dropped_dup = 0
        for item in chunk_texts:
            # A chunk may be a record dict ({"text","headings","page_no"} — the docling
            # path carries citation provenance) or a bare string (audio / legacy data) — accept both
            if isinstance(item, dict):
                text = item.get("text", "")
                provenance = {
                    k: item[k] for k in ("headings", "page_no")
                    if item.get(k) is not None
                }
            else:
                text, provenance = item, {}
            # Force cleanup — applied regardless of source; the Docling pre-parsed path is especially prone to NUL
            text = clean_text(text)
            if not text.strip():
                continue
            # Quality gate: drop empty code-fence placeholders / structure-only residuals (they pollute retrieval)
            if _is_low_value_chunk(text):
                dropped_lowvalue += 1
                continue
            # Dedupe within the document: placeholder residuals often repeat verbatim many times; keep only one
            key = text.strip()
            if key in seen_texts:
                dropped_dup += 1
                continue
            seen_texts.add(key)
            leaf_docs.append(Document(
                text=text,
                metadata={
                    **{k: v for k, v in document.metadata.items() if v is not None},
                    **provenance,  # per-chunk citation fields (hierarchy merges them into the leaf node as-is)
                },
            ))
        if dropped_lowvalue or dropped_dup:
            logger.info(
                f"[LEAF_QUALITY] kept {len(leaf_docs)} leaf chunks; "
                f"dropped {dropped_lowvalue} low-value + {dropped_dup} duplicate")
        return leaf_docs
