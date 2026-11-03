
"""Hierarchical Indexer: Docling + leaf/parent hierarchy + Contextual Retrieval.

Leaves are contextualized, embedded and ranked. Parents are stored alongside
(with a zero embedding) only so auto-merge can fetch them via fetch_node: a
parent's content is a superset of its leaves, so vector and BM25 hits over
parents would just duplicate leaf hits. Co-locating them in the same PGVector
table gives atomic writes.
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from llama_index.core import Document, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore

from src.log import get_api_logger
from src.domain.rag.docling_loader import resolve_assets_dir
# Loading and splitting are delegated to these collaborators.
from src.domain.rag.document_loader import DocumentLoader, clean_text
from src.domain.rag.leaf_splitter import LeafSplitter
from src.domain.rag.embedding.batch_exec import (
    _EMBEDDING_BATCH_SIZE,
    _embed_batch_with_retry,
)
from src.domain.rag.hierarchy import build_hierarchy
from src.domain.rag.chunk_identity import (
    CHUNK_PIPELINE_VERSION,
    breadcrumb,
    chunk_hash,
    embedding_key,
    leaf_node_id,
    parent_node_id,
    retrieval_text,
)
from src.domain.rag.chunk_store import fetch_reusable, replace_file_chunks
from src.domain.rag.context_scope import SECTION_WINDOW_CHARS, document_outline, section_texts
from src.domain.rag.han_script import document_language
from src.domain.rag.hybrid_search import configured_text_search_config

if TYPE_CHECKING:
    from src.domain.rag.context_generator import ContextGenerator

logger = get_api_logger()


# Chunk-level progress callback signature: (stage, done, total)
# stage ∈ {"contextualizing", "embedding", "writing"}
ChunkProgressCallback = Callable[[str, Optional[int], Optional[int]], None]


class IndexingAbortedError(RuntimeError):
    """Raised when should_abort fires (e.g. the file was deleted mid-index).

    Provides cooperative cancellation within the long contextualizing/embedding
    stages, where stage-boundary gates cannot help. The adapter catches it and
    converts it to FileNotFoundError to follow the existing abort path.
    """


async def _to_thread_protected(fn, *args):
    """to_thread with cancellation protection: on cancel, wait for the thread to
    finish before propagating the cancellation upward.

    task.cancel() interrupts the await immediately, but the underlying thread
    keeps running as an orphan. Harmless for read-only calls, but a DB write
    (pgvector bulk insert) would race DROP TABLE / a new job and PGVectorStore.add
    can even re-create a just-dropped table. This guarantees the write thread has
    finished before CancelledError is raised.
    """
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, lambda: fn(*args))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        try:
            await fut  # let the orphan thread finish; discard its result
        except Exception:
            pass  # the thread's own error is irrelevant now — we are being cancelled
        raise


def _raise_if_aborted(
    should_abort: Optional[Callable[[], bool]],
    file_name: str,
    stage: str,
) -> None:
    """In-stage checkpoint: abort this file's indexing once the abort condition holds.

    should_abort is assumed not to raise (the adapter-side probe is fail-open); if
    it does, let the failure surface rather than silently skip the abort.
    """
    if should_abort and should_abort():
        raise IndexingAbortedError(f"{file_name}: indexing aborted during {stage}")


def _report_chunk_progress(
    cb: Optional[ChunkProgressCallback],
    stage: str,
    done: Optional[int],
    total: Optional[int],
) -> None:
    """A failing progress callback must never kill indexing — swallow and debug-log."""
    if cb is None:
        return
    try:
        cb(stage, done, total)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"chunk progress callback failed (ignored): {e}")


# Pre-load NLTK punkt tokenizer (thread-safe).
try:
    import nltk
    _assets = resolve_assets_dir()
    if _assets:
        _nltk_data = _assets / "nltk_data"
        if _nltk_data.is_dir() and str(_nltk_data) not in nltk.data.path:
            nltk.data.path.insert(0, str(_nltk_data))
    try:
        import llama_index.core._static as _llama_static
        _static_root = next(iter(_llama_static.__path__), None)
        if _static_root:
            _llama_cache = Path(_static_root) / "nltk_cache"
            if _llama_cache.is_dir() and str(_llama_cache) not in nltk.data.path:
                nltk.data.path.insert(0, str(_llama_cache))
    except Exception:
        pass
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.warning("NLTK punkt_tab not found")
except Exception:
    pass



def _prefix_warning(failed: int, total: int) -> Optional[str]:
    """File warning for chunks indexed without their contextual prefix."""
    if not failed:
        return None
    return (f"{failed} of {total} chunks lack a context prefix (the model gave no usable "
            "answer) — re-index the file to retry them")


def _join_warnings(*warnings: Optional[str]) -> Optional[str]:
    parts = [w for w in warnings if w]
    return "; ".join(parts) if parts else None


class HierarchicalIndexer:
    """Integrated indexer: Docling + hierarchy + contextual retrieval."""

    def __init__(
        self,
        leaf_chunk_size: int = 256,
        parent_target_tokens: int = 1024,
        chunk_overlap: int = 50,
        context_generator: Optional["ContextGenerator"] = None,
        embedding_model_name: Optional[str] = None,
        embed_model=None,
        asr_provider=None,
        asr_enabled: bool = True,
    ):
        self._leaf_chunk_size = leaf_chunk_size
        self._parent_target_tokens = parent_target_tokens
        self._chunk_overlap = chunk_overlap
        self._context_generator = context_generator
        self._embedding_model_name = embedding_model_name
        # Inject embed_model instead of the global Settings.embed_model: a shared
        # global would block multi-config embedding in one process and A/B reindexing.
        self._embed_model = embed_model

        self._document_loader = DocumentLoader(
            leaf_chunk_size=leaf_chunk_size,
            embedding_model_name=embedding_model_name,
            has_context_generator=context_generator is not None,
            asr_provider=asr_provider,   # None → default docling-whisper
            asr_enabled=asr_enabled,
        )
        self._leaf_splitter = LeafSplitter(
            leaf_chunk_size=leaf_chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model_name,
            has_context_generator=context_generator is not None,
        )

        cr_status = "enabled" if context_generator else "disabled"
        logger.info(
            f"HierarchicalIndexer initialized "
            f"(leaf_size={leaf_chunk_size}, parent_target={parent_target_tokens}, "
            f"contextual_retrieval={cr_status})"
        )

    @staticmethod
    def _text_search_config() -> str:
        """Postgres text-search config for the FTS column ('simple' = segmented mode).

        In segmented mode Chinese is segmented in Python and the tokens are written with
        to_tsvector('simple', ...); any other config (e.g. 'english') tokenizes the
        retrieval text directly. Reads the global config; 'simple' when unavailable.
        """
        return configured_text_search_config()

    @staticmethod
    def _clean_text(text: str) -> str:
        return clean_text(text)

    def load_document_from_file(
        self,
        file_path: str,
        file_id: str,
        file_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_cb: Optional[ChunkProgressCallback] = None,
    ) -> Document:
        """File -> Document (public API; delegates to DocumentLoader.load)."""
        return self._document_loader.load(
            file_path=file_path,
            file_id=file_id,
            file_name=file_name,
            metadata=metadata,
            progress_cb=progress_cb,
        )

    def _split_into_leaf_documents(self, document: Document) -> List[Document]:
        return self._leaf_splitter.split_into_leaf_documents(document)

    async def index_document(
        self,
        document: Document,
        vector_store: PGVectorStore,
        progress_cb: Optional[ChunkProgressCallback] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, int]:
        """Index one document: leaf chunks -> contextual prefix -> embed -> store with parents.

        progress_cb is a chunk-level callback ``(stage, done, total)`` over the
        contextualizing -> embedding -> writing stages (None for silent).
        should_abort returns True to abort this file (checked per chunk while
        contextualizing, per batch while embedding, and before writing), raising
        IndexingAbortedError.

        Re-indexing a file is incremental: leaves whose content hash is unchanged keep
        their contextual prefix and embedding (see chunk_store.fetch_reusable), and the
        file's rows are swapped in one transaction.

        Returns {"leaves", "parents", "llm_calls", "embedded", "pipeline_version",
        "prefix_failures", "warning"}.
        """
        file_name = document.metadata.get("file_name", "unknown")

        # 1. Split to leaf documents
        leaf_docs = self._split_into_leaf_documents(document)
        if not leaf_docs:
            # Producing no chunks must not silently return 0 — the upper layer
            # would record "indexed / 0 chunks" and write content_hash, after
            # which every re-index of the same file is short-circuited and never
            # recoverable. Raising makes the upper layer mark it failed (with a
            # reason) so the user can see it and fix the file and retry.
            raise ValueError(
                f"No indexable content: {file_name} produced 0 chunks "
                "(empty or whitespace-only document)"
            )
        n_chunks = len(leaf_docs)
        logger.info(f"Chunked {file_name}: {n_chunks} leaf chunks")

        # 2. Identity of each leaf: breadcrumb (file › heading path) and content hash.
        # The leaf's text stays the original document text, which is what every read
        # path shows; retrieval uses prefix + breadcrumb + text (chunk_identity).
        file_id_meta = str(document.metadata.get("file_id") or "")
        if not file_id_meta:
            raise ValueError(f"{file_name}: document has no file_id; cannot replace its chunks")
        hashes: List[str] = []
        leaf_ids: List[str] = []
        seen_hashes: Dict[str, int] = {}
        for ld in leaf_docs:
            crumb = breadcrumb(file_name, ld.metadata.get("headings"))
            chash = chunk_hash(ld.text, crumb)
            occurrence = seen_hashes.get(chash, 0)
            seen_hashes[chash] = occurrence + 1
            ld.metadata["breadcrumb"] = crumb
            ld.metadata["chunk_hash"] = chash
            hashes.append(chash)
            leaf_ids.append(leaf_node_id(file_id_meta, chash, occurrence))

        # Incremental re-index: an unchanged leaf keeps its prefix (LLM call) and its
        # embedding when they were made by the same prefix generator / embedding model.
        reusable = await asyncio.to_thread(fetch_reusable, vector_store, file_id_meta)
        ctx_key = self._context_generator.cache_key if self._context_generator else "off"

        prefixes: List[Optional[str]] = [None] * n_chunks
        need_llm: List[int] = []
        for i, chash in enumerate(hashes):
            old = reusable.get(chash)
            # An empty stored prefix under a live generator's key is a failure that an
            # older version cached (it stamped ctx_key on a discarded answer): regenerate.
            if (old is not None and old.get("ctx_key") == ctx_key
                    and (old.get("context_prefix") or ctx_key == "off")):
                prefixes[i] = old.get("context_prefix") or ""
            elif not self._context_generator or not document.text:
                prefixes[i] = ""
            else:
                need_llm.append(i)
        n_reused_prefix = n_chunks - len(need_llm)
        n_prefix_failed = 0

        if need_llm:
            _raise_if_aborted(should_abort, file_name, "contextualizing")
            _report_chunk_progress(progress_cb, "contextualizing", 0, len(need_llm))
            # The model reads the document outline + the chunk's own section (windowed
            # around it), not the first max_doc_chars of the whole document.
            all_headings = [ld.metadata.get("headings") for ld in leaf_docs]
            sections = section_texts(
                [ld.text for ld in leaf_docs], all_headings,
                width=min(self._context_generator.max_doc_chars, SECTION_WINDOW_CHARS),
            )
            generated = await self._context_generator.generate_prefixes(
                chunks=[leaf_docs[i].text for i in need_llm],
                full_document=document.text,
                file_name=file_name,
                on_progress=lambda done, total: _report_chunk_progress(
                    progress_cb, "contextualizing", done, total
                ),
                should_abort=should_abort,
                contexts=[sections[i] for i in need_llm],
                outline=document_outline(all_headings),
                language=document_language(document.text),
            )
            # generate_prefixes no-ops through the rest on abort; this is the real
            # stopping point — the result is discarded and never reaches embedding.
            _raise_if_aborted(should_abort, file_name, "contextualizing")
            for i, prefix in zip(need_llm, generated):
                # Reasoning models occasionally emit control characters: clean before PGVector.
                cleaned = None if prefix is None else self._clean_text(prefix).strip()
                # Empty after cleaning is a failure too: never cache it under ctx_key.
                prefixes[i] = cleaned or None
            n_prefix_failed = sum(1 for i in need_llm if prefixes[i] is None)
            if n_prefix_failed:
                logger.warning(
                    f"{file_name}: {n_prefix_failed} of {len(need_llm)} chunk(s) got no "
                    "contextual prefix (no usable answer from the model); indexed without "
                    "one — a re-index retries them"
                )
            logger.info(f"Contextual prefixes generated for {len(need_llm)} leaves ({file_name})")

        for ld, prefix in zip(leaf_docs, prefixes):
            if prefix:
                ld.metadata["context_prefix"] = prefix
            if prefix is not None:
                # A prefix that failed (None) carries no key, so the next re-index retries it.
                ld.metadata["ctx_key"] = ctx_key

        # 3. Build hierarchy (leaves + parents), with ids stable across re-indexes
        base_metadata = {
            k: v for k, v in document.metadata.items()
            if v is not None and not k.startswith("_")
        }
        leaf_nodes, parent_nodes = build_hierarchy(
            leaf_documents=leaf_docs,
            parent_target_tokens=self._parent_target_tokens,
            base_metadata=base_metadata,
            leaf_ids=leaf_ids,
            parent_id_for=lambda idxs: parent_node_id(file_id_meta, [hashes[i] for i in idxs]),
        )
        search_texts = {n.node_id: retrieval_text(n.text, n.metadata) for n in leaf_nodes}

        # 4. Embed leaves only (parents stay un-embedded); prefer the injected
        # embed_model, falling back to the global one only when not injected.
        embed_model = self._embed_model if self._embed_model is not None else Settings.embed_model
        if embed_model is None:
            raise RuntimeError("Embedding model not configured (neither injected nor global)")
        emb_key = embedding_key(embed_model)

        to_embed: List[TextNode] = []
        for n in leaf_nodes:
            n.metadata["emb_key"] = emb_key
            old = reusable.get(n.metadata["chunk_hash"])
            same_text = (
                old is not None
                and old.get("ctx_key") == n.metadata.get("ctx_key")
                and (old.get("context_prefix") or "") == (n.metadata.get("context_prefix") or "")
            )
            if same_text and old.get("emb_key") == emb_key and old.get("embedding"):
                n.embedding = old["embedding"]
            else:
                to_embed.append(n)
        n_reused_emb = len(leaf_nodes) - len(to_embed)

        leaf_texts = [search_texts[n.node_id] for n in to_embed]
        n_leaves = len(leaf_texts)
        n_batches = (n_leaves + _EMBEDDING_BATCH_SIZE - 1) // _EMBEDDING_BATCH_SIZE
        _report_chunk_progress(progress_cb, "embedding", 0, n_leaves)
        leaf_embeddings: list = []
        for i in range(n_batches):
            _raise_if_aborted(should_abort, file_name, "embedding")
            batch_start = i * _EMBEDDING_BATCH_SIZE
            batch_end = min(batch_start + _EMBEDDING_BATCH_SIZE, n_leaves)
            batch_texts = leaf_texts[batch_start:batch_end]
            try:
                batch_embeddings = await _embed_batch_with_retry(
                    embed_model, batch_texts, file_name, batch_start, batch_end, n_leaves
                )
            except Exception as exc:
                logger.error(
                    f"Embedding batch failed at leaves [{batch_start}-{batch_end}) "
                    f"of {n_leaves} ({file_name}) after retries: {exc}"
                )
                raise
            leaf_embeddings.extend(batch_embeddings)
            _report_chunk_progress(progress_cb, "embedding", batch_end, n_leaves)
            logger.info(
                f"Embedded batch {i + 1}/{n_batches} "
                f"({batch_start}-{batch_end} of {n_leaves})"
            )
        for n, emb in zip(to_embed, leaf_embeddings):
            n.embedding = emb
        if reusable:
            logger.info(
                f"[INCREMENTAL] {file_name}: {n_chunks} leaves; reused {n_reused_prefix} "
                f"prefix(es) and {n_reused_emb} embedding(s); {len(need_llm)} LLM call(s), "
                f"{n_leaves} embedding(s)"
            )

        # Parent nodes: assign a zero vector (PGVector disallows NULL embeddings).
        # Combined with the query-time node_role=leaf filter, parents never affect ranking.
        dims = next((len(n.embedding) for n in leaf_nodes if n.embedding), 1024)
        zero_vec = [0.0] * dims
        for p in parent_nodes:
            p.embedding = zero_vec

        # 5. Replace this file's rows in ONE transaction: old rows out, new rows in, with
        # the full-text vector written in the same INSERT. Re-indexing never leaves a
        # window with no chunks or with NULL search vectors (see chunk_store).
        _raise_if_aborted(should_abort, file_name, "writing")
        _report_chunk_progress(progress_cb, "writing", len(leaf_nodes), len(leaf_nodes))
        all_nodes: List[TextNode] = list(leaf_nodes) + list(parent_nodes)
        tsv_config = self._text_search_config()
        segment = None
        segmenter_name = None
        if tsv_config == "simple":
            # Segmented mode: Chinese is segmented in Python by the configured segmenter
            # (rag.retrieval.segmenter: ckip / jieba / bigram); Postgres's 'simple'
            # parser then splits on the inserted spaces. Same segmenter as Rebuild FTS.
            from src.domain.rag.text_segmenter import get_segmenter

            segmenter = get_segmenter()
            segment = segmenter.segment_batch
            segmenter_name = getattr(segmenter, "name", None)
        # _to_thread_protected (not a bare to_thread) so on cancel the write finishes
        # rather than leaving an orphan insert racing DROP TABLE.
        write = replace_file_chunks
        if segmenter_name:
            write = functools.partial(replace_file_chunks, segmenter_name=segmenter_name)
        await _to_thread_protected(
            write, vector_store, file_id_meta, all_nodes,
            search_texts, tsv_config, segment,
        )

        # 6. Ensure this (possibly just-created) table has an HNSW vector index now,
        # so a new folder gets fast search without waiting for the next boot.
        # Idempotent — near-instant skip when it already exists.
        try:
            from src.domain.rag.chunk_lookup import ChunkLookup
            from src.utils.db_bootstrap import ensure_hnsw_for_table

            table_name = ChunkLookup._get_table_name(vector_store)
            engine = ChunkLookup._get_engine(vector_store)
            created = await asyncio.to_thread(
                ensure_hnsw_for_table, engine, table_name, logger
            )
            if created:
                logger.info(f"✅ HNSW index auto-created for new table: {table_name}")
        except Exception as e:
            # Non-fatal: it still works without HNSW (just a slower seq scan).
            logger.warning(f"⚠️ HNSW post-index hook failed (not fatal): {e}")

        logger.info(
            f"Indexed {file_name}: {len(leaf_nodes)} leaves + {len(parent_nodes)} parents"
        )
        return {
            "leaves": len(leaf_nodes),
            "parents": len(parent_nodes),
            "llm_calls": len(need_llm),
            "embedded": n_leaves,
            "pipeline_version": CHUNK_PIPELINE_VERSION,
            "prefix_failures": n_prefix_failed,
            # Caveats for the file record: pages the parser could not read, chunks that
            # are indexed without their contextual prefix.
            "warning": _join_warnings(
                document.metadata.get("_parse_warning"),
                _prefix_warning(n_prefix_failed, n_chunks),
            ),
        }
