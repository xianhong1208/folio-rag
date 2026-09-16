
"""Hierarchical Indexer: Docling + leaf/parent hierarchy + Contextual Retrieval.

Leaves are contextualized, embedded and ranked. Parents are stored alongside
(with a zero embedding) only so auto-merge can fetch them via fetch_node: a
parent's content is a superset of its leaves, so vector and BM25 hits over
parents would just duplicate leaf hits. Co-locating them in the same PGVector
table gives atomic writes.
"""

from __future__ import annotations

import asyncio
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


# Whisper audio defenses. Importing audio_defense applies the whisper patch, and
# this module is always imported before get_converter() — keep that ordering.
from src.domain.rag.audio_defense import (  # noqa: F401
    _AUDIO_TRIM_EXTENSIONS,
    _filter_whisper_hallucinations,
    _trim_audio_leading_silence,
)



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
    def _ckip_fts_enabled() -> bool:
        """True when FTS runs in CKIP mode (text_search_config == 'simple').

        In CKIP mode Chinese is segmented in Python and stored into the
        `text_search_tsv` column via to_tsvector('simple', ...). Any other
        config (e.g. 'english') keeps llama-index's native tsvector untouched.
        Reads the global config; defaults to 'simple' when unavailable.
        """
        try:
            from src.config.config_manager import Config

            ret = getattr(
                getattr(Config.get_config_model(), "rag", None), "retrieval", None
            )
            cfg = getattr(ret, "text_search_config", "simple") or "simple"
            return cfg == "simple"
        except Exception:
            return True

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
        IndexingAbortedError. Returns {"leaves": N, "parents": M}.
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

        # 2. Apply Contextual Retrieval to leaves
        if self._context_generator and document.text:
            _raise_if_aborted(should_abort, file_name, "contextualizing")
            _report_chunk_progress(progress_cb, "contextualizing", 0, n_chunks)
            chunk_texts = [d.text for d in leaf_docs]
            contextualized = await self._context_generator.generate_batch(
                chunks=chunk_texts,
                full_document=document.text,
                file_name=file_name,
                on_progress=lambda done, total: _report_chunk_progress(
                    progress_cb, "contextualizing", done, total
                ),
                should_abort=should_abort,
            )
            # generate_batch no-ops through the rest on abort; this is the real
            # stopping point — the result is discarded and never reaches embedding.
            _raise_if_aborted(should_abort, file_name, "contextualizing")
            # Clean the LLM contextual prefix too: reasoning models occasionally emit
            # control characters, so make it safe before PGVector.
            leaf_docs = [
                Document(text=self._clean_text(ct), metadata=ld.metadata.copy())
                for ld, ct in zip(leaf_docs, contextualized)
            ]
            logger.info(f"Contextual prefixes applied to {len(leaf_docs)} leaves ({file_name})")

        # 3. Build hierarchy (leaves + parents)
        base_metadata = {
            k: v for k, v in document.metadata.items()
            if v is not None and not k.startswith("_")
        }
        leaf_nodes, parent_nodes = build_hierarchy(
            leaf_documents=leaf_docs,
            parent_target_tokens=self._parent_target_tokens,
            base_metadata=base_metadata,
        )

        # 4. Embed leaves only (parents stay un-embedded); prefer the injected
        # embed_model, falling back to the global one only when not injected.
        embed_model = self._embed_model if self._embed_model is not None else Settings.embed_model
        if embed_model is None:
            raise RuntimeError("Embedding model not configured (neither injected nor global)")

        leaf_texts = [n.text for n in leaf_nodes]
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
        for n, emb in zip(leaf_nodes, leaf_embeddings):
            n.embedding = emb

        # Parent nodes: assign a zero vector (PGVector disallows NULL embeddings).
        # Combined with the query-time node_role=leaf filter, parents never affect ranking.
        embed_dim = len(leaf_embeddings[0]) if leaf_embeddings else 1024
        zero_vec = [0.0] * embed_dim
        for p in parent_nodes:
            p.embedding = zero_vec

        # 5. Bulk insert (leaves + parents in single transaction)
        _raise_if_aborted(should_abort, file_name, "writing")
        _report_chunk_progress(progress_cb, "writing", n_leaves, n_leaves)
        all_nodes: List[TextNode] = list(leaf_nodes) + list(parent_nodes)

        # Idempotently purge this file's stale chunks before writing. Node ids are
        # regenerated with uuid4 each time and PGVector does not de-duplicate, so
        # any re-run after a "write succeeded but finalize failed" case (timeout
        # cancel / record_index_success blew up / a later gate) would stack a
        # second full copy of the vectors into the same table. Silently skipped
        # when the table does not exist (first index).
        file_id_meta = document.metadata.get("file_id")
        if file_id_meta:
            def _purge_stale():
                from src.domain.rag.chunk_lookup import ChunkLookup
                from src.domain.rag.vector_store_manager import VectorStoreManager
                try:
                    table = ChunkLookup._get_table_name(vector_store)
                    removed = VectorStoreManager.delete_chunks_by_file(table, str(file_id_meta))
                    if removed:
                        logger.info(
                            f"[REINDEX_PURGE] removed {removed} stale chunk(s) of "
                            f"file {file_id_meta} before write")
                except Exception as purge_err:  # table missing etc. — normal on first index
                    logger.debug(f"stale-chunk purge skipped: {purge_err}")
            await _to_thread_protected(_purge_stale)
        # Sync SQLAlchemy insert, offloaded off the loop. _to_thread_protected (not
        # a bare to_thread) so on cancel the write finishes rather than leaving an
        # orphan insert racing DROP TABLE.
        await _to_thread_protected(vector_store.add, all_nodes)

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

        # 6b. CKIP mode: overwrite the FTS tsvector with Python-side CKIP tokens so
        # BM25 matches space-less Chinese (llama-index's to_tsvector('simple', text)
        # cannot segment it). Only the tsv changes; stored `text` stays original.
        # Any other config no-ops. Runs off the loop (blocking DB + CKIP inference).
        if self._ckip_fts_enabled():
            try:
                from src.domain.rag.chunk_lookup import ChunkLookup
                from src.domain.rag.vector_store_manager import (
                    rebuild_text_search_tsv,
                )
                from src.domain.rag.ckip_segmenter import get_segmenter

                table_name = ChunkLookup._get_table_name(vector_store)
                engine = ChunkLookup._get_engine(vector_store)
                segmenter = get_segmenter()
                # Scope to this file's rows: re-segmenting the whole table on every
                # add cost ~10 min on a 7k-chunk folder.
                n = await _to_thread_protected(
                    rebuild_text_search_tsv, engine, table_name, segmenter,
                    document.metadata.get("file_id"),
                )
                logger.info(
                    f"✅ CKIP text_search_tsv rebuilt for {n} row(s) of "
                    f"{file_name} in {table_name}"
                )
            except Exception as e:
                # Non-fatal: falls back to llama-index's native 'simple' tsvector
                # (degraded Chinese recall) rather than failing the index write.
                logger.warning(f"⚠️ CKIP tsv rebuild failed (not fatal): {e}")

        logger.info(
            f"Indexed {file_name}: {len(leaf_nodes)} leaves + {len(parent_nodes)} parents"
        )
        return {"leaves": len(leaf_nodes), "parents": len(parent_nodes)}
