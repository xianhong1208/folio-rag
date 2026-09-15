
"""RAG adapter facade (obtained via `get_rag_adapter()`).

Coordinates the sub-modules in this package: rag_context (shared state),
rag_indexing, rag_query, and rag_maintenance.
"""

from typing import Any, Dict, List, Optional

from llama_index.vector_stores.postgres import PGVectorStore

from src.adapter.model import RAGQueryResult
from src.adapter.rag_context import RAGContext
from src.adapter.rag_indexing import RAGIndexingService
from src.adapter.rag_maintenance import RAGMaintenanceService
from src.adapter.rag_query import RAGQueryService
from src.api.router.response import (
    DeleteFolderIndexResponse,
    FileRequest,
    IndexDocumentResponse,
    IndexFolderResponse,
)
from src.domain.rag.index_job_manager import IndexProgressTracker
from src.domain.rag.query_engine import QueryEngine
from src.log import get_adapter_logger


logger = get_adapter_logger()


class RAGAdapter:
    """LlamaIndex-based RAG adapter for indexing, querying, and management,
    backed by PostgreSQL pgvector.

    State lives on RAGContext (`self._ctx`); this class coordinates the
    indexing, query, and maintenance sub-services. The `self._xxx` properties
    are backwards-compat delegates to the context.
    """

    def __init__(self):
        self._ctx: RAGContext = RAGContext.from_config()
        self._indexing = RAGIndexingService(self._ctx)
        self._query_svc = RAGQueryService(self._ctx)
        self._maintenance = RAGMaintenanceService(self._ctx)

    @property
    def _embedding_provider(self):
        return self._ctx.embedding_provider

    @property
    def _embed_dim(self) -> int:
        return self._ctx.embed_dim

    @property
    def _model_name(self) -> str:
        return self._ctx.model_name

    @property
    def _chunker_tokenizer_name(self) -> Optional[str]:
        return self._ctx.chunker_tokenizer_name

    @property
    def _leaf_chunk_size(self) -> int:
        return self._ctx.leaf_chunk_size

    @property
    def _parent_target_tokens(self) -> int:
        return self._ctx.parent_target_tokens

    @property
    def _chunk_overlap(self) -> int:
        return self._ctx.chunk_overlap

    @property
    def _vector_store_manager(self):
        return self._ctx.vector_store_manager

    @property
    def _indexer(self):
        return self._ctx.indexer

    @property
    def _document_indexer(self):
        return self._ctx.indexer  # legacy alias

    @property
    def _indexing_service(self):
        return self._ctx.indexing_service

    @property
    def _auto_merging_enabled(self) -> bool:
        return self._ctx.auto_merging_enabled

    @property
    def _merge_threshold(self) -> float:
        return self._ctx.merge_threshold

    @property
    def _expand_neighbors(self) -> int:
        return self._ctx.expand_neighbors

    @property
    def _reranker(self):
        return self._ctx.reranker

    @property
    def _query_engines(self) -> Dict[str, QueryEngine]:
        return self._ctx.query_engines

    def _invalidate_query_engine_cache(self, folder_id: int) -> None:
        """Remove cached QueryEngine for a folder (call after index changes)."""
        self._ctx.invalidate_query_engine_cache(folder_id)

    def _get_vector_store(
        self, folder_id: int, token: str, folder: Optional[Any] = None
    ) -> PGVectorStore:
        """Get or create the vector store for a folder/token (delegates to RAGContext)."""
        return self._ctx.get_vector_store(folder_id, token, folder)

    async def index_document(
        self,
        file_record: FileRequest,
        token: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        folder_id_override: Optional[int] = None,
    ) -> IndexDocumentResponse:
        """Delegates to RAGIndexingService — single-document indexing."""
        return await self._indexing.index_document(
            file_record=file_record,
            token=token,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            folder_id_override=folder_id_override,
        )

    async def index_folder(
        self,
        folder_id: int,
        token: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        skip_existing: bool = True,
        progress_tracker: Optional["IndexProgressTracker"] = None,
    ) -> IndexFolderResponse:
        """Delegates to RAGIndexingService — whole-folder indexing."""
        return await self._indexing.index_folder(
            folder_id=folder_id,
            token=token,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            skip_existing=skip_existing,
            progress_tracker=progress_tracker,
        )

    async def index_files(
        self,
        file_ids: List[str],
        folder_id: int,
        token: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        progress_tracker: Optional["IndexProgressTracker"] = None,
    ) -> IndexFolderResponse:
        """Delegates to RAGIndexingService — index a specified list of files (used by upload auto-index)."""
        return await self._indexing.index_files(
            file_ids=file_ids,
            folder_id=folder_id,
            token=token,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_tracker=progress_tracker,
        )

    async def query_rag(
        self,
        query: str,
        folder_id: int,
        token: str,
        top_k: int = 5,
        similarity_cutoff: float = 0.7,
        use_cache: bool = True,
        sparse_top_k: Optional[int] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> RAGQueryResult:
        """Delegates to RAGQueryService — flat hybrid query with result cache."""
        return await self._query_svc.query_rag(
            query=query,
            folder_id=folder_id,
            token=token,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
            use_cache=use_cache,
            sparse_top_k=sparse_top_k,
            hybrid_alpha=hybrid_alpha,
        )

    async def query_trace(
        self,
        query: str,
        folder_id: int,
        token: str,
        top_k: int = 5,
        similarity_cutoff: float = 0.7,
        sparse_top_k: Optional[int] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> dict:
        """Delegates to RAGQueryService — retrieval-trace breakdown (diagnostic, bypasses cache)."""
        return await self._query_svc.query_trace(
            query=query,
            folder_id=folder_id,
            token=token,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
            sparse_top_k=sparse_top_k,
            hybrid_alpha=hybrid_alpha,
        )

    async def query_rag_by_folder_name(
        self,
        query: str,
        folder_name: str,
        token: str,
        top_k: int = 5,
        similarity_cutoff: float = 0.7,
        sparse_top_k: Optional[int] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> RAGQueryResult:
        """Delegates to RAGQueryService — query by folder name (with authorization)."""
        return await self._query_svc.query_rag_by_folder_name(
            query=query,
            folder_name=folder_name,
            token=token,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
            sparse_top_k=sparse_top_k,
            hybrid_alpha=hybrid_alpha,
        )

    async def query(
        self,
        *,
        folder_name: str,
        mode: str,
        query: str,
        file_id: Optional[str],
        top_k: int,
        similarity_cutoff: float,
        expand_context: bool,
        token: str,
    ) -> Dict[str, Any]:
        """Delegates to RAGQueryService — MCP agentic three-mode (search / list / read)."""
        return await self._query_svc.query_agentic(
            folder_name=folder_name,
            mode=mode,
            query=query,
            file_id=file_id,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
            expand_context=expand_context,
            token=token,
        )

    async def delete_document_index(self, file_id: int, token: str) -> bool:
        """Delegates to RAGMaintenanceService — remove a document's index from the vector store."""
        return await self._maintenance.delete_document_index(file_id, token)

    async def trigger_auto_index(
        self,
        file_ids: List[str],
        folder_id: int,
        token: str,
        auto_index: bool = True,
    ) -> Dict[str, Any]:
        """Delegates to RAGIndexingService — trigger a background auto-index job."""
        return await self._indexing.trigger_auto_index(
            file_ids=file_ids,
            folder_id=folder_id,
            token=token,
            auto_index=auto_index,
        )

    async def delete_folder_index(
        self,
        folder_id: int,
        token: str,
    ) -> DeleteFolderIndexResponse:
        """Delegates to RAGMaintenanceService — delete all of a folder's indexes (keeping the folder and files)."""
        return await self._maintenance.delete_folder_index(folder_id, token)

    async def get_indexed_files(
        self,
        folder_id: int,
    ) -> List[Dict[str, Any]]:
        """Delegates to RAGMaintenanceService — list indexed files."""
        return await self._maintenance.get_indexed_files(folder_id)

# Lazy singleton: Config must be loaded before the adapter is built.
_rag_adapter_instance = None

def get_rag_adapter() -> RAGAdapter:
    """Get or lazily create the singleton RAGAdapter instance."""
    global _rag_adapter_instance
    if _rag_adapter_instance is None:
        _rag_adapter_instance = RAGAdapter()
    return _rag_adapter_instance


def peek_rag_adapter() -> "RAGAdapter | None":
    """Return the instance only if already built; never triggers construction.

    Used by admin hot-reload: if the adapter isn't built yet, the first
    construction picks up the new ConfigModel, so there is nothing to rebind.
    """
    return _rag_adapter_instance