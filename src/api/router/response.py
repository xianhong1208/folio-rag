
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from uuid import UUID

class UpdateFileRequest(BaseModel):
    """Update-file request model."""
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class BatchDeleteRequest(BaseModel):
    """Batch-delete request model."""
    file_ids: Optional[List[UUID]] = None
    delete_all: bool = False


class CreateFolderRequest(BaseModel):
    """Request model to create a folder in a given space."""
    name: str
    description: Optional[str] = None

class UpdateFolderRequest(BaseModel):
    """Update-folder request model."""
    name: Optional[str] = None
    description: Optional[str] = None

# RAG Response Models


class IndexRequest(BaseModel):
    """Index-file request model.

    Both fields allow ``None`` — when None, the indexing pipeline reads defaults
    (leaf_chunk_size / chunk_overlap) from ``config.rag.chunking``.
    """
    chunk_size: Optional[int] = Field(
        default=None,
        ge=100,
        le=2000,
        # examples drives Swagger's "Example Value"; when this field is omitted, indexing uses the
        # config default, so the example matches the config's current hierarchy_sizes[1] (leaf_chunk_size=256).
        examples=[256],
        description=(
            "Chunk size (characters). When omitted, uses `config.rag.chunking.hierarchy_sizes[1]`"
            " (current default = 256)."
        ),
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        ge=0,
        le=500,
        examples=[50],
        description=(
            "Chunk overlap size. When omitted, uses `config.rag.chunking.chunk_overlap`"
            " (current default = 50). Note: should be < chunk_size."
        ),
    )

    model_config = {
        # Provide two examples: an empty body (uses config defaults, most common) + an explicit override
        "json_schema_extra": {
            "examples": [
                {},
                {"chunk_size": 256, "chunk_overlap": 50},
            ]
        }
    }


class QueryRequest(BaseModel):
    """Query request model.

    All retrieval-tuning fields allow ``None``; when None, the API reads the defaults from
    `rag.retrieval.*` in `config.yaml`, keeping API behavior consistent with the server config.
    This way an ops change to config immediately affects the API's default behavior, eliminating
    the "hardcoded Pydantic values drift out of sync with config" problem.
    """
    query: str = Field(..., examples=["What is the core conclusion of this document?"], description="Query string")
    folder_name: str = Field(..., examples=["my-folder"], description="Folder name")
    # The 4 tuning fields below — example values match config.rag.retrieval.default_*.
    # When omitted they take the config value, so using the config's current values as examples is the most intuitive.
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        examples=[10],
        description="Return the top K results. None = use `config.rag.retrieval.default_top_k` (current = 10).",
    )
    similarity_cutoff: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        examples=[0.25],
        description=(
            "Similarity threshold. None = use `config.rag.retrieval.default_similarity_cutoff` (current = 0.25). "
            "When the reranker is enabled this value is ignored, filtering by `rerank.score_threshold` instead."
        ),
    )
    sparse_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        examples=[12],
        description="Sparse (keyword) retrieval candidate count. None = use `config.rag.retrieval.default_sparse_top_k` (current = 12).",
    )
    hybrid_alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        examples=[0.75],
        description="Weight of vector vs keyword scores in hybrid retrieval (0=pure BM25, 1=pure vector). None = use `config.rag.retrieval.default_hybrid_alpha` (current = 0.75).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                # Most common usage — send only the required fields; all 4 retrieval params use config defaults
                {"query": "What is the core conclusion of this document?", "folder_name": "my-folder"},
                # Explicit override (matches config defaults, for easy copy-editing)
                {
                    "query": "What is the core conclusion of this document?",
                    "folder_name": "my-folder",
                    "top_k": 10,
                    "similarity_cutoff": 0.25,
                    "sparse_top_k": 12,
                    "hybrid_alpha": 0.75,
                },
            ]
        }
    }

# Definitions moved down into the domain layer (src/domain/rag/dto.py); re-exported here to preserve existing import paths.
from src.domain.rag.dto import FileRequest  # noqa: E402,F401

class IndexDocumentResponse(BaseModel):
    """Single file indexing response model"""
    file_id: str = Field(..., description="File ID")
    index_id: str = Field(..., description="Index ID")
    num_chunks: int = Field(..., description="Number of chunks created")
    status: str = Field(..., description="Index status (indexed, already_indexed, failed)")
    indexed_at: str = Field(..., description="Index timestamp in ISO format")
    message: str = Field(..., description="Response message")


class FileIndexResult(BaseModel):
    """File index result model (for batch operations)"""
    file_id: str = Field(..., description="File ID")
    filename: str = Field(..., description="Filename")
    status: str = Field(..., description="Processing status (success, failed, skipped)")
    message: str = Field(..., description="Processing message")
    num_chunks: Optional[int] = Field(None, description="Number of chunks (on success)")


class IndexFolderResponse(BaseModel):
    """Folder batch indexing response model"""
    folder_id: int = Field(..., description="Folder ID")
    total_files: int = Field(..., description="Total number of files")
    successful: int = Field(..., description="Number of successfully indexed files")
    failed: int = Field(..., description="Number of failed indexing operations")
    skipped: int = Field(..., description="Number of skipped files")
    results: List[FileIndexResult] = Field(default_factory=list, description="Detailed results list")
    message: str = Field(..., description="Batch processing summary message")


class DeleteFolderIndexResponse(BaseModel):
    """Folder batch delete index response model"""
    folder_id: int = Field(..., description="Folder ID")
    total_files: int = Field(..., description="Total number of files")
    successful: int = Field(..., description="Number of successfully deleted indexes")
    failed: int = Field(..., description="Number of failed delete operations")
    results: List[FileIndexResult] = Field(default_factory=list, description="Detailed results list")
    message: str = Field(..., description="Batch processing summary message")


class IndexJobStatusResponse(BaseModel):
    """Background indexing job status response"""

    job_id: str = Field(..., description="Job identifier")
    folder_id: int = Field(..., description="Target folder ID")
    status: str = Field(
        ...,
        description="Job status (pending, running, succeeded, partial_success, failed, cancelled)",
    )
    total_files: int = Field(..., description="Total number of files scheduled")
    processed_files: int = Field(..., description="Number of files processed so far")
    current_index: int = Field(..., description="Current file position (1-based)")
    current_file_id: Optional[str] = Field(None, description="Current file ID being processed")
    current_file_name: Optional[str] = Field(None, description="Current file name being processed")
    current_file_stage: Optional[str] = Field(
        None,
        description="Current file chunk-level stage: loading / contextualizing / embedding / writing "
                    "(None between files or after the file completes)",
    )
    current_file_chunks_done: Optional[int] = Field(
        None, description="Chunks completed in the current stage of the current file"
    )
    current_file_chunks_total: Optional[int] = Field(
        None, description="Total leaf chunks of the current file (chunk progress done/total)"
    )
    current_file_eta_seconds: Optional[int] = Field(
        None,
        description="Estimated seconds until the current stage finishes, extrapolated from "
                    "the observed chunk rate (None while calibrating or between stages)",
    )
    eta_seconds: Optional[int] = Field(
        None,
        description="Rough estimated seconds until the whole job finishes "
                    "(avg completed-file duration × remaining files; None until the first file completes)",
    )
    last_file_status: Optional[str] = Field(None, description="Status of the last processed file")
    last_message: Optional[str] = Field(None, description="Message from the last processed file")
    message: Optional[str] = Field(None, description="Job level message")
    error: Optional[str] = Field(None, description="Error details when the job fails")
    skip_existing: bool = Field(default=True, description="Whether existing indexes were skipped")
    started_at: str = Field(..., description="Job start timestamp (ISO 8601 format)")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp (ISO 8601 format)")
    last_updated_at: Optional[str] = Field(
        None,
        description="Last heartbeat timestamp (A3 watchdog uses this to detect staleness)",
    )
    result_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Final indexing summary when job succeeds"
    )
    # per-file timings; empty until at least one file has been processed
    file_timings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-file timing audit: [{file_id, file_name, status, total_ms, load_ms, index_ms, chunks}, ...]"
                    " — chunks = number of leaf chunks produced for that file (persisted record)",
    )
    # Part 3 — file_ids this job will/did process (populated for file-scoped jobs only)
    scope_file_ids: List[str] = Field(
        default_factory=list,
        description="Files this job will process (only populated for start_indexing_files calls)",
    )


class FileIndexStatusResponse(BaseModel):
    """Per-file index status response (Part 3).

    Returned by GET /api/rag/file/{file_id}/index/status. The ``status`` field
    is one of: indexed, indexing, queued, failed, not_indexed.
    """

    file_id: str = Field(..., description="File ID")
    file_name: str = Field(..., description="File name")
    status: str = Field(..., description="indexed / indexing / queued / failed / not_indexed")
    num_chunks: Optional[int] = Field(None, description="Chunk count (when indexed)")
    indexed_at: Optional[str] = Field(None, description="ISO timestamp when last indexed")
    job_id: Optional[str] = Field(None, description="Active job id (when indexing/queued)")
    progress: Optional[Dict[str, int]] = Field(
        None, description="Progress snapshot {current_index, total_files}"
    )
    error: Optional[str] = Field(None, description="Last error message (when failed)")


class IndexingMetadata(BaseModel):
    """Auto-indexing metadata response (for file upload endpoints)"""
    auto_index_enabled: bool = Field(..., description="Whether auto-indexing was triggered")
    job_id: Optional[str] = Field(None, description="Background indexing job ID (if auto_index_enabled)")
    status: Optional[str] = Field(None, description="Job status (pending, running, succeeded, failed)")
    total_files: Optional[int] = Field(None, description="Total files scheduled for indexing")
    message: str = Field(..., description="Indexing status message")


# =============================================================================
# Shared envelope + error models — used by endpoints for response_model= / responses= annotations.
# Rule: any successful response uses {"data": ..., "message": ...}, errors use {"detail": ...}
# =============================================================================


class ErrorDetailResponse(BaseModel):
    """Error envelope — the standard shape of a FastAPI HTTPException.

    The middleware also emits this shape for a DomainException.
    Used in endpoint responses={404: {"model": ErrorDetailResponse}} annotations.
    """
    detail: str = Field(..., examples=["folder 99999 not found"], description="Human-readable error message")


# ---- Folder resource ----


class FolderResource(BaseModel):
    """Folder resource (maps to a db.db.Folder ORM row).

    `model_config.from_attributes=True` lets FastAPI accept the adapter's returned
    ``FolderConfigData`` or a SQLAlchemy ORM Folder directly (no need to dict-ify first).
    ``created_at``/``updated_at`` accept a ``datetime`` — Pydantic v2 automatically serializes them
    to ISO strings for the client (no change to the client experience).
    """
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., examples=[1])
    name: str = Field(..., examples=["my-folder"])
    description: Optional[str] = Field(None)
    file_count: Optional[int] = Field(None, description="Number of files in this folder")
    total_size: Optional[int] = Field(None, description="bytes, sum of all files")
    user_token: Optional[str] = Field(None, description="Owner token (returned only to the owner)")
    vector_table_uuid: Optional[str] = Field(None, description="UUID of the corresponding vector store table")
    created_at: Optional[datetime] = Field(None, description="Serialized to an ISO string")
    updated_at: Optional[datetime] = Field(None, description="Serialized to an ISO string")


class FolderDeletedResponse(BaseModel):
    """Successful response for DELETE /api/folders/{id}."""
    detail: str = Field(..., examples=["Folder 1 deleted successfully"])


# ---- File resource ----


class FileResource(BaseModel):
    """File resource (maps to a db.db.File ORM row).

    Handled the same way as FolderResource: `from_attributes=True` is needed to accept ORM /
    BaseModel objects directly; ``id`` accepts a ``UUID`` and serializes to a string;
    ``upload_time`` / ``updated_time`` accept a ``datetime``, which Pydantic v2 auto-converts to
    ISO strings during serialization.
    """
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., examples=["bdb19c41-0012-4400-9172-67c8dbd08d56"])
    folder_id: int = Field(...)
    file_name: str = Field(..., examples=["report.pdf"])
    file_path: str = Field(..., description="Internal storage/{token}/{folder}/{uuid} path; only the download endpoint resolves it to the physical file")
    file_size: int = Field(..., description="bytes")
    mime_type: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    tags: Optional[List[str]] = Field(None)
    content_hash: Optional[str] = Field(None, description="sha256 hex (used for D3 idempotency)")
    upload_time: Optional[datetime] = Field(None, description="Serialized to an ISO string")
    updated_time: Optional[datetime] = Field(None, description="Serialized to an ISO string")


class FileUploadResponse(BaseModel):
    """Response for POST /api/folders/{folder_id}/file — includes file_info + auto-index trigger status."""
    file_info: FileResource
    indexing: IndexingMetadata
    message: str = Field(..., examples=["File uploaded successfully"])


class FileDeletedResponse(BaseModel):
    """Successful response for DELETE /api/folders/{folder_id}/files/{file_id}."""
    success: bool = Field(..., examples=[True])
    message: str = Field(..., examples=["File deleted successfully"])


# ---- Cache / health ----


class HealthStatusResponse(BaseModel):
    """Basic response for GET /health."""
    status: str = Field(..., examples=["healthy"])
    timestamp: str = Field(..., description="ISO timestamp")


class CacheClearResponse(BaseModel):
    """Response for POST /health/cache/clear."""
    status: str = Field(..., examples=["success"])
    message: str = Field(...)
    items_removed: int = Field(...)
    timestamp: str = Field(...)


# ---- Index lifecycle envelopes ----


class IndexJobEnvelope(BaseModel):
    """Envelope for indexing-job-related endpoints: {data: IndexJobStatusResponse, message}."""
    data: IndexJobStatusResponse
    message: str


class FileIndexStatusEnvelope(BaseModel):
    """Envelope for GET /file/{id}/index/status: {data: FileIndexStatusResponse, message}."""
    data: FileIndexStatusResponse
    message: str


class IndexJobCancelResponse(BaseModel):
    """Response for DELETE /index/jobs/{job_id}."""
    data: Dict[str, Any] = Field(
        ...,
        examples=[{"cancelled": True, "job_id": "uuid-..."}],
        description="{cancelled: bool, job_id: str}",
    )
    message: str = Field(..., examples=["Job cancelled"])


class IndexJobListResponse(BaseModel):
    """Response for GET /index/jobs — counts + jobs list."""
    counts: Dict[str, int] = Field(
        ...,
        examples=[{"pending": 0, "running": 1, "succeeded": 5, "partial_success": 0, "failed": 0, "cancelled": 0, "total": 6}],
        description="Job count per status + total",
    )
    jobs: List[IndexJobStatusResponse] = Field(default_factory=list)


class IndexedFileEntry(BaseModel):
    """A single entry within GET /indexed-files."""
    file_id: str
    file_name: str
    num_chunks: int
    indexed_at: str
    status: str = Field(..., description="indexed / failed / deleted")
    embedding_model: Optional[str]
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]


class IndexedFilesResponse(BaseModel):
    """Envelope for GET /indexed-files."""
    data: List[IndexedFileEntry]
    message: str = Field(..., examples=["Found 3 indexed files"])


class IndexDocumentEnvelope(BaseModel):
    """Response envelope for POST /file/{id}/index (sync)."""
    data: IndexDocumentResponse
    message: str = Field(..., examples=["File indexed successfully with 5 chunks"])


# ---- Query ----


class RAGChunkMetadataDoc(BaseModel):
    """Metadata shape for a retrieved chunk hit."""
    node_id: str = Field(..., description="vector store node id")
    mcp_file_id: Optional[str] = Field(None, description="File UUID")
    file_name: Optional[str]
    folder_name: Optional[str]
    page: Optional[int] = Field(None, description="Source page number (only for paginated documents parsed by docling)")
    headings: Optional[List[str]] = Field(None, description="Heading chain the chunk belongs to (citation provenance)")


class RAGSearchResultDoc(BaseModel):
    """A single hit result."""
    text: str = Field(..., description="Matched fragment content")
    score: float = Field(..., description="Ranking score (rerank score when the reranker is enabled)")
    metadata: RAGChunkMetadataDoc


class QueryResponseData(BaseModel):
    """The data block of POST /api/rag/query."""
    query: str
    results: List[RAGSearchResultDoc] = Field(default_factory=list)
    total_results: int
    retrieval_time_ms: Optional[float] = Field(None)


class QueryResponse(BaseModel):
    """POST /api/rag/query envelope."""
    data: QueryResponseData
    message: str = Field(..., examples=["Query returned 5 results"])
