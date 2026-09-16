
"""Shared domain-layer DTOs.

These are pure data types (pydantic) shared across the domain / adapter / api layers. They
previously lived in adapter.model and api.router.response, which forced the domain (query_engine /
index_service) to import outward -> inverted layering and potential import cycles.

They are now defined in the innermost domain layer, and the outer layers (adapter.model /
api.router.response) re-export them to keep existing import paths unchanged (backward compatible).
Dependencies now always point inward.
"""

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class FileRequest(BaseModel):
    """DTO for a file to be indexed (produced by IndexingService from the File ORM, consumed by the indexing pipeline)."""
    id: UUID
    folder_id: int
    file_name: str
    file_path: str
    file_size: int
    mime_type: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    # sha256(content). Populated by IndexingService._to_file_request from the
    # File ORM row; consumed by RAGIndexingService for idempotent re-index check.
    content_hash: Optional[str] = None


class RAGChunkMetadata(BaseModel):
    """Metadata for a single chunk in a RAG retrieval result."""
    node_id: UUID
    mcp_file_id: str
    file_name: str
    folder_name: Optional[str] = None
    # Citation provenance (only present for data indexed via the docling path; None for old indexes/plain text)
    page: Optional[int] = None
    headings: Optional[List[str]] = None
    # "table" / "picture" when the chunk carries that Docling item; None for plain text.
    content_type: Optional[str] = None


class RAGSearchResult(BaseModel):
    """A single RAG retrieval result."""
    text: str
    score: float
    metadata: RAGChunkMetadata
