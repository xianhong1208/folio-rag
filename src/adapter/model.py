
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from uuid import UUID


class FileConfigData(BaseModel):
    """File configuration information model."""
    id: UUID
    folder_id: int
    file_name: str
    file_path: str
    file_size: int
    mime_type: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    upload_time: datetime
    updated_time: datetime

class FileDownloadData(BaseModel):
    """File-download model, including the file content."""
    id: UUID
    folder_id: int
    file_name: str
    file_content: bytes
    file_size: int
    mime_type: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    upload_time: datetime
    updated_time: datetime



class FolderConfigData(BaseModel):
    """Folder information model."""
    id: int
    name: str
    description: Optional[str] = None
    file_count: int = 0
    total_size: int = 0
    user_token: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# The definitions live in the domain layer (src/domain/rag/dto.py); re-exported
# here to preserve the existing import path.
from src.domain.rag.dto import RAGChunkMetadata, RAGSearchResult  # noqa: E402,F401


class RAGQueryResult(BaseModel):
    """RAG query response model."""
    query: str
    results: List[RAGSearchResult]
    total_results: int
    retrieval_time_ms: Optional[float] = None


class FileIndexData(BaseModel):
    """File index information model."""
    file_id: UUID
    file_name: str
    folder_id: int
    num_chunks: int
    indexed_at: datetime
    status: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


