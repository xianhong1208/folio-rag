"""Unit tests for src/adapter/model.py Pydantic DTOs (validation/serialization)."""

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.adapter.model import (
    FileConfigData,
    FileDownloadData,
    FolderConfigData,
    RAGChunkMetadata,
    RAGSearchResult,
    RAGQueryResult,
    FileIndexData,
)

FILE_UUID = "bdb19c41-0012-4400-9172-67c8dbd08d56"
ISO_TIME = "2026-01-02T03:04:05"


def _file_config_kwargs(**overrides):
    kwargs = dict(
        id=FILE_UUID,
        folder_id=1,
        file_name="report.pdf",
        file_path="storage/t/f/x",
        file_size=1024,
        upload_time=ISO_TIME,
        updated_time=ISO_TIME,
    )
    kwargs.update(overrides)
    return kwargs


def test_file_config_data_coercion():
    """FileConfigData auto-coerces string UUID and ISO string into UUID/datetime (TC-adapter-01)"""
    data = FileConfigData(**_file_config_kwargs())
    assert data.id == UUID(FILE_UUID)
    assert isinstance(data.upload_time, datetime)
    assert data.upload_time.year == 2026


def test_file_config_data_missing_required():
    """FileConfigData missing required fields -> ValidationError (TC-adapter-02)"""
    with pytest.raises(ValidationError) as exc_info:
        FileConfigData(id=FILE_UUID, folder_id=1)
    missing = {e["loc"][0] for e in exc_info.value.errors()}
    assert {"file_name", "file_path", "file_size", "upload_time", "updated_time"} <= missing


def test_file_config_data_optional_defaults():
    """FileConfigData mime_type/description/tags default to None (TC-adapter-03)"""
    data = FileConfigData(**_file_config_kwargs())
    assert data.mime_type is None
    assert data.description is None
    assert data.tags is None


def test_file_download_data_bytes_content():
    """FileDownloadData carries bytes file content (TC-adapter-04)"""
    data = FileDownloadData(
        id=FILE_UUID, folder_id=1, file_name="a.bin",
        file_content=b"\x00\x01binary", file_size=8,
        upload_time=ISO_TIME, updated_time=ISO_TIME,
    )
    assert data.file_content == b"\x00\x01binary"
    assert isinstance(data.file_content, bytes)


def test_folder_config_data_defaults():
    """FolderConfigData defaults file_count=0, total_size=0, user_token=None (TC-adapter-05)"""
    data = FolderConfigData(
        id=1, name="my-folder", created_at=ISO_TIME, updated_at=ISO_TIME,
    )
    assert data.file_count == 0
    assert data.total_size == 0
    assert data.user_token is None
    assert data.description is None


def test_rag_chunk_metadata_optional_folder():
    """RAGChunkMetadata requires node_id/mcp_file_id/file_name; folder_name is optional (TC-adapter-06)"""
    meta = RAGChunkMetadata(node_id=FILE_UUID, mcp_file_id="f-1", file_name="a.txt")
    assert meta.node_id == UUID(FILE_UUID)
    assert meta.folder_name is None


def test_rag_search_result_nested_assembly():
    """RAGSearchResult nested metadata can be assembled directly from a dict (TC-adapter-07)"""
    result = RAGSearchResult(
        text="命中片段",
        score=0.87,
        metadata={"node_id": FILE_UUID, "mcp_file_id": "f-1", "file_name": "a.txt"},
    )
    assert isinstance(result.metadata, RAGChunkMetadata)
    assert result.score == pytest.approx(0.87)


def test_rag_query_result_defaults_and_list():
    """RAGQueryResult carries a results list; retrieval_time_ms defaults to None (TC-adapter-08)"""
    result = RAGQueryResult(
        query="q",
        results=[
            {
                "text": "t", "score": 0.5,
                "metadata": {"node_id": FILE_UUID, "mcp_file_id": "f", "file_name": "a"},
            }
        ],
        total_results=1,
    )
    assert result.retrieval_time_ms is None
    assert len(result.results) == 1
    assert isinstance(result.results[0], RAGSearchResult)


def test_file_index_data_json_serialization():
    """FileIndexData json-mode serialization: UUID->str, datetime->ISO string (TC-adapter-09)"""
    data = FileIndexData(
        file_id=FILE_UUID, file_name="a.txt", folder_id=1, num_chunks=5,
        indexed_at=ISO_TIME, status="indexed", embedding_model="bge-m3",
        chunk_size=256, chunk_overlap=50,
    )
    dumped = data.model_dump(mode="json")
    assert dumped["file_id"] == FILE_UUID
    assert dumped["indexed_at"] == ISO_TIME


def test_rag_search_result_invalid_score_type():
    """RAGSearchResult score given a non-number -> ValidationError (TC-adapter-10)"""
    with pytest.raises(ValidationError):
        RAGSearchResult(
            text="t", score="not-a-number",
            metadata={"node_id": FILE_UUID, "mcp_file_id": "f", "file_name": "a"},
        )
