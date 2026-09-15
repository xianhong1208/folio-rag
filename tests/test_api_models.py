"""Unit tests for src/api/router/response.py request/response models
(validation, boundaries, serialization)."""

from datetime import datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.api.router.response import (
    UpdateFileRequest,
    UpdateFolderRequest,
    BatchDeleteRequest,
    CreateFolderRequest,
    IndexRequest,
    QueryRequest,
    IndexDocumentResponse,
    IndexJobStatusResponse,
    FolderResource,
    FileResource,
    ErrorDetailResponse,
    HealthStatusResponse,
    IndexJobListResponse,
    QueryResponse,
)

FILE_UUID = "bdb19c41-0012-4400-9172-67c8dbd08d56"


# QueryRequest

def test_query_request_defaults():
    """QueryRequest with only required fields; the 4 tuning fields default to None (deferred to config) (TC-api-01)"""
    req = QueryRequest(query="結論是什麼?", folder_name="my-folder")
    assert req.top_k is None
    assert req.similarity_cutoff is None
    assert req.sparse_top_k is None
    assert req.hybrid_alpha is None


def test_query_request_missing_required():
    """QueryRequest missing query / folder_name -> ValidationError (TC-api-02)"""
    with pytest.raises(ValidationError) as exc_info:
        QueryRequest()
    missing = {e["loc"][0] for e in exc_info.value.errors()}
    assert missing == {"query", "folder_name"}


def test_query_request_top_k_bounds():
    """top_k bounds: 1 and 20 valid; 0 and 21 invalid (TC-api-03)"""
    base = dict(query="q", folder_name="f")
    assert QueryRequest(**base, top_k=1).top_k == 1
    assert QueryRequest(**base, top_k=20).top_k == 20
    with pytest.raises(ValidationError):
        QueryRequest(**base, top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(**base, top_k=21)


def test_query_request_float_bounds():
    """similarity_cutoff / hybrid_alpha limited to 0.0~1.0; out of range is invalid (TC-api-04)"""
    base = dict(query="q", folder_name="f")
    assert QueryRequest(**base, similarity_cutoff=0.0).similarity_cutoff == 0.0
    assert QueryRequest(**base, hybrid_alpha=1.0).hybrid_alpha == 1.0
    with pytest.raises(ValidationError):
        QueryRequest(**base, similarity_cutoff=1.1)
    with pytest.raises(ValidationError):
        QueryRequest(**base, hybrid_alpha=-0.1)


# IndexRequest

def test_index_request_empty_body_ok():
    """IndexRequest empty body is valid; both fields default to None (config defaults used) (TC-api-05)"""
    req = IndexRequest()
    assert req.chunk_size is None
    assert req.chunk_overlap is None


def test_index_request_chunk_size_bounds():
    """chunk_size bounds: 100 and 2000 valid; 99 and 2001 invalid (TC-api-06)"""
    assert IndexRequest(chunk_size=100).chunk_size == 100
    assert IndexRequest(chunk_size=2000).chunk_size == 2000
    with pytest.raises(ValidationError):
        IndexRequest(chunk_size=99)
    with pytest.raises(ValidationError):
        IndexRequest(chunk_size=2001)


def test_index_request_chunk_overlap_bounds():
    """chunk_overlap bounds: 0 and 500 valid; -1 and 501 invalid (TC-api-07)"""
    assert IndexRequest(chunk_overlap=0).chunk_overlap == 0
    assert IndexRequest(chunk_overlap=500).chunk_overlap == 500
    with pytest.raises(ValidationError):
        IndexRequest(chunk_overlap=-1)
    with pytest.raises(ValidationError):
        IndexRequest(chunk_overlap=501)


# File / folder request models

def test_update_requests_all_optional():
    """UpdateFileRequest / UpdateFolderRequest all fields optional, default None (TC-api-08)"""
    file_req = UpdateFileRequest()
    assert file_req.description is None
    assert file_req.tags is None
    folder_req = UpdateFolderRequest()
    assert folder_req.name is None
    assert folder_req.description is None


def test_batch_delete_request_defaults_and_uuid_coercion():
    """BatchDeleteRequest defaults delete_all=False, file_ids=None; strings coerced to UUID (TC-api-09)"""
    req = BatchDeleteRequest()
    assert req.delete_all is False
    assert req.file_ids is None
    req2 = BatchDeleteRequest(file_ids=[FILE_UUID])
    assert req2.file_ids == [UUID(FILE_UUID)]


def test_create_folder_request_name_required():
    """CreateFolderRequest missing required name -> ValidationError (TC-api-10)"""
    assert CreateFolderRequest(name="my-folder").description is None
    with pytest.raises(ValidationError):
        CreateFolderRequest(description="no name")


# Index response models

def test_index_document_response_required():
    """IndexDocumentResponse all six fields required; missing any -> ValidationError (TC-api-11)"""
    resp = IndexDocumentResponse(
        file_id="f-1", index_id="i-1", num_chunks=5,
        status="indexed", indexed_at="2026-01-01T00:00:00", message="ok",
    )
    assert resp.num_chunks == 5
    with pytest.raises(ValidationError):
        IndexDocumentResponse(file_id="f-1")


def test_index_job_status_response_defaults():
    """IndexJobStatusResponse defaults skip_existing=True, file_timings/scope_file_ids empty list (TC-api-12)"""
    resp = IndexJobStatusResponse(
        job_id="j-1", folder_id=1, status="running",
        total_files=3, processed_files=1, current_index=2,
        started_at="2026-01-01T00:00:00",
    )
    assert resp.skip_existing is True
    assert resp.file_timings == []
    assert resp.scope_file_ids == []
    assert resp.completed_at is None
    assert resp.result_summary is None


# from_attributes (ORM-shaped input)

def test_folder_resource_from_orm_shape():
    """FolderResource accepts an ORM-shaped object directly (from_attributes=True) (TC-api-13)"""
    orm_row = SimpleNamespace(
        id=1, name="my-folder", description=None, file_count=3,
        total_size=999, user_token="tok", vector_table_uuid=None,
        created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2),
    )
    res = FolderResource.model_validate(orm_row)
    assert res.id == 1
    assert res.file_count == 3
    assert res.created_at == datetime(2026, 1, 1)


def test_file_resource_from_orm_shape_and_json():
    """FileResource accepts ORM-shaped input; json serialization UUID->str, datetime->ISO (TC-api-14)"""
    orm_row = SimpleNamespace(
        id=UUID(FILE_UUID), folder_id=2, file_name="report.pdf",
        file_path="storage/t/f/x", file_size=1024, mime_type="application/pdf",
        description=None, tags=["a"], content_hash="deadbeef",
        upload_time=datetime(2026, 1, 2, 3, 4, 5), updated_time=datetime(2026, 1, 2, 3, 4, 5),
    )
    res = FileResource.model_validate(orm_row)
    assert res.id == UUID(FILE_UUID)
    dumped = res.model_dump(mode="json")
    assert dumped["id"] == FILE_UUID
    assert dumped["upload_time"] == "2026-01-02T03:04:05"


# envelope / health models

def test_health_status_response_required():
    """HealthStatusResponse requires status and timestamp (TC-api-15)"""
    resp = HealthStatusResponse(status="healthy", timestamp="2026-01-01T00:00:00Z")
    assert resp.status == "healthy"
    with pytest.raises(ValidationError):
        HealthStatusResponse(status="healthy")


def test_index_job_list_response_default_empty_jobs():
    """IndexJobListResponse.jobs defaults to empty list; counts is required (TC-api-16)"""
    resp = IndexJobListResponse(counts={"running": 1, "total": 1})
    assert resp.jobs == []
    with pytest.raises(ValidationError):
        IndexJobListResponse()


def test_query_response_nested_assembly():
    """QueryResponse nested assembly: data.results defaults to empty list, retrieval_time_ms None (TC-api-17)"""
    resp = QueryResponse(
        data={"query": "q", "total_results": 0},
        message="Query returned 0 results",
    )
    assert resp.data.results == []
    assert resp.data.retrieval_time_ms is None
    assert resp.data.total_results == 0


def test_error_detail_response_required():
    """ErrorDetailResponse.detail is required (TC-api-18)"""
    assert ErrorDetailResponse(detail="folder not found").detail == "folder not found"
    with pytest.raises(ValidationError):
        ErrorDetailResponse()
