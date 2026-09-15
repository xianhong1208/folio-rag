"""Unit tests for src/domain/exceptions.py — the domain exception hierarchy and HTTP status mapping.

No dependency on the DB / vLLM / any external service — pure logic tests.
Related docs: docs/testing/specs/SPEC-exceptions.md, docs/testing/test-cases/TC-exceptions.md
"""

import pytest

from src.domain.exceptions import (
    DomainException,
    ResourceNotFoundError,
    FolderNotFoundError,
    RAGFileNotFoundError as DomainFileNotFoundError,  # renamed to no longer shadow the builtin
    FileIndexNotFoundError,
    UnauthorizedAccessError,
    InvalidTokenError,
    ValidationError,
    ConflictError,
    RAGOperationError,
    FileIndexingError,
    QueryExecutionError,
    get_http_status_for_exception,
)


def test_domain_exception_attributes_and_default_details():
    """message / error_code are stored correctly; details defaults to an empty dict when not given (TC-exceptions-01)."""
    exc = DomainException(message="something broke", error_code="SOME_CODE")
    assert exc.message == "something broke"
    assert exc.error_code == "SOME_CODE"
    assert exc.details == {}


def test_domain_exception_str_and_inheritance():
    """str(exc) equals message, and it is a standard Exception subclass (TC-exceptions-02)."""
    exc = DomainException(message="msg here", error_code="X")
    assert str(exc) == "msg here"
    assert isinstance(exc, Exception)


def test_to_dict_without_details():
    """When details is empty, to_dict contains only the error and message keys (TC-exceptions-03)."""
    exc = DomainException(message="m", error_code="CODE")
    assert exc.to_dict() == {"error": "CODE", "message": "m"}


def test_to_dict_with_details():
    """When details is non-empty, to_dict includes the details key (TC-exceptions-04)."""
    exc = DomainException(message="m", error_code="CODE", details={"k": "v"})
    assert exc.to_dict() == {"error": "CODE", "message": "m", "details": {"k": "v"}}


def test_resource_not_found_default_message_and_code():
    """Auto-composed message, error_code of {TYPE}_NOT_FOUND, and details with the stringified identifier (TC-exceptions-05)."""
    exc = ResourceNotFoundError(resource_type="folder", identifier=42)
    assert exc.message == "folder not found: 42"
    assert exc.error_code == "FOLDER_NOT_FOUND"
    assert exc.details == {"resource_type": "folder", "identifier": "42"}


def test_resource_not_found_custom_message():
    """A custom message overrides the default format (TC-exceptions-06)."""
    exc = ResourceNotFoundError(resource_type="file", identifier="abc", message="custom words")
    assert exc.message == "custom words"
    assert exc.error_code == "FILE_NOT_FOUND"


def test_folder_not_found_by_id():
    """When built from folder_id, the identifier uses the id (TC-exceptions-07)."""
    exc = FolderNotFoundError(folder_id=7)
    assert exc.error_code == "FOLDER_NOT_FOUND"
    assert exc.details["identifier"] == "7"


def test_folder_not_found_by_name():
    """When only folder_name is given, the identifier uses the name (TC-exceptions-08)."""
    exc = FolderNotFoundError(folder_name="財報")
    assert exc.details["identifier"] == "財報"
    assert "財報" in exc.message


def test_file_not_found():
    """FILE_NOT_FOUND code with the file_id identifier (TC-exceptions-09)."""
    exc = DomainFileNotFoundError(file_id="f-123")
    assert exc.error_code == "FILE_NOT_FOUND"
    assert exc.details == {"resource_type": "file", "identifier": "f-123"}


def test_file_index_not_found():
    """FILE_INDEX_NOT_FOUND code with a "not yet indexed" message (TC-exceptions-10)."""
    exc = FileIndexNotFoundError(file_id="f-9")
    assert exc.error_code == "FILE_INDEX_NOT_FOUND"
    assert exc.message == "File f-9 has not been indexed yet"
    assert exc.details["identifier"] == "f-9"


def test_unauthorized_without_reason():
    """Without a reason, the message contains only the resource information (TC-exceptions-11)."""
    exc = UnauthorizedAccessError(resource_type="folder", resource_id=5)
    assert exc.error_code == "UNAUTHORIZED_ACCESS"
    assert exc.message == "Not authorized to access folder: 5"
    assert exc.details == {"resource_type": "folder", "resource_id": "5"}


def test_unauthorized_with_reason():
    """When a reason is given, it is appended to the end of the message (TC-exceptions-12)."""
    exc = UnauthorizedAccessError(resource_type="file", resource_id="x", reason="not owner")
    assert exc.message == "Not authorized to access file: x - not owner"


def test_invalid_token_default_and_custom_reason():
    """INVALID_TOKEN code; both the default and a custom reason serve directly as the message (TC-exceptions-13)."""
    default_exc = InvalidTokenError()
    assert default_exc.error_code == "INVALID_TOKEN"
    assert default_exc.message == "Invalid or expired token"
    custom_exc = InvalidTokenError(reason="token revoked")
    assert custom_exc.message == "token revoked"


def test_validation_error_with_value():
    """When value is not None, details contains the stringified invalid_value (TC-exceptions-14)."""
    exc = ValidationError(field="top_k", message="must be positive", value=-3)
    assert exc.error_code == "VALIDATION_ERROR"
    assert exc.message == "Validation failed for top_k: must be positive"
    assert exc.details == {"field": "top_k", "invalid_value": "-3"}


def test_validation_error_without_value():
    """When value is None, details does not contain invalid_value (TC-exceptions-15)."""
    exc = ValidationError(field="query", message="empty")
    assert exc.details == {"field": "query"}
    assert "invalid_value" not in exc.details


def test_conflict_with_resource():
    """When resource is given, it goes into details (TC-exceptions-16)."""
    exc = ConflictError(message="already indexing", resource="file f-1")
    assert exc.error_code == "CONFLICT"
    assert exc.details == {"resource": "file f-1"}


def test_conflict_without_resource():
    """Without a resource, details is an empty dict (TC-exceptions-17)."""
    exc = ConflictError(message="state conflict")
    assert exc.details == {}


def test_rag_operation_error_code_from_operation():
    """error_code is composed as RAG_{OP}_ERROR from the uppercased operation (TC-exceptions-18)."""
    exc = RAGOperationError(operation="indexing", reason="disk full")
    assert exc.error_code == "RAG_INDEXING_ERROR"
    assert exc.message == "RAG indexing failed: disk full"
    assert exc.details == {}


def test_file_indexing_error():
    """RAG_INDEXING_ERROR code, with file_id in details (TC-exceptions-19)."""
    exc = FileIndexingError(file_id="f-7", reason="parse failed")
    assert exc.error_code == "RAG_INDEXING_ERROR"
    assert exc.details == {"file_id": "f-7"}
    assert "parse failed" in exc.message


def test_query_execution_error_with_folder():
    """When folder_id is given, details contains both query and folder_id (TC-exceptions-20)."""
    exc = QueryExecutionError(query="財報重點", reason="timeout", folder_id=3)
    assert exc.error_code == "RAG_QUERY_ERROR"
    assert exc.details == {"query": "財報重點", "folder_id": 3}


def test_query_execution_error_without_folder():
    """Without folder_id, details contains only query (TC-exceptions-21)."""
    exc = QueryExecutionError(query="q", reason="r")
    assert exc.details == {"query": "q"}
    assert "folder_id" not in exc.details


@pytest.mark.parametrize(
    "exc, expected_status",
    [
        (ValidationError(field="f", message="bad"), 400),
        (InvalidTokenError(), 401),
        (UnauthorizedAccessError(resource_type="folder", resource_id=1), 403),
        (FolderNotFoundError(folder_id=1), 404),
        (DomainFileNotFoundError(file_id="f1"), 404),
        (FileIndexNotFoundError(file_id="f1"), 404),
        (ConflictError(message="c"), 409),
        (FileIndexingError(file_id="f1", reason="r"), 500),
        (QueryExecutionError(query="q", reason="r"), 500),
    ],
    ids=[
        "validation-400", "token-401", "unauthorized-403",
        "folder-404", "file-404", "file-index-404",
        "conflict-409", "indexing-500", "query-500",
    ],
)
def test_http_status_mapping(exc, expected_status):
    """Each known error_code maps to its corresponding HTTP status (TC-exceptions-22)."""
    assert get_http_status_for_exception(exc) == expected_status


def test_http_status_unknown_code_defaults_to_500():
    """An unknown error_code (including unlisted RAG operations) always returns 500 (TC-exceptions-23)."""
    assert get_http_status_for_exception(DomainException("m", "TOTALLY_UNKNOWN")) == 500
    # RAG_EMBEDDING_ERROR is not in the mapping table and also falls through to the default 500
    assert get_http_status_for_exception(RAGOperationError("embedding", "r")) == 500


def test_all_exceptions_inherit_domain_exception():
    """The whole family are DomainException subclasses, catchable by a single except (TC-exceptions-24)."""
    instances = [
        ResourceNotFoundError("folder", 1),
        FolderNotFoundError(folder_id=1),
        DomainFileNotFoundError(file_id="f"),
        FileIndexNotFoundError(file_id="f"),
        UnauthorizedAccessError("file", "x"),
        InvalidTokenError(),
        ValidationError("f", "m"),
        ConflictError("m"),
        RAGOperationError("query", "r"),
        FileIndexingError("f", "r"),
        QueryExecutionError("q", "r"),
    ]
    for exc in instances:
        assert isinstance(exc, DomainException)
