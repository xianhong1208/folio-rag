
"""Domain-level exceptions for the RAG system.

Unified exception hierarchy for business logic errors. The API error-handler
middleware catches these and converts them to HTTP responses, keeping the
domain layer independent of HTTP/FastAPI.
"""

from typing import Optional, Dict, Any


class DomainException(Exception):
    """Base exception for all domain-level errors

    All domain exceptions should inherit from this class.
    The API error handler middleware catches these and converts them
    to appropriate HTTP responses.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code (e.g., "FOLDER_NOT_FOUND")
        details: Optional additional details about the error
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON responses"""
        result = {
            "error": self.error_code,
            "message": self.message
        }
        if self.details:
            result["details"] = self.details
        return result


class ResourceNotFoundError(DomainException):
    """Base class for resource not found errors"""

    def __init__(self, resource_type: str, identifier: Any, message: str = None):
        if message is None:
            message = f"{resource_type} not found: {identifier}"
        super().__init__(
            message=message,
            error_code=f"{resource_type.upper()}_NOT_FOUND",
            details={"resource_type": resource_type, "identifier": str(identifier)}
        )


class FolderNotFoundError(ResourceNotFoundError):
    """Raised when a folder cannot be found"""

    def __init__(self, folder_id: int = None, folder_name: str = None):
        identifier = folder_id if folder_id else folder_name
        super().__init__(resource_type="folder", identifier=identifier)


class RAGFileNotFoundError(ResourceNotFoundError):
    """Raised when a file cannot be found.

    Deliberately not named FileNotFoundError, which would shadow Python's built-in of the same
    name. Any module importing this and writing `except FileNotFoundError` to catch an OS
    file-not-found would accidentally catch this domain class, letting the real OS exception slip
    through and surface as a 500.
    """

    def __init__(self, file_id: str):
        super().__init__(resource_type="file", identifier=file_id)


class FileIndexNotFoundError(ResourceNotFoundError):
    """Raised when a file index record cannot be found"""

    def __init__(self, file_id: str):
        super().__init__(
            resource_type="file_index",
            identifier=file_id,
            message=f"File {file_id} has not been indexed yet"
        )


class UnauthorizedAccessError(DomainException):
    """Raised when a user attempts to access a resource they don't own"""

    def __init__(self, resource_type: str, resource_id: Any, reason: str = None):
        message = f"Not authorized to access {resource_type}: {resource_id}"
        if reason:
            message += f" - {reason}"

        super().__init__(
            message=message,
            error_code="UNAUTHORIZED_ACCESS",
            details={"resource_type": resource_type, "resource_id": str(resource_id)}
        )


class InvalidTokenError(DomainException):
    """Raised when an authentication token is invalid"""

    def __init__(self, reason: str = "Invalid or expired token"):
        super().__init__(message=reason, error_code="INVALID_TOKEN")


class ValidationError(DomainException):
    """Raised when input validation fails"""

    def __init__(self, field: str, message: str, value: Any = None):
        details = {"field": field}
        if value is not None:
            details["invalid_value"] = str(value)

        super().__init__(
            message=f"Validation failed for {field}: {message}",
            error_code="VALIDATION_ERROR",
            details=details
        )


class ConflictError(DomainException):
    """Raised when an operation conflicts with current state"""

    def __init__(self, message: str, resource: str = None):
        details = {}
        if resource:
            details["resource"] = resource
        super().__init__(message=message, error_code="CONFLICT", details=details)


class RAGOperationError(DomainException):
    """Base class for RAG operation failures"""

    def __init__(self, operation: str, reason: str, details: Dict[str, Any] = None):
        super().__init__(
            message=f"RAG {operation} failed: {reason}",
            error_code=f"RAG_{operation.upper()}_ERROR",
            details=details or {}
        )


class FileIndexingError(RAGOperationError):
    """Raised when file indexing fails"""

    def __init__(self, file_id: str, reason: str):
        super().__init__(operation="indexing", reason=reason, details={"file_id": file_id})


class QueryExecutionError(RAGOperationError):
    """Raised when RAG query execution fails"""

    def __init__(self, query: str, reason: str, folder_id: int = None):
        details = {"query": query}
        if folder_id:
            details["folder_id"] = folder_id
        super().__init__(operation="query", reason=reason, details=details)


def get_http_status_for_exception(exception: DomainException) -> int:
    """Map domain exception to HTTP status code"""
    status_mapping = {
        "VALIDATION_ERROR": 400,
        "INVALID_TOKEN": 401,
        "UNAUTHORIZED_ACCESS": 403,
        "FOLDER_NOT_FOUND": 404,
        "FILE_NOT_FOUND": 404,
        "FILE_INDEX_NOT_FOUND": 404,
        "CONFLICT": 409,
        "RAG_INDEXING_ERROR": 500,
        "RAG_QUERY_ERROR": 500,
    }
    return status_mapping.get(exception.error_code, 500)
