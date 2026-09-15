
"""RAG Query REST API

Handles retrieval queries only:
- POST /api/rag/query                query indexed content (via hybrid + rerank)

Indexing endpoints have moved to rag_indexing.py.
Note: this endpoint uses RAGAdapter.query_rag (flat-style hybrid retrieval). For Auto-Merging +
context expansion, use the MCP tool (Agentic_{folder_name}), or extend this endpoint with a
mode parameter in a future version.

Defaults: if the four retrieval parameters (top_k / similarity_cutoff / sparse_top_k /
hybrid_alpha) are not provided, they are read live from `rag.retrieval.*` in `config.yaml`,
staying consistent with the server's overall config.
"""

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from db.cached_folderdb import CachedFolderDB
from src.adapter.rag import get_rag_adapter
from src.api.dependencies.auth import extract_token
from src.api.router.response import ErrorDetailResponse, QueryRequest, QueryResponse
from src.auth.dependencies import authenticate_request
from src.config.config_manager import Config
from src.domain.exceptions import FolderNotFoundError
from src.log import get_api_logger, log_err, log_op, log_warn


_QUERY_REQUEST_EXAMPLES = {
    "minimal": {
        "summary": "Minimal usage (config defaults)",
        "description": "Only required fields. The 4 retrieval params are read from `config.rag.retrieval.default_*`.",
        "value": {
            "query": "What is the core conclusion of this document?",
            "folder_name": "my-folder",
        },
    },
    "explicit_match_config": {
        "summary": "Explicitly match current config values",
        "description": "Copy the config defaults into the body to tune a single retrieval param.",
        "value": {
            "query": "What is the core conclusion of this document?",
            "folder_name": "my-folder",
            "top_k": 10,
            "similarity_cutoff": 0.25,
            "sparse_top_k": 12,
            "hybrid_alpha": 0.75,
        },
    },
    "vector_heavy": {
        "summary": "Vector-only retrieval (BM25 off)",
        "description": "hybrid_alpha=1.0 uses vector scores only, ignoring keywords. Good for semantically close but differently worded queries.",
        "value": {
            "query": "What is the main conclusion?",
            "folder_name": "my-folder",
            "hybrid_alpha": 1.0,
        },
    },
    "keyword_heavy": {
        "summary": "Keyword-only retrieval (vector off)",
        "description": "hybrid_alpha=0.0 uses BM25 only. Good for exact terms (e.g. model numbers, names).",
        "value": {
            "query": "AMD AI 395",
            "folder_name": "my-folder",
            "hybrid_alpha": 0.0,
            "sparse_top_k": 30,
        },
    },
}

logger = get_api_logger()

router = APIRouter(tags=["RAG: Query"], dependencies=[Depends(authenticate_request)])


# Hard fallback if config can't be loaded at all — same values as the Pydantic
# defaults shipped before this commit, so behavior degrades gracefully.
_FALLBACK_RETRIEVAL_DEFAULTS = {
    "top_k": 5,
    "similarity_cutoff": 0.4,
    "sparse_top_k": 12,
    "hybrid_alpha": 0.75,
}


def _resolve_retrieval_defaults() -> dict:
    """Read retrieval defaults live from config.yaml.

    Returns:
        A dict with top_k / similarity_cutoff / sparse_top_k / hybrid_alpha; falls back to the
        module constant _FALLBACK_RETRIEVAL_DEFAULTS for any field missing from config.
    """
    out = dict(_FALLBACK_RETRIEVAL_DEFAULTS)
    try:
        cfg = Config.get_config_model()
        ret = getattr(getattr(cfg, "rag", None), "retrieval", None)
        if ret is None:
            return out
        if getattr(ret, "default_top_k", None) is not None:
            out["top_k"] = ret.default_top_k
        if getattr(ret, "default_similarity_cutoff", None) is not None:
            out["similarity_cutoff"] = ret.default_similarity_cutoff
        if getattr(ret, "default_sparse_top_k", None) is not None:
            out["sparse_top_k"] = ret.default_sparse_top_k
        if getattr(ret, "default_hybrid_alpha", None) is not None:
            out["hybrid_alpha"] = ret.default_hybrid_alpha
    except Exception as e:
        logger.warning(f"Failed to read rag.retrieval defaults from config (using fallback): {e}")
    return out


# Response = {data: {query, results[], total_results, retrieval_time_ms}, message}
# results use hybrid (vector + BM25) + optional rerank; when the reranker is enabled, score is the rerank score
@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder_name does not exist or no permission"},
        422: {"model": ErrorDetailResponse, "description": "request body validation failed"},
        500: {"model": ErrorDetailResponse},
    },
)
async def query_rag(
    request: Request,
    query_request: QueryRequest = Body(..., openapi_examples=_QUERY_REQUEST_EXAMPLES),
    token: str = Depends(extract_token),
):
    """Query by folder name + natural language — hybrid retrieval + rerank

    Args:
        query_request: query / folder_name / top_k / similarity_cutoff /
                       sparse_top_k / hybrid_alpha — missing fields use config defaults
        token: User token (auto-extracted from Authorization header)

    Returns:
        {data: {query, results[], total_results}, message}
    """
    try:
        # Resolve config-driven defaults for any missing retrieval params so
        # API behavior tracks config.yaml instead of stale Pydantic literals.
        defaults = _resolve_retrieval_defaults()
        top_k = query_request.top_k if query_request.top_k is not None else defaults["top_k"]
        similarity_cutoff = (
            query_request.similarity_cutoff
            if query_request.similarity_cutoff is not None
            else defaults["similarity_cutoff"]
        )
        sparse_top_k = (
            query_request.sparse_top_k
            if query_request.sparse_top_k is not None
            else defaults["sparse_top_k"]
        )
        hybrid_alpha = (
            query_request.hybrid_alpha
            if query_request.hybrid_alpha is not None
            else defaults["hybrid_alpha"]
        )

        log_op(
            logger, "API_QUERY", token=token,
            msg=(
                f"folder={query_request.folder_name} "
                f"query='{query_request.query[:50]}' "
                f"top_k={top_k} cutoff={similarity_cutoff} "
                f"sparse={sparse_top_k} alpha={hybrid_alpha}"
            ),
        )

        folder = CachedFolderDB.get_by_name_and_token(query_request.folder_name, token)
        if not folder:
            raise FolderNotFoundError(folder_name=query_request.folder_name)

        import time as _time
        _t0 = _time.perf_counter()
        result = await get_rag_adapter().query_rag(
            query=query_request.query,
            folder_id=folder.id,
            token=token,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
            sparse_top_k=sparse_top_k,
            hybrid_alpha=hybrid_alpha,
        )
        _latency_ms = int((_time.perf_counter() - _t0) * 1000)

        # Query telemetry (best-effort; feeds Control Center's top-queries/zero-result/latency analytics)
        try:
            from db.query_log_db import QueryLogDB
            from src.log import mask_token
            QueryLogDB.record(
                query=query_request.query, folder_id=folder.id,
                result_count=result.total_results, latency_ms=_latency_ms,
                token_prefix=mask_token(token) if token else None)
        except Exception as _e:  # noqa: BLE001
            logger.debug(f"query log skipped: {_e}")

        return {
            "data": result.model_dump(),
            "message": f"Query returned {result.total_results} results",
        }

    except FolderNotFoundError as e:
        log_warn(logger, "API_QUERY_404", token=token, msg=f"folder={query_request.folder_name}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_QUERY_FAIL", e, token=token)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
