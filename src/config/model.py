
"""Configuration models.

All Pydantic config models are collected here for reuse across the project.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str
    port: int
    transport: str


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str
    echo: bool
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None


class DynamicToolsConfig(BaseModel):
    """Dynamic tool-filtering configuration.

    enabled: false -> every authenticated token sees all tools
    enabled: true  -> a token only sees the dynamic tools created for it (static tools are visible to everyone)
    """
    enabled: bool = False


class AuthConfig(BaseModel):
    """Bearer-token verification against an OAuth 2.1 authorization server (MCP Center).

    Access tokens are RS256 JWTs verified OFFLINE through the issuer's JWKS; no
    per-request network round-trip is needed once the signing keys are cached.
    `audience` must match the resource URI registered for this server in MCP Center
    (defaults to the server's own MCP URL, i.e. http://<host>:<port>/mcp).
    """
    enabled: bool

    # -- MCP Center (OAuth 2.1 authorization server) --
    issuer: Optional[str] = None  # MCP Center base URL (RS256 JWKS issuer); required when enabled
    audience: Optional[str] = None  # Resource id; empty -> defaults to <server base>/mcp
    required_scopes: List[str] = []  # Scopes every token must carry (empty = none required)

    # DEPRECATED: legacy remote Token Server URL. No longer used in the auth path
    # (kept only so old configs still validate). Use `issuer` instead.
    token_server_url: Optional[str] = None

    # -- Connection / cache settings (JWKS caching still benefits from cache_ttl) --
    cache_ttl: int = 60  # Verification-result cache TTL (seconds)
    request_timeout: float = 5.0  # HTTP request timeout (seconds); e.g. JWKS fetch
    retry_count: int = 2  # Retry count (legacy; unused by the offline verifier)
    dynamic_tools: Optional[DynamicToolsConfig] = None


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str
    format: str
    file: Optional[str]
    max_size: str
    backup_count: int


class AppConfig(BaseModel):
    """Application configuration.

    There is no version field: the single source of truth is [project].version in pyproject.toml,
    parsed by get_version() in src/version.py.
    """
    name: str = "Agentic RAG"
    title: str = "Agentic RAG MCP Server"
    description: str = "Agentic RAG — Hierarchical RAG with auto-merging retrieval"
    lifespan: Optional[str] = None
    # Per-folder MCP tool name prefix (default Agentic_)
    tool_prefix: str = "Agentic"


class RouterConfig(BaseModel):
    """API router configuration."""
    module: str
    router_name: str
    prefix: str


class McpToolsConfig(BaseModel):
    """MCP tools configuration."""
    module: str
    function_name: str


class ModuleConfig(BaseModel):
    """Module configuration."""
    api_router: Optional[RouterConfig]
    mcp_tools: Optional[McpToolsConfig]
    description: str


class ModulesConfig(BaseModel):
    """Top-level modules configuration."""
    model_config = ConfigDict(extra="allow")

    enabled: List[str]


class RAGEmbeddingConfig(BaseModel):
    """RAG embedding configuration."""
    provider: str  # "openai", "azure", "vllm", "ollama"
    model: str
    dimension: int
    api_key: Optional[str] = None  # Direct API key
    api_key_env: Optional[str] = None  # Environment variable name for API key
    base_url: Optional[str] = None  # Also used as azure_endpoint for Azure provider
    api_version: Optional[str] = None  # Azure API version (e.g., "2024-02-01")
    azure_deployment: Optional[str] = None  # Azure deployment name
    # e5-family needs "query: "/"passage: " prefixes; empty (default) for bge-family.
    # Changing a prefix changes the vector space, so existing indexes need a full reindex.
    query_prefix: str = ""
    passage_prefix: str = ""


class RAGChunkingConfig(BaseModel):
    """RAG chunking configuration — supports flat chunking and hierarchical (auto-merging) chunking."""
    # Flat-chunking fields (for flat mode; optional and unused in this project)
    default_chunk_size: Optional[int] = None
    default_overlap: Optional[int] = None
    max_chunk_size: int = 2000
    # Hierarchical-chunking fields
    hierarchy_sizes: Optional[List[int]] = None  # [parent_target_tokens, leaf_chunk_size]
    chunk_overlap: int = 50


class AutoMergingConfig(BaseModel):
    """Auto-Merging Retriever configuration."""
    enabled: bool = True
    merge_threshold: float = 0.5  # promote to the parent once the hit ratio >= this value


class RAGRetrievalConfig(BaseModel):
    """RAG retrieval configuration."""
    default_top_k: int
    default_similarity_cutoff: float
    default_sparse_top_k: int = 12
    default_hybrid_alpha: float = 0.75
    hybrid_search: bool
    text_search_config: str = "simple"  # CKIP segments in Python; Postgres tokenizes the space-joined result with `simple`
    # "rrf" = Reciprocal Rank Fusion (BM25 affects ordering); "concat" = llama_index native
    # (dense dominates, effectively vector-only; legacy).
    hybrid_fusion: str = "rrf"
    rrf_k: int = 60  # RRF constant (larger = smoother; 60 is the common convention)
    return_resource_files: bool = False
    auto_merging: Optional[AutoMergingConfig] = None
    expand_context_default: bool = True
    expand_context_neighbors: int = 2
    # Max concurrent queries (REST + MCP share one semaphore; excess is queued). 8 keeps every
    # connection pool (5+10 each) within headroom.
    max_concurrent_queries: int = 8


class RAGVectorStoreConfig(BaseModel):
    """RAG vector-store configuration."""
    type: str
    table_prefix: str


class RAGLLMConfig(BaseModel):
    """RAG LLM configuration (for LLM-dependent features such as Contextual Retrieval)."""
    provider: str = "azure"  # "azure", "openai", "vllm", "ollama"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # Azure endpoint
    api_version: Optional[str] = "2024-12-01-preview"
    azure_deployment: Optional[str] = None


class ContextualRetrievalConfig(BaseModel):
    """Contextual Retrieval configuration."""
    enabled: bool = False
    max_context_length: int = 150
    max_concurrent: int = 5
    max_doc_chars: int = 60000
    max_tokens: int = 1024
    reasoning_effort: Optional[str] = None
    # Choose whether to add the context prefix to leaf / parent / all chunks
    apply_to: str = "all"  # "leaf" / "parent" / "all"


class RerankConfig(BaseModel):
    """Reranker configuration."""
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    base_url: str = "http://localhost:8787"
    api_key: Optional[str] = None
    top_n: Optional[int] = None  # keep the top N results after reranking (None = keep all)
    score_threshold: float = 0.0  # rerank score threshold
    # Input template applied before POST /v1/score; {text} is the placeholder. Instruction-tuned
    # rerankers (e.g. mxbai) need the official template for a sane score distribution; traditional
    # cross-encoders (bge-family) keep the default = raw text sent through as-is.
    query_template: str = "{text}"
    document_template: str = "{text}"


class DoclingConfig(BaseModel):
    """Docling document-parsing configuration."""
    device: str = "cpu"  # "cpu", "cuda", "cuda:0", "cuda:4", etc.
    ocr_enabled: bool = True  # whether to enable OCR (needed for scanned PDFs)
    # OCR model tier (PP-OCRv6): tiny / small / medium (medium most accurate, slower).
    # Weights at assets/docling_models/RapidOcr/torch/PP-OCRv6/{det,rec}/...; missing files fail at load.
    ocr_model_scale: str = "small"
    hf_offline: bool = False  # offline-deployment switch: True -> set HF_HUB_OFFLINE=1, and fail startup if a local model/tokenizer isn't found under assets/
    compile_models: bool = False  # torch.compile model optimization (faster inference but ~120s slower startup; suits rarely-restarted environments)


class AsrConfig(BaseModel):
    """Audio-transcription (ASR) source selection.

    Only the minimal field set is exposed (every config field is a permanent maintenance surface);
    trimming/hallucination filtering stays always-on in code (audio_defense). provider maps to the
    factory in src/domain/rag/asr_provider.py.
    """
    enabled: bool = True                    # False = don't transcribe audio (even if the model is present)
    # docling-whisper (local) | openai-compatible (cloud/self-hosted HTTP;
    # compatible with OpenAI / Groq / vLLM whisper / faster-whisper-server) |
    # fireredasr (local FireRedASR-AED-L, Traditional-Chinese output; weights see fireredasr_provider.py)
    provider: str = "docling-whisper"
    # ---- Cloud (openai-compatible) only: POST {base_url}/audio/transcriptions ----
    base_url: Optional[str] = None          # e.g. http://host:8000/v1
    api_key: Optional[str] = None
    model: Optional[str] = None             # cloud model name, e.g. whisper-1


class IndexingConfig(BaseModel):
    """Behavior control for background indexing jobs."""
    # Max concurrent files per job. Kept low (4) because docling convert is serialized by
    # DOCLING_INFER_LOCK; a high value would starve the shared executor. Override via
    # RAG_INDEXING_CONCURRENCY.
    max_concurrent_jobs: int = 4

    # Per-file limit (seconds); override via RAG_PER_FILE_TIMEOUT_SECONDS. Default is effectively
    # unlimited (10 days), relying on the 60s keepalive heartbeat instead of a hard timeout.
    per_file_timeout_seconds: int = 864000  # 10 days

    # Whole-job limit (seconds); override via RAG_JOB_TIMEOUT_SECONDS. 0/negative disables.
    # Must be >= per_file_timeout_seconds or the job dies before its files finish.
    job_timeout_seconds: int = 864000  # 10 days


class RAGConfig(BaseModel):
    """RAG system configuration."""
    enabled: bool = True
    embedding: RAGEmbeddingConfig
    chunking: RAGChunkingConfig
    retrieval: RAGRetrievalConfig
    vector_store: RAGVectorStoreConfig
    llm: Optional[RAGLLMConfig] = None
    contextual_retrieval: Optional[ContextualRetrievalConfig] = None
    rerank: Optional[RerankConfig] = None
    docling: Optional[DoclingConfig] = None
    asr: Optional[AsrConfig] = None
    indexing: Optional[IndexingConfig] = None


class ConfigModel(BaseModel):
    """Complete configuration model."""
    server: ServerConfig
    database: DatabaseConfig
    auth: AuthConfig
    logging: LoggingConfig
    app: AppConfig
    modules: ModulesConfig
    rag: Optional[RAGConfig] = None


__all__ = [
    "ServerConfig",
    "DatabaseConfig",
    "DynamicToolsConfig",
    "AuthConfig",
    "LoggingConfig",
    "AppConfig",
    "RouterConfig",
    "McpToolsConfig",
    "ModuleConfig",
    "ModulesConfig",
    "RAGEmbeddingConfig",
    "RAGChunkingConfig",
    "AutoMergingConfig",
    "RAGRetrievalConfig",
    "RAGVectorStoreConfig",
    "RAGLLMConfig",
    "ContextualRetrievalConfig",
    "RerankConfig",
    "DoclingConfig",
    "RAGConfig",
    "ConfigModel",
]
