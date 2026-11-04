
"""Configuration models.

All Pydantic config models are collected here for reuse across the project.
"""

import logging as _logging
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

# Secrets are excluded from repr()/str() (repr=False), so logging a whole config
# object, or an exception message that embeds one, can never print them.


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str
    port: int
    transport: str


class RetentionConfig(BaseModel):
    """How long append-only tables keep rows (days; 0 = keep for ever).

    Purged at startup and then every ``interval_hours`` (db/retention_db.py),
    ``batch_size`` rows per transaction.
    """
    index_jobs_days: int = Field(30, ge=0)        # terminal IndexJobs rows only
    query_logs_days: int = Field(90, ge=0)        # the console analytics window goes up to 90 days
    settings_audit_days: int = Field(365, ge=0)
    batch_size: int = Field(5000, ge=100, le=100000)
    interval_hours: float = Field(24, gt=0, le=24 * 7)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(repr=False)  # usually embeds the password
    echo: bool
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = Field(None, repr=False)
    # How each folder's PGVectorStore connects. "shared" (default): every store reuses
    # the two shared engines above (pool_size + max_overflow connections each, no
    # matter how many folders exist). "per_folder": each cached store gets its own
    # small sync + async pools (1 + 2 connections each), disposed on cache eviction.
    vector_store_engines: Literal["shared", "per_folder"] = "shared"
    # Max PGVectorStore objects kept in the LRU cache (least recently used evicted).
    vector_store_cache_size: int = Field(64, ge=1, le=10000)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)


class DynamicToolsConfig(BaseModel):
    """Dynamic tool-filtering configuration.

    enabled: false -> every authenticated token sees all tools
    enabled: true  -> a token only sees the dynamic tools created for it (static tools are visible to everyone)
    """
    enabled: bool = False


class ConsoleAuthConfig(BaseModel):
    """Admin-console login (``auth.console``). Every field has a default, so existing
    config files need no edits.

    The console signs in by pasting an MCP Gate access token (a PAT) issued for this
    service; the server verifies it once and sets a stateless, HS256-signed session
    cookie carrying only the token's identity (jti / sub / scopes), never the token.
    """
    # Session lifetime cap; a session never outlives the token it was created from.
    session_hours: float = Field(8, gt=0)
    # HMAC key for the session cookie (supports ${ENV_VAR}). Empty -> a random key is
    # generated once and persisted under storage/ so sessions survive restarts.
    session_secret: Optional[str] = Field(None, repr=False)
    # Cookie Secure flag: "auto" = only when the request arrived over https.
    secure_cookie: Union[bool, Literal["auto"]] = "auto"
    # Scope that unlocks global changes (runtime settings, connectivity probes, audit log).
    admin_scope: str = "rag:admin"


class RevocationConfig(BaseModel):
    """MCP Gate's revocation feed (``auth.revocation``).

    Folio verifies tokens offline, so without this a token revoked in MCP Gate keeps
    working until it expires. When enabled, a background task polls
    ``<issuer>/oauth/revoked`` and every verification path rejects listed tokens.
    The credentials belong to a confidential client registered in the MCP Gate
    console and bound to this server as its resource server.
    """
    enabled: bool = False
    client_id: Optional[str] = None
    # Must be an ${ENV_VAR} reference; a literal secret in config.yaml is refused at startup.
    client_secret: Optional[str] = Field(None, repr=False)
    poll_interval_sec: float = Field(30, ge=5, le=3600)
    # "open": keep accepting tokens when the feed is unreachable (logged loudly);
    # "closed": reject every token once no poll has succeeded for
    # stale_after_intervals x poll_interval_sec.
    fail_mode: Literal["open", "closed"] = "open"
    stale_after_intervals: int = Field(4, ge=2, le=100)


class AuthConfig(BaseModel):
    """Bearer-token verification against an OAuth 2.1 authorization server (MCP Gate).

    Access tokens are RS256 JWTs verified OFFLINE through the issuer's JWKS; no
    per-request network round-trip is needed once the signing keys are cached.
    `audience` must match the resource URI registered for this server in MCP Gate
    (defaults to the server's own MCP URL, i.e. http://<host>:<port>/mcp).
    """
    enabled: bool

    # -- MCP Gate (OAuth 2.1 authorization server) --
    issuer: Optional[str] = None  # MCP Gate base URL (RS256 JWKS issuer); required when enabled
    audience: Optional[str] = None  # Resource id; empty -> defaults to <server base>/mcp
    required_scopes: List[str] = []  # Scopes every token must carry (empty = none required)

    # DEPRECATED: legacy remote Token Server URL. No longer used in the auth path
    # (kept only so old configs still validate). Use `issuer` instead.
    token_server_url: Optional[str] = None

    # -- Unused since the offline verifier replaced the token server; kept so existing
    # configs still validate. --
    cache_ttl: int = 60
    request_timeout: float = 5.0
    retry_count: int = 2
    dynamic_tools: Optional[DynamicToolsConfig] = None
    # Admin-console login (session cookie, admin scope). Defaults apply when omitted.
    console: ConsoleAuthConfig = Field(default_factory=ConsoleAuthConfig)
    # MCP Gate revocation feed (off by default; needs a resource-server client).
    revocation: RevocationConfig = Field(default_factory=RevocationConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str
    format: str
    file: Optional[str]
    max_size: str
    backup_count: int


# Folio defaults for the app fields whose pre-rename defaults are upgraded on load.
_FOLIO_APP_DEFAULTS = {
    "name": "Folio",
    "title": "Folio MCP Server",
    "description": "Folio — hierarchical RAG with auto-merging retrieval",
    "tool_prefix": "Folio",
}
# What the code and the config template shipped before the project was renamed
# from Agentic RAG to Folio.
_PRE_RENAME_APP_VALUES = {
    "name": ("Agentic RAG",),
    "title": ("Agentic RAG MCP Server",),
    "description": (
        "Agentic RAG — Hierarchical RAG with auto-merging retrieval",
        "Agentic RAG — Hierarchical chunking + AutoMerging retrieval for LLM agents",
    ),
    "tool_prefix": ("Agentic",),
}


class AppConfig(BaseModel):
    """Application configuration.

    There is no version field: the single source of truth is [project].version in pyproject.toml,
    parsed by get_version() in src/version.py.
    """
    name: str = _FOLIO_APP_DEFAULTS["name"]
    title: str = _FOLIO_APP_DEFAULTS["title"]
    description: str = _FOLIO_APP_DEFAULTS["description"]
    lifespan: Optional[str] = None
    # Per-folder MCP tool name prefix: tools are named Folio_<folder>
    tool_prefix: str = _FOLIO_APP_DEFAULTS["tool_prefix"]

    # "app.<field>: old -> new" for every pre-rename default replaced on load;
    # main.py logs a deprecation warning for each at startup.
    _renamed_defaults: List[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="wrap")
    @classmethod
    def _upgrade_pre_rename_defaults(cls, data, handler):
        """Treat the values shipped before the rename to Folio as "not set".

        A config.yaml copied from the old template still says ``tool_prefix:
        "Agentic"`` and ``name: "Agentic RAG"``. Those were defaults, not a
        choice, so they load as the Folio defaults; any other value is kept.
        """
        upgraded: List[str] = []
        if isinstance(data, dict):
            data = dict(data)
            for field, old_values in _PRE_RENAME_APP_VALUES.items():
                if data.get(field) in old_values:
                    new = _FOLIO_APP_DEFAULTS[field]
                    upgraded.append(f"app.{field}: {data[field]!r} -> {new!r}")
                    data[field] = new
        model = handler(data)
        if isinstance(model, cls):
            model._renamed_defaults = upgraded
        return model

    @property
    def renamed_defaults(self) -> List[str]:
        return list(self._renamed_defaults)


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
    # openai | openai-compatible | azure | vllm | ollama (HTTP) | huggingface / local (in-process)
    provider: str
    model: str
    dimension: int
    api_key: Optional[str] = Field(None, repr=False)  # Prefer "${ENV_VAR}"; literal keys are rejected in profile: cloud
    api_key_env: Optional[str] = None  # Environment variable name for API key
    # OpenAI-compatible endpoint (honoured by every HTTP provider, including `openai`);
    # also used as azure_endpoint for the Azure provider.
    base_url: Optional[str] = None
    api_version: Optional[str] = None  # Azure API version (e.g., "2024-02-01")
    azure_deployment: Optional[str] = None  # Azure deployment name
    # e5-family needs "query: "/"passage: " prefixes; empty (default) for bge-family.
    # Changing a prefix changes the vector space, so existing indexes need a full reindex.
    query_prefix: str = ""
    passage_prefix: str = ""
    # Send `dimensions=<dimension>` with each request (Matryoshka models such as
    # text-embedding-3-*; needed when the model's native size exceeds pgvector's HNSW limit).
    request_dimensions: bool = False
    timeout: float = Field(60.0, gt=0)  # per-request timeout (seconds)
    max_retries: int = Field(3, ge=0, le=10)  # 429 / 5xx / timeout retries (SDK backoff)
    # Texts per embedding request (llama-index defaults to 10, i.e. 5 requests per batch of 50).
    embed_batch_size: int = Field(64, ge=1, le=2048)
    # The model's input window. Before embedding, a leaf's retrieval text (prefix +
    # breadcrumb + text, plus passage_prefix) is fitted into it by shortening the
    # breadcrumb, then the prefix — never the chunk text. 512 for e5 / bge-large.
    max_input_tokens: int = Field(512, ge=64)


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
    # Which candidates the reranker scores (ignored without a reranker):
    # per_list = the dense top rerank_dense_candidates (at least 3 x top_k) plus the BM25
    #            top rerank_sparse_candidates, about 35 pairs instead of 60 at the same
    #            recall on the golden set (evals/baselines/rag-2026-11-04-search-latency.md);
    # fused    = the top max(3 x top_k, rerank_candidates) of the fused list.
    rerank_pool: Literal["per_list", "fused"] = "per_list"
    rerank_dense_candidates: int = Field(30, ge=0)
    rerank_sparse_candidates: int = Field(10, ge=0)
    # Fused candidates handed to the reranker with rerank_pool: fused: max(3 x top_k, this).
    rerank_candidates: int = 60
    default_hybrid_alpha: float = 0.75
    hybrid_search: bool
    text_search_config: str = "simple"  # Chinese is segmented in Python; Postgres tokenizes the space-joined result with `simple`
    # Python-side segmenter for the `simple` config, used by indexing, Rebuild FTS and queries alike:
    # ckip (accurate, torch model, `--extra local`) | jieba (pure Python, `--extra jieba`) |
    # bigram (no dependency). Unset = the profile's default: ckip, or bigram under
    # profile: cloud (which loads no in-process model; explicit ckip is rejected there).
    # Changing it flags existing folders for Rebuild FTS (no re-embedding needed).
    segmenter: Optional[Literal["ckip", "jieba", "bigram"]] = None
    # Device for the CKIP segmenter: cpu | cuda | cuda:N | auto (cuda:0 when available).
    ckip_device: str = "cpu"
    # "rrf" = Reciprocal Rank Fusion (BM25 affects ordering); "concat" = llama_index native
    # (dense dominates, effectively vector-only; legacy).
    hybrid_fusion: str = "rrf"
    rrf_k: int = 60  # RRF constant (larger = smoother; 60 is the common convention)
    # MCP search: attach the top-ranked hit's original file as an embedded resource
    # (off by default — a whole file can dwarf the search result itself). Only that
    # one file is attached, and only when it is at most resource_file_max_bytes.
    return_resource_files: bool = False
    resource_file_max_bytes: int = Field(1_000_000, ge=0)
    auto_merging: Optional[AutoMergingConfig] = None
    expand_context_default: bool = True
    # Leaves of context on each side of an unmerged hit (MCP search). Windows stay
    # inside the hit's section and adjacent windows of a file are merged.
    expand_context_neighbors: int = 1
    # Estimated-token budget of one MCP search response (CJK ≈ 0.7 tokens/char,
    # other text ≈ 0.25). Lowest-ranked results are dropped (and reported) to fit;
    # 0 disables the budget.
    mcp_response_token_budget: int = Field(6000, ge=0)
    # MCP search relevance cutoff on the reranker's score (0..1; 0 = off). A search whose
    # best cross-encoder score is below it is answered with low_relevance: true and a hint
    # to say "not found"; results below it are flagged low_relevance (never dropped - that
    # is rag.rerank.score_threshold). Model-specific like score_threshold: 0.9 is
    # calibrated for mxbai-rerank-v2 on the golden set (evals/baselines). Ignored
    # without a reranker.
    min_relevance: float = Field(0.0, ge=0.0, le=1.0)
    # Max concurrent queries (REST + MCP share one semaphore; excess is queued). 8 keeps every
    # connection pool (5+10 each) within headroom.
    max_concurrent_queries: int = 8


class RAGVectorStoreConfig(BaseModel):
    """RAG vector-store configuration."""
    type: str
    table_prefix: str


class RAGLLMConfig(BaseModel):
    """Default LLM for every LLM role (contextual prefixes, answers, eval question generation).

    Each role can override any field through ``rag.contextual_llm`` / ``rag.answer_llm`` /
    ``rag.eval_llm`` (see LLMRoleConfig); unset fields inherit from here.
    """
    # openai | openai-compatible | vllm | ollama (all OpenAI-compatible; base_url honoured) | azure
    provider: str = "azure"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = Field(None, repr=False)
    base_url: Optional[str] = None  # OpenAI-compatible endpoint (Azure: the resource endpoint)
    api_version: Optional[str] = "2024-12-01-preview"
    azure_deployment: Optional[str] = None
    timeout: float = Field(120.0, gt=0)  # per-request timeout (seconds)
    max_retries: int = Field(2, ge=0, le=10)  # 429 / 5xx / timeout retries (SDK backoff, honours Retry-After)


class LLMRoleConfig(BaseModel):
    """Per-role LLM override. Every field is optional; unset fields inherit from ``rag.llm``."""
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = Field(None, repr=False)
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    azure_deployment: Optional[str] = None
    timeout: Optional[float] = Field(None, gt=0)
    max_retries: Optional[int] = Field(None, ge=0, le=10)


class ContextualRetrievalConfig(BaseModel):
    """Contextual Retrieval configuration."""
    enabled: bool = False
    max_context_length: int = 150
    max_concurrent: int = 5  # per file
    # Cap on prefix requests in flight across every file indexed at once (process-wide).
    max_concurrent_global: int = Field(16, ge=1)
    max_doc_chars: int = 60000
    max_tokens: int = 1024
    reasoning_effort: Optional[str] = None
    # Choose whether to add the context prefix to leaf / parent / all chunks
    apply_to: str = "all"  # "leaf" / "parent" / "all"
    # Cost guards (0 = unlimited). Chunks over a cap get the deterministic fallback prefix
    # (file name) instead of an LLM call, and a warning is logged.
    max_calls_per_file: int = Field(0, ge=0)
    # Estimated tokens (prompt: 1 per CJK char, 1 per 4 other chars; + max_tokens) per UTC day.
    max_tokens_per_day: int = Field(0, ge=0)


class RerankConfig(BaseModel):
    """Reranker configuration."""
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    base_url: str = "http://localhost:8787"
    api_key: Optional[str] = Field(None, repr=False)
    # HTTP API style:
    #   score   — vLLM POST /v1/score {text_1, text_2}            (default; previous behaviour)
    #   cohere  — Cohere POST /v1/rerank {query, documents}        → results[{index, relevance_score}]
    #   jina    — Jina POST /v1/rerank {query, documents}          → results[{index, relevance_score}]
    #   generic — OpenAI-compatible servers exposing POST /rerank (vLLM, Infinity, gateways);
    #             tolerant parser for results/data and relevance_score/score
    api: Literal["score", "cohere", "jina", "generic"] = "score"
    # Explicit path appended to base_url (e.g. "/v2/rerank"); empty = the style's default.
    endpoint_path: Optional[str] = None
    timeout: float = Field(10.0, gt=0)  # query-path latency budget per request (seconds)
    max_retries: int = Field(1, ge=0, le=5)  # retries on 429 / 5xx / timeout before falling back
    top_n: Optional[int] = None  # keep the top N results after reranking (None = keep all)
    score_threshold: float = 0.0  # rerank score threshold
    # Input template applied before POST /v1/score; {text} is the placeholder. Instruction-tuned
    # rerankers (e.g. mxbai) need the official template for a sane score distribution; traditional
    # cross-encoders (bge-family) keep the default = raw text sent through as-is.
    query_template: str = "{text}"
    document_template: str = "{text}"
    # mxbai-rerank-*-v2 needs the Qwen chat prompt for calibrated scores; when True it is applied
    # automatically if the templates above are the default or the old chat-less form. Turn it off
    # when the rerank server already applies a chat template (vLLM --chat-template), or the
    # prompt is wrapped twice.
    auto_template: bool = True
    # Knowledge-gap analytics: a question whose latest search's best rerank score is below this
    # (or that returned nothing) is listed as a gap. Model-specific like score_threshold.
    gap_threshold: float = 0.5


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
    api_key: Optional[str] = Field(None, repr=False)
    model: Optional[str] = None             # cloud model name, e.g. whisper-1
    language: Optional[str] = None          # ISO-639-1 hint (e.g. "zh"); empty = auto-detect
    # Long audio is resampled to 16 kHz mono and uploaded in segments cut at silences, so
    # no single request hits the provider's upload limit (OpenAI: 25 MB ≈ 13 min of 16k pcm).
    segment_seconds: float = Field(600.0, ge=30.0, le=780.0)
    timeout: float = Field(300.0, gt=0)     # per-segment request timeout (seconds)
    max_retries: int = Field(3, ge=0, le=10)  # 429 / 5xx / timeout / empty-or-runaway retries


class VisionModelConfig(BaseModel):
    """Vision-capable chat model used to transcribe PDF pages / images (cloud parsing).

    Called through the OpenAI-compatible ``POST {base_url}/chat/completions`` with an
    ``image_url`` data-URI part; ``azure`` uses the Azure OpenAI client instead.
    """
    provider: str = "openai-compatible"  # openai | openai-compatible | vllm | ollama | azure
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = Field(None, repr=False)
    api_version: Optional[str] = None
    azure_deployment: Optional[str] = None
    timeout: float = Field(120.0, gt=0)
    max_retries: int = Field(3, ge=0, le=10)
    max_tokens: int = Field(4096, ge=256)


class ParsingConfig(BaseModel):
    """Where PDFs and images are parsed.

    ``local`` = in-process docling (layout / TableFormer / OCR models); ``cloud`` = every
    page is rendered and transcribed to Markdown by ``vision``. Unset follows the profile
    (``cloud`` profile → cloud, otherwise local).
    """
    pdf: Optional[Literal["local", "cloud"]] = None
    images: Optional[Literal["local", "cloud"]] = None
    vision: Optional[VisionModelConfig] = None
    dpi: int = Field(150, ge=72, le=300)  # page render resolution sent to the vision model
    max_concurrency: int = Field(4, ge=1, le=32)  # pages in flight per file
    # Trailing characters of the previous page passed with each page, so headings, lists
    # and tables that continue across a page break are recognised as continuations.
    context_chars: int = Field(800, ge=0, le=8000)
    max_pages_per_file: int = Field(500, ge=0)  # 0 = unlimited; larger files fail up front
    max_pages_per_day: int = Field(0, ge=0)  # process-wide UTC-day page budget; 0 = unlimited
    # A page that still fails after retries is replaced by a visible marker ("mark") or
    # fails the whole file ("fail"). min_success_ratio fails the file if too few pages parse.
    on_page_failure: Literal["mark", "fail"] = "mark"
    min_success_ratio: float = Field(0.5, ge=0.0, le=1.0)
    prompt: Optional[str] = None  # override the built-in language-neutral transcription prompt


# Upper bound for rag.indexing.max_concurrent_jobs (and FOLIO_INDEXING_CONCURRENCY).
MAX_INDEXING_CONCURRENCY = 32


class IndexingConfig(BaseModel):
    """Behavior control for background indexing jobs."""
    # Max files indexed at once, process-wide (one semaphore shared by every job). Kept
    # low (4) because docling convert is serialized by DOCLING_INFER_LOCK and each file
    # fans out contextual-prefix LLM calls and embedding batches. Range 1..32; a larger
    # value (old configs shipped 100) is capped at 32 with a warning. Override via
    # FOLIO_INDEXING_CONCURRENCY.
    max_concurrent_jobs: int = Field(4, ge=1)

    # Threads in the dedicated indexing pool (src/domain/rag/indexing_executor.py).
    # Indexing's blocking work (docling, prefix LLM calls, embedding, pgvector writes)
    # runs there instead of the event loop's default executor, so the query path's
    # to_thread calls never queue behind a large indexing job.
    executor_workers: int = Field(16, ge=1, le=64)

    @field_validator("max_concurrent_jobs")
    @classmethod
    def _cap_concurrency(cls, v: int) -> int:
        if v > MAX_INDEXING_CONCURRENCY:
            _logging.getLogger(__name__).warning(
                f"rag.indexing.max_concurrent_jobs={v} is above {MAX_INDEXING_CONCURRENCY}; "
                f"using {MAX_INDEXING_CONCURRENCY} (set 4-8 for normal use)")
            return MAX_INDEXING_CONCURRENCY
        return v

    # Per-file limit (seconds); override via FOLIO_PER_FILE_TIMEOUT_SECONDS. Default is effectively
    # unlimited (10 days), relying on the 60s keepalive heartbeat instead of a hard timeout.
    per_file_timeout_seconds: int = 864000  # 10 days

    # Whole-job limit (seconds); override via FOLIO_JOB_TIMEOUT_SECONDS. 0/negative disables.
    # Must be >= per_file_timeout_seconds or the job dies before its files finish.
    job_timeout_seconds: int = 864000  # 10 days


Profile = Literal["local", "hybrid", "cloud"]


class RAGConfig(BaseModel):
    """RAG system configuration."""
    enabled: bool = True
    # Deployment profile (see src/config/profile.py):
    #   local  — anything goes (default; previous behaviour)
    #   hybrid — local models and cloud endpoints mixed; literal API keys are warned about
    #   cloud  — every role is an HTTP endpoint; in-process model providers are refused,
    #            API keys must be ${ENV} references, and no local model stack is imported
    profile: Profile = "local"
    embedding: RAGEmbeddingConfig
    chunking: RAGChunkingConfig
    retrieval: RAGRetrievalConfig
    vector_store: RAGVectorStoreConfig
    llm: Optional[RAGLLMConfig] = None
    # Per-role LLM overrides (unset fields inherit from `llm`)
    contextual_llm: Optional[LLMRoleConfig] = None
    answer_llm: Optional[LLMRoleConfig] = None
    eval_llm: Optional[LLMRoleConfig] = None
    contextual_retrieval: Optional[ContextualRetrievalConfig] = None
    rerank: Optional[RerankConfig] = None
    docling: Optional[DoclingConfig] = None
    parsing: Optional[ParsingConfig] = None
    asr: Optional[AsrConfig] = None
    indexing: Optional[IndexingConfig] = None

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> "RAGConfig":
        """Profile defaults that need no validation: a cloud deployment without an `asr`
        section must not fall back to local Whisper, so speech-to-text is off until an
        OpenAI-compatible endpoint is configured."""
        if self.profile == "cloud" and self.asr is None:
            self.asr = AsrConfig(enabled=False, provider="openai-compatible")
        return self


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
    "ConsoleAuthConfig",
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
    "LLMRoleConfig",
    "ContextualRetrievalConfig",
    "RerankConfig",
    "DoclingConfig",
    "VisionModelConfig",
    "ParsingConfig",
    "AsrConfig",
    "IndexingConfig",
    "Profile",
    "RAGConfig",
    "ConfigModel",
]
