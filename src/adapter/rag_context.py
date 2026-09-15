
"""Shared dependencies and cached state for RAG operations.

RAGContext holds the state built during initialization (embedding provider,
indexer, vector store manager, reranker, sizes, caches). It is built once via
`from_config()` and shared by the indexing / query / maintenance sub-services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from llama_index.core import Settings
from llama_index.vector_stores.postgres import PGVectorStore

from src.config.config_manager import Config
from src.domain.exceptions import RAGOperationError
from src.domain.rag.embedding import (
    EmbeddingProviderFactory,
    OpenAIEmbeddingProvider,
)
from src.domain.rag.hierarchical_indexer import HierarchicalIndexer
from src.domain.rag.index_service import IndexingService
from src.domain.rag.vector_store_manager import VectorStoreManager
from src.log import get_adapter_logger, log_err, mask_token

if TYPE_CHECKING:
    from src.domain.rag.query_engine import QueryEngine
    from src.domain.rag.reranker import Reranker

logger = get_adapter_logger()


@dataclass
class RAGContext:
    """Dependencies and caches shared by RAGAdapter and its sub-services."""

    embedding_provider: Any
    embed_dim: int
    model_name: str
    chunker_tokenizer_name: Optional[str]

    leaf_chunk_size: int
    parent_target_tokens: int
    chunk_overlap: int

    vector_store_manager: VectorStoreManager
    indexer: HierarchicalIndexer
    indexing_service: IndexingService

    auto_merging_enabled: bool
    merge_threshold: float
    expand_neighbors: int

    reranker: Optional["Reranker"] = None

    query_engines: Dict[str, "QueryEngine"] = field(default_factory=dict)

    @classmethod
    def from_config(cls) -> "RAGContext":
        """Build the RAG runtime state from the Config singleton.

        Falls back to the OpenAI default on failure, so a config error starts the
        adapter degraded rather than crashing the server.
        """
        rag_config = None
        try:
            config_model = Config.get_config_model()
            rag_config = config_model.rag
        except Exception as e:
            logger.error(f"Failed to load RAG configuration: {e}")

        provider_info = cls._load_embedding_provider(rag_config)

        leaf_size, parent_target, overlap = cls._load_chunking_sizes(rag_config)
        Settings.chunk_size = leaf_size
        Settings.chunk_overlap = overlap

        text_search_config = "simple"
        hybrid_search = True
        try:
            if rag_config and rag_config.retrieval:
                text_search_config = getattr(
                    rag_config.retrieval, "text_search_config", "simple"
                )
                hybrid_search = getattr(rag_config.retrieval, "hybrid_search", True)
        except Exception:
            pass

        vector_store_manager = VectorStoreManager(
            embed_dim=provider_info["embed_dim"],
            text_search_config=text_search_config,
            hybrid_search=hybrid_search,
        )
        logger.info("[INIT] VectorStoreManager initialized")

        context_generator = None
        try:
            if rag_config:
                from src.domain.rag.context_generator import ContextGenerator
                context_generator = ContextGenerator.from_config(rag_config)
        except Exception as e:
            logger.warning(f"Contextual Retrieval initialization failed: {e}")

        # ASR provider from config (unset -> default docling-whisper)
        from src.domain.rag.asr_provider import create_asr_provider
        asr_cfg = getattr(rag_config, "asr", None) if rag_config else None
        indexer = HierarchicalIndexer(
            leaf_chunk_size=leaf_size,
            parent_target_tokens=parent_target,
            chunk_overlap=overlap,
            context_generator=context_generator,
            embedding_model_name=provider_info["chunker_tokenizer_name"],
            embed_model=provider_info.get("embed_model"),  # injected, not read from the global
            asr_provider=create_asr_provider(asr_cfg),
            asr_enabled=(asr_cfg.enabled if asr_cfg else True),
        )
        logger.info("[INIT] HierarchicalIndexer initialized")

        am_cfg = getattr(rag_config, "retrieval", None) if rag_config else None
        am_cfg = getattr(am_cfg, "auto_merging", None) if am_cfg else None
        auto_merging_enabled = bool(getattr(am_cfg, "enabled", True)) if am_cfg else True
        merge_threshold = float(getattr(am_cfg, "merge_threshold", 0.5)) if am_cfg else 0.5
        expand_neighbors = (
            int(getattr(rag_config.retrieval, "expand_context_neighbors", 2))
            if rag_config and rag_config.retrieval else 2
        )

        indexing_service = IndexingService()
        logger.info("[INIT] IndexingService initialized")

        reranker = None
        try:
            if rag_config:
                rerank_config = getattr(rag_config, "rerank", None)
                if rerank_config and rerank_config.enabled:
                    from src.domain.rag.reranker import Reranker
                    reranker = Reranker.from_config(rerank_config)
                    if reranker:
                        logger.info("[INIT] Reranker initialized")
        except Exception as e:
            logger.warning(f"Reranker initialization failed: {e}")

        return cls(
            embedding_provider=provider_info["provider"],
            embed_dim=provider_info["embed_dim"],
            model_name=provider_info["model_name"],
            chunker_tokenizer_name=provider_info["chunker_tokenizer_name"],
            leaf_chunk_size=leaf_size,
            parent_target_tokens=parent_target,
            chunk_overlap=overlap,
            vector_store_manager=vector_store_manager,
            indexer=indexer,
            indexing_service=indexing_service,
            auto_merging_enabled=auto_merging_enabled,
            merge_threshold=merge_threshold,
            expand_neighbors=expand_neighbors,
            reranker=reranker,
        )

    @staticmethod
    def _load_embedding_provider(rag_config) -> Dict[str, Any]:
        """Try to build the provider from config, falling back to the OpenAI default on failure."""
        if rag_config and rag_config.enabled:
            embedding_config = rag_config.embedding
            logger.info(
                f"Loading RAG configuration: provider={embedding_config.provider}, "
                f"model={embedding_config.model}, dimension={embedding_config.dimension}"
            )
            try:
                provider = EmbeddingProviderFactory.create_from_config(embedding_config)
                embed_model = provider.get_embed_model()
                # e5-family models need query/passage prefixes (from config);
                # bge leaves them empty, so no wrapping happens.
                q_pre = getattr(embedding_config, "query_prefix", "") or ""
                p_pre = getattr(embedding_config, "passage_prefix", "") or ""
                if q_pre or p_pre:
                    from src.domain.rag.embedding.prefixed import PrefixedEmbedding
                    embed_model = PrefixedEmbedding(
                        embed_model, query_prefix=q_pre, passage_prefix=p_pre)
                    logger.info(
                        f"Embedding prefixes enabled: query='{q_pre}' passage='{p_pre}'")
                Settings.embed_model = embed_model
                # Chunker tokenizer is the embedding model's HF tokenizer (used to
                # count chunk tokens). vllm / ollama also serve HF model names
                # (e.g. BAAI/bge-m3); only openai / azure names are not HF repos.
                chunker_tok = (
                    provider.get_model_name()
                    if provider.get_provider_name() in ("huggingface", "vllm", "ollama")
                    else None
                )
                logger.info(
                    f"Embedding provider initialized successfully: "
                    f"{provider.get_provider_name()} - {provider.get_model_name()} "
                    f"(dimension={provider.get_dimension()})"
                )
                return {
                    "provider": provider,
                    "embed_dim": provider.get_dimension(),
                    "model_name": provider.get_model_name(),
                    "chunker_tokenizer_name": chunker_tok,
                    # Final embed_model (incl. any PrefixedEmbedding wrapper),
                    # injected into the indexer so it needn't read Settings.embed_model.
                    "embed_model": embed_model,
                }
            except Exception as e:
                logger.error(f"Failed to create embedding provider: {e}")
                logger.warning("Falling back to default OpenAI configuration")

        if not rag_config or not rag_config.enabled:
            logger.warning("RAG configuration not found, using defaults")
        return RAGContext._setup_default_provider()

    @staticmethod
    def _setup_default_provider() -> Dict[str, Any]:
        """OpenAI text-embedding-3-small fallback."""
        try:
            provider = OpenAIEmbeddingProvider(
                model_name="text-embedding-3-small",
                dimension=1536,
            )
            Settings.embed_model = provider.get_embed_model()
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            logger.info("Default OpenAI configuration loaded successfully")
            return {
                "provider": provider,
                "embed_dim": provider.get_dimension(),
                "model_name": provider.get_model_name(),
                "chunker_tokenizer_name": None,
                "embed_model": provider.get_embed_model(),
            }
        except Exception as e:
            logger.error(f"Failed to setup default configuration: {e}")
            raise RuntimeError("Unable to initialize embedding provider")

    @staticmethod
    def _load_chunking_sizes(rag_config) -> tuple[int, int, int]:
        """Returns (leaf_size, parent_target, chunk_overlap)."""
        if not rag_config or not rag_config.enabled:
            return 256, 1024, 50
        chunking_config = rag_config.chunking
        sizes = getattr(chunking_config, "hierarchy_sizes", None) or [1024, 256]
        if len(sizes) < 2:
            sizes = [1024, 256]
        parent_target = sizes[0]
        leaf_size = sizes[-1]
        overlap = getattr(chunking_config, "chunk_overlap", 50)
        logger.info(
            f"Hierarchical chunking: leaf_size={leaf_size}, "
            f"parent_target={parent_target}, overlap={overlap}"
        )
        return leaf_size, parent_target, overlap

    def get_vector_store(
        self,
        folder_id: int,
        token: str,
        folder: Optional[Any] = None,
    ) -> PGVectorStore:
        """Get or create the vector store for the given folder/token.

        Raises:
            RAGOperationError: failed to obtain the vector store.
            DomainException: the folder does not exist or the token lacks permission.
        """
        folder = folder or self.indexing_service.get_folder(folder_id, token)
        vector_table_uuid = str(folder.vector_table_uuid)

        try:
            vector_store = self.vector_store_manager.get_or_create(
                folder_id=folder_id,
                vector_table_uuid=vector_table_uuid,
            )
            logger.debug(
                f"[VECTOR_STORE] fid={folder_id} tok={mask_token(token)} | "
                f"obtained table=data_{folder_id}_{vector_table_uuid}"
            )
            return vector_store
        except Exception as e:
            log_err(logger, "VECTOR_STORE_FAIL", e, folder_id=folder_id, token=token)
            raise RAGOperationError(
                operation="vector_store_access",
                reason=str(e),
                details={"folder_id": folder_id, "vector_table_uuid": vector_table_uuid},
            )

    def invalidate_query_engine_cache(self, folder_id: int) -> None:
        """Remove the given folder's QueryEngine from the cache (call after an index change)."""
        key = f"{folder_id}"
        if key in self.query_engines:
            del self.query_engines[key]
            logger.debug(f"Invalidated QueryEngine cache for folder {folder_id}")

    def invalidate_all_query_engines(self) -> None:
        """Clear the whole QueryEngine cache. Required after hot-changing
        embedding / reranker: cached engines hold a stale reranker and an index
        built with the old embed_model."""
        n = len(self.query_engines)
        self.query_engines.clear()
        logger.info(f"[RUNTIME] cleared {n} cached QueryEngine(s)")

    def apply_runtime_changes(self, groups: list[str]) -> None:
        """Wire the already-updated ConfigModel onto the live objects, per rebind group.

        Changes are applied in place: sub-services hold a reference to this
        context, so replacing it would orphan them. `groups` comes from
        runtime_overrides.rebind_groups_for(applied paths).
        """
        rag_config = Config.get_config_model().rag
        if "rag.retrieval" in groups:
            self._rebind_retrieval(rag_config)
        if "rag.rerank" in groups:
            self._rebind_reranker(rag_config)
        # llm / contextual / asr are frozen into the indexer's construction chain,
        # so rebuild it (cheap: embed_model is injected, no model reload).
        if any(g in groups for g in ("rag.llm", "rag.contextual_retrieval", "rag.asr")) \
                and "rag.embedding" not in groups:
            # Reuse the injected embed_model; reading Settings would build the
            # OpenAI default when unset.
            self._rebuild_indexer(rag_config, embed_model=getattr(self.indexer, "_embed_model", None))
        if "rag.embedding" in groups:
            self._rebind_embedding(rag_config)

    def _rebind_retrieval(self, rag_config) -> None:
        """Rebind the retrieval scalars frozen on the context (the rest are read
        from config per request)."""
        r = rag_config.retrieval
        am = getattr(r, "auto_merging", None)
        if am is not None:
            self.auto_merging_enabled = bool(am.enabled)
            self.merge_threshold = float(am.merge_threshold)
        self.expand_neighbors = int(getattr(r, "expand_context_neighbors", self.expand_neighbors))
        logger.info(
            f"[RUNTIME] retrieval rebound: auto_merging={self.auto_merging_enabled} "
            f"threshold={self.merge_threshold} expand={self.expand_neighbors}")

    def _rebind_reranker(self, rag_config) -> None:
        """Rebuild the Reranker (off -> None; on -> new instance) and clear the
        QueryEngine cache, which holds a stale reranker reference. Also handles a
        reranker disabled at boot, which would otherwise stay None forever."""
        rerank_config = getattr(rag_config, "rerank", None)
        new_reranker = None
        if rerank_config and rerank_config.enabled:
            from src.domain.rag.reranker import Reranker
            new_reranker = Reranker.from_config(rerank_config)
        self.reranker = new_reranker
        self.invalidate_all_query_engines()
        logger.info(f"[RUNTIME] reranker rebound: {'on' if new_reranker else 'off'}")

    def _rebuild_indexer(self, rag_config, embed_model=None) -> None:
        """Rebuild the HierarchicalIndexer (context generator / asr provider are
        frozen into its construction chain). Swapping the field takes effect for
        the next job; an in-flight job finishes on the old indexer it still holds."""
        from src.domain.rag.asr_provider import create_asr_provider
        from src.domain.rag.context_generator import ContextGenerator

        context_generator = None
        try:
            context_generator = ContextGenerator.from_config(rag_config)
        except Exception as e:
            logger.warning(f"[RUNTIME] ContextGenerator rebuild failed (disabled): {e}")
        asr_cfg = getattr(rag_config, "asr", None)
        self.indexer = HierarchicalIndexer(
            leaf_chunk_size=self.leaf_chunk_size,
            parent_target_tokens=self.parent_target_tokens,
            chunk_overlap=self.chunk_overlap,
            context_generator=context_generator,
            embedding_model_name=self.chunker_tokenizer_name,
            embed_model=embed_model,
            asr_provider=create_asr_provider(asr_cfg),
            asr_enabled=(asr_cfg.enabled if asr_cfg else True),
        )
        logger.info("[RUNTIME] indexer rebuilt (llm/contextual/asr rebound)")

    def _rebind_embedding(self, rag_config) -> None:
        """Embedding hot-change: rebuild the provider chain, swap
        Settings.embed_model, update context fields, change the VSM dimension and
        clear its cache, rebuild the indexer, and clear the QueryEngine cache.
        Existing folders' vectors are incompatible with the new model; the API
        layer surfaces this warning to the frontend.
        """
        provider_info = self._load_embedding_provider(rag_config)
        self.embedding_provider = provider_info["provider"]
        self.embed_dim = provider_info["embed_dim"]
        self.model_name = provider_info["model_name"]
        self.chunker_tokenizer_name = provider_info["chunker_tokenizer_name"]
        self.vector_store_manager.update_embed_dim(provider_info["embed_dim"])
        self._rebuild_indexer(rag_config, embed_model=provider_info.get("embed_model"))
        self.invalidate_all_query_engines()
        logger.info(
            f"[RUNTIME] embedding rebound: {self.model_name} (dim={self.embed_dim}) — "
            "existing folders must be reindexed")
