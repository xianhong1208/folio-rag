
"""HuggingFace embedding provider implementation"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .base import IEmbeddingProvider
from src.log import get_api_logger

logger = get_api_logger()


class HuggingFaceEmbeddingProvider(IEmbeddingProvider):
    """HuggingFace embedding provider

    Uses HuggingFace transformers library to load and run embedding models locally.
    No API key required - models are downloaded and run on local hardware.

    Popular models:
    - BAAI/bge-small-en-v1.5 (384 dimensions)
    - BAAI/bge-base-en-v1.5 (768 dimensions)
    - BAAI/bge-large-en-v1.5 (1024 dimensions)
    - sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
    - intfloat/e5-large-v2 (1024 dimensions)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        embed_batch_size: int = 10,
        device: str = "cpu"
    ):
        """Initialize HuggingFace provider

        Args:
            model_name: HuggingFace model identifier
            dimension: Embedding dimension
            embed_batch_size: Batch size for encoding (higher = faster but more memory)
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        self._model_name = model_name
        self._dimension = dimension
        self._embed_batch_size = embed_batch_size
        self._device = device

        # Create embedding model instance
        try:
            self._embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=embed_batch_size,
                device=device
            )

            # Get actual dimension from model if possible
            try:
                actual_dim = self._embed_model.model.get_sentence_embedding_dimension()
                if actual_dim != dimension:
                    logger.warning(
                        f"Configured dimension ({dimension}) differs from model dimension ({actual_dim}). "
                        f"Using model dimension."
                    )
                    self._dimension = actual_dim
            except AttributeError:
                # Some models don't have this method, use configured dimension
                pass

            logger.info(
                f"HuggingFace embedding provider initialized: {model_name} "
                f"(dim={self._dimension}, batch_size={embed_batch_size}, device={device})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embedding provider: {e}")
            raise

    def get_embed_model(self) -> HuggingFaceEmbedding:
        """Get the HuggingFace embedding model instance"""
        return self._embed_model

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name"""
        return self._model_name

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "huggingface"

    def validate_config(self) -> None:
        """Validate HuggingFace configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not self._model_name:
            raise ValueError("HuggingFace model_name cannot be empty")

        if self._dimension <= 0:
            raise ValueError(f"Invalid dimension: {self._dimension}")

        if self._embed_batch_size <= 0:
            raise ValueError(f"Invalid batch size: {self._embed_batch_size}")
