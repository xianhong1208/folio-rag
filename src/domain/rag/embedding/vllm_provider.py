
"""vLLM embedding provider implementation"""

from typing import Optional
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from .base import IEmbeddingProvider
from src.log import get_api_logger
logger = get_api_logger()


class VLLMEmbeddingProvider(IEmbeddingProvider):
    """vLLM embedding provider

    vLLM provides an OpenAI-compatible API for serving embedding models locally.
    This provider uses OpenAILikeEmbedding to communicate with vLLM server.

    Setup:
    1. Install vLLM: pip install vllm
    2. Start server: python -m vllm.entrypoints.openai.api_server \
                     --model BAAI/bge-m3 --port 1234
    3. Configure base_url: http://localhost:1234/v1

    Popular models for vLLM:
    - BAAI/bge-m3 (1024 dimensions)
    - BAAI/bge-large-en-v1.5 (1024 dimensions)
    - intfloat/e5-large-v2 (1024 dimensions)
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: Optional[str] = None,  # vLLM doesn't validate API key
        dimension: int = 1024
    ):
        """Initialize vLLM provider

        Args:
            model_name: Model name (should match what's running on vLLM server)
            base_url: vLLM server URL (include /v1 suffix)
            api_key: Dummy API key (vLLM doesn't validate it)
            dimension: Embedding dimension
        """
        self._model_name = model_name
        self._base_url = base_url
        self._dimension = dimension
        self._api_key = api_key

        # Create embedding model instance
        try:
            self._embed_model = OpenAILikeEmbedding(
                model_name=self._model_name,
                api_key=self._api_key,
                api_base=self._base_url
            )
            logger.info(
                f"vLLM embedding provider initialized: {self._model_name} at {self._base_url} (dim={self._dimension})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM embedding provider: {e}")
            raise

    def get_embed_model(self) -> OpenAILikeEmbedding:
        """Get the vLLM embedding model instance"""
        return self._embed_model

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name"""
        return self._model_name

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "vllm"

    def validate_config(self) -> None:
        """Validate vLLM configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not self._model_name:
            raise ValueError("vLLM model_name cannot be empty")

        if not self._base_url:
            raise ValueError("vLLM base_url cannot be empty")

        if not self._base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid base_url: {self._base_url}. Must start with http:// or https://")

        if self._dimension <= 0:
            raise ValueError(f"Invalid dimension: {self._dimension}")

    def get_base_url(self) -> str:
        """Get vLLM server base URL"""
        return self._base_url
