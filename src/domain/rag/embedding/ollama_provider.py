
"""Ollama embedding provider implementation"""

from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from .base import IEmbeddingProvider
from src.log import get_api_logger
from typing import Optional
logger = get_api_logger()


class OllamaEmbeddingProvider(IEmbeddingProvider):
    """Ollama embedding provider

    Ollama provides an OpenAI-compatible API for running embedding models locally.
    This provider uses OpenAILikeEmbedding to communicate with Ollama server.

    Setup:
    1. Install Ollama: https://ollama.ai/
    2. Pull model: ollama pull nomic-embed-text
    3. Ollama server runs on: http://localhost:11434/v1 by default

    Popular Ollama embedding models:
    - nomic-embed-text (768 dimensions) - Best for English
    - mxbai-embed-large (1024 dimensions) - High quality
    - all-minilm (384 dimensions) - Fast and lightweight
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: Optional[str] = None,  # Ollama doesn't validate API key
        dimension: int = 768
    ):
        """Initialize Ollama provider

        Args:
            model_name: Model name (must be pulled with 'ollama pull')
            base_url: Ollama server URL (include /v1 suffix)
            api_key: Dummy API key (Ollama doesn't validate it)
            dimension: Embedding dimension
        """
        self._model_name = model_name
        self._base_url = base_url
        self._dimension = dimension
        self._api_key = api_key

        # Create embedding model instance
        try:
            self._embed_model = OpenAILikeEmbedding(
                model_name=model_name,
                api_key=api_key,
                api_base=base_url
            )
            logger.info(
                f"Ollama embedding provider initialized: {model_name} at {base_url} (dim={dimension})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embedding provider: {e}")
            raise

    def get_embed_model(self) -> OpenAILikeEmbedding:
        """Get the Ollama embedding model instance"""
        return self._embed_model

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name"""
        return self._model_name

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "ollama"

    def validate_config(self) -> None:
        """Validate Ollama configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not self._model_name:
            raise ValueError("Ollama model_name cannot be empty")

        if not self._base_url:
            raise ValueError("Ollama base_url cannot be empty")

        if not self._base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid base_url: {self._base_url}. Must start with http:// or https://")

        if self._dimension <= 0:
            raise ValueError(f"Invalid dimension: {self._dimension}")

    def get_base_url(self) -> str:
        """Get Ollama server base URL"""
        return self._base_url
