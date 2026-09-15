
"""OpenAI embedding provider implementation"""

import os
from typing import Optional
from llama_index.embeddings.openai import OpenAIEmbedding
from .base import IEmbeddingProvider
from src.log import get_api_logger

logger = get_api_logger()


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    """OpenAI embedding provider

    Uses OpenAI's API for generating embeddings.
    Requires OPENAI_API_KEY environment variable.

    Supported models:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        dimension: int = 1536
    ):
        """Initialize OpenAI provider

        Args:
            model_name: OpenAI model name
            api_key: API key (if None, reads from environment)
            api_key_env: Environment variable name for API key
            dimension: Embedding dimension
        """
        self._model_name = model_name
        self._dimension = dimension
        self._api_key_env = api_key_env

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                # Try fallback to default env var
                api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            logger.warning(
                f"OpenAI API key not found in {api_key_env} or OPENAI_API_KEY. "
                "Provider will fail if used."
            )

        self._api_key = api_key

        # Create embedding model instance
        try:
            self._embed_model = OpenAIEmbedding(
                model=model_name,
                api_key=api_key
            )
            logger.info(f"OpenAI embedding provider initialized: {model_name} (dim={dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding provider: {e}")
            raise

    def get_embed_model(self) -> OpenAIEmbedding:
        """Get the OpenAI embedding model instance"""
        return self._embed_model

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name"""
        return self._model_name

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "openai"

    def validate_config(self) -> None:
        """Validate OpenAI configuration

        Raises:
            ValueError: If API key is not set
        """
        if not self._api_key:
            raise ValueError(
                f"OpenAI API key not found. Set {self._api_key_env} environment variable."
            )
