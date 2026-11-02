
"""OpenAI embedding provider implementation"""

import os
from typing import Optional
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from .base import IEmbeddingProvider
from src.log import get_api_logger

logger = get_api_logger()


OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    """OpenAI / OpenAI-compatible embedding provider

    Calls ``POST {base_url}/embeddings``. ``base_url`` is honoured (it used to be
    dropped); empty means the official OpenAI endpoint. Uses OpenAILikeEmbedding, which
    accepts any model id — llama-index's OpenAIEmbedding rejects names outside OpenAI's own
    catalogue, so an OpenAI-compatible cloud serving e.g. BAAI/bge-m3 would fail.
    Requires an API key (config ``api_key: "${OPENAI_API_KEY}"`` or the env var).

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
        dimension: int = 1536,
        base_url: Optional[str] = None,
        tuning: Optional[dict] = None,
        provider_name: str = "openai",
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
        self._base_url = base_url or OPENAI_DEFAULT_BASE_URL
        self._provider_name = provider_name

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
            self._embed_model = OpenAILikeEmbedding(
                model_name=model_name,
                api_key=api_key or "missing",
                api_base=self._base_url,
                **(tuning or {}),  # timeout / max_retries / dimensions
            )
            logger.info(
                f"OpenAI-compatible embedding provider initialized: {model_name} at "
                f"{self._base_url} (dim={dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding provider: {e}")
            raise

    def get_embed_model(self) -> OpenAILikeEmbedding:
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
        return self._provider_name

    def validate_config(self) -> None:
        """Validate OpenAI configuration

        Raises:
            ValueError: If API key is not set
        """
        if not self._api_key:
            raise ValueError(
                f"OpenAI API key not found. Set {self._api_key_env} environment variable."
            )
