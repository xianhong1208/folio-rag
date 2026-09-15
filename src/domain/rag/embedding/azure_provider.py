
"""Azure OpenAI embedding provider implementation"""

import os
from typing import Optional
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from .base import IEmbeddingProvider
from src.log import get_api_logger

logger = get_api_logger()


class AzureOpenAIEmbeddingProvider(IEmbeddingProvider):
    """Azure OpenAI embedding provider

    Uses Azure's OpenAI Service for generating embeddings.
    Requires azure_endpoint, api_key, api_version, and azure_deployment.

    Supported models:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        azure_deployment: Optional[str] = None,
        dimension: int = 1536,
    ):
        """Initialize Azure OpenAI provider

        Args:
            model_name: Model name (e.g., text-embedding-3-small)
            api_key: Azure OpenAI API key (if None, reads from AZURE_OPENAI_API_KEY env)
            azure_endpoint: Azure endpoint URL (e.g., https://<resource>.openai.azure.com/)
            api_version: Azure API version (default: 2024-02-01)
            azure_deployment: Azure deployment name (if None, uses model_name)
            dimension: Embedding dimension
        """
        self._model_name = model_name
        self._dimension = dimension
        self._api_version = api_version
        self._azure_deployment = azure_deployment or model_name

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")

        if not api_key:
            logger.warning(
                "Azure OpenAI API key not found. "
                "Set AZURE_OPENAI_API_KEY environment variable or provide api_key in config."
            )

        self._api_key = api_key

        # Get endpoint from parameter or environment
        if azure_endpoint is None:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

        self._azure_endpoint = azure_endpoint

        # Create embedding model instance
        try:
            self._embed_model = AzureOpenAIEmbedding(
                model=model_name,
                deployment_name=self._azure_deployment,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
            logger.info(
                f"Azure OpenAI embedding provider initialized: "
                f"{model_name} (deployment={self._azure_deployment}, dim={dimension})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI embedding provider: {e}")
            raise

    def get_embed_model(self) -> AzureOpenAIEmbedding:
        """Get the Azure OpenAI embedding model instance"""
        return self._embed_model

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name"""
        return self._model_name

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "azure"

    def validate_config(self) -> None:
        """Validate Azure OpenAI configuration

        Raises:
            ValueError: If required configuration is missing
        """
        if not self._api_key:
            raise ValueError(
                "Azure OpenAI API key not found. "
                "Set AZURE_OPENAI_API_KEY environment variable or provide api_key in config."
            )
        if not self._azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not found. "
                "Set AZURE_OPENAI_ENDPOINT environment variable or provide base_url in config."
            )
        if not self._azure_deployment:
            raise ValueError("Azure deployment name is required.")
