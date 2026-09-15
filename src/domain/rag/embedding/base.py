
"""Base interface for embedding providers

This module defines the contract that all embedding providers must follow,
enabling the Strategy Pattern implementation.
"""

from abc import ABC, abstractmethod
from typing import Any


class IEmbeddingProvider(ABC):
    """Interface for embedding providers

    All embedding providers must implement this interface to ensure
    consistent behavior across different implementations.

    Benefits of this approach:
    - Easy to add new providers (just implement this interface)
    - Easy to swap providers at runtime
    - Each provider is isolated and testable
    - No need to modify existing code when adding new providers (OCP)
    """

    @abstractmethod
    def get_embed_model(self) -> Any:
        """Get the LlamaIndex embedding model instance

        Returns:
            LlamaIndex embedding model (OpenAIEmbedding, HuggingFaceEmbedding, etc.)

        This method returns the actual LlamaIndex embedding model instance
        that will be used by Settings.embed_model.
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension

        Returns:
            int: Dimension of the embedding vectors

        This is used when creating PGVector tables to ensure the
        correct vector dimension.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name

        Returns:
            str: Name/identifier of the embedding model

        This is used for logging and tracking which model was used
        for indexing.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name

        Returns:
            str: Name of the provider (e.g., 'openai', 'huggingface', 'vllm')

        This is used for logging and debugging.
        """
        pass

    def validate_config(self) -> None:
        """Validate provider configuration

        Raises:
            ValueError: If configuration is invalid

        Override this method to add provider-specific validation.
        For example, OpenAI provider can check if API key is set.
        """
        pass

    def __str__(self) -> str:
        """String representation"""
        return f"{self.get_provider_name()}:{self.get_model_name()}"

    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return f"<{self.__class__.__name__}(model='{self.get_model_name()}', dim={self.get_dimension()})>"
