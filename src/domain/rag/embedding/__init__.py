
"""Embedding providers using Strategy Pattern

This module provides a clean abstraction for different embedding providers,
making it easy to add new providers without modifying existing code.
"""

from .base import IEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .huggingface_provider import HuggingFaceEmbeddingProvider
from .vllm_provider import VLLMEmbeddingProvider
from .ollama_provider import OllamaEmbeddingProvider
from .azure_provider import AzureOpenAIEmbeddingProvider
from .factory import EmbeddingProviderFactory

__all__ = [
    'IEmbeddingProvider',
    'OpenAIEmbeddingProvider',
    'HuggingFaceEmbeddingProvider',
    'VLLMEmbeddingProvider',
    'OllamaEmbeddingProvider',
    'AzureOpenAIEmbeddingProvider',
    'EmbeddingProviderFactory',
]
