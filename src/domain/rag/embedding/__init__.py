"""Embedding providers using Strategy Pattern

This module provides a clean abstraction for different embedding providers,
making it easy to add new providers without modifying existing code.

The HuggingFace provider is imported lazily (PEP 562 ``__getattr__``): importing it pulls
in sentence-transformers -> transformers -> torch, which a cloud deployment must never
load. Every HTTP provider is cheap to import.
"""

from .base import IEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
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


def __getattr__(name):
    if name == "HuggingFaceEmbeddingProvider":
        from .huggingface_provider import HuggingFaceEmbeddingProvider
        return HuggingFaceEmbeddingProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
