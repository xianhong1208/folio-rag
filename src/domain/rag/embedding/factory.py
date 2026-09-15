
"""Factory for creating embedding providers

This factory encapsulates the logic for instantiating the correct embedding
provider based on configuration, following the Factory Pattern.
"""

from .base import IEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .huggingface_provider import HuggingFaceEmbeddingProvider
from .vllm_provider import VLLMEmbeddingProvider
from .ollama_provider import OllamaEmbeddingProvider
from .azure_provider import AzureOpenAIEmbeddingProvider
from src.log import get_api_logger

logger = get_api_logger()


class EmbeddingProviderFactory:
    """Factory for creating embedding providers

    This class implements the Factory Pattern to create the appropriate
    embedding provider based on configuration.

    Benefits:
    - Centralizes provider creation logic
    - Easy to add new providers (just add a new case)
    - Configuration validation before creation
    - Clear error messages for misconfiguration

    Usage:
        from src.config.config_manager import Config

        config = Config.get_config_model()
        provider = EmbeddingProviderFactory.create_from_config(config.rag.embedding)
        embed_model = provider.get_embed_model()
    """

    @staticmethod
    def create(
        provider_type: str,
        model_name: str,
        dimension: int,
        **kwargs
    ) -> IEmbeddingProvider:
        """Create an embedding provider

        Args:
            provider_type: Provider type ('openai', 'huggingface', 'vllm', 'ollama')
            model_name: Model identifier
            dimension: Embedding dimension
            **kwargs: Additional provider-specific arguments

        Returns:
            IEmbeddingProvider instance

        Raises:
            ValueError: If provider_type is unknown or configuration is invalid

        Example:
            provider = EmbeddingProviderFactory.create(
                provider_type='openai',
                model_name='text-embedding-3-small',
                dimension=1536,
                api_key='sk-...'
            )
        """
        provider_type = provider_type.lower().strip()

        if provider_type == "openai":
            return EmbeddingProviderFactory._create_openai(model_name, dimension, **kwargs)

        elif provider_type in ["huggingface", "local"]:
            return EmbeddingProviderFactory._create_huggingface(model_name, dimension, **kwargs)

        elif provider_type == "vllm":
            return EmbeddingProviderFactory._create_vllm(model_name, dimension, **kwargs)

        elif provider_type == "ollama":
            return EmbeddingProviderFactory._create_ollama(model_name, dimension, **kwargs)

        elif provider_type in ["azure", "azure_openai"]:
            return EmbeddingProviderFactory._create_azure(model_name, dimension, **kwargs)

        else:
            raise ValueError(
                f"Unknown embedding provider: '{provider_type}'. "
                f"Supported providers: openai, azure, huggingface, local, vllm, ollama"
            )

    @staticmethod
    def create_from_config(embedding_config) -> IEmbeddingProvider:
        """Create provider from configuration object

        Args:
            embedding_config: EmbeddingConfig from config.yaml

        Returns:
            IEmbeddingProvider instance

        Raises:
            ValueError: If configuration is invalid

        Example:
            from src.config.config_manager import Config

            config = Config.get_config_model()
            provider = EmbeddingProviderFactory.create_from_config(config.rag.embedding)
        """
        provider_type = embedding_config.provider
        model_name = embedding_config.model
        dimension = embedding_config.dimension
        base_url = getattr(embedding_config, 'base_url', None)
        api_key = getattr(embedding_config, 'api_key', None)
        logger.debug(f"Embedding config: {embedding_config}")
        kwargs = {}

        # Add provider-specific parameters
        if provider_type in ["vllm", "ollama"]:
            kwargs['base_url'] = base_url
            kwargs['api_key'] = api_key

        elif provider_type == "openai":
            if api_key:
                kwargs['api_key'] = api_key

        elif provider_type in ["azure", "azure_openai"]:
            kwargs['api_key'] = api_key
            kwargs['azure_endpoint'] = base_url
            kwargs['api_version'] = getattr(embedding_config, 'api_version', '2024-02-01')
            kwargs['azure_deployment'] = getattr(embedding_config, 'azure_deployment', None)

        elif provider_type in ["huggingface", "local"]:
            # HuggingFace-specific settings could be added here
            kwargs['base_url'] = base_url
            kwargs['api_key'] = api_key

        logger.info(f"Creating embedding provider: {provider_type} with model {model_name}")

        provider = EmbeddingProviderFactory.create(
            provider_type=provider_type,
            model_name=model_name,
            dimension=dimension,
            **kwargs
        )

        # Validate configuration
        provider.validate_config()

        return provider

    @staticmethod
    def _create_openai(
        model_name: str,
        dimension: int,
        **kwargs
    ) -> OpenAIEmbeddingProvider:
        """Create OpenAI provider"""
        api_key = kwargs.get('api_key')
        api_key_env = kwargs.get('api_key_env', 'OPENAI_API_KEY')

        return OpenAIEmbeddingProvider(
            model_name=model_name,
            api_key=api_key,
            api_key_env=api_key_env,
            dimension=dimension
        )

    @staticmethod
    def _create_huggingface(
        model_name: str,
        dimension: int,
        **kwargs
    ) -> HuggingFaceEmbeddingProvider:
        """Create HuggingFace provider"""
        embed_batch_size = kwargs.get('embed_batch_size', 10)
        device = kwargs.get('device', 'cpu')

        return HuggingFaceEmbeddingProvider(
            model_name=model_name,
            dimension=dimension,
            embed_batch_size=embed_batch_size,
            device=device
        )

    @staticmethod
    def _create_vllm(
        model_name: str,
        dimension: int,
        **kwargs
    ) -> VLLMEmbeddingProvider:
        """Create vLLM provider"""
        base_url = kwargs.get('base_url', 'http://localhost:1234/v1')
        api_key = kwargs.get('api_key', None)

        return VLLMEmbeddingProvider(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            dimension=dimension
        )

    @staticmethod
    def _create_ollama(
        model_name: str,
        dimension: int,
        **kwargs
    ) -> OllamaEmbeddingProvider:
        """Create Ollama provider"""
        base_url = kwargs.get('base_url', 'http://localhost:11434/v1')
        api_key = kwargs.get('api_key', None)

        return OllamaEmbeddingProvider(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            dimension=dimension
        )

    @staticmethod
    def _create_azure(
        model_name: str,
        dimension: int,
        **kwargs
    ) -> AzureOpenAIEmbeddingProvider:
        """Create Azure OpenAI provider"""
        api_key = kwargs.get('api_key')
        azure_endpoint = kwargs.get('azure_endpoint')
        api_version = kwargs.get('api_version', '2024-02-01')
        azure_deployment = kwargs.get('azure_deployment')

        return AzureOpenAIEmbeddingProvider(
            model_name=model_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=azure_deployment,
            dimension=dimension,
        )

    @staticmethod
    def list_supported_providers() -> list[str]:
        """Get list of supported provider types

        Returns:
            List of supported provider type strings
        """
        return ['openai', 'azure', 'huggingface', 'local', 'vllm', 'ollama']
