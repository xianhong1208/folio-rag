"""API keys and DB passwords must never reach the logs.

The embedding factory used to log its whole config object at DEBUG, and the
pydantic repr of that object includes api_key in plain text.
"""
import pytest

from src.config.model import (
    AsrConfig,
    DatabaseConfig,
    RAGEmbeddingConfig,
    RAGLLMConfig,
    RerankConfig,
)
from src.domain.rag.embedding import factory as factory_mod

SECRET = "sk-very-secret-value-123"


def _embedding_config(**kw):
    base = dict(provider="openai", model="text-embedding-3-small", dimension=1536,
                api_key=SECRET, base_url="http://embed.local/v1", table_prefix="data")
    base.update(kw)
    return RAGEmbeddingConfig(**base)


def _logged_text(logger_mock) -> str:
    parts = []
    for method in ("debug", "info", "warning", "error", "exception"):
        for call in getattr(logger_mock, method).call_args_list:
            parts.extend(str(a) for a in call.args)
            parts.extend(str(v) for v in call.kwargs.values())
    return "\n".join(parts)


@pytest.mark.parametrize("provider", ["openai", "vllm", "azure", "ollama", "huggingface"])
def test_embedding_factory_never_logs_the_api_key(mocker, provider):
    logger = mocker.patch.object(factory_mod, "logger")
    create = mocker.patch.object(factory_mod.EmbeddingProviderFactory, "create")
    factory_mod.EmbeddingProviderFactory.create_from_config(_embedding_config(provider=provider))
    text = _logged_text(logger)
    assert SECRET not in text
    assert "api_key=set" in text  # still says whether a key is configured
    # The key itself is still handed to the provider.
    assert SECRET in str(create.call_args)


def test_embedding_factory_reports_unset_key(mocker):
    logger = mocker.patch.object(factory_mod, "logger")
    mocker.patch.object(factory_mod.EmbeddingProviderFactory, "create")
    factory_mod.EmbeddingProviderFactory.create_from_config(_embedding_config(api_key=None))
    assert "api_key=unset" in _logged_text(logger)


@pytest.mark.parametrize("cfg", [
    _embedding_config(),
    RAGLLMConfig(api_key=SECRET),
    RerankConfig(api_key=SECRET),
    AsrConfig(api_key=SECRET),
])
def test_config_repr_hides_api_key(cfg):
    assert SECRET not in repr(cfg) and SECRET not in str(cfg)
    assert cfg.api_key == SECRET  # value itself unchanged


def test_database_config_repr_hides_password_and_url():
    cfg = DatabaseConfig(url=f"postgresql://postgres:{SECRET}@db/rag", echo=False,
                         pool_size=5, max_overflow=10, pool_timeout=30,
                         pool_recycle=3600, password=SECRET, port=5444)
    assert SECRET not in repr(cfg) and SECRET not in str(cfg)
    assert "5444" in repr(cfg)  # non-secret fields are still shown
