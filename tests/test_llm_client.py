"""LLM client factory (src/domain/rag/llm_client.py) + embedding base_url.

HTTP goes to an in-process fake server (tests/fake_model_server.py); no keys, no network.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.domain.rag.llm_client import create_openai_client, model_name, role_llm
from tests.fake_model_server import FakeModelServer


def _llm(**kw):
    base = dict(provider="openai-compatible", model="m", base_url="http://x/v1", api_key=None,
                api_version=None, azure_deployment=None, timeout=12.5, max_retries=4)
    base.update(kw)
    return SimpleNamespace(**base)


class TestCreateClient:
    def test_openai_honours_base_url(self):
        c = create_openai_client(_llm(provider="openai", api_key="sk-x", base_url="https://gw.example/v1"))
        assert str(c.base_url).startswith("https://gw.example/v1")
        assert c.timeout == 12.5 and c.max_retries == 4

    def test_openai_without_base_url_uses_official_endpoint(self):
        c = create_openai_client(_llm(provider="openai", api_key="sk-x", base_url=None))
        assert "api.openai.com" in str(c.base_url)

    def test_openai_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert create_openai_client(_llm(provider="openai", api_key=None)).api_key == "sk-env"

    def test_openai_requires_a_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key is required"):
            create_openai_client(_llm(provider="openai", api_key=None))

    @pytest.mark.parametrize("provider", ["openai-compatible", "vllm", "ollama"])
    def test_compatible_requires_base_url(self, provider):
        with pytest.raises(ValueError, match="requires base_url"):
            create_openai_client(_llm(provider=provider, base_url=None))

    def test_compatible_placeholder_key(self):
        c = create_openai_client(_llm())
        assert c.api_key == "not-needed"

    def test_defaults_when_tuning_missing(self):
        c = create_openai_client(SimpleNamespace(provider="vllm", base_url="http://x/v1", api_key=None))
        assert c.timeout == 120.0 and c.max_retries == 2

    def test_azure(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="api_key is required"):
            create_openai_client(_llm(provider="azure", api_key=None))
        with pytest.raises(ValueError, match="endpoint"):
            create_openai_client(_llm(provider="azure", api_key="k", base_url=None))
        c = create_openai_client(_llm(provider="azure", api_key="k", base_url="https://r.openai.azure.com"))
        assert c.max_retries == 4

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_openai_client(_llm(provider="groq-native"))

    def test_real_round_trip_against_fake_server(self):
        with FakeModelServer() as srv:
            srv.chat_reply = lambda body: f"echo:{body['messages'][0]['content']}"
            c = create_openai_client(_llm(base_url=srv.url + "/v1", api_key="k"))
            r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
            assert r.choices[0].message.content == "echo:hi"
            assert srv.requests[-1][2]["Authorization"] == "Bearer k"

    def test_sdk_retries_on_429(self):
        with FakeModelServer() as srv:
            srv.fail_next = [(429, "0"), (503, None)]
            c = create_openai_client(_llm(base_url=srv.url + "/v1", max_retries=2))
            r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
            assert r.choices[0].message.content == "ok"
            assert len(srv.posts("/chat/completions")) == 3


class TestRoleResolution:
    def test_model_name_prefers_azure_deployment(self):
        assert model_name(_llm(azure_deployment="dep")) == "dep"
        assert model_name(_llm()) == "m"

    def test_role_llm_reads_live_config(self):
        from src.config.model import RAGConfig
        rag = RAGConfig.model_validate({
            "embedding": {"provider": "openai", "model": "e", "dimension": 8},
            "chunking": {}, "retrieval": {"default_top_k": 1, "default_similarity_cutoff": 0,
                                          "hybrid_search": False},
            "vector_store": {"type": "pgvector", "table_prefix": "v"},
            "llm": {"provider": "openai", "model": "base"},
            "answer_llm": {"model": "big"},
        })
        with patch("src.config.config_manager.Config.get_config_model",
                   return_value=SimpleNamespace(rag=rag)):
            assert role_llm("answer_llm").model == "big"
            assert role_llm("eval_llm").model == "base"

    def test_role_llm_unset_raises(self):
        with patch("src.config.config_manager.Config.get_config_model", return_value=None):
            with pytest.raises(ValueError, match="No LLM configured"):
                role_llm("answer_llm")

    def test_context_generator_uses_contextual_role(self):
        from src.domain.rag.context_generator import ContextGenerator
        cfg = SimpleNamespace(
            contextual_retrieval=SimpleNamespace(enabled=True, max_context_length=100),
            llm=SimpleNamespace(provider="vllm", model="base", base_url="http://x/v1",
                                api_key=None, api_version=None, azure_deployment=None),
            contextual_llm=None,
        )
        assert ContextGenerator.from_config(cfg)._model == "base"


class TestEmbeddingBaseUrl:
    """Regression: the `openai` embedding provider used to ignore base_url."""

    def _cfg(self, **kw):
        from src.config.model import RAGEmbeddingConfig
        base = dict(provider="openai", model="BAAI/bge-m3", dimension=8, api_key="sk-x",
                    base_url="https://gw.example/v1", timeout=7.0, max_retries=5)
        base.update(kw)
        return RAGEmbeddingConfig(**base)

    def test_openai_provider_keeps_base_url_and_tuning(self):
        from src.domain.rag.embedding.factory import EmbeddingProviderFactory
        p = EmbeddingProviderFactory.create_from_config(self._cfg())
        m = p.get_embed_model()
        assert m.api_base == "https://gw.example/v1"
        assert m.timeout == 7.0 and m.max_retries == 5
        assert p.get_provider_name() == "openai"

    def test_request_dimensions(self):
        from src.domain.rag.embedding.factory import EmbeddingProviderFactory
        m = EmbeddingProviderFactory.create_from_config(
            self._cfg(request_dimensions=True, dimension=256)).get_embed_model()
        assert m.dimensions == 256

    def test_openai_compatible_requires_base_url(self):
        from src.domain.rag.embedding.factory import EmbeddingProviderFactory
        with pytest.raises(ValueError, match="requires base_url"):
            EmbeddingProviderFactory.create_from_config(self._cfg(provider="openai-compatible", base_url=None))

    @pytest.mark.parametrize("provider", ["vllm", "ollama", "openai-compatible"])
    def test_http_providers_get_tuning(self, provider):
        from src.domain.rag.embedding.factory import EmbeddingProviderFactory
        p = EmbeddingProviderFactory.create_from_config(self._cfg(provider=provider))
        assert p.get_embed_model().timeout == 7.0 and p.get_provider_name() == provider

    def test_embeds_through_configured_endpoint(self):
        from src.domain.rag.embedding.factory import EmbeddingProviderFactory
        with FakeModelServer(dim=8) as srv:
            p = EmbeddingProviderFactory.create_from_config(
                self._cfg(base_url=srv.url + "/v1", max_retries=0))
            vec = p.get_embed_model().get_text_embedding("hello")
            assert len(vec) == 8
            assert srv.posts("/embeddings")[0]["model"] == "BAAI/bge-m3"
            assert srv.requests[0][2]["Authorization"] == "Bearer sk-x"
