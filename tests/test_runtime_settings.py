
"""Runtime settings hot-reload (feat/runtime-settings).

Contract:
- Anything outside the whitelist is rejected; type validation is all-or-nothing
  (one item invalid → none written).
- In-place mutation: a consumer holding a section reference (context_generator
  holds llm_config) sees the change immediately.
- warn-level items (the embedding family) carry an impact warning.
- An optional section (rag.asr may be None) is auto-created when patched.
- service: persists the write + rebinds; when the adapter is not yet built, only
  the config is written (first build picks up the new value).
- rebind: the reranker can go off→on (fixing boot-disabled being forever None),
  and the QE cache must be cleared.
- VSM.update_embed_dim: same dimension is a no-op, a changed dimension clears the
  store cache.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config import runtime_overrides as ro
from src.config.model import ConfigModel


def _mk_config() -> ConfigModel:
    return ConfigModel.model_validate({
        "server": {"host": "0.0.0.0", "port": 1, "transport": "http"},
        "database": {"url": "postgresql://x/y", "echo": False, "pool_size": 1,
                     "max_overflow": 1, "pool_timeout": 1, "pool_recycle": 1},
        "auth": {"enabled": False, "token_server_url": "http://x"},
        "logging": {"level": "INFO", "format": "plain", "file": "x.log",
                    "max_size": "1MB", "backup_count": 1},
        "app": {},
        "modules": {"enabled": []},
        "rag": {
            "embedding": {"provider": "vllm", "model": "intfloat/multilingual-e5-large",
                          "dimension": 1024, "base_url": "http://x/v1"},
            "chunking": {"hierarchy_sizes": [1024, 256], "chunk_overlap": 50},
            "retrieval": {"default_top_k": 10, "default_similarity_cutoff": 0.25,
                          "hybrid_search": True,
                          "auto_merging": {"enabled": True, "merge_threshold": 0.5}},
            "vector_store": {"type": "pgvector", "table_prefix": "vector_store"},
            "llm": {"provider": "vllm", "model": "m", "base_url": "http://x/v1"},
            "rerank": {"enabled": False, "model": "r", "base_url": "http://x"},
            "contextual_retrieval": {"enabled": True},
            "asr": {"enabled": True, "provider": "docling-whisper"},
        },
    })


class TestValidateAndApply:
    def test_whitelist_rejects_unknown_path(self):
        cfg = _mk_config()
        with pytest.raises(ValueError, match="cannot be changed at runtime"):
            ro.validate_and_apply(cfg, {"database.url": "postgresql://evil"})

    def test_type_error_is_all_or_nothing(self):
        cfg = _mk_config()
        before = cfg.rag.retrieval.default_top_k
        with pytest.raises(ValueError, match="Settings validation failed"):
            ro.validate_and_apply(cfg, {
                "rag.retrieval.default_top_k": 99,          # valid
                "rag.rerank.score_threshold": "not-a-num",  # invalid
            })
        assert cfg.rag.retrieval.default_top_k == before  # the valid item was not written either

    def test_in_place_mutation_visible_via_held_reference(self):
        # A consumer (e.g. context_generator) holds a section reference — an in-place change must be immediately visible
        cfg = _mk_config()
        llm_ref = cfg.rag.llm
        ro.validate_and_apply(cfg, {"rag.llm.model": "new-model"})
        assert llm_ref.model == "new-model"

    def test_warn_level_returns_warning(self):
        cfg = _mk_config()
        applied, warnings = ro.validate_and_apply(
            cfg, {"rag.embedding.model": "BAAI/bge-m3"})
        assert applied == {"rag.embedding.model": "BAAI/bge-m3"}
        assert warnings and "reindexed" in warnings[0]

    def test_live_level_no_warning(self):
        cfg = _mk_config()
        _, warnings = ro.validate_and_apply(cfg, {"rag.rerank.score_threshold": 0.5})
        assert warnings == []

    def test_optional_section_auto_created(self):
        # rag.asr unset (None) → patch auto-creates the default section before writing
        cfg = _mk_config()
        cfg.rag.asr = None
        ro.validate_and_apply(cfg, {"rag.asr.provider": "fireredasr"})
        assert cfg.rag.asr is not None
        assert cfg.rag.asr.provider == "fireredasr"

    def test_coercion_follows_pydantic(self):
        # "12" (a numeric string) → pydantic's lenient coercion, consistent with yaml behavior
        cfg = _mk_config()
        applied, _ = ro.validate_and_apply(cfg, {"rag.retrieval.default_top_k": "12"})
        assert applied["rag.retrieval.default_top_k"] == 12


class TestEffectiveView:
    def test_secrets_masked(self):
        cfg = _mk_config()
        cfg.rag.rerank.api_key = "super-secret"
        eff = ro.get_effective(cfg)
        assert eff["rag.rerank.api_key"] == "•••"
        assert eff["rag.rerank.score_threshold"] == cfg.rag.rerank.score_threshold

    def test_all_whitelist_paths_resolvable(self):
        # Every whitelist path must resolve to a ConfigModel field — this turns red when the schema is renamed
        cfg = _mk_config()
        eff = ro.get_effective(cfg, mask_secrets=False)
        assert set(eff.keys()) == set(ro.EDITABLE.keys())

    def test_rebind_groups_dedup_and_order(self):
        gs = ro.rebind_groups_for([
            "rag.rerank.enabled", "rag.rerank.model", "rag.embedding.model"])
        assert gs == ["rag.embedding", "rag.rerank"]


_SVC = "src.adapter.runtime_settings_service"


class TestService:
    def _with_cfg(self, cfg):
        return patch(f"{_SVC}.Config.get_config_model", return_value=cfg)

    def test_apply_persists_each_key(self):
        cfg = _mk_config()
        from src.adapter import runtime_settings_service as svc
        with self._with_cfg(cfg), \
             patch(f"{_SVC}.RuntimeSettingsDB") as db, \
             patch(f"{_SVC}._rebind") as rb:
            applied, _ = svc.apply_settings(
                {"rag.rerank.score_threshold": 0.4}, updated_by="tok…")
        db.upsert.assert_called_once_with(
            "rag.rerank.score_threshold", 0.4, updated_by="tok…")
        rb.assert_called_once()
        assert cfg.rag.rerank.score_threshold == 0.4

    def test_invalid_patch_never_touches_db(self):
        cfg = _mk_config()
        from src.adapter import runtime_settings_service as svc
        with self._with_cfg(cfg), \
             patch(f"{_SVC}.RuntimeSettingsDB") as db:
            with pytest.raises(ValueError):
                svc.apply_settings({"server.port": 1})
        db.upsert.assert_not_called()

    def test_rebind_noop_when_adapter_not_built(self):
        from src.adapter import runtime_settings_service as svc
        with patch("src.adapter.rag.peek_rag_adapter", return_value=None):
            assert svc._rebind(["rag.rerank.enabled"]) == []

    def test_rebind_dispatches_to_ctx(self):
        from src.adapter import runtime_settings_service as svc
        adapter = MagicMock()
        with patch("src.adapter.rag.peek_rag_adapter", return_value=adapter):
            groups = svc._rebind(["rag.rerank.enabled", "rag.llm.model"])
        adapter._ctx.apply_runtime_changes.assert_called_once_with(groups)
        assert groups == ["rag.llm", "rag.rerank"]

    def test_startup_overlay_skips_invalid_rows(self):
        cfg = _mk_config()
        from src.adapter import runtime_settings_service as svc
        with self._with_cfg(cfg), patch("db.db.get_engine"), \
             patch(f"{_SVC}.RuntimeSettingsDB") as db:
            db.load_all.return_value = {
                "rag.retrieval.default_top_k": 15,
                "no.such.path": 1,                      # leftover from a whitelist revision → skipped
                "rag.rerank.score_threshold": "junk",   # bad data → skipped
            }
            applied = svc.load_overrides_on_startup()
        assert applied == 1
        assert cfg.rag.retrieval.default_top_k == 15

    def test_startup_overlay_binds_the_engine_before_reading(self):
        """main.py overlays the overrides before it calls init_db. Without binding the
        session first, every read failed ("Could not locate a bind") and the saved
        console settings were silently dropped on each restart."""
        cfg = _mk_config()
        from src.adapter import runtime_settings_service as svc
        order = []
        with self._with_cfg(cfg), \
             patch("db.db.get_engine", side_effect=lambda: order.append("bind")) as ge, \
             patch(f"{_SVC}.RuntimeSettingsDB") as db:
            db.load_all.side_effect = lambda: order.append("read") or {"rag.retrieval.default_top_k": 9}
            assert svc.load_overrides_on_startup() == 1
        ge.assert_called_once()
        assert order == ["bind", "read"]
        assert cfg.rag.retrieval.default_top_k == 9

    def test_reset_without_override_returns_false(self):
        from src.adapter import runtime_settings_service as svc
        cfg = _mk_config()
        with self._with_cfg(cfg), patch(f"{_SVC}.RuntimeSettingsDB") as db, \
             patch(f"{_SVC}._rebind") as rb:
            db.delete.return_value = False
            assert svc.reset_setting("rag.rerank.score_threshold") is False
        rb.assert_not_called()


class TestCtxRebind:
    def _ctx(self):
        from src.adapter.rag_context import RAGContext
        return RAGContext(
            embedding_provider=MagicMock(), embed_dim=1024, model_name="m",
            chunker_tokenizer_name=None, leaf_chunk_size=256,
            parent_target_tokens=1024, chunk_overlap=50,
            vector_store_manager=MagicMock(), indexer=MagicMock(),
            indexing_service=MagicMock(), auto_merging_enabled=True,
            merge_threshold=0.5, expand_neighbors=2, reranker=None,
            query_engines={"1": MagicMock()},
        )

    def test_rerank_off_to_on_and_cache_cleared(self):
        # Disabled at boot → reranker None; a hot-reload enable must be able to bring it into being (fixes an old defect)
        ctx = self._ctx()
        cfg = _mk_config()
        cfg.rag.rerank.enabled = True
        fake = MagicMock()
        with patch("src.adapter.rag_context.Config.get_config_model", return_value=cfg), \
             patch("src.domain.rag.reranker.Reranker") as R:
            R.from_config.return_value = fake
            ctx.apply_runtime_changes(["rag.rerank"])
        assert ctx.reranker is fake
        assert ctx.query_engines == {}  # the cache froze the old reranker → must be cleared

    def test_rerank_disable_sets_none(self):
        ctx = self._ctx()
        ctx.reranker = MagicMock()
        cfg = _mk_config()
        cfg.rag.rerank.enabled = False
        with patch("src.adapter.rag_context.Config.get_config_model", return_value=cfg):
            ctx.apply_runtime_changes(["rag.rerank"])
        assert ctx.reranker is None

    def test_retrieval_scalars_rebound(self):
        ctx = self._ctx()
        cfg = _mk_config()
        cfg.rag.retrieval.auto_merging.merge_threshold = 0.7
        cfg.rag.retrieval.expand_context_neighbors = 9
        with patch("src.adapter.rag_context.Config.get_config_model", return_value=cfg):
            ctx.apply_runtime_changes(["rag.retrieval"])
        assert ctx.merge_threshold == 0.7
        assert ctx.expand_neighbors == 9

    def test_llm_group_rebuilds_indexer(self):
        ctx = self._ctx()
        old_indexer = ctx.indexer
        cfg = _mk_config()
        with patch("src.adapter.rag_context.Config.get_config_model", return_value=cfg), \
             patch("src.adapter.rag_context.HierarchicalIndexer") as HI, \
             patch("src.domain.rag.context_generator.ContextGenerator") as CG:
            CG.from_config.return_value = None
            ctx.apply_runtime_changes(["rag.llm"])
        assert ctx.indexer is HI.return_value
        assert ctx.indexer is not old_indexer

    def test_embedding_group_full_rebind(self):
        ctx = self._ctx()
        cfg = _mk_config()
        pi = {"provider": MagicMock(), "embed_dim": 768, "model_name": "new-m",
              "chunker_tokenizer_name": "new-m", "embed_model": MagicMock()}
        with patch("src.adapter.rag_context.Config.get_config_model", return_value=cfg), \
             patch.object(type(ctx), "_load_embedding_provider", return_value=pi), \
             patch("src.adapter.rag_context.HierarchicalIndexer"), \
             patch("src.domain.rag.context_generator.ContextGenerator") as CG:
            CG.from_config.return_value = None
            ctx.apply_runtime_changes(["rag.embedding"])
        assert ctx.embed_dim == 768 and ctx.model_name == "new-m"
        ctx.vector_store_manager.update_embed_dim.assert_called_once_with(768)
        assert ctx.query_engines == {}


class TestVsmDimUpdate:
    def _vsm(self):
        from src.domain.rag.vector_store_manager import VectorStoreManager
        with patch("src.domain.rag.vector_store_manager.Config.get_config_model") as g:
            g.return_value.database.url = "postgresql://x/y"
            g.return_value.database.port = 5432
            return VectorStoreManager(embed_dim=1024)

    def test_same_dim_noop_keeps_cache(self):
        vsm = self._vsm()
        vsm._vector_stores["t"] = MagicMock()
        vsm.update_embed_dim(1024)
        assert "t" in vsm._vector_stores

    def test_new_dim_clears_cache(self):
        vsm = self._vsm()
        vsm._vector_stores["t"] = MagicMock()
        vsm.update_embed_dim(768)
        assert vsm._embed_dim == 768
        assert vsm._vector_stores == {}


class TestServiceViewAndReset:
    def _with_cfg(self, cfg):
        return patch(f"{_SVC}.Config.get_config_model", return_value=cfg)

    def test_settings_view_marks_overrides_and_secrets(self):
        cfg = _mk_config()
        from src.adapter import runtime_settings_service as svc
        with self._with_cfg(cfg), patch(f"{_SVC}.RuntimeSettingsDB") as db:
            db.load_all.return_value = {"rag.rerank.score_threshold": 0.4}
            view = svc.get_settings_view()
        by_path = {s["path"]: s for s in view["settings"]}
        assert by_path["rag.rerank.score_threshold"]["overridden"] is True
        assert by_path["rag.rerank.enabled"]["overridden"] is False
        assert by_path["rag.embedding.api_key"]["secret"] is True
        assert by_path["rag.embedding.model"]["level"] == "warn"

    def test_reset_restores_yaml_value_and_rebinds(self, tmp_path):
        cfg = _mk_config()
        cfg.rag.rerank.score_threshold = 0.9  # currently overridden at runtime
        yaml_file = tmp_path / "c.yaml"
        yaml_file.write_text(
            "rag:\n  rerank:\n    score_threshold: 0.25\n", encoding="utf-8")
        from src.adapter import runtime_settings_service as svc
        from src.config.config_manager import Config
        with self._with_cfg(cfg), \
             patch(f"{_SVC}.RuntimeSettingsDB") as db, \
             patch(f"{_SVC}._rebind") as rb, \
             patch.object(Config, "_config_path", str(yaml_file)):
            db.delete.return_value = True
            assert svc.reset_setting("rag.rerank.score_threshold") is True
        assert cfg.rag.rerank.score_threshold == 0.25  # restored to the yaml factory value
        rb.assert_called_once_with(["rag.rerank.score_threshold"])

    def test_reset_unknown_path_rejected(self):
        from src.adapter import runtime_settings_service as svc
        with pytest.raises(ValueError, match="not editable"):
            svc.reset_setting("database.url")

    def test_yaml_value_falls_back_to_field_default(self):
        # The field is absent from yaml (or the file is unreadable) → the pydantic field-declared default
        cfg = _mk_config()
        from src.adapter import runtime_settings_service as svc
        from src.config.config_manager import Config
        with self._with_cfg(cfg), patch.object(Config, "_config_path", None):
            v = svc._yaml_value("rag.contextual_retrieval.reasoning_effort")
        from src.config.model import ContextualRetrievalConfig
        assert v == ContextualRetrievalConfig.model_fields["reasoning_effort"].get_default()


class TestValueConstraints:
    """The settings API rejects values that would hang or empty the service."""

    @pytest.mark.parametrize("path, value", [
        ("rag.contextual_retrieval.max_concurrent", 0),      # Semaphore(0) hangs indexing
        ("rag.contextual_retrieval.max_concurrent", -3),
        ("rag.rerank.score_threshold", 5),                   # empties every result
        ("rag.rerank.score_threshold", -0.1),
        ("rag.retrieval.default_top_k", 0),
        ("rag.retrieval.default_top_k", 500),
        ("rag.retrieval.default_similarity_cutoff", 1.5),
        ("rag.retrieval.default_hybrid_alpha", -1),
        ("rag.retrieval.auto_merging.merge_threshold", 2),
        ("rag.retrieval.expand_context_neighbors", -1),
        ("rag.retrieval.default_sparse_top_k", 0),
        ("rag.rerank.top_n", 0),
        ("rag.embedding.dimension", 0),
        ("rag.retrieval.hybrid_fusion", "foo"),
        ("rag.embedding.provider", "nope"),
        ("rag.llm.provider", "nope"),
        ("rag.asr.provider", "nope"),
        ("rag.contextual_retrieval.reasoning_effort", "max"),
        ("rag.rerank.base_url", "localhost:8787"),            # scheme missing
        ("rag.rerank.query_template", "no placeholder"),
        ("rag.embedding.model", "   "),
        ("rag.retrieval.segmenter", "mecab"),
        ("rag.rerank.gap_threshold", 1.5),
        # cloud-profile keys
        ("rag.llm.provider", "huggingface"),
        ("rag.answer_llm.provider", "nope"),
        ("rag.parsing.vision.provider", "huggingface"),
        ("rag.parsing.vision.base_url", "api.example.com/v1"),
        ("rag.parsing.vision.model", "  "),
        ("rag.rerank.api", "voyage"),
        ("rag.rerank.endpoint_path", "v2/rerank"),
        ("rag.rerank.max_retries", 9),
        ("rag.rerank.timeout", 0),
        ("rag.parsing.pdf", "remote"),
        ("rag.parsing.on_page_failure", "skip"),
        ("rag.parsing.max_concurrency", 0),
        ("rag.parsing.max_pages_per_file", -1),
        ("rag.asr.segment_seconds", 10),
        ("rag.llm.max_retries", 11),
        ("rag.contextual_retrieval.max_calls_per_file", -1),
    ])
    def test_out_of_range_or_unknown_rejected(self, path, value):
        cfg = _mk_config()
        before = ro.get_effective(cfg, mask_secrets=False)[path]
        with pytest.raises(ValueError, match="Settings validation failed"):
            ro.validate_and_apply(cfg, {path: value})
        assert ro.get_effective(cfg, mask_secrets=False)[path] == before

    @pytest.mark.parametrize("path, value", [
        ("rag.contextual_retrieval.max_concurrent", 1),
        ("rag.rerank.score_threshold", 0.0),
        ("rag.rerank.score_threshold", 1),
        ("rag.rerank.top_n", None),
        ("rag.retrieval.default_top_k", 20),
        ("rag.retrieval.hybrid_fusion", "concat"),
        ("rag.contextual_retrieval.reasoning_effort", None),
        ("rag.rerank.base_url", "https://rerank.example.com"),
        ("rag.asr.base_url", None),
        ("rag.asr.base_url", ""),
        ("rag.retrieval.segmenter", "jieba"),
        ("rag.retrieval.segmenter", "bigram"),
        ("rag.rerank.gap_threshold", 0.3),
        ("rag.rerank.auto_template", False),
        ("rag.retrieval.segmenter", None),          # unset = follow the profile
        ("rag.retrieval.segmenter", ""),
        # every new cloud-profile setting the console can save
        ("rag.llm.provider", "openai-compatible"),
        ("rag.embedding.provider", "openai-compatible"),
        ("rag.embedding.request_dimensions", True),
        ("rag.embedding.timeout", 30),
        ("rag.embedding.max_retries", 0),
        ("rag.llm.timeout", 90.5),
        ("rag.llm.max_retries", 3),
        ("rag.contextual_llm.provider", "openai-compatible"),
        ("rag.contextual_llm.model", "gpt-4o-mini"),
        ("rag.contextual_llm.base_url", "https://api.example.com/v1"),
        ("rag.answer_llm.provider", ""),            # empty = inherit rag.llm
        ("rag.answer_llm.model", None),
        ("rag.eval_llm.provider", "azure"),
        ("rag.eval_llm.base_url", ""),
        ("rag.contextual_retrieval.max_calls_per_file", 200),
        ("rag.contextual_retrieval.max_tokens_per_day", 0),
        ("rag.rerank.api", "cohere"),
        ("rag.rerank.api", "Jina"),
        ("rag.rerank.endpoint_path", "/v2/rerank"),
        ("rag.rerank.endpoint_path", ""),
        ("rag.rerank.timeout", 5),
        ("rag.rerank.max_retries", 2),
        ("rag.asr.language", "zh"),
        ("rag.asr.language", ""),
        ("rag.asr.segment_seconds", 600),
        ("rag.asr.timeout", 300),
        ("rag.asr.max_retries", 3),
        ("rag.parsing.pdf", "LOCAL"),               # cloud needs a vision model (profile check)
        ("rag.parsing.images", ""),                 # empty = follow the profile
        ("rag.parsing.vision.provider", "vllm"),
        ("rag.parsing.vision.model", "Qwen2.5-VL"),
        ("rag.parsing.vision.base_url", "https://vlm.example.com/v1"),
        ("rag.parsing.max_concurrency", 8),
        ("rag.parsing.max_pages_per_file", 0),
        ("rag.parsing.max_pages_per_day", 5000),
        ("rag.parsing.on_page_failure", "fail"),
    ])
    def test_valid_values_accepted(self, path, value):
        cfg = _mk_config()
        applied, _ = ro.validate_and_apply(cfg, {path: value})
        assert path in applied

    def test_provider_is_normalised(self):
        cfg = _mk_config()
        applied, _ = ro.validate_and_apply(cfg, {"rag.llm.provider": "  OpenAI "})
        assert applied["rag.llm.provider"] == "openai" and cfg.rag.llm.provider == "openai"

    def test_every_new_cloud_path_has_a_rule_or_is_free_text(self):
        """Each editable path is range-checked, choice-checked, nullable or plain text/bool,
        so the console can save every setting without a 500."""
        free = {"model", "base_url", "api_key", "query_prefix", "passage_prefix",
                "query_template", "document_template", "enabled", "auto_template",
                "request_dimensions", "expand_context_default", "language", "endpoint_path"}
        for path in ro.EDITABLE:
            leaf = path.rsplit(".", 1)[1]
            assert (path in ro._RANGES or ro._choices(path) or path in ro._NULLABLE
                    or leaf in free), path

    def test_cloud_profile_rejects_explicit_ckip_with_422_text(self):
        from src.config.model import ParsingConfig, VisionModelConfig
        cfg = _mk_config()
        cfg.rag.profile = "cloud"
        cfg.rag.asr.provider = "openai-compatible"
        cfg.rag.parsing = ParsingConfig(vision=VisionModelConfig(
            model="v", base_url="https://vlm.example.com/v1"))
        with pytest.raises(ValueError, match="rejects this change.*segmenter: ckip"):
            ro.validate_and_apply(cfg, {"rag.retrieval.segmenter": "ckip"})
        assert cfg.rag.retrieval.segmenter is None
        applied, _ = ro.validate_and_apply(cfg, {"rag.retrieval.segmenter": "jieba"})
        assert applied["rag.retrieval.segmenter"] == "jieba"

    def test_segmenter_is_normalised(self):
        cfg = _mk_config()
        applied, _ = ro.validate_and_apply(cfg, {"rag.retrieval.segmenter": " Bigram "})
        assert applied["rag.retrieval.segmenter"] == "bigram"

    def test_embedding_providers_follow_the_factory(self):
        from src.domain.rag.embedding.factory import EmbeddingProviderFactory
        for p in EmbeddingProviderFactory.list_supported_providers():
            ro.validate_and_apply(_mk_config(), {"rag.embedding.provider": p})


class TestResetMissingSection:
    """Resetting a key whose whole section is absent must not 500 / half-apply."""

    def test_yaml_value_walks_schema_when_section_is_none(self):
        cfg = _mk_config()
        cfg.rag.asr = None  # optional section absent from config.yaml
        from src.adapter import runtime_settings_service as svc
        from src.config.config_manager import Config
        from src.config.model import AsrConfig
        with patch(f"{_SVC}.Config.get_config_model", return_value=cfg), \
             patch.object(Config, "_config_path", None):
            assert svc._yaml_value("rag.asr.provider") == AsrConfig.model_fields["provider"].get_default()

    def test_reset_with_missing_section_succeeds(self, tmp_path):
        cfg = _mk_config()
        cfg.rag.asr = None
        yaml_file = tmp_path / "c.yaml"
        yaml_file.write_text("rag:\n  rerank:\n    enabled: false\n", encoding="utf-8")
        from src.adapter import runtime_settings_service as svc
        from src.config.config_manager import Config
        with patch(f"{_SVC}.Config.get_config_model", return_value=cfg), \
             patch(f"{_SVC}.RuntimeSettingsDB") as db, patch(f"{_SVC}._rebind"), \
             patch(f"{_SVC}._audit"), patch.object(Config, "_config_path", str(yaml_file)):
            db.delete.return_value = True
            assert svc.reset_setting("rag.asr.provider") is True
        assert cfg.rag.asr.provider == "docling-whisper"

    def test_reset_resolves_value_before_deleting_override(self, tmp_path):
        """If the factory value cannot be applied, the override row must survive."""
        cfg = _mk_config()
        yaml_file = tmp_path / "c.yaml"
        yaml_file.write_text("rag:\n  rerank:\n    score_threshold: 9\n", encoding="utf-8")
        from src.adapter import runtime_settings_service as svc
        from src.config.config_manager import Config
        with patch(f"{_SVC}.Config.get_config_model", return_value=cfg), \
             patch(f"{_SVC}.RuntimeSettingsDB") as db, patch(f"{_SVC}._rebind"), \
             patch.object(Config, "_config_path", str(yaml_file)):
            with pytest.raises(ValueError):
                svc.reset_setting("rag.rerank.score_threshold")
            db.delete.assert_not_called()

    def test_reset_expands_env_placeholders(self, tmp_path, monkeypatch):
        cfg = _mk_config()
        monkeypatch.setenv("RERANK_URL_T", "http://rr:1")
        yaml_file = tmp_path / "c.yaml"
        yaml_file.write_text('rag:\n  rerank:\n    base_url: "${RERANK_URL_T:-http://x}"\n',
                             encoding="utf-8")
        from src.adapter import runtime_settings_service as svc
        from src.config.config_manager import Config
        with patch(f"{_SVC}.Config.get_config_model", return_value=cfg), \
             patch.object(Config, "_config_path", str(yaml_file)):
            assert svc._yaml_value("rag.rerank.base_url") == "http://rr:1"
