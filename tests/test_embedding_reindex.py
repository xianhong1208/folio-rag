"""Changing the embedding model flags the files indexed with the old one.

Each FileIndex records the embedding model (and, since 1.4.2, the dimension) it was
indexed with. A file whose recorded model or dimension differs from the configured
one is "re-index recommended" with reason ``embedding``, through the same flag and
folder count as an older chunking pipeline (reason ``pipeline``).
"""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.rag.chunk_identity import (
    CHUNK_PIPELINE_VERSION,
    REINDEX_EMBEDDING,
    REINDEX_PIPELINE,
    configured_embedding,
    reindex_reason,
    reindex_recommended,
)

V = CHUNK_PIPELINE_VERSION
NOW = ("bge-m3", 1024)


class TestReason:
    def test_same_model_and_dimension_is_fine(self):
        assert reindex_reason("indexed", V, "bge-m3", 1024, NOW) is None

    def test_another_model_is_flagged(self):
        assert reindex_reason("indexed", V, "e5-large", 1024, NOW) == REINDEX_EMBEDDING
        assert reindex_recommended("indexed", V, "e5-large", 1024, NOW)

    def test_another_dimension_is_flagged(self):
        assert reindex_reason("indexed", V, "bge-m3", 768, NOW) == REINDEX_EMBEDDING

    def test_embedding_wins_over_pipeline(self):
        assert reindex_reason("indexed", "2", "e5-large", None, NOW) == REINDEX_EMBEDDING

    def test_older_pipeline_keeps_its_reason(self):
        assert reindex_reason("indexed", "2", "bge-m3", 1024, NOW) == REINDEX_PIPELINE
        assert reindex_reason("indexed", None, "bge-m3", 1024, NOW) == REINDEX_PIPELINE

    def test_unrecorded_values_flag_nothing(self):
        assert reindex_reason("indexed", V, None, None, NOW) is None

    def test_unknown_configuration_flags_nothing(self):
        assert reindex_reason("indexed", V, "e5-large", 768, (None, None)) is None

    @pytest.mark.parametrize("status", ["failed", "interrupted", None])
    def test_only_indexed_files(self, status):
        assert reindex_reason(status, V, "e5-large", 768, NOW) is None

    def test_reads_the_live_config_by_default(self):
        cfg = SimpleNamespace(rag=SimpleNamespace(embedding=SimpleNamespace(model="m2", dimension=8)))
        with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
            assert configured_embedding() == ("m2", 8)
            assert reindex_reason("indexed", V, "m1", 8) == REINDEX_EMBEDDING
            assert reindex_reason("indexed", V, "m2", 8) is None

    def test_unreadable_config_is_unknown(self):
        with patch("src.config.config_manager.Config.get_config_model", side_effect=RuntimeError):
            assert configured_embedding() == (None, None)


class TestSqlForm:
    def _sql(self, current):
        from sqlalchemy.dialects import postgresql
        from db.admin_stats import _file_needs_reindex
        return str(_file_needs_reindex(current).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))

    def test_compares_model_and_dimension(self):
        sql = self._sql(("bge-m3", 1024))
        assert "embedding_model != 'bge-m3'" in sql and "embedding_model IS NOT NULL" in sql
        assert "embedding_dimension != 1024" in sql and "embedding_dimension IS NOT NULL" in sql
        assert f"pipeline_version != '{V}'" in sql and "status = 'indexed'" in sql

    def test_unknown_configuration_keeps_only_the_pipeline_rule(self):
        sql = self._sql((None, None))
        assert "embedding_model" not in sql and "embedding_dimension" not in sql
        assert "pipeline_version" in sql


class TestListings:
    async def test_cached_rest_list_is_flagged_against_the_current_model(self):
        """GET /indexed-files is cached; the flag follows a runtime model change at once."""
        from src.adapter.rag_maintenance import RAGMaintenanceService
        row = {"file_id": "f", "status": "indexed", "pipeline_version": V,
               "embedding_model": "old", "embedding_dimension": 8}
        cache = MagicMock()
        cache.get.return_value = [row]
        svc = RAGMaintenanceService(MagicMock())
        with patch("src.adapter.rag_maintenance.CacheService.get_instance", return_value=cache), \
                patch("src.adapter.rag_maintenance.configured_embedding", return_value=("old", 8)):
            before = await svc.get_indexed_files(1)
        with patch("src.adapter.rag_maintenance.CacheService.get_instance", return_value=cache), \
                patch("src.adapter.rag_maintenance.configured_embedding", return_value=("new", 8)):
            after = await svc.get_indexed_files(1)
        assert before[0]["reindex_recommended"] is False and before[0]["reindex_reason"] is None
        assert after[0]["reindex_recommended"] is True and after[0]["reindex_reason"] == "embedding"
        assert "reindex_reason" not in row  # the cached rows themselves are not changed

    async def test_fresh_rest_list_carries_dimension_and_reason(self):
        from src.adapter.rag_maintenance import RAGMaintenanceService
        rec = SimpleNamespace(file_id="f", num_chunks=3, indexed_at=SimpleNamespace(isoformat=lambda: "t"),
                              status="indexed", embedding_model="old", embedding_dimension=8,
                              chunk_size=256, chunk_overlap=50, index_warning=None, pipeline_version=V)
        ctx = MagicMock()
        ctx.indexing_service.list_indices_for_folder.return_value = [rec]
        ctx.indexing_service.get_files_by_ids.return_value = {"f": SimpleNamespace(file_name="a.txt")}
        cache = MagicMock()
        cache.get.return_value = None
        with patch("src.adapter.rag_maintenance.CacheService.get_instance", return_value=cache), \
                patch("src.adapter.rag_maintenance.configured_embedding", return_value=("new", 8)), \
                patch("src.adapter.rag_maintenance.run_db", new=AsyncMock(side_effect=lambda f, *a: f(*a))):
            out = await RAGMaintenanceService(ctx).get_indexed_files(1)
        assert out[0]["embedding_dimension"] == 8 and out[0]["reindex_reason"] == "embedding"

    def test_file_status_endpoint_compares_the_embedding(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.api.router import rag_indexing as ri
        from src.api.dependencies.auth import extract_token
        from src.auth.dependencies import authenticate_request
        app = FastAPI()
        app.include_router(ri.router)
        app.dependency_overrides[extract_token] = lambda: "tok"
        app.dependency_overrides[authenticate_request] = lambda: None
        fid = "9e68c18e-3209-44d3-b7e9-2211b3660d66"
        app.dependency_overrides[ri.get_file_by_id] = lambda: SimpleNamespace(id=fid, file_name="a.txt")
        record = SimpleNamespace(status="indexed", num_chunks=3, indexed_at=None, index_warning=None,
                                 pipeline_version=V, embedding_model="old", embedding_dimension=8)
        with patch("db.fileindexdb.FileIndexDB.get_by_file", return_value=record), \
                patch("src.domain.rag.chunk_identity.configured_embedding", return_value=("new", 8)):
            r = TestClient(app).get(f"/file/{fid}/index/status")
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["reindex_recommended"] is True and data["reindex_reason"] == "embedding"


class TestRecording:
    def test_indexing_records_the_dimension(self):
        """The success record carries the dimension the vectors were written with."""
        from src.adapter import rag_indexing
        src = inspect.getsource(rag_indexing)
        call = src[src.index("ctx.indexing_service.record_index_success"):]
        call = call[:call.index(")\n")]
        assert "embedding_model=ctx.model_name" in call and "embedding_dimension=ctx.embed_dim" in call

    def test_service_passes_it_to_the_db(self):
        from src.domain.rag.index_service import IndexingService
        with patch("src.domain.rag.index_service.FileIndexDB.upsert_indexed") as up:
            IndexingService.record_index_success(
                MagicMock(), file_id="f", folder_id=1, index_id="i", vector_store_table="t",
                chunk_size=256, chunk_overlap=50, embedding_model="m", num_chunks=3,
                embedding_dimension=1024)
        assert up.call_args.kwargs["embedding_dimension"] == 1024
