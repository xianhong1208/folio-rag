"""The retrieval eval must never stop the server.

scripts/rag_eval.py is loaded in-process by POST /api/admin/eval/run and runs as a
task on the server's event loop. It used to raise SystemExit for "folder not found"
and "nothing indexed"; inside the loop that ended the whole process. Now:

- the module raises ordinary EvalError subclasses;
- the endpoint checks the folder before scheduling a run (404 / 409);
- the background runner also swallows SystemExit as a last line of defence;
- only the CLI entry point converts EvalError into a non-zero exit.
"""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import ProgrammingError

from src.api.router import admin_overview as ov
from src.api.router.admin_settings import router

_FOLDER = SimpleNamespace(id=7, name="docs", vector_table_uuid="abc", user_token="sub:k")


@pytest.fixture()
def client(as_identity, make_identity):
    # Signed in as the owner of _FOLDER; folder lookups resolve only folder 7.
    app = FastAPI()
    app.include_router(router)
    with patch("db.cached_folderdb.CachedFolderDB.get_by_id",
               side_effect=lambda fid: _FOLDER if fid == 7 else None):
        yield TestClient(as_identity(app, make_identity("sub:k")))


@pytest.fixture(autouse=True)
def _reset_eval_state():
    def reset():
        ov._EVAL.update(running=False, folder=None, done=0, total=0,
                        result=None, error=None, started_at=None, owner=None)
    reset()
    yield
    reset()


def _missing_table_error():
    orig = Exception('relation "data_7_abc" does not exist')
    orig.pgcode = "42P01"
    return ProgrammingError("SELECT 1", {}, orig)


def _engine_raising(exc):
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value.execute.side_effect = exc
    return engine


class TestModuleRaisesOrdinaryExceptions:
    def test_unknown_folder_is_not_systemexit(self):
        mod = ov._load_eval_mod()
        with patch("db.folderdb.FolderDB.get", return_value=[]):
            with pytest.raises(mod.FolderNotFound, match="folder not found: 999999"):
                mod._resolve_folder("999999")
        assert issubclass(mod.FolderNotFound, Exception)
        assert not issubclass(mod.FolderNotFound, SystemExit)

    def test_missing_vector_table_reads_as_no_chunks(self):
        mod = ov._load_eval_mod()
        with patch("db.db.get_engine", return_value=_engine_raising(_missing_table_error())):
            assert mod._sample_chunks(_FOLDER, 5) == []
            assert mod.has_indexed_chunks(_FOLDER) is False
            assert mod._existing_gold_ids(_FOLDER, [{"gold_node_id": "x"}]) == set()

    def test_other_sql_errors_still_raise(self):
        mod = ov._load_eval_mod()
        orig = Exception("syntax error")
        orig.pgcode = "42601"
        with patch("db.db.get_engine",
                   return_value=_engine_raising(ProgrammingError("x", {}, orig))):
            with pytest.raises(ProgrammingError):
                mod.has_indexed_chunks(_FOLDER)

    def test_generate_with_no_chunks_raises_eval_error(self, monkeypatch):
        mod = ov._load_eval_mod()
        monkeypatch.setattr(mod, "_sample_chunks", lambda folder, n: [])
        fake_cfg = SimpleNamespace(rag=SimpleNamespace(llm=SimpleNamespace(model="m")))
        with patch("src.config.config_manager.Config.get_config_model", return_value=fake_cfg), \
             patch("src.domain.rag.context_generator.ContextGenerator._create_sync_client"):
            with pytest.raises(mod.NoIndexedChunks):
                mod.generate_eval_set(_FOLDER, 5, "reports/unused.jsonl")

    def test_check_ready_returns_folder_when_indexed(self, monkeypatch):
        mod = ov._load_eval_mod()
        monkeypatch.setattr(mod, "_resolve_folder", lambda ref: _FOLDER)
        monkeypatch.setattr(mod, "has_indexed_chunks", lambda folder: True)
        assert mod.check_ready("7") is _FOLDER


class TestEndpoint:
    def test_unknown_folder_returns_404_and_server_keeps_serving(self, client):
        with patch("db.folderdb.FolderDB.get", return_value=[]), \
             patch("asyncio.create_task") as create_task:
            r = client.post("/api/admin/eval/run", json={"folder_id": 999999})
            r2 = client.post("/api/admin/eval/run", json={"folder_id": "abc"})
        assert r.status_code == 404 and "not found" in r.json()["detail"]
        assert r2.status_code == 422  # folder_id must be an id
        create_task.assert_not_called()
        assert ov._EVAL["running"] is False
        # The process is still up and answering.
        assert client.get("/api/admin/eval/status").status_code == 200

    def test_unindexed_folder_returns_409(self, client):
        with patch("db.folderdb.FolderDB.get", return_value=[_FOLDER]), \
             patch("db.db.get_engine", return_value=_engine_raising(_missing_table_error())), \
             patch("asyncio.create_task") as create_task:
            r = client.post("/api/admin/eval/run", json={"folder_id": 7})
        assert r.status_code == 409
        assert "no indexed leaf chunks" in r.json()["detail"]
        create_task.assert_not_called()
        assert ov._EVAL["running"] is False

    def test_run_started_by_another_request_during_check_is_rejected(self, client):
        mod = ov._load_eval_mod()

        def _check(ref):
            ov._EVAL["running"] = True  # a concurrent request won the race
            return _FOLDER
        mod.check_ready = _check
        with patch.object(ov, "_load_eval_mod", return_value=mod), \
             patch("asyncio.create_task") as create_task:
            r = client.post("/api/admin/eval/run", json={"folder_id": 7})
        assert r.status_code == 409
        create_task.assert_not_called()


class TestBackgroundRunner:
    async def test_systemexit_from_the_module_does_not_escape(self):
        fake = MagicMock()
        fake.run_eval_async = AsyncMock(side_effect=SystemExit("bye"))
        ov._EVAL["running"] = True
        with patch.object(ov, "_load_eval_mod", return_value=fake):
            task = asyncio.create_task(ov._run_eval_bg("7", 5, 10, False))
            await task  # would raise SystemExit here if it escaped
        assert ov._EVAL["running"] is False
        assert ov._EVAL["error"] == "bye"
        # The loop is still usable.
        assert await asyncio.sleep(0, result="alive") == "alive"

    async def test_folder_deleted_after_check_is_reported_not_fatal(self):
        ov._EVAL["running"] = True
        with patch("db.folderdb.FolderDB.get", return_value=[]):
            await ov._run_eval_bg("7", 5, 10, False)
        assert ov._EVAL["running"] is False
        assert "folder not found" in ov._EVAL["error"]


class TestCli:
    def test_cli_exits_non_zero_with_message(self, monkeypatch):
        mod = ov._load_eval_mod()

        def _boom(*a, **k):
            raise mod.FolderNotFound("folder not found: nope")
        monkeypatch.setattr(mod, "run_eval", _boom)
        monkeypatch.setattr(sys, "argv", ["rag_eval.py", "--folder", "nope"])
        with patch("src.config.config_manager.Config.set_config"), \
             patch("src.config.dotenv.load_dotenv"), \
             patch("db.runtime_settings_db.RuntimeSettingsDB.ensure_table"):
            with pytest.raises(SystemExit) as exc:
                mod.main()
        # SystemExit with a string argument exits with status 1 and prints it.
        assert exc.value.code == "error: folder not found: nope"
