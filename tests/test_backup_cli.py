"""``main.py backup`` / ``main.py restore``: arguments, output and exit codes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.backup import cli
from src.backup import service as S
from src.backup.archive import ArchiveError
from src.backup.pg import PgToolError
from src.config.model import BackupConfig

MANIFEST = {"folders": [{"id": 1, "name": "docs", "chunks": 5}], "storage": {"files": 2},
            "embedding": {"model": "e5", "dimension": 8}}


@pytest.fixture
def env(monkeypatch, tmp_path):
    from src.config.config_manager import Config
    conf = BackupConfig(dir=str(tmp_path), pg_bin_dir="/opt/pg/bin")
    monkeypatch.setattr(cli, "_load_config", lambda d: SimpleNamespace(rag="RAG"))
    monkeypatch.setattr(Config, "get_database_config",
                        classmethod(lambda cls: SimpleNamespace(url="postgresql://u:p@h/db", port=None, backup=conf)))
    monkeypatch.setattr(S, "default_storage_root", lambda: tmp_path / "storage")
    return SimpleNamespace(tmp=tmp_path, conf=conf)


class TestBackup:
    def test_writes_into_the_configured_dir_and_prunes(self, env, monkeypatch, capsys):
        seen = {}

        def fake(**kw):
            seen.update(kw)
            return S.BackupResult(path=env.tmp / "folio-backup-x.tar", size=1234, manifest=MANIFEST,
                                  warnings=["1 file missing"])
        monkeypatch.setattr(S, "create_backup", fake)
        monkeypatch.setattr(S, "prune", lambda d, keep: [d / "old.tar"])
        assert cli.main(["backup", "--keep", "3"]) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "wrote" in out and "1,234 bytes" in out and "warning: 1 file missing" in out
        assert "removed old archive old.tar" in out
        assert seen["out_dir"] == env.tmp and seen["out_path"] is None
        assert seen["bin_dir"] == "/opt/pg/bin" and seen["rag_conf"] == "RAG"
        assert seen["db_url"] == "postgresql://u:p@h/db" and seen["allow_running_jobs"] is False

    def test_output_file(self, env, monkeypatch):
        seen = {}
        monkeypatch.setattr(S, "create_backup", lambda **kw: seen.update(kw) or S.BackupResult(
            path=Path(kw["out_path"]), size=1, manifest=MANIFEST))
        assert cli.main(["backup", "--output", "/x/y.tar", "--allow-running-jobs"]) == 0
        assert seen["out_path"] == Path("/x/y.tar") and seen["out_dir"] is None and seen["allow_running_jobs"]

    def test_keep_must_be_positive(self, env):
        assert cli.main(["backup", "--keep", "0"]) == cli.EXIT_REFUSED

    def test_output_and_dir_are_exclusive(self, env):
        with pytest.raises(SystemExit):
            cli.main(["backup", "--output", "a.tar", "--dir", "b"])

    @pytest.mark.parametrize("exc,code,word", [
        (S.JobsRunning("2 index job(s)"), cli.EXIT_REFUSED, "refused"),
        (ArchiveError("bad"), cli.EXIT_REFUSED, "refused"),
        (PgToolError("pg_dump missing"), cli.EXIT_FAILED, "failed"),
        (RuntimeError("boom"), cli.EXIT_FAILED, "failed: RuntimeError"),
    ])
    def test_errors_map_to_exit_codes(self, env, monkeypatch, capsys, exc, code, word):
        def fail(**kw):
            raise exc
        monkeypatch.setattr(S, "create_backup", fail)
        assert cli.main(["backup"]) == code
        assert word in capsys.readouterr().out


class TestRestore:
    def test_check(self, env, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(S, "restore_backup", lambda archive, **kw: seen.update(kw, archive=archive) or
                            S.RestoreResult(manifest=MANIFEST, restored=False, warnings=["w1"]))
        assert cli.main(["restore", "a.tar", "--check"]) == 0
        out = capsys.readouterr().out
        assert "intact" in out and "warning: w1" in out
        assert seen["check_only"] and not seen["force"] and not seen["allow_mismatch"]
        assert seen["archive"] == Path("a.tar") and seen["storage_root"] == env.tmp / "storage"

    def test_restore(self, env, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(S, "restore_backup", lambda archive, **kw: seen.update(kw) or S.RestoreResult(
            manifest=MANIFEST, restored=True, folders=[{"id": 1, "name": "docs", "chunks": 5}]))
        assert cli.main(["restore", "a.tar", "--force", "--allow-mismatch"]) == 0
        out = capsys.readouterr().out
        assert "restored folder 1 'docs': 5 chunks" in out and "Start Folio" in out
        assert seen["force"] and seen["allow_mismatch"] and not seen["check_only"]

    def test_refusal(self, env, monkeypatch, capsys):
        def refuse(archive, **kw):
            raise S.BackupRefused("the database 'folio' already has 8 table(s)")
        monkeypatch.setattr(S, "restore_backup", refuse)
        assert cli.main(["restore", "a.tar"]) == cli.EXIT_REFUSED
        assert "refused: the database" in capsys.readouterr().out


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


def test_load_config_rejects_an_invalid_config(monkeypatch):
    from src.config.config_manager import Config
    monkeypatch.setattr(Config, "set_config", classmethod(lambda cls, d: None))
    monkeypatch.setattr(Config, "get_config_model", classmethod(lambda cls: None))
    with pytest.raises(RuntimeError, match="invalid"):
        cli._load_config("nowhere")


def test_load_config(monkeypatch):
    from src.config.config_manager import Config
    import src.log as log
    calls = {}
    monkeypatch.setattr(Config, "set_config", classmethod(lambda cls, d: calls.setdefault("dir", d)))
    monkeypatch.setattr(Config, "get_config_model", classmethod(lambda cls: "MODEL"))
    monkeypatch.setattr(Config, "get_logging_config", classmethod(lambda cls: SimpleNamespace(log_dir="lg")))
    monkeypatch.setattr(log, "setup_logger", lambda **kw: calls.setdefault("log", kw))
    assert cli._load_config("cfg") == "MODEL"
    assert calls["dir"] == "cfg" and calls["log"]["console_level"] == "ERROR" and calls["log"]["log_base_dir"] == "lg"
