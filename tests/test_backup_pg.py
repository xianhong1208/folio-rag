"""pg_dump / pg_restore wrappers: tool lookup, password handling, TOC filtering."""

import subprocess
from types import SimpleNamespace

import pytest

from src.backup import pg

URL = "postgresql://folio:s3cr%40t@db.example:5444/folio?sslmode=require"

TOC = """;
; Archive created at 2026-11-14
;
2; 3079 16386 EXTENSION - vector
3764; 0 0 COMMENT - EXTENSION vector
5; 2615 2200 SCHEMA - public pg_database_owner
3760; 0 0 COMMENT - SCHEMA public pg_database_owner
222; 1259 16766 TABLE public FileIndices folio
3768; 0 0 COMMENT public COLUMN "data_1_x".text_search_tsv folio
3700; 0 16766 TABLE DATA public FileIndices folio
"""


class _Runs:
    def __init__(self, outputs=None, rc=0, stderr=""):
        self.calls = []
        self.outputs = list(outputs or [])
        self.rc, self.stderr = rc, stderr

    def __call__(self, args, env=None, capture_output=None, text=None, timeout=None):
        self.calls.append((args, env))
        out = self.outputs.pop(0) if self.outputs else ""
        return SimpleNamespace(returncode=self.rc, stdout=out, stderr=self.stderr)


@pytest.fixture
def tools(tmp_path, monkeypatch):
    for name in ("pg_dump", "pg_restore"):
        p = tmp_path / name
        p.write_text("#!/bin/sh\n")
        p.chmod(0o755)
    return str(tmp_path)


class TestFindTool:
    def test_bin_dir_first(self, tools):
        assert pg.find_tool("pg_dump", tools) == f"{tools}/pg_dump"

    def test_missing_in_bin_dir(self, tmp_path):
        with pytest.raises(pg.PgToolError, match="pg_bin_dir"):
            pg.find_tool("pg_dump", str(tmp_path))

    def test_missing_on_path_says_what_to_install(self, monkeypatch):
        monkeypatch.setattr(pg.shutil, "which", lambda name: None)
        with pytest.raises(pg.PgToolError, match="postgresql-client"):
            pg.find_tool("pg_restore")

    def test_on_path(self, monkeypatch):
        monkeypatch.setattr(pg.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert pg.find_tool("pg_dump") == "/usr/bin/pg_dump"


class TestConninfo:
    def test_the_password_moves_to_the_environment(self):
        target, env = pg.conninfo(URL)
        assert "s3cr" not in target
        assert env == {"PGPASSWORD": "s3cr@t"}
        assert target.startswith("postgresql://folio@db.example:5444/folio") and "sslmode=require" in target

    def test_no_password_no_environment(self):
        target, env = pg.conninfo("postgresql+psycopg2://u@h/db")
        assert env == {} and target == "postgresql://u@h/db"


class TestDump:
    def test_arguments_and_password(self, tools, tmp_path, monkeypatch):
        runs = _Runs()
        monkeypatch.setattr(pg.subprocess, "run", runs)
        pg.dump(URL, tmp_path / "d.dump", snapshot="00000003-1", bin_dir=tools)
        args, env = runs.calls[0]
        assert args[0] == f"{tools}/pg_dump"
        assert "--snapshot=00000003-1" in args and "--format=custom" in args
        assert "--no-owner" in args and "--no-privileges" in args
        assert not any("s3cr" in a for a in args)
        assert env["PGPASSWORD"] == "s3cr@t"

    def test_failure_reports_the_last_lines(self, tools, tmp_path, monkeypatch):
        monkeypatch.setattr(pg.subprocess, "run", _Runs(rc=1, stderr="a\nb\npg_dump: error: server version mismatch"))
        with pytest.raises(pg.PgToolError, match="server version mismatch"):
            pg.dump(URL, tmp_path / "d.dump", bin_dir=tools)

    def test_timeout(self, tools, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired("pg_dump", 5)
        monkeypatch.setattr(pg.subprocess, "run", boom)
        with pytest.raises(pg.PgToolError, match="did not finish"):
            pg.dump(URL, tmp_path / "d.dump", bin_dir=tools, timeout=5)

    def test_tool_vanished(self, tools, tmp_path, monkeypatch):
        def gone(*a, **k):
            raise FileNotFoundError("pg_dump")
        monkeypatch.setattr(pg.subprocess, "run", gone)
        with pytest.raises(pg.PgToolError):
            pg.dump(URL, tmp_path / "d.dump", bin_dir=tools)

    def test_failure_without_output(self, tools, tmp_path, monkeypatch):
        monkeypatch.setattr(pg.subprocess, "run", _Runs(rc=3))
        with pytest.raises(pg.PgToolError, match="exit code 3"):
            pg.dump(URL, tmp_path / "d.dump", bin_dir=tools)


class TestRestore:
    def test_filter_toc_skips_extension_and_public_schema(self):
        kept, skipped = pg.filter_toc(TOC)
        assert "EXTENSION" not in kept and "SCHEMA - public" not in kept and "COMMENT - SCHEMA" not in kept
        assert "TABLE public FileIndices" in kept and "text_search_tsv" in kept and "TABLE DATA" in kept
        assert len(skipped) == 4

    def test_restore_runs_list_then_one_transaction(self, tools, tmp_path, monkeypatch):
        runs = _Runs(outputs=[TOC, ""])
        monkeypatch.setattr(pg.subprocess, "run", runs)
        skipped = pg.restore(URL, tmp_path / "d.dump", tmp_path / "toc.list", bin_dir=tools)
        assert len(skipped) == 4
        list_args, _ = runs.calls[0]
        assert "--list" in list_args
        args, env = runs.calls[1]
        assert "--single-transaction" in args and "--exit-on-error" in args
        assert f"--use-list={tmp_path / 'toc.list'}" in args
        assert env["PGPASSWORD"] == "s3cr@t" and not any("s3cr" in a for a in args)
        assert "EXTENSION" not in (tmp_path / "toc.list").read_text()

    def test_tool_version(self, monkeypatch):
        monkeypatch.setattr(pg.subprocess, "run", _Runs(outputs=["pg_dump (PostgreSQL) 17.2\n"]))
        assert pg.tool_version("/usr/bin/pg_dump") == "pg_dump (PostgreSQL) 17.2"
