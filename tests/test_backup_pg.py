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

    def test_same_major_restores_directly_in_one_transaction(self, tools, tmp_path, monkeypatch):
        runs = _Runs(outputs=[TOC, ""])
        monkeypatch.setattr(pg.subprocess, "run", runs)
        monkeypatch.setattr(pg, "tool_major", lambda tool: 16)
        skipped = pg.restore(URL, tmp_path / "d.dump", tmp_path / "toc.list", bin_dir=tools,
                             server=(16, {"statement_timeout"}))
        assert len(skipped) == 4
        list_args, _ = runs.calls[0]
        assert "--list" in list_args
        args, env = runs.calls[1]
        assert "--single-transaction" in args and "--exit-on-error" in args
        assert f"--use-list={tmp_path / 'toc.list'}" in args
        assert env["PGPASSWORD"] == "s3cr@t" and not any("s3cr" in a for a in args)
        assert "EXTENSION" not in (tmp_path / "toc.list").read_text()

    def test_older_pg_restore_also_restores_directly(self, tools, monkeypatch):
        monkeypatch.setattr(pg, "tool_major", lambda tool: 15)
        assert pg.restore_tools(17, tools) == (f"{tools}/pg_restore", None)

    def test_server_info_is_read_when_not_given(self, tools, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(pg, "server_info", lambda url: seen.append(url) or (17, set()))
        monkeypatch.setattr(pg, "tool_major", lambda tool: 17)
        monkeypatch.setattr(pg.subprocess, "run", _Runs(outputs=[TOC, ""]))
        pg.restore(URL, tmp_path / "d.dump", tmp_path / "toc.list", bin_dir=tools)
        assert seen == [URL]

    def test_tool_version(self, monkeypatch):
        monkeypatch.setattr(pg.subprocess, "run", _Runs(outputs=["pg_dump (PostgreSQL) 17.2\n"]))
        assert pg.tool_version("/usr/bin/pg_dump") == "pg_dump (PostgreSQL) 17.2"


class TestVersions:
    def test_tool_major(self, monkeypatch):
        monkeypatch.setattr(pg.subprocess, "run", _Runs(outputs=["pg_restore (PostgreSQL) 18.6 (Ubuntu 18.6-1)\n"]))
        assert pg.tool_major("/usr/bin/pg_restore") == 18

    def test_tool_major_unreadable(self, monkeypatch):
        monkeypatch.setattr(pg.subprocess, "run", _Runs(outputs=["something else\n"]))
        with pytest.raises(pg.PgToolError, match="cannot tell the version"):
            pg.tool_major("/usr/bin/pg_restore")

    def test_an_older_pg_dump_is_refused_before_it_runs(self, tools, tmp_path, monkeypatch):
        runs = _Runs(outputs=["pg_dump (PostgreSQL) 16.4\n"])
        monkeypatch.setattr(pg.subprocess, "run", runs)
        with pytest.raises(pg.PgToolError, match=r"version 16 but the database server is PostgreSQL 17.*"
                                                 r"postgresql-client-17"):
            pg.dump(URL, tmp_path / "d.dump", bin_dir=tools, server_major=17)
        assert len(runs.calls) == 1 and "--version" in runs.calls[0][0]

    @pytest.mark.parametrize("tool_v", ["17.0", "18.6"])
    def test_a_same_or_newer_pg_dump_runs(self, tools, tmp_path, monkeypatch, tool_v):
        runs = _Runs(outputs=[f"pg_dump (PostgreSQL) {tool_v}\n", ""])
        monkeypatch.setattr(pg.subprocess, "run", runs)
        pg.dump(URL, tmp_path / "d.dump", bin_dir=tools, server_major=17)
        assert "--format=custom" in runs.calls[1][0]


PREAMBLE = b"""--
-- PostgreSQL database dump
--

\\restrict abc123

-- Dumped from database version 15.19
-- Dumped by pg_dump version 18.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SELECT pg_catalog.set_config('search_path', '', false);
SET row_security = off;

--
-- Name: data_1_u; Type: TABLE; Schema: public; Owner: -
--

SET transaction_timeout = 0;
CREATE TABLE public.data_1_u (id bigint);
COPY public.data_1_u (id) FROM stdin;
SET transaction_timeout = 0;
\\.

\\unrestrict abc123
"""
KNOWN_15 = {"statement_timeout", "lock_timeout", "client_encoding", "row_security", "search_path"}


class TestFilterPreamble:
    def test_drops_only_unknown_opening_sets(self):
        dropped = []
        out = b"".join(pg.filter_preamble(PREAMBLE.splitlines(keepends=True), KNOWN_15, dropped))
        assert dropped == ["SET transaction_timeout = 0;"]
        assert out == PREAMBLE.replace(b"SET transaction_timeout = 0;\n", b"", 1)
        # after the first object nothing is touched, COPY data included
        assert out.count(b"SET transaction_timeout = 0;") == 2

    def test_a_server_that_knows_the_setting_keeps_it(self):
        dropped = []
        out = b"".join(pg.filter_preamble(PREAMBLE.splitlines(keepends=True),
                                          KNOWN_15 | {"transaction_timeout"}, dropped))
        assert out == PREAMBLE and dropped == []

    def test_the_preamble_ends_at_the_first_other_statement(self):
        script = b"SET foo_timeout = 0;\nBEGIN;\nSET foo_timeout = 0;\n"
        dropped = []
        out = b"".join(pg.filter_preamble(script.splitlines(keepends=True), set(), dropped))
        assert out == b"BEGIN;\nSET foo_timeout = 0;\n" and len(dropped) == 1

    def test_the_setting_name_is_matched_case_insensitively(self):
        dropped = []
        out = list(pg.filter_preamble([b"SET Lock_Timeout TO 0;\n"], {"lock_timeout"}, dropped))
        assert out == [b"SET Lock_Timeout TO 0;\n"] and dropped == []


def _script(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


@pytest.fixture
def newer_tools(tmp_path):
    """A pg_restore 18 that writes PREAMBLE as SQL, and a psql that records its stdin."""
    bin_dir = tmp_path / "bin18"
    bin_dir.mkdir()
    (bin_dir / "preamble.sql").write_bytes(PREAMBLE)
    _script(bin_dir / "pg_restore", f"""
case "$*" in
  *--version*) echo "pg_restore (PostgreSQL) 18.6";;
  *--list*) cat <<'TOC'
{TOC}TOC
;;
  *--file=-*) [ -n "$FAKE_RESTORE_SLOW" ] && exec sleep 5
              [ -n "$FAKE_RESTORE_BIG" ] && {{ cat "{bin_dir}/preamble.sql"; head -c 4000000 /dev/zero | tr '\\0' 'x'; }}
              [ -n "$FAKE_RESTORE_FAIL" ] && {{ echo "pg_restore: error: bad archive" >&2; exit 1; }}
              cat "{bin_dir}/preamble.sql";;
esac
""")
    _script(bin_dir / "psql", f"""
[ -n "$FAKE_PSQL_EARLY" ] && {{ echo 'psql: error: connection refused' >&2; exit 2; }}
echo "$@" > "{bin_dir}/psql.args"
env | grep '^PGPASSWORD=' > "{bin_dir}/psql.env"
cat > "{bin_dir}/psql.stdin"
[ -n "$FAKE_PSQL_FAIL" ] && {{ echo 'psql:<stdin>:3: ERROR:  syntax error at or near "x"' >&2; exit 3; }}
exit 0
""")
    return bin_dir


class TestRestoreThroughSql:
    def test_newer_pg_restore_pipes_filtered_sql_into_psql(self, newer_tools, tmp_path):
        skipped = pg.restore(URL, tmp_path / "d.dump", tmp_path / "toc.list", bin_dir=str(newer_tools),
                             server=(15, KNOWN_15))
        assert "SET transaction_timeout = 0;" in skipped and len(skipped) == 5
        sent = (newer_tools / "psql.stdin").read_bytes()
        assert sent == PREAMBLE.replace(b"SET transaction_timeout = 0;\n", b"", 1)
        args = (newer_tools / "psql.args").read_text()
        assert "--single-transaction" in args and "ON_ERROR_STOP=1" in args and "--no-psqlrc" in args
        assert "s3cr" not in args and "PGPASSWORD=s3cr@t" in (newer_tools / "psql.env").read_text()
        assert "EXTENSION" not in (tmp_path / "toc.list").read_text()

    def test_restore_tools_takes_psql_from_the_same_directory(self, newer_tools):
        tool, psql = pg.restore_tools(16, str(newer_tools))
        assert psql == str(newer_tools / "psql")

    def test_psql_comes_from_pg_restores_directory_before_path(self, newer_tools, monkeypatch):
        # Another psql earlier on PATH (an older version) must not be picked.
        monkeypatch.setattr(pg.shutil, "which", lambda name: str(newer_tools / name) if name == "pg_restore"
                            else f"/opt/old/bin/{name}")
        assert pg.restore_tools(15) == (str(newer_tools / "pg_restore"), str(newer_tools / "psql"))

    def test_a_missing_psql_is_reported(self, newer_tools, monkeypatch):
        (newer_tools / "psql").unlink()
        with pytest.raises(pg.PgToolError, match="psql not found"):
            pg.restore_tools(15, str(newer_tools))

    def test_psql_errors_are_reported(self, newer_tools, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_PSQL_FAIL", "1")
        with pytest.raises(pg.PgToolError, match="through psql.*syntax error"):
            pg.restore(URL, tmp_path / "d.dump", tmp_path / "toc.list", bin_dir=str(newer_tools),
                       server=(15, KNOWN_15))

    def test_pg_restore_errors_are_reported(self, newer_tools, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_RESTORE_FAIL", "1")
        with pytest.raises(pg.PgToolError, match="pg_restore failed: pg_restore: error: bad archive"):
            pg.restore(URL, tmp_path / "d.dump", tmp_path / "toc.list", bin_dir=str(newer_tools),
                       server=(15, KNOWN_15))

    def test_psql_stopping_early_does_not_hang(self, newer_tools, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_PSQL_EARLY", "1")
        monkeypatch.setenv("FAKE_RESTORE_BIG", "1")
        with pytest.raises(pg.PgToolError, match="through psql.*connection refused"):
            pg.restore(URL, tmp_path / "d.dump", tmp_path / "toc.list", bin_dir=str(newer_tools),
                       server=(15, KNOWN_15), timeout=30)

    def test_a_slow_restore_times_out(self, newer_tools, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_RESTORE_SLOW", "1")
        with pytest.raises(pg.PgToolError, match="did not finish"):
            pg.restore(URL, tmp_path / "d.dump", tmp_path / "toc.list", bin_dir=str(newer_tools),
                       server=(15, KNOWN_15), timeout=1)
