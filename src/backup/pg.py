"""``pg_dump`` / ``pg_restore`` for Folio's database.

The password never goes on a command line (where ``ps`` shows it): the tools get the
URL without it and read ``PGPASSWORD`` from their environment. Neither tool needs a
superuser: the dump is taken with ``--no-owner --no-privileges``, and the restore skips
the ``vector`` extension entry (Folio's start-up bootstrap creates the extension the
same way it always does) and the ``public`` schema entries, which only the schema's
owner may change.

Client tools newer than the server are normal (``apt install postgresql-client`` gives
the newest major). ``pg_dump`` copes with an older server, but ``pg_restore`` opens every
restore with a fixed list of ``SET`` statements for its own version, and ``SET transaction_timeout``
(new in 17) is an error on a 15 or 16 server. So when ``pg_restore`` is newer than the
target server, the restore goes through SQL instead: ``pg_restore --file=-`` piped into
``psql --single-transaction -v ON_ERROR_STOP=1``, dropping only those opening ``SET`` statements
that name a setting the target server does not have. A ``pg_dump`` older than the server
is refused up front: it cannot dump a newer server.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import IO, Dict, Iterable, List, Optional, Set, Tuple

from sqlalchemy.engine import URL, make_url


class PgToolError(RuntimeError):
    """pg_dump / pg_restore is missing or failed."""


def find_tool(name: str, bin_dir: Optional[str] = None) -> str:
    """Absolute path of ``pg_dump`` / ``pg_restore``: ``bin_dir`` first, then PATH."""
    if bin_dir:
        candidate = Path(bin_dir) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise PgToolError(f"{name} not found in database.backup.pg_bin_dir ({bin_dir})")
    found = shutil.which(name)
    if not found:
        raise PgToolError(
            f"{name} is not installed or not on PATH. Install the PostgreSQL client tools "
            f"(e.g. apt install postgresql-client-17), the same major version as the server "
            f"or newer, or set database.backup.pg_bin_dir.")
    return found


def conninfo(db_url: str) -> Tuple[str, Dict[str, str]]:
    """(libpq URL without the password, extra environment) for a SQLAlchemy URL."""
    url = make_url(db_url)
    env: Dict[str, str] = {}
    if url.password:
        env["PGPASSWORD"] = str(url.password)
    # URL.set() ignores None, so the URL is rebuilt without the password.
    plain = URL.create("postgresql", username=url.username, host=url.host, port=url.port,
                       database=url.database, query=url.query)
    return plain.render_as_string(hide_password=False), env


def _run(args: List[str], env_extra: Dict[str, str], what: str, timeout: Optional[float]) -> str:
    env = dict(os.environ)
    env.update(env_extra)
    try:
        proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise PgToolError(f"{what}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise PgToolError(f"{what} did not finish within {timeout:.0f} s") from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = " | ".join(detail[-5:]) if detail else f"exit code {proc.returncode}"
        raise PgToolError(f"{what} failed: {tail}")
    return proc.stdout


def tool_version(tool: str) -> str:
    out = _run([tool, "--version"], {}, f"{Path(tool).name} --version", 30)
    return out.strip()


_VERSION = re.compile(r"\(PostgreSQL\)\s+(\d+)")


def tool_major(tool: str) -> int:
    """Major version of a client tool (``pg_dump (PostgreSQL) 17.2`` -> 17)."""
    text_ = tool_version(tool)
    m = _VERSION.search(text_)
    if not m:
        raise PgToolError(f"cannot tell the version of {tool} from {text_!r}")
    return int(m.group(1))


def server_info(db_url: str) -> Tuple[int, Set[str]]:
    """(server major version, names of every setting the server has)."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool

    engine = create_engine(db_url, poolclass=NullPool)
    try:
        with engine.connect() as conn:
            num = int(conn.execute(text("SHOW server_version_num")).scalar())
            names = {r[0] for r in conn.execute(text("SELECT name FROM pg_settings"))}
    finally:
        engine.dispose()
    return num // 10000, names


def check_dump_tool(tool_major_: int, server_major: int, tool: str) -> None:
    """pg_dump cannot dump a server newer than itself: say so before trying."""
    if tool_major_ < server_major:
        raise PgToolError(
            f"{tool} is version {tool_major_} but the database server is PostgreSQL {server_major}: "
            f"pg_dump must be the same major version as the server or newer. Install "
            f"postgresql-client-{server_major} (or newer), or point database.backup.pg_bin_dir at it.")


def dump(db_url: str, out_file: Path, *, snapshot: Optional[str] = None,
         bin_dir: Optional[str] = None, timeout: Optional[float] = None,
         server_major: Optional[int] = None) -> None:
    """``pg_dump --format=custom`` of the whole database into ``out_file``.

    ``snapshot`` (from ``pg_export_snapshot()``) makes the dump see exactly what the
    caller's open transaction sees. ``server_major`` (when known) refuses a pg_dump
    older than the server with a clear message.
    """
    tool = find_tool("pg_dump", bin_dir)
    if server_major is not None:
        check_dump_tool(tool_major(tool), server_major, tool)
    target, env = conninfo(db_url)
    args = [tool, "--format=custom", "--no-owner", "--no-privileges",
            f"--file={out_file}", f"--dbname={target}"]
    if snapshot:
        args.insert(1, f"--snapshot={snapshot}")
    _run(args, env, "pg_dump", timeout)


# TOC entries a non-superuser restore must not replay (see the module docstring).
_SKIP_TOC = re.compile(
    r"^\s*\d+;\s+\d+\s+\d+\s+(EXTENSION\s+-\s|COMMENT\s+-\s+EXTENSION\s|SCHEMA\s+-\s+public\b"
    r"|COMMENT\s+-\s+SCHEMA\s+-?\s*public\b)")


def filter_toc(toc: str) -> Tuple[str, List[str]]:
    """(TOC list for ``pg_restore -L`` without the skipped entries, the skipped lines)."""
    kept, skipped = [], []
    for line in toc.splitlines():
        if _SKIP_TOC.match(line):
            skipped.append(line.strip())
        else:
            kept.append(line)
    return "\n".join(kept) + "\n", skipped


_PREAMBLE_SET = re.compile(rb"^SET\s+([A-Za-z_][A-Za-z0-9_.]*)\s*(=|TO)\s", re.IGNORECASE)


def filter_preamble(lines: Iterable[bytes], known: Set[str], dropped: List[str]) -> Iterable[bytes]:
    """Drop the opening ``SET name = ...;`` lines for settings the server does not have.

    Only the preamble pg_restore writes before the first object is touched: the
    comments, ``\\restrict`` and the ``SET`` / ``set_config`` lines up to the first
    ``-- Name:`` / ``-- TOC entry`` comment or any other statement. Everything after it
    (DDL, ``COPY`` data, function bodies) passes through byte for byte.
    """
    known_lower = {k.lower() for k in known}
    in_preamble = True
    for line in lines:
        if in_preamble:
            stripped = line.strip()
            if stripped.startswith(b"-- Name:") or stripped.startswith(b"-- TOC entry") or stripped.startswith(
                    b"-- Data for Name:"):
                in_preamble = False
            elif stripped == b"" or stripped.startswith(b"--") or stripped.startswith(b"\\") \
                    or stripped.lower().startswith(b"select pg_catalog.set_config("):
                pass
            else:
                m = _PREAMBLE_SET.match(stripped)
                if m is None:
                    in_preamble = False
                elif m.group(1).decode("ascii").lower() not in known_lower:
                    dropped.append(stripped.decode("utf-8", "replace"))
                    continue
        yield line


def _sibling_tool(tool: str, name: str, bin_dir: Optional[str]) -> str:
    """``name`` from the same directory as ``tool`` (same version), else bin_dir / PATH."""
    sibling = Path(tool).with_name(name)
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return find_tool(name, bin_dir)


def _tail(f: IO[bytes]) -> str:
    f.seek(0)
    lines = f.read().decode("utf-8", "replace").strip().splitlines()
    return " | ".join(lines[-5:])


def _restore_via_sql(tool: str, psql: str, dump_file: Path, list_file: Path, target: str,
                     env_extra: Dict[str, str], known: Set[str], timeout: Optional[float]) -> List[str]:
    """pg_restore --file=- | preamble filter | psql --single-transaction. Returns dropped SETs."""
    env = dict(os.environ)
    env.update(env_extra)
    dropped: List[str] = []
    with tempfile.TemporaryFile() as rerr, tempfile.TemporaryFile() as perr:
        try:
            producer = subprocess.Popen(
                [tool, "--no-owner", "--no-privileges", "--exit-on-error", f"--use-list={list_file}",
                 "--file=-", str(dump_file)], stdout=subprocess.PIPE, stderr=rerr, env=env)
        except OSError as e:
            raise PgToolError(f"pg_restore: {e}") from e
        try:
            consumer = subprocess.Popen(
                [psql, "--no-psqlrc", "--quiet", "--single-transaction", "--set=ON_ERROR_STOP=1",
                 f"--dbname={target}"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=perr, env=env)
        except OSError as e:
            producer.kill()
            producer.wait()
            raise PgToolError(f"psql: {e}") from e
        timed_out = threading.Event()

        def _expire() -> None:
            timed_out.set()
            producer.kill()
            consumer.kill()

        timer = threading.Timer(timeout, _expire) if timeout else None
        if timer is not None:
            timer.daemon = True
            timer.start()
        broken = False
        try:
            for line in filter_preamble(producer.stdout, known, dropped):
                try:
                    consumer.stdin.write(line)
                except BrokenPipeError:
                    broken = True
                    break
            if broken:
                # psql stopped reading (it hit an error); pg_restore has nothing left to do.
                producer.kill()
        finally:
            producer.stdout.close()
            try:
                consumer.stdin.close()
            except BrokenPipeError:
                pass
            prc = producer.wait()
            crc = consumer.wait()
            if timer is not None:
                timer.cancel()
        if timed_out.is_set():
            raise PgToolError(f"the restore did not finish within {timeout:.0f} s")
        # psql is the side that knows why the SQL failed; check it first.
        if crc != 0:
            raise PgToolError(f"pg_restore (through psql) failed: {_tail(perr) or f'exit code {crc}'}")
        if prc != 0:
            raise PgToolError(f"pg_restore failed: {_tail(rerr) or f'exit code {prc}'}")
    return dropped


def restore_tools(server_major: int, bin_dir: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """(pg_restore, psql or None): psql only when pg_restore is newer than the server.

    Called before anything in the target is changed, so a missing tool is reported
    while the database is still as it was.
    """
    tool = find_tool("pg_restore", bin_dir)
    if tool_major(tool) > server_major:
        return tool, _sibling_tool(tool, "psql", bin_dir)
    return tool, None


def restore(db_url: str, dump_file: Path, list_file: Path, *, bin_dir: Optional[str] = None,
            timeout: Optional[float] = None, server: Optional[Tuple[int, Set[str]]] = None
            ) -> List[str]:
    """Restore ``dump_file`` into the (empty) database in one transaction.

    Returns the TOC lines that were skipped (and, on the SQL route, the opening ``SET`` statements
    that were dropped). Raises PgToolError when the restore fails; it runs in a single
    transaction, so nothing is left half-restored. ``server`` is ``server_info(db_url)``,
    read here when not given.
    """
    server_major, known = server if server is not None else server_info(db_url)
    tool, psql = restore_tools(server_major, bin_dir)
    toc = _run([tool, "--list", str(dump_file)], {}, "pg_restore --list", timeout)
    kept, skipped = filter_toc(toc)
    list_file.write_text(kept, encoding="utf-8")
    target, env = conninfo(db_url)
    if psql is not None:
        dropped = _restore_via_sql(tool, psql, dump_file, list_file, target, env, known, timeout)
        return skipped + dropped
    _run([tool, "--no-owner", "--no-privileges", "--exit-on-error", "--single-transaction",
          f"--use-list={list_file}", f"--dbname={target}", str(dump_file)],
         env, "pg_restore", timeout)
    return skipped
