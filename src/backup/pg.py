"""``pg_dump`` / ``pg_restore`` for Folio's database.

The password never goes on a command line (where ``ps`` shows it): the tools get the
URL without it and read ``PGPASSWORD`` from their environment. Neither tool needs a
superuser: the dump is taken with ``--no-owner --no-privileges``, and the restore skips
the ``vector`` extension entry (Folio's start-up bootstrap creates the extension the
same way it always does) and the ``public`` schema entries, which only the schema's
owner may change.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def dump(db_url: str, out_file: Path, *, snapshot: Optional[str] = None,
         bin_dir: Optional[str] = None, timeout: Optional[float] = None) -> None:
    """``pg_dump --format=custom`` of the whole database into ``out_file``.

    ``snapshot`` (from ``pg_export_snapshot()``) makes the dump see exactly what the
    caller's open transaction sees.
    """
    tool = find_tool("pg_dump", bin_dir)
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


def restore(db_url: str, dump_file: Path, list_file: Path, *, bin_dir: Optional[str] = None,
            timeout: Optional[float] = None) -> List[str]:
    """Restore ``dump_file`` into the (empty) database in one transaction.

    Returns the TOC lines that were skipped. Raises PgToolError when pg_restore fails;
    ``--single-transaction`` means nothing is left half-restored.
    """
    tool = find_tool("pg_restore", bin_dir)
    toc = _run([tool, "--list", str(dump_file)], {}, "pg_restore --list", timeout)
    kept, skipped = filter_toc(toc)
    list_file.write_text(kept, encoding="utf-8")
    target, env = conninfo(db_url)
    _run([tool, "--no-owner", "--no-privileges", "--exit-on-error", "--single-transaction",
          f"--use-list={list_file}", f"--dbname={target}", str(dump_file)],
         env, "pg_restore", timeout)
    return skipped
