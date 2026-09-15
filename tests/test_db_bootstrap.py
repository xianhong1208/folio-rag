
"""Unit tests for src/utils/db_bootstrap.py _validate_dbname (pure logic).

CREATE DATABASE cannot bind the identifier as a param, so the allowlist is the
first line of defense against injection. The DB-touching helpers belong to the
integration scope and are not covered here.
"""

import pytest

from src.utils.db_bootstrap import _MAX_IDENTIFIER_BYTES, _validate_dbname


def test_validate_accepts_typical_dbname():
    """A conventional name (letter-leading + underscore) passes through as-is (TC-db-bootstrap-01)"""
    assert _validate_dbname("agentic_rag") == "agentic_rag"


def test_validate_accepts_allowed_special_chars():
    """Hyphen/digit/$ within the allowed character set pass (TC-db-bootstrap-02)"""
    assert _validate_dbname("db-2026_$x") == "db-2026_$x"


def test_validate_rejects_leading_digit():
    """A leading digit violates the allowlist -> ValueError (TC-db-bootstrap-03)"""
    with pytest.raises(ValueError):
        _validate_dbname("1abc")


def test_validate_rejects_sql_injection_payload():
    """An injection payload with quotes/semicolons must be blocked (TC-db-bootstrap-04)

    This is the core value of the CREATE DATABASE allowlist: an identifier cannot
    be a bind param, so the allowlist is the first line of defense (the second is
    doubled `"` escaping).
    """
    with pytest.raises(ValueError):
        _validate_dbname('evil"; DROP DATABASE x; --')


@pytest.mark.parametrize("bad", ["", "a b", "中文庫", "tab\tname"])
def test_validate_rejects_empty_whitespace_and_nonascii(bad):
    """Empty string / whitespace / non-ASCII are all rejected (TC-db-bootstrap-05)"""
    with pytest.raises(ValueError):
        _validate_dbname(bad)


def test_validate_rejects_over_63_bytes():
    """Reject names over Postgres NAMEDATALEN-1 (63 bytes) -- otherwise they get silently truncated (TC-db-bootstrap-06)"""
    with pytest.raises(ValueError):
        _validate_dbname("x" * (_MAX_IDENTIFIER_BYTES + 1))
    # Exactly 63 bytes must pass (boundary)
    assert _validate_dbname("x" * _MAX_IDENTIFIER_BYTES) == "x" * _MAX_IDENTIFIER_BYTES
