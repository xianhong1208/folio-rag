
"""Unit tests for src/utils/db_bootstrap.py _validate_dbname (pure logic).

CREATE DATABASE cannot bind the identifier as a param, so the allowlist is the
first line of defense against injection. The DB-touching helpers belong to the
integration scope and are not covered here.
"""

import pytest

from src.utils.db_bootstrap import _MAX_IDENTIFIER_BYTES, _validate_dbname


def test_validate_accepts_typical_dbname():
    """A conventional name (letter-leading + underscore) passes through as-is"""
    assert _validate_dbname("agentic_rag") == "agentic_rag"


def test_validate_accepts_allowed_special_chars():
    """Hyphen/digit/$ within the allowed character set pass"""
    assert _validate_dbname("db-2026_$x") == "db-2026_$x"


def test_validate_rejects_leading_digit():
    """A leading digit violates the allowlist -> ValueError"""
    with pytest.raises(ValueError):
        _validate_dbname("1abc")


def test_validate_rejects_sql_injection_payload():
    """An injection payload with quotes/semicolons must be blocked

    This is the core value of the CREATE DATABASE allowlist: an identifier cannot
    be a bind param, so the allowlist is the first line of defense (the second is
    doubled `"` escaping).
    """
    with pytest.raises(ValueError):
        _validate_dbname('evil"; DROP DATABASE x; --')


@pytest.mark.parametrize("bad", ["", "a b", "中文庫", "tab\tname"])
def test_validate_rejects_empty_whitespace_and_nonascii(bad):
    """Empty string / whitespace / non-ASCII are all rejected"""
    with pytest.raises(ValueError):
        _validate_dbname(bad)


def test_validate_rejects_over_63_bytes():
    """Reject names over Postgres NAMEDATALEN-1 (63 bytes) -- otherwise they get silently truncated"""
    with pytest.raises(ValueError):
        _validate_dbname("x" * (_MAX_IDENTIFIER_BYTES + 1))
    # Exactly 63 bytes must pass (boundary)
    assert _validate_dbname("x" * _MAX_IDENTIFIER_BYTES) == "x" * _MAX_IDENTIFIER_BYTES


def test_hnsw_hook_also_ensures_lookup_indexes():
    """The post-write hook covers lookup indexes too, so new folders get them at once."""
    from unittest.mock import patch
    from src.utils import db_bootstrap as b
    with patch.object(b, "_ensure_hnsw_index", return_value=False) as h, \
         patch.object(b, "ensure_lookup_indexes_for_table", return_value=2) as lk:
        assert b.ensure_hnsw_for_table("E", "data_1_x") is False
    h.assert_called_once_with("E", "data_1_x", None)
    lk.assert_called_once_with("E", "data_1_x", None)


def test_lookup_index_failure_is_not_fatal():
    from unittest.mock import MagicMock
    from src.utils.db_bootstrap import ensure_lookup_indexes_for_table
    engine = MagicMock()
    engine.begin.side_effect = RuntimeError("db down")
    log = MagicMock()
    assert ensure_lookup_indexes_for_table(engine, "data_1_x", log) == 0
    assert log.warning.call_count == 2
