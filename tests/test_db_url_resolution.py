
"""Injection of the database.port field.

create_engine/create_async_engine only read the URL and ignore a separately
configured port field; when the URL omits the port it defaults to 5432,
connecting to the wrong instance. _resolve_db_url fills in the field port when
the URL carries none.
"""

from types import SimpleNamespace

from db.db import _resolve_db_url


def _cfg(url, port):
    return SimpleNamespace(url=url, port=port)


def test_injects_field_port_when_url_has_none():
    r = _resolve_db_url(_cfg("postgresql://postgres:pw@localhost/db", 5444))
    assert ":5444/" in r

def test_url_port_takes_precedence_over_field():
    r = _resolve_db_url(_cfg("postgresql://postgres:pw@localhost:5432/db", 5444))
    assert ":5432/" in r and ":5444" not in r

def test_no_port_anywhere_left_untouched():
    r = _resolve_db_url(_cfg("postgresql://postgres:pw@localhost/db", None))
    assert "localhost/db" in r and ":54" not in r

def test_password_preserved():
    # render_as_string(hide_password=False) — used for connecting, must not be masked
    r = _resolve_db_url(_cfg("postgresql://postgres:secret@localhost/db", 5444))
    assert "secret" in r
