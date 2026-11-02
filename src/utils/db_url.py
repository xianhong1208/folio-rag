"""The one place that turns the `database` config section into a connection URL.

`database.url` may omit the port and carry it in the separate `database.port`
field. SQLAlchemy, psycopg2 and Alembic only read the URL, so a portless URL
silently means 5432 and a user who set `port: 5444` ends up talking to a different
server (typically "password authentication failed"). Every connection path —
runtime engines, startup bootstrap, migrations, Alembic, the vector store and the
integration tests — must go through resolve_db_url so they all agree.
"""

from __future__ import annotations


def resolve_db_url(db_conf) -> str:
    """Return the full connection URL for a `database` config section.

    The `database.port` field is injected when the URL has no port of its own; an
    explicit port in the URL wins. The password is kept (this is for connecting,
    never for logging).
    """
    from sqlalchemy.engine import make_url

    url = make_url(db_conf.url)
    port = getattr(db_conf, "port", None)
    if url.port is None and port:
        url = url.set(port=int(port))
    return url.render_as_string(hide_password=False)


def alembic_safe_url(url: str) -> str:
    """Escape a URL for Alembic's ``Config.set_main_option``.

    Alembic stores options in a ConfigParser with %-interpolation, so a
    percent-encoded character in the URL (e.g. ``%40`` for ``@`` in a password)
    would be read as an interpolation and fail. ``%%`` is read back as ``%``.
    """
    return url.replace("%", "%%")


def db_url_from_config_file(config_path) -> str:
    """Resolve the connection URL straight from a config YAML file.

    For callers that cannot go through the Config singleton (the integration-test
    harness reads the config before choosing a database). Applies the same
    ``${VAR:-default}`` expansion as Config and the same port rule as
    resolve_db_url.
    """
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from src.config.config_manager import expand_env_vars

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    db = expand_env_vars(raw.get("database") or {})
    if not db.get("url"):
        raise RuntimeError(f"{config_path} has no database.url")
    return resolve_db_url(SimpleNamespace(url=db["url"], port=db.get("port")))
