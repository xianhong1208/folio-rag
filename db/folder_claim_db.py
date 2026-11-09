"""SQL for claiming legacy (per-token) folders — see src/adapter/folder_claim.py.

Every function except ``count_legacy`` / ``has_legacy`` runs on the caller's
connection, inside the caller's transaction (PostgreSQL only: advisory locks).
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import text

# Owner keys that are not legacy per-token keys: a person (``sub:…``) and the
# open-mode owner. Everything else non-empty is a jti from before 1.5.0.
_LEGACY_FILTER = (
    "user_token IS NOT NULL AND user_token <> '' "
    "AND user_token NOT LIKE 'sub:%' AND user_token <> :open_owner"
)


def lock_owner(conn, owner_key: str) -> None:
    """Serialise claims for one owner across processes (released at commit/rollback).

    Two tokens of the same person claiming at once must not both pick the same free
    name; two requests with the same token must not both move the same folder.
    """
    conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
                 {"k": f"folio-folder-owner:{owner_key}"})


def legacy_folders_for_update(conn, jti: str) -> list:
    """The folders still owned by ``jti``, row-locked (id order), as (id, name, description)."""
    return list(conn.execute(text(
        'SELECT id, name, description FROM "Folders" WHERE user_token = :jti '
        'ORDER BY id FOR UPDATE'), {"jti": jti}).all())


def owner_names(conn, owner_key: str) -> set[str]:
    """Names of the folders ``owner_key`` already owns."""
    return set(conn.execute(text('SELECT name FROM "Folders" WHERE user_token = :o'),
                            {"o": owner_key}).scalars().all())


def reassign(conn, folder_id: int, jti: str, owner_key: str, new_name: str) -> int:
    """Move one folder from ``jti`` to ``owner_key`` (and ``new_name``); returns rows changed.

    Conditional on the folder still belonging to ``jti``, so it can never move a folder
    twice or take one from anybody else.
    """
    return conn.execute(text(
        'UPDATE "Folders" SET user_token = :o, name = :n, updated_at = now() '
        'WHERE id = :id AND user_token = :jti'),
        {"o": owner_key, "n": new_name, "id": folder_id, "jti": jti}).rowcount


def has_legacy(jti: str) -> bool:
    """Cheap pre-check (no locks): does ``jti`` still own any folder?"""
    from db.db import get_engine
    with get_engine().connect() as conn:
        return conn.execute(text('SELECT 1 FROM "Folders" WHERE user_token = :jti LIMIT 1'),
                            {"jti": jti}).first() is not None


def count_legacy(open_owner: str = "console") -> Optional[int]:
    """How many folders are still owned by a legacy per-token key (server-wide)."""
    from db.db import get_engine
    with get_engine().connect() as conn:
        return int(conn.execute(text(f'SELECT count(*) FROM "Folders" WHERE {_LEGACY_FILTER}'),
                                {"open_owner": open_owner}).scalar() or 0)
