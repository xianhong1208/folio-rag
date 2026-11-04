"""Folder scoping for the admin console.

A console session *is* a token's identity: its owner key is the token's ``jti``,
and every folder / file / index / eval / search operation is limited to folders
bound to that key — exactly what a REST or MCP caller holding the token can see.
Not owning a folder is reported exactly like it not existing (404), as on the
REST API, so the console cannot be used to probe other owners' folder ids.

In open (auth disabled) mode the identity owns every folder.
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

from src.auth.console_session import ConsoleIdentity


def owned_folder(folder_id: int, ident: ConsoleIdentity):
    """Return the folder if ``ident`` owns it; 404 otherwise.

    Open mode can reach any folder, but an ownerless one still cannot be managed
    (there is no owner token to act with): 409.
    """
    from db.cached_folderdb import CachedFolderDB
    folder = CachedFolderDB.get_by_id(folder_id)
    if not folder or not ident.owns(folder.user_token):
        raise HTTPException(404, f"Folder {folder_id} not found")
    if not folder.user_token:
        raise HTTPException(
            409, f"Folder {folder_id} has no owner token — console cannot manage it")
    return folder


def owned_folder_ids(ident: ConsoleIdentity) -> Optional[set[int]]:
    """Ids of the folders ``ident`` owns; None means "every folder" (open mode)."""
    if ident.owner_key is None:
        return None
    from db.cached_folderdb import CachedFolderDB
    return {f.id for f in CachedFolderDB.get_by_user_token(ident.owner_key, use_cache=False)}


def aggregate_owner(ident: ConsoleIdentity) -> Optional[str]:
    """Owner filter for cross-folder aggregates (overview KPIs, jobs, analytics).

    Always the session's own folders — ``rag:admin`` included: it unlocks global
    settings, never other owners' folders, jobs or queries. None only in open mode
    (auth disabled), where the identity owns every folder anyway.
    """
    return ident.owner_key
