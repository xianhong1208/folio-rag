"""Claim on first use: move a person's legacy per-token folders to their person key.

Before 1.5.0 a folder belonged to the token that created it (owner key = the token's
``jti``). Now it belongs to the person (``"sub:<sub>"``, see src/auth/owner_key.py).
Existing folders are not migrated in bulk — Folio cannot know which person a jti
belonged to. Instead, the first verified request that carries a ``jti`` still owning
folders moves them to that token's ``sub``: the token is signed by MCP Gate, so the
jti -> sub pairing cannot be forged.

For each folder the claim, in one database transaction:

- picks a free name when the person already has a folder of that name (``docs`` ->
  ``docs-2``, ``docs-3``, …), so two folders are never merged;
- moves the storage directory and rewrites its files' stored paths through the same
  helper a rename uses (``FolderAdapter._move_storage``);
- reassigns the row, conditional on it still belonging to that jti.

After the commit it refreshes the folder caches and the MCP tool registry, and writes
an audit-log entry for every folder that had to be renamed. A failure rolls the
database back and moves any directory it already moved back.

Concurrency: claims for one person are serialised by an in-process lock (striped by
owner key) *and* a PostgreSQL advisory transaction lock on the owner key, and the
folder rows are locked ``FOR UPDATE`` — two requests with the same token (in one
process or several) cannot both move, or half-move, a folder; the second finds
nothing left to claim. A jti found to own nothing is remembered, so later requests
with it skip the database entirely (no folder is ever created under a jti again).

Folders whose jti never shows up again (revoked / expired tokens) stay where they
are; ``count_legacy_folders`` reports how many remain.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional

from src.adapter.db_offload import run_db
from src.auth.owner_key import TokenOwner, is_person_key
from src.log import get_adapter_logger
from src.utils.names import MAX_NAME_LEN

logger = get_adapter_logger()

# Audit-log action for a folder renamed while being claimed (SettingsAudit.action is 16 chars).
AUDIT_ACTION = "claim_rename"
# Owner key of folders created in open (auth disabled) mode: not a legacy jti.
OPEN_MODE_OWNER = "console"

_LOCK_STRIPES = 64
_owner_locks = [threading.Lock() for _ in range(_LOCK_STRIPES)]
# jtis known to own no folder any more (bounded: forgetting one only costs a cheap re-check).
_CHECKED_MAX = 50_000
_checked: dict[str, None] = {}
_checked_lock = threading.Lock()


@dataclass(frozen=True)
class ClaimedFolder:
    folder_id: int
    old_name: str
    new_name: str
    description: Optional[str] = None

    @property
    def renamed(self) -> bool:
        return self.old_name != self.new_name


def _owner_lock(owner_key: str) -> threading.Lock:
    return _owner_locks[hash(owner_key) % _LOCK_STRIPES]


def _is_checked(jti: str) -> bool:
    with _checked_lock:
        return jti in _checked


def _mark_checked(jti: str) -> None:
    with _checked_lock:
        if len(_checked) >= _CHECKED_MAX:
            _checked.pop(next(iter(_checked)))  # oldest first (insertion order)
        _checked[jti] = None


def reset_claim_state() -> None:
    """Forget which jtis were checked (tests)."""
    with _checked_lock:
        _checked.clear()


def free_name(name: str, taken: set[str], exists: Callable[[str], bool] = lambda n: False) -> str:
    """``name`` if it is free, else the first free ``name-2``, ``name-3``, …

    Free = not in ``taken`` (the person's folder names) and ``exists(candidate)`` is
    false (no directory of that name on disk). A long name is shortened so the suffix
    still fits the folder-name limit; the result only uses characters folder names
    already allow (the original's plus ``-`` and digits).
    """
    if name not in taken and not exists(name):
        return name
    n = 2
    while True:
        suffix = f"-{n}"
        candidate = name[:MAX_NAME_LEN - len(suffix)] + suffix
        if candidate not in taken and not exists(candidate):
            return candidate
        n += 1


def _safe(name: str) -> bool:
    from src.storage.file_storage import _validate_safe_name
    try:
        _validate_safe_name(name)
        return True
    except ValueError:
        return False


def _claim_in_db(jti: str, owner_key: str) -> list[ClaimedFolder]:
    """Move every folder ``jti`` owns to ``owner_key`` in one transaction (blocking)."""
    from db import folder_claim_db as cdb
    from db.db import get_engine
    from src.adapter.folder import FolderAdapter
    from src.storage.file_storage import FileStorage, owner_dir_name

    try:
        owner_dir_name(jti)
        storage_ok = True
    except ValueError:  # a legacy key that never had a directory of its own
        storage_ok = False

    moved: list[tuple[str, str]] = []  # (old_name, new_name) whose directory moved
    claimed: list[ClaimedFolder] = []
    try:
        with get_engine().begin() as conn:
            cdb.lock_owner(conn, owner_key)
            rows = cdb.legacy_folders_for_update(conn, jti)
            if not rows:
                return []
            taken = cdb.owner_names(conn, owner_key)
            for row in rows:
                safe = storage_ok and _safe(row.name)
                new_name = free_name(
                    row.name, taken,
                    (lambda n: FileStorage.folder_dir(owner_key, n).exists()) if safe else
                    (lambda n: False))
                taken.add(new_name)
                if safe and FolderAdapter._move_storage(
                        row.id, jti, owner_key, row.name, new_name, conn=conn):
                    moved.append((row.name, new_name))
                if cdb.reassign(conn, row.id, jti, owner_key, new_name) != 1:
                    raise RuntimeError(f"folder {row.id} changed owner while being claimed")
                claimed.append(ClaimedFolder(row.id, row.name, new_name, row.description))
    except Exception:
        for old_name, new_name in reversed(moved):
            if not FileStorage.move_folder(owner_key, new_name, jti, old_name):
                logger.error(f"Claim rollback: could not move {new_name} back to {old_name}")
        raise
    return claimed


def _after_commit(jti: str, owner_key: str, claimed: list[ClaimedFolder],
                  actor: Optional[str]) -> None:
    """Caches, MCP tool registry and audit log for folders that just changed owner."""
    from db.cached_folderdb import CachedFolderDB
    from src.adapter.folder_events import notify_folder_owner_changed

    for c in claimed:
        CachedFolderDB.invalidate_folder_caches(
            SimpleNamespace(id=c.folder_id, user_token=jti, name=c.old_name))
        CachedFolderDB.invalidate_folder_caches(
            SimpleNamespace(id=c.folder_id, user_token=owner_key, name=c.new_name))
        notify_folder_owner_changed(c.old_name, jti, c.new_name, owner_key, c.description)
        if c.renamed:
            from db.settings_audit_db import SettingsAuditDB
            SettingsAuditDB.record(f"folder:{c.folder_id}", AUDIT_ACTION, c.old_name,
                                   c.new_name, (actor or owner_key)[:64])
            logger.info(f"Claimed folder {c.folder_id} renamed {c.old_name!r} -> {c.new_name!r} "
                        f"(the owner already had a folder of that name)")


def claim_legacy_folders(jti: Optional[str], owner_key: Optional[str], *,
                         actor: Optional[str] = None) -> list[ClaimedFolder]:
    """Move the folders ``jti`` still owns to ``owner_key`` — once (blocking).

    Only call it with a jti and owner key taken from a **verified** token. Returns
    the folders moved by this call (empty when there was nothing left to claim).
    Raises on a database / storage failure; nothing is half-moved.
    """
    if not jti or not is_person_key(owner_key) or jti == owner_key:
        return []
    if jti == OPEN_MODE_OWNER or _is_checked(jti):
        return []
    from db import folder_claim_db as cdb
    with _owner_lock(owner_key):
        # Re-checked under the lock: a concurrent request with this token may have
        # finished the claim (caches and tools included) while this one waited.
        if _is_checked(jti):
            return []
        claimed = _claim_in_db(jti, owner_key) if cdb.has_legacy(jti) else []
        if claimed:
            logger.info(f"Claimed {len(claimed)} legacy folder(s) of token {jti[:8]}… for "
                        f"{owner_key[:12]}…")
            _after_commit(jti, owner_key, claimed, actor)
        _mark_checked(jti)
    return claimed


async def claim_for(owner: Optional[TokenOwner], *, actor: Optional[str] = None
                    ) -> list[ClaimedFolder]:
    """Async, never-raising wrapper for request paths: a failed claim is logged and
    retried on the token's next request; the request itself goes on."""
    if owner is None or not owner.jti:
        return []
    try:
        return await run_db(claim_legacy_folders, owner.jti, owner.owner_key,
                            actor=actor or owner.sub)
    except Exception as e:  # noqa: BLE001 - never fail the request over a claim
        logger.error(f"Claiming legacy folders of token {owner.jti[:8]}… failed: {e}")
        return []


def count_legacy_folders() -> Optional[int]:
    """How many folders still belong to a legacy per-token key; None if unknown."""
    try:
        from db import folder_claim_db as cdb
        return cdb.count_legacy(OPEN_MODE_OWNER)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Counting legacy-owned folders failed: {e}")
        return None
