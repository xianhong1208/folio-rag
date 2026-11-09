"""Claim on first use (src/adapter/folder_claim.py): legacy per-token folders move to
the person once, idempotently, safely under concurrency; name collisions get a free
name and an audit entry, never a merge.

The SQL lives in db/folder_claim_db.py and runs against PostgreSQL in
tests/integration/test_folder_claim_db.py; here it is faked so the orchestration —
locking, once-only, storage moves and their rollback, tools, caches, audit — is
tested in isolation.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from src.adapter import folder_claim as fc
from src.auth.owner_key import TokenOwner

JTI, OWNER = "8f14e45f-ceea-4f6a-9c43-1f2c3d4e5f60", "sub:user-1"


# -- free names --------------------------------------------------------------------------


class TestFreeName:
    def test_free_name_is_kept(self):
        assert fc.free_name("docs", {"notes"}) == "docs"

    def test_taken_name_gets_the_first_free_suffix(self):
        assert fc.free_name("docs", {"docs"}) == "docs-2"
        assert fc.free_name("docs", {"docs", "docs-2", "docs-3"}) == "docs-4"

    def test_a_directory_on_disk_counts_as_taken(self):
        assert fc.free_name("docs", set(), exists=lambda n: n in ("docs", "docs-2")) == "docs-3"

    def test_long_name_still_fits(self):
        from src.storage.file_storage import _validate_safe_name
        from src.utils.names import MAX_NAME_LEN
        long = "a" * MAX_NAME_LEN
        new = fc.free_name(long, {long})
        assert len(new) == MAX_NAME_LEN and new.endswith("-2")
        _validate_safe_name(new)

    def test_cjk_name_stays_a_valid_folder_name(self):
        from src.storage.file_storage import _validate_safe_name
        new = fc.free_name("產品 手冊", {"產品 手冊"})
        assert new == "產品 手冊-2"
        _validate_safe_name(new)


# -- once, idempotent, concurrent ------------------------------------------------------


def _claimed(*pairs):
    return [fc.ClaimedFolder(i, old, new, None) for i, (old, new) in enumerate(pairs, 1)]


@pytest.fixture()
def fake_db(monkeypatch):
    """has_legacy is true until the (faked) transaction has run once."""
    state = SimpleNamespace(left=True, runs=0, after=[])

    def claim_in_db(jti, owner_key):
        state.runs += 1
        time.sleep(0.05)  # widen the race window
        if not state.left:
            return []
        state.left = False
        return _claimed(("docs", "docs-2"))

    from db import folder_claim_db
    monkeypatch.setattr(folder_claim_db, "has_legacy", lambda jti: state.left)
    monkeypatch.setattr(fc, "_claim_in_db", claim_in_db)
    monkeypatch.setattr(fc, "_after_commit", lambda *a, **k: state.after.append(a))
    return state


class TestOnce:
    @pytest.mark.parametrize("jti, owner", [
        (None, OWNER), ("", OWNER),
        (JTI, "8f14e45f-not-a-person"),   # owner key without the sub: prefix
        (JTI, None),
        (OWNER, OWNER),                   # nothing to move onto itself
        ("console", OWNER),               # the open-mode owner is not a jti
    ])
    def test_nothing_is_claimed_without_a_jti_and_a_person(self, fake_db, jti, owner):
        assert fc.claim_legacy_folders(jti, owner) == []
        assert fake_db.runs == 0

    def test_claims_once_then_never_touches_the_db_again(self, fake_db, monkeypatch):
        first = fc.claim_legacy_folders(JTI, OWNER)
        assert [(c.old_name, c.new_name) for c in first] == [("docs", "docs-2")]
        from db import folder_claim_db
        monkeypatch.setattr(folder_claim_db, "has_legacy",
                            lambda jti: pytest.fail("checked jti must skip the DB"))
        assert fc.claim_legacy_folders(JTI, OWNER) == []
        assert fake_db.runs == 1 and len(fake_db.after) == 1

    def test_jti_without_folders_is_remembered(self, monkeypatch):
        from db import folder_claim_db
        calls = []
        monkeypatch.setattr(folder_claim_db, "has_legacy", lambda jti: calls.append(jti) or False)
        monkeypatch.setattr(fc, "_claim_in_db", lambda *a: pytest.fail("nothing to claim"))
        assert fc.claim_legacy_folders(JTI, OWNER) == []
        assert fc.claim_legacy_folders(JTI, OWNER) == []
        assert calls == [JTI]

    def test_concurrent_requests_with_one_token_claim_once(self, fake_db):
        results, barrier = [], threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(fc.claim_legacy_folders(JTI, OWNER))
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert fake_db.runs == 1
        assert sum(1 for r in results if r) == 1 and len(fake_db.after) == 1

    def test_two_tokens_of_one_person_are_serialised(self, monkeypatch):
        """Different jtis, same person: the claims never overlap (one picks names after
        the other committed), so two folders cannot both get the same free name."""
        from db import folder_claim_db
        monkeypatch.setattr(folder_claim_db, "has_legacy", lambda jti: True)
        monkeypatch.setattr(fc, "_after_commit", lambda *a, **k: None)
        active, overlap = [0], []

        def claim_in_db(jti, owner_key):
            active[0] += 1
            overlap.append(active[0])
            time.sleep(0.05)
            active[0] -= 1
            return []
        monkeypatch.setattr(fc, "_claim_in_db", claim_in_db)
        threads = [threading.Thread(target=fc.claim_legacy_folders, args=(f"jti-{i}", OWNER))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(overlap) == 4 and max(overlap) == 1

    def test_failure_is_retried_on_the_next_request(self, monkeypatch):
        from db import folder_claim_db
        monkeypatch.setattr(folder_claim_db, "has_legacy", lambda jti: True)
        monkeypatch.setattr(fc, "_after_commit", lambda *a, **k: None)
        attempts = []

        def flaky(jti, owner_key):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("db down")
            return _claimed(("docs", "docs"))
        monkeypatch.setattr(fc, "_claim_in_db", flaky)
        with pytest.raises(RuntimeError):
            fc.claim_legacy_folders(JTI, OWNER)
        assert len(fc.claim_legacy_folders(JTI, OWNER)) == 1

    async def test_claim_for_never_fails_the_request(self, monkeypatch):
        monkeypatch.setattr(fc, "claim_legacy_folders",
                            MagicMock(side_effect=RuntimeError("db down")))
        owner = TokenOwner(owner_key=OWNER, sub="user-1", jti=JTI)
        assert await fc.claim_for(owner) == []
        assert await fc.claim_for(None) == []
        assert await fc.claim_for(TokenOwner(owner_key=OWNER, sub="user-1")) == []

    async def test_claim_for_passes_the_token_owner(self, monkeypatch):
        spy = MagicMock(return_value=[])
        monkeypatch.setattr(fc, "claim_legacy_folders", spy)
        await fc.claim_for(TokenOwner(owner_key=OWNER, sub="user-1", jti=JTI))
        spy.assert_called_once_with(JTI, OWNER, actor="user-1")


# -- the transaction ----------------------------------------------------------------------


class FakeConn:
    def __init__(self):
        self.executed = []

    def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        return MagicMock()


@pytest.fixture()
def tx(tmp_path, monkeypatch):
    """A fake engine/transaction + fake SQL helpers over a real storage directory."""
    from db import folder_claim_db as cdb
    from src.storage.file_storage import FileStorage
    monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path)
    state = SimpleNamespace(conn=FakeConn(), committed=False, rolled_back=False,
                            rows=[], owner_names=set(), reassign_rowcount={}, locked=[],
                            reassigned=[])

    @contextmanager
    def begin():
        try:
            yield state.conn
            state.committed = True
        except Exception:
            state.rolled_back = True
            raise
    engine = SimpleNamespace(begin=begin)
    monkeypatch.setattr("db.db.get_engine", lambda: engine)
    monkeypatch.setattr(cdb, "lock_owner", lambda conn, o: state.locked.append(o))
    monkeypatch.setattr(cdb, "legacy_folders_for_update", lambda conn, jti: state.rows)
    monkeypatch.setattr(cdb, "owner_names", lambda conn, o: set(state.owner_names))

    def reassign(conn, fid, jti, owner, name):
        state.reassigned.append((fid, jti, owner, name))
        return state.reassign_rowcount.get(fid, 1)
    monkeypatch.setattr(cdb, "reassign", reassign)
    state.root = tmp_path
    return state


def _row(fid, name, desc=None):
    return SimpleNamespace(id=fid, name=name, description=desc)


def _mkfolder(root, owner_dir, name):
    d = root / owner_dir / name
    d.mkdir(parents=True)
    (d / "file-1").write_text(name)
    return d


class TestTransaction:
    def test_moves_rows_files_and_renames_collisions(self, tx):
        _mkfolder(tx.root, JTI, "docs")
        _mkfolder(tx.root, JTI, "notes")
        tx.rows = [_row(1, "docs", "d"), _row(2, "notes")]
        tx.owner_names = {"docs"}  # the person already has a "docs"
        out = fc._claim_in_db(JTI, OWNER)
        assert [(c.folder_id, c.old_name, c.new_name, c.renamed) for c in out] == [
            (1, "docs", "docs-2", True), (2, "notes", "notes", False)]
        assert tx.locked == [OWNER] and tx.committed
        assert tx.reassigned == [(1, JTI, OWNER, "docs-2"), (2, JTI, OWNER, "notes")]
        assert (tx.root / "sub_user-1" / "docs-2" / "file-1").read_text() == "docs"
        assert (tx.root / "sub_user-1" / "notes" / "file-1").read_text() == "notes"
        assert not (tx.root / JTI / "docs").exists()
        # stored paths rewritten inside the same transaction
        rewrites = [p for s, p in tx.conn.executed if 'UPDATE "Files"' in s]
        assert rewrites[0] == {"old_path": f"storage/{JTI}/docs/",
                               "new_path": "storage/sub_user-1/docs-2/", "folder_id": 1}

    def test_two_claimed_folders_with_one_name_both_survive(self, tx):
        tx.rows = [_row(1, "docs"), _row(2, "docs")]  # e.g. a ghost duplicate
        out = fc._claim_in_db(JTI, OWNER)
        assert [c.new_name for c in out] == ["docs", "docs-2"]

    def test_an_orphan_directory_counts_as_taken(self, tx):
        _mkfolder(tx.root, JTI, "docs")
        (tx.root / "sub_user-1" / "docs").mkdir(parents=True)  # left over on disk
        tx.rows = [_row(1, "docs")]
        (c,) = fc._claim_in_db(JTI, OWNER)
        assert c.new_name == "docs-2" and (tx.root / "sub_user-1" / "docs-2" / "file-1").exists()

    def test_folder_without_directory_only_changes_owner(self, tx):
        tx.rows = [_row(1, "docs")]
        (c,) = fc._claim_in_db(JTI, OWNER)
        assert c.new_name == "docs" and tx.reassigned == [(1, JTI, OWNER, "docs")]

    def test_ghost_row_with_unsafe_name_is_reassigned_without_storage(self, tx):
        tx.rows = [_row(1, "../evil")]
        (c,) = fc._claim_in_db(JTI, OWNER)
        assert c.new_name == "../evil"
        assert not any('UPDATE "Files"' in s for s, _ in tx.conn.executed)

    def test_nothing_left_commits_nothing(self, tx):
        assert fc._claim_in_db(JTI, OWNER) == [] and tx.reassigned == []

    def test_failure_rolls_back_and_moves_directories_back(self, tx):
        _mkfolder(tx.root, JTI, "docs")
        _mkfolder(tx.root, JTI, "notes")
        tx.rows = [_row(1, "docs"), _row(2, "notes")]
        tx.reassign_rowcount = {2: 0}  # someone else moved folder 2 meanwhile
        with pytest.raises(RuntimeError):
            fc._claim_in_db(JTI, OWNER)
        assert tx.rolled_back and not tx.committed
        assert (tx.root / JTI / "docs" / "file-1").exists()
        assert (tx.root / JTI / "notes" / "file-1").exists()
        assert not (tx.root / "sub_user-1" / "docs").exists()

    def test_target_directory_that_cannot_be_used_aborts(self, tx, monkeypatch):
        from src.adapter.folder import FolderStorageMoveError
        from src.storage.file_storage import FileStorage
        _mkfolder(tx.root, JTI, "docs")
        tx.rows = [_row(1, "docs")]
        monkeypatch.setattr(FileStorage, "move_folder", lambda *a: False)
        with pytest.raises(FolderStorageMoveError):
            fc._claim_in_db(JTI, OWNER)
        assert tx.rolled_back and tx.reassigned == []


# -- after the commit ---------------------------------------------------------------------


class TestAfterCommit:
    def test_tools_caches_and_audit(self):
        claimed = [fc.ClaimedFolder(1, "docs", "docs-2", "d"), fc.ClaimedFolder(2, "notes", "notes")]
        with patch("src.adapter.folder_events.notify_folder_owner_changed") as moved, \
             patch("db.cached_folderdb.CachedFolderDB.invalidate_folder_caches") as inval, \
             patch("db.settings_audit_db.SettingsAuditDB.record") as audit:
            fc._after_commit(JTI, OWNER, claimed, "user-1")
        assert moved.call_args_list == [call("docs", JTI, "docs-2", OWNER, "d"),
                                        call("notes", JTI, "notes", OWNER, None)]
        # both the old (jti, name) and the new (owner, name) cache entries go
        keys = {(c.args[0].user_token, c.args[0].name) for c in inval.call_args_list}
        assert keys == {(JTI, "docs"), (OWNER, "docs-2"), (JTI, "notes"), (OWNER, "notes")}
        # an audit entry for the renamed folder only
        audit.assert_called_once_with("folder:1", "claim_rename", "docs", "docs-2", "user-1")
        assert len(fc.AUDIT_ACTION) <= 16  # SettingsAudit.action is VARCHAR(16)

    def test_claim_runs_after_commit_with_the_actor(self, monkeypatch):
        from db import folder_claim_db
        monkeypatch.setattr(folder_claim_db, "has_legacy", lambda jti: True)
        monkeypatch.setattr(fc, "_claim_in_db", lambda *a: _claimed(("docs", "docs")))
        after = MagicMock()
        monkeypatch.setattr(fc, "_after_commit", after)
        fc.claim_legacy_folders(JTI, OWNER, actor="user-1")
        after.assert_called_once()
        assert after.call_args.args[:2] == (JTI, OWNER) and after.call_args.args[3] == "user-1"


class TestOwnerChangedHook:
    @pytest.fixture(autouse=True)
    def hooks(self):
        from src.adapter import folder_events as fe
        reg, unreg = MagicMock(), MagicMock()
        fe.set_folder_tool_hooks(register=reg, unregister=unreg)
        yield SimpleNamespace(reg=reg, unreg=unreg, fe=fe)
        fe.clear_folder_tool_hooks()

    def test_moves_the_tool_to_the_new_owner(self, hooks):
        assert hooks.fe.notify_folder_owner_changed("docs", JTI, "docs-2", OWNER, "d")
        hooks.unreg.assert_called_once_with("docs", user_token=JTI)
        hooks.reg.assert_called_once_with("docs-2", "d", OWNER)

    def test_register_still_runs_when_unregister_fails(self, hooks):
        hooks.unreg.side_effect = RuntimeError("x")
        assert not hooks.fe.notify_folder_owner_changed("docs", JTI, "docs", OWNER, None)
        hooks.reg.assert_called_once()

    def test_register_failure_returns_false(self, hooks):
        hooks.reg.side_effect = RuntimeError("x")
        assert not hooks.fe.notify_folder_owner_changed("docs", JTI, "docs", OWNER, None)

    def test_unwired(self, hooks):
        hooks.fe.clear_folder_tool_hooks()
        assert not hooks.fe.notify_folder_owner_changed("docs", JTI, "docs", OWNER, None)


def test_checked_set_is_bounded(monkeypatch):
    monkeypatch.setattr(fc, "_CHECKED_MAX", 3)
    for i in range(5):
        fc._mark_checked(f"j{i}")
    assert not fc._is_checked("j0") and fc._is_checked("j4")


def test_claim_for_is_awaitable_from_async_code():
    """claim_for offloads the blocking claim to a worker thread."""
    import asyncio
    with patch.object(fc, "claim_legacy_folders", return_value=[]) as c:
        asyncio.run(fc.claim_for(TokenOwner(owner_key=OWNER, sub="u", jti=JTI)))
    assert c.called
