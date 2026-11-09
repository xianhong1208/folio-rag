"""Claim on first use against real PostgreSQL: rows, stored file paths, directories,
tools and audit move together, exactly once, also when several connections race
(the advisory lock + FOR UPDATE, not only the in-process lock)."""

from __future__ import annotations

import threading
import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from db import folder_claim_db
from src.adapter import folder_claim as fc

_REAL_HAS_LEGACY = folder_claim_db.has_legacy  # the root conftest stubs it per test


@pytest.fixture()
def env(itest_db, tmp_path, monkeypatch):
    from src.adapter import folder_events
    from src.storage.file_storage import FileStorage
    monkeypatch.setattr(folder_claim_db, "has_legacy", _REAL_HAS_LEGACY)
    monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path / "storage")
    monkeypatch.setattr(FileStorage, "STORAGE_FILE_ROOT", tmp_path)
    reg, unreg = MagicMock(), MagicMock()
    folder_events.set_folder_tool_hooks(register=reg, unregister=unreg)
    yield type("Env", (), {"root": tmp_path, "reg": reg, "unreg": unreg})
    folder_events.clear_folder_tool_hooks()


def _folder(owner, name, with_file=True):
    from db.filedb import FileDB
    from db.folderdb import FolderDB
    from src.storage.file_storage import FileStorage
    f = FolderDB.create(name=name, description=f"{name} desc", user_token=owner)
    if with_file:
        fid = str(uuid.uuid4())
        path = FileStorage.save_file(owner, name, fid, name.encode())
        FileDB.create(folder_id=f.id, file_name="a.txt", file_path=path, file_size=1,
                      mime_type="text/plain")
    return f


def _row(fid):
    from db.db import get_engine
    with get_engine().connect() as c:
        owner, name = c.execute(text('SELECT user_token, name FROM "Folders" WHERE id=:i'),
                                {"i": fid}).one()
        paths = c.execute(text('SELECT file_path FROM "Files" WHERE folder_id=:i'),
                          {"i": fid}).scalars().all()
    return owner, name, paths


def _ids():
    jti = str(uuid.uuid4())
    return jti, f"sub:user-{uuid.uuid4().hex[:8]}"


def test_claim_moves_rows_paths_dirs_tools_and_audits_renames(env):
    jti, owner = _ids()
    docs, notes = _folder(jti, "docs"), _folder(jti, "notes")
    _folder(owner, "docs", with_file=False)  # the person already has a "docs"
    before = folder_claim_db.count_legacy()

    out = fc.claim_legacy_folders(jti, owner, actor="user")

    assert {(c.folder_id, c.new_name) for c in out} == {(docs.id, "docs-2"), (notes.id, "notes")}
    odir = owner.replace("sub:", "sub_")
    o, n, paths = _row(docs.id)
    assert (o, n) == (owner, "docs-2")
    assert paths and all(p.startswith(f"storage/{odir}/docs-2/") for p in paths)
    assert (env.root / paths[0]).read_bytes() == b"docs"
    assert not (env.root / "storage" / jti).joinpath("docs").exists()
    assert _row(notes.id)[:2] == (owner, "notes")
    assert folder_claim_db.count_legacy() == before - 2
    env.reg.assert_any_call("docs-2", "docs desc", owner)
    env.unreg.assert_any_call("docs", user_token=jti)
    from db.settings_audit_db import SettingsAuditDB
    entry = next(e for e in SettingsAuditDB.recent(20) if e["key"] == f"folder:{docs.id}")
    assert (entry["action"], entry["old"], entry["new"]) == ("claim_rename", "docs", "docs-2")

    # idempotent: nothing left, nothing moves again
    fc.reset_claim_state()
    assert fc.claim_legacy_folders(jti, owner) == []
    assert _row(docs.id)[:2] == (owner, "docs-2")


def test_racing_connections_move_each_folder_once(env):
    """Bypasses the in-process lock: the database alone must serialise the claims."""
    jti, owner = _ids()
    folders = [_folder(jti, f"f{i}") for i in range(3)]
    results, errors, barrier = [], [], threading.Barrier(6)

    def worker():
        barrier.wait()
        try:
            results.append(fc._claim_in_db(jti, owner))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sorted(len(r) for r in results) == [0, 0, 0, 0, 0, 3]
    for f in folders:
        o, name, paths = _row(f.id)
        assert (o, name) == (owner, f.name)
        assert (env.root / paths[0]).read_bytes() == f.name.encode()


def test_two_tokens_of_one_person_never_get_the_same_name(env):
    jti_a, owner = _ids()
    jti_b = str(uuid.uuid4())
    a, b = _folder(jti_a, "docs"), _folder(jti_b, "docs")
    barrier = threading.Barrier(2)

    def worker(jti):
        barrier.wait()
        fc._claim_in_db(jti, owner)
    threads = [threading.Thread(target=worker, args=(j,)) for j in (jti_a, jti_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    names = {_row(a.id)[1], _row(b.id)[1]}
    assert names == {"docs", "docs-2"}
    for f in (a, b):
        o, name, paths = _row(f.id)
        assert o == owner and (env.root / paths[0]).read_bytes() == b"docs"


def test_count_ignores_people_open_mode_and_ownerless(env):
    jti, owner = _ids()
    before = folder_claim_db.count_legacy()
    _folder(owner, "mine", with_file=False)
    _folder("console", f"open-{uuid.uuid4().hex[:6]}", with_file=False)
    from db.folderdb import FolderDB
    FolderDB.create(name="ghost", user_token=None)
    assert folder_claim_db.count_legacy() == before
    _folder(jti, "legacy", with_file=False)
    assert folder_claim_db.count_legacy() == before + 1
    assert folder_claim_db.has_legacy(jti) and not folder_claim_db.has_legacy(str(uuid.uuid4()))


def test_reassign_never_takes_a_folder_that_moved_on(env):
    from db.db import get_engine
    jti, owner = _ids()
    f = _folder(jti, "docs", with_file=False)
    with get_engine().begin() as conn:
        assert folder_claim_db.reassign(conn, f.id, "some-other-jti", owner, "docs") == 0
        assert folder_claim_db.reassign(conn, f.id, jti, owner, "docs") == 1
        assert folder_claim_db.reassign(conn, f.id, jti, owner, "docs") == 0  # already moved
    assert _row(f.id)[:2] == (owner, "docs")


def test_claimed_rows_are_locked_until_the_claim_commits(env):
    """Nothing else can rename or delete a folder while it is being moved."""
    from sqlalchemy.exc import OperationalError
    from db.db import get_engine
    jti, owner = _ids()
    f = _folder(jti, "docs", with_file=False)
    with get_engine().begin() as conn:
        assert [r.id for r in folder_claim_db.legacy_folders_for_update(conn, jti)] == [f.id]
        with get_engine().connect() as other:
            other.execute(text("SET lock_timeout = '200ms'"))
            with pytest.raises(OperationalError):
                other.execute(text('UPDATE "Folders" SET name = \'x\' WHERE id = :i'), {"i": f.id})
            other.rollback()
