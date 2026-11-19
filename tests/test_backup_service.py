"""Backup / restore orchestration with a scripted database (the real thing runs in
tests/integration/test_backup_db.py)."""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.backup import archive as A
from src.backup import service as S

NOW = datetime(2026, 11, 14, 4, 0, tzinfo=timezone.utc)
RAG = SimpleNamespace(embedding=SimpleNamespace(model="e5", dimension=8), profile="cloud",
                      retrieval=SimpleNamespace(segmenter="bigram"))


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def scalar(self):
        if not self.rows:
            return None
        r = self.rows[0]
        return r[0] if isinstance(r, tuple) else r

    def mappings(self):
        return self

    def all(self):
        return list(self.rows)

    def __iter__(self):
        return iter(self.rows)


class FakeDB:
    """Answers the queries the service sends, by SQL fragment."""

    def __init__(self, tables=("Folders", "Files", "FileIndices", "IndexJobs", "RuntimeSettings",
                               "alembic_version", "data_1_u1"), jobs=(), overrides=None):
        self.tables = set(tables)
        self.jobs = list(jobs)
        self.overrides = overrides or {}
        self.sql = []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.sql.append(sql)
        if "to_regclass" in sql:
            name = params["n"].split('"')[1]
            return _Result([name in self.tables])
        if "FROM alembic_version" in sql:
            return _Result(["rev2"])
        if 'FROM "RuntimeSettings"' in sql:
            return _Result(list(self.overrides.items()))
        if 'FROM "IndexJobs"' in sql:
            return _Result(self.jobs)
        if "pg_export_snapshot" in sql:
            return _Result(["00000003-1"])
        if 'FROM "Folders"' in sql:
            return _Result([{"id": 1, "name": "docs", "vector_table_uuid": "u1"}])
        if 'GROUP BY folder_id' in sql:
            return _Result([(1, 2)])
        if 'count(*) FROM "data_1_u1"' in sql:
            return _Result([5])
        if 'SELECT file_path FROM "Files"' in sql:
            return _Result([("storage/sub_u/docs/f1",), ("storage/sub_u/docs/gone",), ("/elsewhere/x",)])
        if 'FROM "FileIndices"' in sql:
            return _Result([("e5", 8, 2)])
        if "atttypmod" in sql:
            return _Result([(8,)])
        if "server_version_num" in sql:
            return _Result(["170001"])
        if "server_version" in sql:
            return _Result(["17.1"])
        raise AssertionError(f"unexpected SQL: {sql}")

    # connection protocol
    def execution_options(self, **kw):
        return self

    def begin(self):
        return self

    def connect(self):
        return self

    def dispose(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def storage(tmp_path):
    root = tmp_path / "storage"
    (root / "sub_u" / "docs").mkdir(parents=True)
    (root / "sub_u" / "docs" / "f1").write_bytes(b"file one")
    (root / ".console_session_secret").write_text("s")
    return root


@pytest.fixture
def fake(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(S, "_engine", lambda url: db)

    def fake_dump(url, out_file, snapshot=None, bin_dir=None, server_major=None):
        assert snapshot == "00000003-1"
        assert server_major == 17
        Path(out_file).write_bytes(b"PGDMP")
    monkeypatch.setattr(S.pg, "dump", fake_dump)
    monkeypatch.setattr(S.pg, "find_tool", lambda name, bin_dir=None: f"/bin/{name}")
    monkeypatch.setattr(S.pg, "tool_version", lambda tool: "pg_dump (PostgreSQL) 17.1")
    monkeypatch.setattr(S, "current_version", lambda: "1.4.2")
    return db


def _backup(tmp_path, storage, **kw):
    return S.create_backup(db_url="postgresql://u:p@h/db", storage_root=storage, out_dir=tmp_path / "b",
                           rag_conf=RAG, now=NOW, **kw)


class TestCreateBackup:
    def test_archive_and_manifest(self, tmp_path, storage, fake):
        r = _backup(tmp_path, storage)
        assert r.path.name == "folio-backup-20261114T040000Z.tar" and r.size == r.path.stat().st_size
        m = A.read_manifest(r.path)
        assert m["alembic_revision"] == "rev2" and m["folio_version"] == "1.4.2"
        assert m["embedding"] == {"model": "e5", "dimension": 8, "from_runtime_settings": []}
        assert m["folders"] == [{"id": 1, "name": "docs", "files": 2, "chunks": 5, "vector_table": "data_1_u1"}]
        assert m["indexed_embeddings"] == [{"model": "e5", "dimension": 8, "files": 2}]
        assert m["vector_dimensions"] == [8]
        assert set(m["checksums"]) == {"database.dump", "storage/sub_u/docs/f1", "storage/.console_session_secret"}
        assert m["storage"]["missing"] == ["storage/sub_u/docs/gone"] and m["storage"]["outside_storage"] == 1
        assert any("not in storage/" in w for w in r.warnings)
        A.verify_and_extract(r.path, m)
        # the snapshot was read in a REPEATABLE READ transaction before the dump
        assert any("pg_export_snapshot" in s for s in fake.sql)
        assert not list((tmp_path / "b").glob(".folio-backup-*"))  # temp dir cleaned

    def test_cex35_the_scratch_directory_is_left_out(self, tmp_path, storage, fake):
        """CEX-35 -- storage/.folio-work (exports, uploads, import staging) is never backed up."""
        work = storage / ".folio-work" / "console" / "export-x"
        work.mkdir(parents=True)
        (work / "a.tar").write_bytes(b"tar")
        (storage / ".folio-work" / "import-1").mkdir()
        (storage / ".folio-work" / "import-1" / "folder.json").write_text("{}")
        m = A.read_manifest(_backup(tmp_path, storage).path)
        assert not any(".folio-work" in k for k in m["checksums"])
        assert "storage/sub_u/docs/f1" in m["checksums"]

    def test_runtime_settings_win_over_the_config(self, tmp_path, storage, fake):
        fake.overrides = {"rag.embedding.model": {"value": "bge-m3"}, "rag.embedding.dimension": {"value": 1024},
                          "rag.retrieval.segmenter": {"value": "jieba"}}
        m = _backup(tmp_path, storage).manifest
        assert m["embedding"] == {"model": "bge-m3", "dimension": 1024,
                                  "from_runtime_settings": ["model", "dimension"]}
        assert m["segmenter"] == {"name": "jieba", "from_runtime_settings": True}

    def test_refused_while_an_index_job_runs(self, tmp_path, storage, fake):
        fake.jobs = [{"job_id": "j1", "folder_id": 1, "status": "running", "kind": "index",
                      "last_updated_at": (NOW - timedelta(minutes=1)).isoformat()}]
        with pytest.raises(S.JobsRunning, match="--allow-running-jobs"):
            _backup(tmp_path, storage)
        assert not list((tmp_path / "b").glob("*.tar*"))

    def test_running_jobs_allowed_are_recorded(self, tmp_path, storage, fake):
        fake.jobs = [{"job_id": "j1", "folder_id": 1, "status": "running", "kind": "index",
                      "last_updated_at": (NOW - timedelta(minutes=1)).isoformat()}]
        m = _backup(tmp_path, storage, allow_running_jobs=True).manifest
        assert m["running_jobs"] == [{"job_id": "j1", "folder_id": 1, "status": "running", "kind": "index"}]

    def test_a_stale_job_does_not_block(self, tmp_path, storage, fake):
        fake.jobs = [{"job_id": "j1", "folder_id": 1, "status": "running", "kind": "index",
                      "last_updated_at": (NOW - timedelta(hours=2)).isoformat()}]
        assert _backup(tmp_path, storage).manifest["running_jobs"] == []

    def test_not_a_folio_database(self, tmp_path, storage, fake):
        fake.tables = set()
        with pytest.raises(S.BackupRefused, match="no Folio tables"):
            _backup(tmp_path, storage)

    def test_existing_output_is_never_overwritten(self, tmp_path, storage, fake):
        out = tmp_path / "x.tar"
        out.write_text("keep me")
        with pytest.raises(S.BackupRefused, match="already exists"):
            S.create_backup(db_url="postgresql://h/db", storage_root=storage, out_path=out, rag_conf=RAG)
        assert out.read_text() == "keep me"

    def test_needs_a_destination(self, storage):
        with pytest.raises(ValueError):
            S.create_backup(db_url="postgresql://h/db", storage_root=storage)

    def test_a_failed_dump_leaves_no_archive(self, tmp_path, storage, fake, monkeypatch):
        def boom(*a, **k):
            raise S.pg.PgToolError("pg_dump failed: nope")
        monkeypatch.setattr(S.pg, "dump", boom)
        with pytest.raises(S.pg.PgToolError):
            _backup(tmp_path, storage)
        assert [p.name for p in (tmp_path / "b").iterdir()] == [".folio-backup.lock"]

    def test_one_backup_at_a_time(self, tmp_path, storage, fake):
        import fcntl
        import os
        d = tmp_path / "b"
        d.mkdir()
        fd = os.open(d / ".folio-backup.lock", os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            # flock is per open file description: a second open in this process conflicts too
            result = {}

            def run():
                try:
                    _backup(tmp_path, storage)
                except S.BackupRefused as e:
                    result["e"] = e
            t = threading.Thread(target=run)
            t.start()
            t.join()
            assert "already being written" in str(result["e"])
        finally:
            os.close(fd)

    def test_backup_dir_inside_storage_is_not_archived(self, tmp_path, storage, fake):
        r = S.create_backup(db_url="postgresql://h/db", storage_root=storage, out_dir=storage / "backups",
                            rag_conf=RAG, now=NOW)
        assert not any("backups" in k for k in r.manifest["checksums"])

    def test_unarchivable_and_special_files_are_skipped(self, tmp_path, storage, fake):
        (storage / "sub_u" / "docs" / "bad\\name").write_text("x")
        (storage / "sub_u" / "docs" / "link").symlink_to(storage / "sub_u" / "docs" / "f1")
        r = _backup(tmp_path, storage)
        assert any("cannot be archived safely" in w for w in r.warnings)
        assert any("not a regular file" in w for w in r.warnings)


class TestHelpers:
    def test_active_jobs_without_the_table(self):
        assert S.active_jobs(FakeDB(tables=())) == []

    def test_unparseable_job_time_counts_as_live(self):
        db = FakeDB(jobs=[{"job_id": "j", "folder_id": 2, "status": "pending", "kind": "index",
                           "last_updated_at": "garbage"}])
        assert len(S.active_jobs(db, NOW)) == 1

    @pytest.mark.parametrize("path,expected", [
        ("storage/sub_u/a/b", "storage/sub_u/a/b"),
        ("other/x", None),
        ("storage", None),
    ])
    def test_db_paths_map_to_archive_names(self, tmp_path, path, expected):
        assert S._arcname_for_db_path(path, tmp_path / "storage") == expected

    def test_absolute_db_paths(self, tmp_path):
        root = tmp_path / "storage"
        assert S._arcname_for_db_path(str(root / "a" / "b"), root) == "storage/a/b"
        assert S._arcname_for_db_path("/elsewhere/b", root) is None

    def test_list_and_prune(self, tmp_path):
        for h in range(5):
            (tmp_path / A.archive_name(NOW + timedelta(hours=h))).write_text("x")
        (tmp_path / "manual.tar").write_text("x")
        (tmp_path / (A.archive_name(NOW) + ".partial")).write_text("x")
        assert len(S.list_archives(tmp_path)) == 5
        gone = S.prune(tmp_path, 2)
        assert [p.name for p in gone] == [A.archive_name(NOW + timedelta(hours=h)) for h in range(3)]
        left = sorted(p.name for p in tmp_path.iterdir())
        assert "manual.tar" in left and len(S.list_archives(tmp_path)) == 2
        assert S.prune(tmp_path, 5) == []
        with pytest.raises(ValueError):
            S.prune(tmp_path, 0)
        assert S.list_archives(tmp_path / "nope") == []

    def test_backup_dir_relative_and_absolute(self, tmp_path):
        from src.config.model import BackupConfig
        from src.storage.file_storage import FileStorage
        assert S.backup_dir(BackupConfig(dir=str(tmp_path))) == tmp_path
        assert S.backup_dir(BackupConfig(dir="bk")) == FileStorage.STORAGE_FILE_ROOT / "bk"

    def test_settings_fall_back_to_defaults(self, monkeypatch):
        from src.config.config_manager import Config
        monkeypatch.setattr(Config, "get_database_config", classmethod(lambda cls: None))
        assert S.backup_settings().keep == 7

    def test_code_revisions_are_the_migration_chain(self):
        revs = S.code_revisions()
        assert len(revs) > 5 and revs.index("20261113e0b001") > revs.index("20261102c0de01")

    def test_summary(self):
        s = json.loads(S.manifest_summary({"folders": [{"chunks": 3}, {"chunks": 4}],
                                           "embedding": {"model": "e5", "dimension": 8}}))
        assert s["chunks"] == 7 and s["embedding"] == "e5 (8)"


def _m(**over):
    m = {"format": A.FORMAT, "format_version": 1, "created_at": NOW.isoformat(), "folio_version": "1.4.2",
         "alembic_revision": "rev2", "embedding": {"model": "e5", "dimension": 8, "from_runtime_settings": []},
         "segmenter": {"name": "bigram", "from_runtime_settings": False},
         "indexed_embeddings": [{"model": "e5", "dimension": 8, "files": 2}], "vector_dimensions": [8],
         "folders": [], "running_jobs": [], "storage": {"missing_count": 0}}
    m.update(over)
    return m


def _compat(m, rag=RAG, version="1.4.2"):
    return S.compatibility(m, revisions=["rev1", "rev2"], version=version, rag_conf=rag)


class TestCompatibility:
    def test_same_setup(self):
        assert _compat(_m()) == ([], [])

    def test_older_backup_is_fine(self):
        assert _compat(_m(folio_version="1.3.0", alembic_revision="rev1")) == ([], [])

    def test_unknown_revision(self):
        blocking, _ = _compat(_m(alembic_revision="rev9", folio_version="9.0.0"))
        assert "does not know" in blocking[0]

    def test_newer_version(self):
        blocking, _ = _compat(_m(folio_version="1.10.0"))
        assert "newer than this Folio" in blocking[0]

    def test_other_model(self):
        rag = SimpleNamespace(embedding=SimpleNamespace(model="bge-m3", dimension=8), profile="cloud",
                              retrieval=SimpleNamespace(segmenter="bigram"))
        blocking, _ = _compat(_m(), rag)
        assert blocking == ["the archive's vectors were made with embedding model e5; this Folio uses bge-m3"]

    def test_other_dimension(self):
        rag = SimpleNamespace(embedding=SimpleNamespace(model="e5", dimension=1024), profile="cloud",
                              retrieval=SimpleNamespace(segmenter="bigram"))
        blocking, _ = _compat(_m(), rag)
        assert "dimension 8" in blocking[0]

    def test_a_runtime_override_comes_back_with_the_database(self):
        rag = SimpleNamespace(embedding=SimpleNamespace(model="bge-m3", dimension=1024), profile="cloud",
                              retrieval=SimpleNamespace(segmenter="bigram"))
        m = _m(embedding={"model": "e5", "dimension": 8, "from_runtime_settings": ["model", "dimension"]})
        assert _compat(m, rag)[0] == []

    def test_mixed_models_in_the_data(self):
        m = _m(indexed_embeddings=[{"model": "e5", "dimension": 8}, {"model": "old", "dimension": 8}])
        assert "old" in _compat(m)[0][0]

    def test_empty_instance_falls_back_to_the_recorded_model(self):
        m = _m(indexed_embeddings=[], vector_dimensions=[], embedding={"model": "x", "dimension": 4,
                                                                       "from_runtime_settings": []})
        blocking, _ = _compat(m)
        assert len(blocking) == 2

    def test_warnings(self):
        m = _m(segmenter={"name": "ckip", "from_runtime_settings": False},
               running_jobs=[{"job_id": "j"}], storage={"missing_count": 2})
        blocking, warnings = _compat(m)
        assert blocking == [] and len(warnings) == 3
        assert "Rebuild FTS" in warnings[0]


@pytest.fixture
def restore_env(tmp_path, storage, fake, monkeypatch):
    """A real archive plus a scripted target database."""
    archive = _backup(tmp_path, storage).path
    state = {"tables": [], "connections": 0}
    calls = {}
    monkeypatch.setattr(S, "_target_state", lambda url: dict(state))
    monkeypatch.setattr(S, "_ensure_vector", lambda url: calls.setdefault("vector", True))
    monkeypatch.setattr(S, "_drop_tables", lambda url, tables: calls.setdefault("dropped", tables))
    monkeypatch.setattr(S, "_restored_counts", lambda url: [{"id": 1, "name": "docs", "chunks": 5}])
    import src.utils.db_bootstrap as boot
    monkeypatch.setattr(boot, "_ensure_database_exists", lambda url, logger: calls.setdefault("db", True))

    monkeypatch.setattr(S.pg, "server_info", lambda url: (15, {"statement_timeout"}))
    monkeypatch.setattr(S.pg, "restore_tools", lambda major, bin_dir=None: calls.setdefault("tools", major))

    def fake_restore(url, dump_file, list_file, bin_dir=None, server=None):
        calls["restored"] = Path(dump_file).read_bytes()
        calls["server"] = server
        return []
    monkeypatch.setattr(S.pg, "restore", fake_restore)
    target = tmp_path / "target" / "storage"
    return SimpleNamespace(archive=archive, state=state, calls=calls, target=target)


def _restore(env, **kw):
    return S.restore_backup(env.archive, db_url="postgresql://u:p@h/db", storage_root=env.target,
                            rag_conf=RAG, revisions=["rev1", "rev2"], version="1.4.2", now=NOW, **kw)


class TestRestore:
    def test_into_an_empty_instance(self, restore_env):
        r = _restore(restore_env)
        assert r.restored and r.moved_aside is None and r.warnings[-1].startswith("1 file(s)")
        assert restore_env.calls["restored"] == b"PGDMP" and "dropped" not in restore_env.calls
        assert (restore_env.target / "sub_u/docs/f1").read_bytes() == b"file one"
        assert not list(restore_env.target.glob(".folio-work/restore-*"))

    def test_check_only_changes_nothing(self, restore_env):
        r = _restore(restore_env, check_only=True)
        assert not r.restored and restore_env.calls == {} and not restore_env.target.exists()

    def test_a_database_with_tables_needs_force(self, restore_env):
        restore_env.state["tables"] = ["Folders"]
        with pytest.raises(S.BackupRefused, match="already has 1 table"):
            _restore(restore_env)
        assert "restored" not in restore_env.calls
        r = _restore(restore_env, force=True)
        assert restore_env.calls["dropped"] == ["Folders"] and r.restored

    def test_the_server_info_is_handed_to_pg_restore(self, restore_env):
        _restore(restore_env)
        assert restore_env.calls["tools"] == 15
        assert restore_env.calls["server"] == (15, {"statement_timeout"})

    def test_a_missing_client_tool_stops_before_anything_is_dropped(self, restore_env, monkeypatch):
        def no_psql(major, bin_dir=None):
            raise S.pg.PgToolError("psql is not installed or not on PATH")
        monkeypatch.setattr(S.pg, "restore_tools", no_psql)
        restore_env.state["tables"] = ["Folders"]
        (restore_env.target / "sub_x").mkdir(parents=True)
        with pytest.raises(S.pg.PgToolError, match="psql"):
            _restore(restore_env, force=True)
        assert "dropped" not in restore_env.calls and "restored" not in restore_env.calls
        assert (restore_env.target / "sub_x").is_dir()
        assert not list(restore_env.target.glob(".folio-work/restore-*"))

    def test_a_used_storage_needs_force_and_is_moved_aside(self, restore_env):
        (restore_env.target / "sub_x").mkdir(parents=True)
        (restore_env.target / "sub_x" / "old").write_text("old")
        with pytest.raises(S.BackupRefused, match="storage/ already holds"):
            _restore(restore_env)
        r = _restore(restore_env, force=True)
        assert r.moved_aside == restore_env.target / ".folio-work" / "before-restore-20261114T040000Z"
        assert (r.moved_aside / "sub_x" / "old").read_text() == "old"
        assert (restore_env.target / "sub_u/docs/f1").exists() and not (restore_env.target / "sub_x").exists()

    def test_housekeeping_files_alone_do_not_need_force(self, restore_env):
        restore_env.target.mkdir(parents=True)
        (restore_env.target / ".console_session_secret").write_text("new instance")
        (restore_env.target / ".folio-work" / "console").mkdir(parents=True)  # CEX-36
        r = _restore(restore_env)
        assert r.moved_aside is not None
        assert (restore_env.target / ".console_session_secret").read_text() == "s"

    def test_open_connections_refuse_even_with_force(self, restore_env):
        restore_env.state["connections"] = 3
        with pytest.raises(S.BackupRefused, match="stop Folio"):
            _restore(restore_env, force=True)

    def test_a_mismatch_needs_allow_mismatch(self, restore_env):
        rag = SimpleNamespace(embedding=SimpleNamespace(model="bge-m3", dimension=8), profile="cloud",
                              retrieval=SimpleNamespace(segmenter="bigram"))
        with pytest.raises(S.BackupRefused, match="--allow-mismatch"):
            S.restore_backup(restore_env.archive, db_url="postgresql://h/db", storage_root=restore_env.target,
                             rag_conf=rag, revisions=["rev2"], version="1.4.2")
        r = S.restore_backup(restore_env.archive, db_url="postgresql://h/db", storage_root=restore_env.target,
                             rag_conf=rag, revisions=["rev2"], version="1.4.2", allow_mismatch=True, check_only=True)
        assert r.warnings[0].startswith("ignored: ")

    def test_a_damaged_archive_never_reaches_the_database(self, restore_env):
        data = bytearray(restore_env.archive.read_bytes())
        i = data.find(b"file one")
        data[i] ^= 1
        restore_env.archive.write_bytes(bytes(data))
        with pytest.raises(A.ArchiveError, match="checksum mismatch"):
            _restore(restore_env)
        assert "restored" not in restore_env.calls
        assert [p.name for p in restore_env.target.iterdir()] == [".folio-work"]
        assert not list(restore_env.target.glob(".folio-work/restore-*"))

    def test_chunk_counts_are_compared(self, restore_env, monkeypatch):
        monkeypatch.setattr(S, "_restored_counts", lambda url: [{"id": 1, "name": "docs", "chunks": 4}])
        r = _restore(restore_env)
        assert any("has 4 chunks, the manifest says 5" in w for w in r.warnings)


class TestQuietLogger:
    def test_only_warnings_reach_the_printer(self):
        seen = []
        q = S._QuietLogger(seen.append)
        q.info("a")
        q.warning("b")
        q.error("c")
        assert seen == ["b", "c"]



class TestRestoreInPlace:
    """restore swaps storage/'s contents inside storage/ (Docker: a volume's mount point whose parent is
    not writable)."""

    def test_cex41_storage_itself_is_never_renamed(self, restore_env, monkeypatch):
        """CEX-41 -- storage/ behaves like a mount point: renaming it fails (EBUSY). The restore
        still works and storage/ is the same directory afterwards."""
        import errno
        target = restore_env.target
        (target / "sub_x").mkdir(parents=True)
        (target / "sub_x" / "old").write_text("old")
        inode = target.stat().st_ino
        real = S.os.replace

        def replace(src, dst):
            if Path(src) == target or Path(dst) == target:
                raise OSError(errno.EBUSY, "Device or resource busy", str(src))
            return real(src, dst)
        monkeypatch.setattr(S.os, "replace", replace)
        r = _restore(restore_env, force=True)
        assert r.restored and target.stat().st_ino == inode
        assert (target / "sub_u/docs/f1").read_bytes() == b"file one" and not (target / "sub_x").exists()
        assert (r.moved_aside / "sub_x" / "old").read_text() == "old"
        assert sorted(p.name for p in target.parent.iterdir()) == ["storage"]  # nothing beside it

    def test_cex42_the_archive_is_unpacked_inside_storage(self, restore_env, monkeypatch):
        """CEX-42 -- staging under storage/.folio-work (writable, same filesystem), removed afterwards."""
        seen = {}
        real = S.pg.restore

        def spy(url, dump_file, list_file, bin_dir=None, server=None):
            seen["dump"] = Path(dump_file)
            return real(url, dump_file, list_file, bin_dir=bin_dir, server=server)
        monkeypatch.setattr(S.pg, "restore", spy)
        _restore(restore_env)
        assert seen["dump"].parent == restore_env.target / ".folio-work" / "restore-20261114T040000Z"
        assert not seen["dump"].parent.exists()

    def test_cex43_a_failed_swap_puts_storage_back(self, restore_env, monkeypatch):
        """CEX-43 -- a move that fails half way: the previous contents are back, nothing restored stays."""
        target = restore_env.target
        (target / "sub_x").mkdir(parents=True)
        (target / "sub_x" / "old").write_text("old")
        (target / ".console_session_secret").write_text("mine")
        real = S.os.replace

        def replace(src, dst):
            if Path(dst) == target / "sub_u":
                raise OSError(28, "No space left on device")
            return real(src, dst)
        monkeypatch.setattr(S.os, "replace", replace)
        with pytest.raises(OSError, match="No space"):
            _restore(restore_env, force=True)
        assert (target / "sub_x" / "old").read_text() == "old"
        assert (target / ".console_session_secret").read_text() == "mine"
        assert sorted(p.name for p in target.iterdir()) == [".console_session_secret", ".folio-work", "sub_x"]
        assert list((target / ".folio-work").iterdir()) == []

    def test_cex44_folio_work_stays_in_place(self, restore_env):
        """CEX-44 -- the console's pending exports in storage/.folio-work survive a restore."""
        pending = restore_env.target / ".folio-work" / "console" / "export-1"
        pending.mkdir(parents=True)
        (pending / "a.tar").write_bytes(b"tar")
        r = _restore(restore_env)
        assert (pending / "a.tar").read_bytes() == b"tar" and r.moved_aside is None
        assert (restore_env.target / "sub_u/docs/f1").exists()

    def test_cex45_a_missing_storage_directory_is_created(self, restore_env):
        """CEX-45 -- piece missing: no storage/ at all (a fresh volume path); the restore creates it."""
        assert not restore_env.target.exists()
        assert _restore(restore_env).restored
        assert (restore_env.target / "sub_u/docs/f1").read_bytes() == b"file one"


def test_cex46_an_unwritable_backup_directory_is_a_clear_refusal(tmp_path, storage, fake, monkeypatch):
    """CEX-46 -- piece missing: no writable backup directory (Docker without the folio-backups volume
    and a read-only /app); a refusal that says where to point database.backup.dir, not a traceback."""
    real = S.os.open

    def no_write(path, *a, **k):
        if str(path).endswith(S._LOCK_NAME):
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *a, **k)
    monkeypatch.setattr(S.os, "open", no_write)
    with pytest.raises(S.BackupRefused, match="cannot write to the backup directory.*folio-backups"):
        _backup(tmp_path, storage)
