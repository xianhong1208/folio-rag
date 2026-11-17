"""One-folder export / import: names, owners, compatibility, metadata rewriting, the archive
layout and the CLI. The database round trip is in tests/integration/test_folder_export_db.py."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.backup import archive as A
from src.backup import cli
from src.backup import folder as F
from src.backup import service as S
from src.config.model import BackupConfig

OLD, NEW = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 11, 16, 3, 4, 5, tzinfo=timezone.utc)
TARGET = {"embedding": {"model": "e5", "dimension": 8}, "segmenter": {"name": "bigram"}}


def _manifest(**over):
    m = {"format": F.FORMAT, "format_version": 1, "created_at": "x", "folio_version": "1.4.2",
         "alembic_revision": "rev2", "embedding": {"model": "e5", "dimension": 8},
         "indexed_embeddings": [{"model": "e5", "dimension": 8, "files": 1}],
         "vector_table": {"exists": True, "dimension": 8, "segmenter": "bigram"},
         "folder": {"name": "docs", "files": 1, "chunks": 4, "missing_files": 0},
         "checksums": {F.FOLDER_JSON: "0" * 64, F.CHUNKS_NAME: "1" * 64}}
    m.update(over)
    return m


def _compat(m, files_only=False, target=TARGET, version="1.4.2"):
    return F.compatibility(m, target=target, revisions=["rev1", "rev2"], version=version, files_only=files_only)


class TestNamesAndOwners:
    def test_export_name_is_safe_and_dated(self):
        assert F.export_name("my docs/../x", NOW) == "folio-folder-my_docs_.._x-20261116T030405Z.tar"
        assert F.export_name("茶葉", NOW) == "folio-folder-茶葉-20261116T030405Z.tar"
        assert F.export_name("...", NOW).startswith("folio-folder-folder-")

    @pytest.mark.parametrize("value, key", [("console", "console"), ("abc-123", "sub:abc-123"),
                                            ("sub:abc-123", "sub:abc-123"), ("  u1 ", "sub:u1")])
    def test_owner_keys(self, value, key):
        assert F.resolve_owner(value) == key

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_owner_is_refused(self, value):
        with pytest.raises(F.FolderRefused, match="an owner is required"):
            F.resolve_owner(value)

    def test_an_empty_sub_is_refused(self):
        with pytest.raises(F.FolderRefused, match="not a usable owner"):
            F.resolve_owner("sub:")

    def test_a_long_sub_becomes_a_digest_key(self):
        assert F.resolve_owner("x" * 300).startswith("sub:sha256:")


class TestCompatibility:
    def test_same_model_imports(self):
        assert _compat(_manifest()) == ([], [])

    def test_another_model_is_refused_with_the_way_out(self):
        blocking, _ = _compat(_manifest(indexed_embeddings=[{"model": "bge", "dimension": 8}]))
        assert len(blocking) == 1 and "bge" in blocking[0] and "--files-only" in blocking[0]

    def test_another_dimension_is_refused(self):
        blocking, _ = _compat(_manifest(vector_table={"exists": True, "dimension": 4}))
        assert "dimension 4" in blocking[0]

    def test_a_model_whose_dimension_differs_is_refused(self):
        blocking, _ = _compat(_manifest(indexed_embeddings=[{"model": "e5", "dimension": 4}]))
        assert blocking

    def test_files_only_lifts_the_model_check(self):
        m = _manifest(indexed_embeddings=[{"model": "bge", "dimension": 4}])
        blocking, warnings = _compat(m, files_only=True)
        assert blocking == [] and "index the folder" in warnings[0]

    def test_a_folder_without_chunks_needs_no_model(self):
        m = _manifest(indexed_embeddings=[], folder={"name": "d", "files": 1, "chunks": 0})
        assert _compat(m, target={"embedding": {"model": "x", "dimension": 2}}) == ([], [])

    def test_newer_folio_is_refused(self):
        assert "newer Folio" in _compat(_manifest(alembic_revision="rev9"))[0][0]
        assert "newer than this" in _compat(_manifest(folio_version="9.0.0"))[0][0]
        assert _compat(_manifest(folio_version="1.4.1")) == ([], [])

    def test_another_segmenter_warns(self):
        _, warnings = _compat(_manifest(), target={**TARGET, "segmenter": {"name": "jieba"}})
        assert "Rebuild FTS" in warnings[0]

    def test_missing_files_warn(self):
        _, warnings = _compat(_manifest(folder={"name": "d", "files": 2, "chunks": 0, "missing_files": 1}))
        assert "already missing" in warnings[0]


def _chunk(fid=OLD, **extra):
    meta = {"file_id": fid, "file_path": f"storage/sub_a/docs/{fid}", "folder_id": 3, "folder_name": "docs",
            "ref_doc_id": f"file_{fid}", "doc_id": "None", "parent_node_id": "p1", **extra}
    node = {"id_": "n1", "metadata": dict(meta),
            "relationships": {"1": {"node_id": fid}, "4": [{"node_id": "p1"}]}}
    return json.dumps({**meta, "_node_content": json.dumps(node)})


class TestRemap:
    def test_file_folder_and_path_are_rewritten_everywhere(self):
        raw, fid = F.remap_chunk_metadata(_chunk(), {OLD: NEW}, {OLD: f"storage/sub_b/docs-2/{NEW}"}, 9, "docs-2")
        assert fid == NEW and OLD not in raw
        meta = json.loads(raw)
        inner = json.loads(meta["_node_content"])
        for d in (meta, inner["metadata"]):
            assert d["file_id"] == NEW and d["file_path"] == f"storage/sub_b/docs-2/{NEW}"
            assert d["folder_id"] == 9 and d["folder_name"] == "docs-2" and d["ref_doc_id"] == f"file_{NEW}"
            assert d["doc_id"] == "None" and d["parent_node_id"] == "p1"
        assert inner["relationships"] == {"1": {"node_id": NEW}, "4": [{"node_id": "p1"}]}

    def test_a_chunk_of_an_unknown_file_is_left_out(self):
        assert F.remap_chunk_metadata(_chunk(fid="33333333-3333-4333-8333-333333333333"), {OLD: NEW}, {}, 1,
                                      "d") == (None, None)
        assert F.remap_chunk_metadata("", {OLD: NEW}, {}, 1, "d") == (None, None)
        assert F.remap_chunk_metadata("[1]", {OLD: NEW}, {}, 1, "d") == (None, None)

    def test_unreadable_node_content_is_kept_as_is(self):
        raw = json.dumps({"file_id": OLD, "_node_content": "{not json"})
        out, _ = F.remap_chunk_metadata(raw, {OLD: NEW}, {}, 1, "d")
        assert json.loads(out) == {"file_id": NEW, "_node_content": "{not json"}


class TestArchiveParts:
    def test_folder_doc_validation(self, tmp_path):
        p = tmp_path / "folder.json"
        good = {"folder": {"name": "d"}, "files": [{"id": OLD, "file_name": "a.md"}], "file_indices": []}
        p.write_text(json.dumps(good))
        assert F._load_folder_doc(p) == good
        for bad in ({"folder": {}, "files": [{"id": "../x", "file_name": "a"}], "file_indices": []},
                    {"folder": {}, "files": [], "file_indices": ["x"]},
                    {"files": []}, [1]):
            p.write_text(json.dumps(bad))
            with pytest.raises(A.ArchiveError):
                F._load_folder_doc(p)
        p.write_text("{")
        with pytest.raises(A.ArchiveError, match="not valid JSON"):
            F._load_folder_doc(p)

    def test_chunks_reader(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text('{"text": "a"}\n\n{"text": "b"}\n')
        assert [c["text"] for c in F._read_chunks(p)] == ["a", "b"]
        p.write_text('{"text": 1}\n')
        with pytest.raises(A.ArchiveError, match="line 1 is not a chunk"):
            list(F._read_chunks(p))
        p.write_text('nope\n')
        with pytest.raises(A.ArchiveError, match="not valid JSON"):
            list(F._read_chunks(p))

    def test_the_layout_takes_only_its_own_entries(self):
        assert A.safe_member_path(f"files/{OLD}", F.FOLDER_EXPORT) == f"files/{OLD}"
        assert A.safe_member_path(F.CHUNKS_NAME, F.FOLDER_EXPORT) == F.CHUNKS_NAME
        for name in ("database.dump", "storage/x", "files", "other.json"):
            with pytest.raises(A.ArchiveError):
                A.safe_member_path(name, F.FOLDER_EXPORT)
        with pytest.raises(A.ArchiveError):
            A.safe_member_path("files/x", A.BACKUP)

    def test_a_backup_is_not_a_folder_export_and_vice_versa(self, tmp_path):
        w = A.ArchiveWriter(tmp_path / "f.tar", F.FOLDER_EXPORT)
        for name in (F.FOLDER_JSON, F.CHUNKS_NAME):
            (tmp_path / name).write_text("{}")
            w.add_file(name, tmp_path / name)
        w.add_manifest(_manifest(checksums=dict(w.checksums)))
        path = w.commit()
        m = A.read_manifest(path, F.FOLDER_EXPORT)
        assert A.verify_and_extract(path, m, layout=F.FOLDER_EXPORT)["entries"] == 2
        with pytest.raises(A.ArchiveError, match="not a Folio backup manifest"):
            A.read_manifest(path)
        with pytest.raises(A.ArchiveError, match="lists no checksums"):
            A.validate_manifest(_manifest(checksums={F.FOLDER_JSON: "0" * 64}), F.FOLDER_EXPORT)
        with pytest.raises(A.ArchiveError, match="format 2"):
            A.validate_manifest(_manifest(format_version=2), F.FOLDER_EXPORT)

    def test_manifest_summary(self):
        s = json.loads(F.manifest_summary(_manifest()))
        assert s["folder"] == "docs" and s["embedding"] == "e5 (8)" and s["chunks"] == 4


class TestValues:
    def test_jsonable(self):
        import uuid
        from decimal import Decimal
        assert F._jsonable(NOW) == NOW.isoformat()
        assert F._jsonable(uuid.UUID(OLD)) == OLD
        assert F._jsonable(Decimal("3")) == 3 and F._jsonable(Decimal("1.5")) == 1.5
        assert F._jsonable([1]) == [1]

    def test_stored_file_stays_inside_storage(self, tmp_path):
        root = tmp_path / "storage"
        (root / "a").mkdir(parents=True)
        assert F._stored_file("storage/a/f", root) == (root / "a" / "f").resolve()
        assert F._stored_file("storage/../etc/passwd", root) is None
        assert F._stored_file("/etc/passwd", root) is None


class TestImportGuards:
    def test_a_bad_new_name_is_refused_before_the_database(self, tmp_path, monkeypatch):
        monkeypatch.setattr(F, "read_manifest", lambda p, layout: _manifest())
        monkeypatch.setattr(S, "_engine", lambda url: pytest.fail("the database was touched"))
        for name in ("../x", "a/b", "x" * 300):
            with pytest.raises(F.FolderRefused):
                F.import_folder(tmp_path / "a.tar", db_url="postgresql://u@h/d", storage_root=tmp_path,
                                owner="console", name=name)


# ---------------------------------------------------------------- CLI

@pytest.fixture
def env(monkeypatch, tmp_path):
    from src.config.config_manager import Config
    conf = BackupConfig(dir=str(tmp_path))
    monkeypatch.setattr(cli, "_load_config", lambda d: SimpleNamespace(rag="RAG"))
    monkeypatch.setattr(Config, "get_database_config",
                        classmethod(lambda cls: SimpleNamespace(url="postgresql://u:p@h/db", port=None, backup=conf)))
    monkeypatch.setattr(S, "default_storage_root", lambda: tmp_path / "storage")
    return SimpleNamespace(tmp=tmp_path)


class TestCli:
    def test_export_by_id(self, env, monkeypatch, capsys):
        seen = {}

        def fake(**kw):
            seen.update(kw)
            return F.ExportResult(path=env.tmp / "x.tar", size=2048, manifest=_manifest(), warnings=["w1"])
        monkeypatch.setattr(F, "export_folder", fake)
        assert cli.main(["export-folder", "--id", "3"]) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "2,048 bytes" in out and "'docs'" in out and "warning: w1" in out
        assert seen["folder_id"] == 3 and seen["out_dir"] == env.tmp and seen["out_path"] is None

    def test_export_by_name_needs_an_owner(self, env, capsys):
        assert cli.main(["export-folder", "--name", "docs"]) == cli.EXIT_REFUSED
        assert "--owner" in capsys.readouterr().out

    def test_export_refusal_exit_code(self, env, monkeypatch, capsys):
        def refuse(**kw):
            raise F.FolderRefused("no such folder")
        monkeypatch.setattr(F, "export_folder", refuse)
        assert cli.main(["export-folder", "--name", "d", "--owner", "u", "--output", "o.tar"]) == cli.EXIT_REFUSED
        assert "refused: no such folder" in capsys.readouterr().out

    def test_import(self, env, monkeypatch, capsys):
        seen = {}

        def fake(archive, **kw):
            seen.update(kw, archive=archive)
            return F.ImportResult(manifest=_manifest(), imported=True, folder_id=7, name="docs-2", owner="sub:u",
                                  files=1, chunks=4, indexed=True, warnings=["renamed"])
        monkeypatch.setattr(F, "import_folder", fake)
        assert cli.main(["import-folder", "a.tar", "--owner", "u"]) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "imported folder 7 'docs-2' for sub:u: 1 file(s), 4 chunks" in out and "warning: renamed" in out
        assert seen["archive"] == Path("a.tar") and seen["owner"] == "u" and not seen["files_only"]

    def test_import_files_only_and_check(self, env, monkeypatch, capsys):
        monkeypatch.setattr(F, "import_folder", lambda archive, **kw: F.ImportResult(
            manifest=_manifest(), imported=not kw["check_only"], folder_id=7, name="d", owner="console",
            files=1, indexed=not kw["files_only"]))
        assert cli.main(["import-folder", "a.tar", "--owner", "console", "--files-only"]) == 0
        assert "not indexed yet" in capsys.readouterr().out
        assert cli.main(["import-folder", "a.tar", "--owner", "console", "--check"]) == 0
        assert "can be imported" in capsys.readouterr().out

    def test_import_needs_an_owner(self, env):
        with pytest.raises(SystemExit):
            cli.main(["import-folder", "a.tar"])


class _Res(list):
    def scalar(self):
        return self[0] if self else None


class _FakeConn:
    """Answers the few queries import_folder makes before it changes anything."""

    def __init__(self, tables=True):
        self.tables = tables

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "to_regclass" in sql:
            return _Res([self.tables])
        if "RuntimeSettings" in sql:
            return _Res([])
        raise AssertionError(f"unexpected SQL before the checks passed: {sql}")

    def connect(self):
        return self

    def dispose(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _export(tmp_path, **manifest):
    w = A.ArchiveWriter(tmp_path / "f.tar", F.FOLDER_EXPORT)
    for name in (F.FOLDER_JSON, F.CHUNKS_NAME):
        (tmp_path / name).write_text("{}")
        w.add_file(name, tmp_path / name)
    w.add_manifest(_manifest(checksums=dict(w.checksums), **manifest))
    return w.commit()


class TestImportChecks:
    RAG = SimpleNamespace(embedding=SimpleNamespace(model="e5", dimension=8), retrieval=None)

    def _import(self, path, tmp_path, **kw):
        return F.import_folder(path, db_url="postgresql://u@h/d", storage_root=tmp_path / "storage",
                               owner="console", rag_conf=self.RAG, revisions=["rev1", "rev2"], version="1.4.2",
                               **kw)

    def test_check_only_verifies_and_changes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "_engine", lambda url: _FakeConn())
        path = _export(tmp_path)
        r = self._import(path, tmp_path, check_only=True)
        assert not r.imported and r.owner == "console" and not (tmp_path / "storage").exists()

    def test_a_mismatch_is_refused_before_anything_changes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "_engine", lambda url: _FakeConn())
        path = _export(tmp_path, indexed_embeddings=[{"model": "bge", "dimension": 8}])
        with pytest.raises(F.FolderRefused, match="--files-only"):
            self._import(path, tmp_path)
        assert not (tmp_path / "storage").exists()

    def test_a_database_without_folio_tables_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "_engine", lambda url: _FakeConn(tables=False))
        with pytest.raises(F.FolderRefused, match="start Folio once"):
            self._import(_export(tmp_path), tmp_path)

    def test_a_damaged_archive_is_refused_by_check(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "_engine", lambda url: _FakeConn())
        path = _export(tmp_path)
        data = bytearray(path.read_bytes())
        i = data.index(b"{}")
        data[i:i + 2] = b"[]"
        path.write_bytes(bytes(data))
        with pytest.raises(A.ArchiveError, match="checksum mismatch"):
            self._import(path, tmp_path, check_only=True)


class TestConsoleHooks:
    """What the console (src/api/router/admin_folder_archive.py) relies on."""

    def test_cex28_problems_carry_codes_without_the_cli_hint(self):
        """CEX-28"""
        m = _manifest(indexed_embeddings=[{"model": "bge", "dimension": 8}], alembic_revision="rev9",
                      folio_version="9.0.0")
        problems, _ = F.check_compatibility(m, target=TARGET, revisions=["rev1"], version="1.4.2",
                                            files_only=False)
        assert [c for c, _ in problems] == ["newer_version", "embedding_mismatch"]
        assert not any("--files-only" in msg for _, msg in problems)
        assert F.check_compatibility(_manifest(), target=TARGET, revisions=["rev1", "rev2"], version="1.4.2",
                                     files_only=False) == ([], [])

    def test_cex29_refused_import_says_which_problem(self, tmp_path, monkeypatch):
        """CEX-29"""
        monkeypatch.setattr(S, "_engine", lambda url: _FakeConn())
        path = _export(tmp_path, indexed_embeddings=[{"model": "bge", "dimension": 8}])
        with pytest.raises(F.FolderRefused) as e:
            TestImportChecks()._import(path, tmp_path)
        assert e.value.code == "embedding_mismatch" and "--files-only" in str(e.value)
        assert len(e.value.reasons) == 1 and "--files-only" not in e.value.reasons[0]
        (tmp_path / "n").mkdir()
        path2 = _export(tmp_path / "n", folio_version="9.0.0")
        with pytest.raises(F.FolderRefused) as e:
            TestImportChecks()._import(path2, tmp_path)
        assert e.value.code == "newer_version"

    def test_cex30_owner_key_is_used_as_given(self, tmp_path, monkeypatch):
        """CEX-30 -- the console passes its session's owner key; it is not re-resolved."""
        monkeypatch.setattr(S, "_engine", lambda url: _FakeConn())
        path = _export(tmp_path)
        r = F.import_folder(path, db_url="postgresql://u@h/d", storage_root=tmp_path / "storage",
                            owner_key="sub:sha256:" + "a" * 64, rag_conf=TestImportChecks.RAG,
                            revisions=["rev1", "rev2"], version="1.4.2", check_only=True)
        assert r.owner == "sub:sha256:" + "a" * 64
        with pytest.raises(F.FolderRefused, match="not a usable owner"):
            F.import_folder(path, db_url="postgresql://u@h/d", storage_root=tmp_path / "storage",
                            owner_key="../x", rag_conf=TestImportChecks.RAG, revisions=["rev1", "rev2"],
                            version="1.4.2", check_only=True)

    def test_cex31_refused_defaults(self):
        """CEX-31"""
        e = F.FolderRefused("no tables")
        assert e.code == "refused" and e.reasons == ["no tables"] and str(e) == "no tables"
