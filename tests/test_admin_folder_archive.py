"""Console folder export / import (src/api/router/admin_folder_archive.py).

Pins down: only your own folders are exported (the owner is re-checked inside the
export); an export is downloaded once, and only by the identity that made it; an
imported folder always belongs to the session (never to anything in the request);
the upload is streamed and refused past the size limit; the CLI's refusals map to
409 / 400 with codes the console understands; temporary files are removed on every
path; cookie writes need the CSRF header. The export / import themselves
(src/backup/folder.py) are mocked here and covered by their own tests.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router import admin_folder_archive as AR
from src.api.router.admin_settings import router
from src.backup import folder as F
from src.backup.archive import ArchiveError
from src.backup.service import JobsRunning

ORIGIN = "http://testserver"
CSRF = {"X-RAG-Console": "1", "Origin": ORIGIN}
FOLDERS = {
    1: SimpleNamespace(id=1, name="docs", user_token="sub:owner-a"),
    2: SimpleNamespace(id=2, name="docs", user_token="sub:owner-b"),
}
TAR = {"Content-Type": "application/x-tar"}
REAL_CHECK = AR.check_expanded_size


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    storage = tmp_path / "app" / "storage"
    storage.mkdir(parents=True)
    monkeypatch.setattr(AR, "_storage_root", lambda: storage)
    monkeypatch.setattr(AR, "_db_url", lambda: "postgresql://u@h/d")
    monkeypatch.setattr(AR, "_rag_conf", lambda: None)
    monkeypatch.setattr(AR, "import_limit_bytes", lambda: 1000)
    # The uploads below are not real tars; CEX-17 / CEX-18 / CEX-19 use the real check.
    monkeypatch.setattr(AR, "check_expanded_size", lambda path, limit: None)
    with patch("db.cached_folderdb.CachedFolderDB.get_by_id", side_effect=FOLDERS.get), \
         patch("db.cached_folderdb.CachedFolderDB.invalidate_folder_caches"):
        yield SimpleNamespace(storage=storage, work=storage / ".folio-work" / "console")


def _client(as_identity, ident):
    app = FastAPI()
    app.include_router(router)
    return TestClient(as_identity(app, ident))


@pytest.fixture()
def as_a(as_identity, make_identity):
    return _client(as_identity, make_identity("sub:owner-a"))


@pytest.fixture()
def as_b(as_identity, make_identity):
    return _client(as_identity, make_identity("sub:owner-b"))


def _fake_export(**kw):
    out = Path(kw["out_dir"]) / "folio-folder-docs-20261116T000000Z.tar"
    out.write_bytes(b"archive-bytes")
    return F.ExportResult(path=out, size=13, manifest={"folder": {"name": "docs", "files": 2, "chunks": 5}},
                          warnings=["w1"])


def _leftovers(env):
    return sorted(p.name for p in env.work.iterdir()) if env.work.exists() else []


# ---------------------------------------------------------------- export

class TestExport:
    def test_cex01_foreign_folder_is_404_and_nothing_runs(self, as_b, env):
        """CEX-01"""
        with patch("src.backup.folder.export_folder") as ex:
            r = as_b.post("/api/admin/manage/folders/1/export")
        assert r.status_code == 404 and not ex.called
        assert "docs" not in r.text

    def test_cex02_export_checks_the_session_owner_and_downloads_once(self, as_a, env):
        """CEX-02"""
        with patch("src.backup.folder.export_folder", side_effect=_fake_export) as ex:
            r = as_a.post("/api/admin/manage/folders/1/export")
        assert r.status_code == 200, r.text
        assert ex.call_args.kwargs["folder_id"] == 1
        assert ex.call_args.kwargs["require_owner"] == "sub:owner-a"
        d = r.json()["data"]
        assert d["file_name"].endswith(".tar") and d["chunks"] == 5 and d["warnings"] == ["w1"]
        got = as_a.get(d["download_url"])
        assert got.status_code == 200 and got.content == b"archive-bytes"
        assert d["file_name"] in got.headers["content-disposition"]
        assert as_a.get(d["download_url"]).status_code == 404  # once only
        assert _leftovers(env) == []  # deleted after the download

    def test_cex03_another_identity_cannot_download(self, as_a, as_b, env):
        """CEX-03"""
        with patch("src.backup.folder.export_folder", side_effect=_fake_export):
            url = as_a.post("/api/admin/manage/folders/1/export").json()["data"]["download_url"]
        assert as_b.get(url).status_code == 404
        assert as_a.get(url).status_code == 200  # B's attempt did not consume it

    @pytest.mark.parametrize("export_id", ["../../etc", "x" * 32, "0" * 31, "A" * 32])
    def test_cex04_unknown_or_malformed_ids_are_404(self, as_a, export_id):
        """CEX-04"""
        assert as_a.get(f"/api/admin/manage/exports/{export_id}").status_code == 404

    def test_cex05_running_index_job_is_409_and_cleaned_up(self, as_a, env):
        """CEX-05"""
        with patch("src.backup.folder.export_folder", side_effect=JobsRunning("1 index job(s) are running")):
            r = as_a.post("/api/admin/manage/folders/1/export")
        assert r.status_code == 409
        assert r.json()["detail"]["error"] == "index_job_running"
        assert "index job" in r.json()["detail"]["error_description"]
        assert _leftovers(env) == []

    def test_cex06_owner_changed_during_export_is_404(self, as_a, env):
        """CEX-06"""
        with patch("src.backup.folder.export_folder", side_effect=F.FolderRefused("x", code="not_found")):
            r = as_a.post("/api/admin/manage/folders/1/export")
        assert r.status_code == 404 and _leftovers(env) == []

    def test_cex07_other_refusal_and_crash_clean_up(self, as_a, env):
        """CEX-07"""
        with patch("src.backup.folder.export_folder", side_effect=F.FolderRefused("no tables")):
            r = as_a.post("/api/admin/manage/folders/1/export")
        assert r.status_code == 400 and r.json()["detail"]["error"] == "export_refused"
        with patch("src.backup.folder.export_folder", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                as_a.post("/api/admin/manage/folders/1/export")
        assert _leftovers(env) == []

    def test_cex08_open_mode_exports_any_folder_without_owner_check(self, as_identity, env):
        """CEX-08"""
        from src.auth.console_session import ConsoleIdentity
        c = _client(as_identity, ConsoleIdentity(mode="open", via="open", is_admin=True))
        with patch("src.backup.folder.export_folder", side_effect=_fake_export) as ex:
            r = c.post("/api/admin/manage/folders/2/export")
        assert r.status_code == 200 and ex.call_args.kwargs["require_owner"] is None
        assert c.get(r.json()["data"]["download_url"]).status_code == 200


# ---------------------------------------------------------------- import

def _ok(_path=None, **kw):
    return F.ImportResult(manifest={}, imported=True, folder_id=9, name="docs-2", owner=kw["owner_key"],
                          files=2, chunks=5, indexed=not kw["files_only"], warnings=["renamed"])


class TestImport:
    def test_cex09_owner_is_the_session_never_the_request(self, as_b, env):
        """CEX-09"""
        with patch("src.backup.folder.import_folder", side_effect=_ok) as im:
            r = as_b.post("/api/admin/manage/folders/import?owner=sub:owner-a&owner_key=sub:owner-a",
                          content=b"tar-bytes", headers={**TAR, "X-Owner": "sub:owner-a"})
        assert r.status_code == 200, r.text
        kw = im.call_args.kwargs
        assert kw["owner_key"] == "sub:owner-b" and kw.get("owner") is None
        assert kw["files_only"] is False and kw["name"] is None
        assert r.json()["data"] == {"folder_id": 9, "name": "docs-2", "files": 2, "chunks": 5,
                                    "indexed": True, "warnings": ["renamed"]}
        assert _leftovers(env) == []

    def test_cex10_the_upload_reaches_the_import_whole(self, as_a, env):
        """CEX-10"""
        seen = {}

        def _read(path, **kw):
            seen["bytes"] = Path(path).read_bytes()
            seen["dir"] = Path(path).parent
            return _ok(**kw)
        with patch("src.backup.folder.import_folder", side_effect=_read):
            r = as_a.post("/api/admin/manage/folders/import?name=%20copy%20&files_only=true",
                          content=b"a" * 999, headers=TAR)
        assert r.status_code == 200
        assert seen["bytes"] == b"a" * 999 and seen["dir"] == env.work
        assert r.json()["data"]["indexed"] is False

    def test_cex11_open_mode_imports_for_console(self, as_identity, env):
        """CEX-11"""
        from src.auth.console_session import ConsoleIdentity
        c = _client(as_identity, ConsoleIdentity(mode="open", via="open", is_admin=True))
        with patch("src.backup.folder.import_folder", side_effect=_ok) as im:
            assert c.post("/api/admin/manage/folders/import", content=b"x", headers=TAR).status_code == 200
        assert im.call_args.kwargs["owner_key"] == "console"

    def test_cex12_declared_size_over_the_limit_is_413_before_reading(self, as_a, env):
        """CEX-12"""
        with patch("src.backup.folder.import_folder") as im, \
             patch.object(AR.tempfile, "mkstemp", wraps=AR.tempfile.mkstemp) as mk:
            r = as_a.post("/api/admin/manage/folders/import", content=b"a" * 1001, headers=TAR)
        assert r.status_code == 413 and r.json()["detail"]["error"] == "too_large"
        assert not mk.called  # refused from Content-Length alone: nothing of the body was stored
        assert not im.called and _leftovers(env) == []

    def test_cex13_streamed_body_over_the_limit_is_413(self, as_a, env):
        """CEX-13 -- chunked (no Content-Length): counted as it arrives."""
        def body():
            for _ in range(20):
                yield b"a" * 100
        with patch("src.backup.folder.import_folder") as im:
            r = as_a.post("/api/admin/manage/folders/import", content=body(), headers=TAR)
        assert r.status_code == 413
        assert not im.called and _leftovers(env) == []

    def test_cex14_empty_upload_is_400(self, as_a, env):
        """CEX-14"""
        with patch("src.backup.folder.import_folder") as im:
            r = as_a.post("/api/admin/manage/folders/import", content=b"", headers=TAR)
        assert r.status_code == 400 and r.json()["detail"]["error"] == "empty_upload" and not im.called
        assert _leftovers(env) == []

    @pytest.mark.parametrize("exc, status, code", [
        (F.FolderRefused("x; pass --files-only", code="embedding_mismatch",
                         reasons=["its vectors were made with bge"]), 409, "embedding_mismatch"),
        (F.FolderRefused("newer", code="newer_version", reasons=["made by a newer Folio"]), 409, "newer_version"),
        (F.FolderRefused("bad name"), 400, "import_refused"),
        (ArchiveError("checksum mismatch for 'chunks.jsonl'"), 400, "damaged_archive"),
    ])
    def test_cex15_refusals_map_to_codes_and_clean_up(self, as_a, env, exc, status, code):
        """CEX-15"""
        with patch("src.backup.folder.import_folder", side_effect=exc):
            r = as_a.post("/api/admin/manage/folders/import", content=b"x", headers=TAR)
        assert r.status_code == status and r.json()["detail"]["error"] == code
        assert "--files-only" not in r.json()["detail"]["error_description"]
        assert _leftovers(env) == []

    def test_cex16_a_crash_in_the_import_removes_the_upload(self, as_a, env):
        """CEX-16"""
        with patch("src.backup.folder.import_folder", side_effect=RuntimeError("db down")):
            with pytest.raises(RuntimeError):
                as_a.post("/api/admin/manage/folders/import", content=b"x", headers=TAR)
        assert _leftovers(env) == []

    def test_cex17_a_non_tar_upload_is_a_damaged_archive(self, as_a, env, monkeypatch):
        """CEX-17 -- real check_expanded_size, import never reached."""
        monkeypatch.setattr(AR, "check_expanded_size", REAL_CHECK)
        with patch("src.backup.folder.import_folder") as im:
            r = as_a.post("/api/admin/manage/folders/import", content=b"not a tar" * 10, headers=TAR)
        assert r.status_code == 400 and r.json()["detail"]["error"] == "damaged_archive"
        assert not im.called and _leftovers(env) == []


# ---------------------------------------------------------------- helpers

def _tar(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as t:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestHelpers:
    def test_cex18_expanded_size_limit(self, tmp_path):
        """CEX-18"""
        p = tmp_path / "a.tar"
        p.write_bytes(_tar([("a", b"x" * 600), ("b", b"y" * 600)]))
        REAL_CHECK(p, 1200)
        with pytest.raises(ArchiveError, match="more than the import limit"):
            REAL_CHECK(p, 1199)

    def test_cex19_sparse_entries_are_refused(self, tmp_path):
        """CEX-19"""
        p = tmp_path / "s.tar"
        p.write_bytes(_tar([("a", b"x" * 10)]))
        with patch.object(tarfile.TarInfo, "issparse", return_value=True):
            with pytest.raises(ArchiveError, match="sparse"):
                REAL_CHECK(p, 10_000)
        REAL_CHECK(p, 10_000)  # the same entry, not sparse, passes

    def test_cex20_sweep_removes_only_stale_entries(self, tmp_path):
        """CEX-20"""
        tmp_path = tmp_path / "w"
        tmp_path.mkdir()
        old_dir = tmp_path / "export-old"
        old_dir.mkdir()
        (old_dir / "a.tar").write_bytes(b"x")
        fresh_dir = tmp_path / "export-fresh"
        fresh_dir.mkdir()
        (fresh_dir / "a.tar").write_bytes(b"x")
        old_file = tmp_path / "upload-1.tar"
        old_file.write_bytes(b"x")
        past = time.time() - AR.EXPORT_TTL_S - 10
        for p in (old_dir / "a.tar", old_dir, old_file):
            os.utime(p, (past, past))
        # A directory whose own mtime is old but holds a file still being written stays.
        busy = tmp_path / "export-busy"
        busy.mkdir()
        (busy / "a.tar.partial").write_bytes(b"x")
        os.utime(busy, (past, past))
        assert AR.sweep(tmp_path) == 2
        assert sorted(p.name for p in tmp_path.iterdir()) == ["export-busy", "export-fresh"]
        assert AR.sweep(tmp_path / "missing") == 0

    def test_cex21_work_dir_is_inside_storage_and_private(self, env):
        """CEX-21 -- inside the storage root (the only writable place in the Docker image, and
        the import's filesystem), under Folio's scratch directory."""
        d = AR.work_dir()
        assert d == env.storage / ".folio-work" / "console"
        assert oct(d.stat().st_mode & 0o777) == "0o700"
        assert oct(d.parent.stat().st_mode & 0o777) == "0o700"

    def test_cex22_limit_comes_from_the_backup_config(self, monkeypatch):
        """CEX-22"""
        monkeypatch.undo()
        from src.config.model import BackupConfig
        with patch("src.backup.service.backup_settings", return_value=BackupConfig(folder_import_max_mb=3)):
            assert AR.import_limit_bytes() == 3 * 1024 * 1024
        assert BackupConfig().folder_import_max_mb == 2048


# ---------------------------------------------------------------- real guards: CSRF / bearer

def _real_app():
    app = FastAPI()
    app.include_router(router)
    return app


def _signed_in(tokens, sub):
    c = TestClient(_real_app())
    r = c.post("/api/console/session", json={"token": tokens.make(jti=f"jti-{sub}", sub=sub)}, headers=CSRF)
    assert r.status_code == 200, r.text
    return c


class TestRealGuards:
    def test_cex23_cookie_import_without_csrf_is_403_and_nothing_is_written(self, console_auth, tokens, env):
        """CEX-23"""
        c = _signed_in(tokens, "owner-a")
        with patch("src.backup.folder.import_folder") as im:
            r = c.post("/api/admin/manage/folders/import", content=b"x", headers={**TAR, "Origin": ORIGIN})
            assert r.status_code == 403
            r = c.post("/api/admin/manage/folders/import", content=b"x",
                       headers={**TAR, "X-RAG-Console": "1", "Origin": "http://localhost:4568"})
            assert r.status_code == 403
        assert not im.called and _leftovers(env) == []

    def test_cex24_cookie_export_without_csrf_is_403(self, console_auth, tokens, env):
        """CEX-24"""
        c = _signed_in(tokens, "owner-a")
        with patch("src.backup.folder.export_folder", side_effect=_fake_export) as ex:
            assert c.post("/api/admin/manage/folders/1/export").status_code == 403
            assert not ex.called
            r = c.post("/api/admin/manage/folders/1/export", headers=CSRF)
            assert r.status_code == 200
            # The download is a GET: the cookie alone is enough.
            assert c.get(r.json()["data"]["download_url"]).status_code == 200

    def test_cex25_bearer_pat_needs_no_csrf_and_imports_for_its_person(self, console_auth, tokens, env):
        """CEX-25"""
        c = TestClient(_real_app())
        auth = {"Authorization": f"Bearer {tokens.make(jti='jti-b', sub='owner-b')}"}
        with patch("src.backup.folder.import_folder", side_effect=_ok) as im:
            r = c.post("/api/admin/manage/folders/import", content=b"x", headers={**TAR, **auth})
        assert r.status_code == 200 and im.call_args.kwargs["owner_key"] == "sub:owner-b"

    def test_cex26_anonymous_is_401(self, console_auth, env):
        """CEX-26"""
        c = TestClient(_real_app())
        assert c.post("/api/admin/manage/folders/import", content=b"x", headers={**TAR, **CSRF}).status_code == 401
        assert c.get("/api/admin/manage/exports/" + "0" * 32).status_code == 401
        assert _leftovers(env) == []


def test_cex27_export_json_records_the_owner(as_a, env):
    """CEX-27"""
    with patch("src.backup.folder.export_folder", side_effect=_fake_export):
        export_id = as_a.post("/api/admin/manage/folders/1/export").json()["data"]["export_id"]
    meta = json.loads((env.work / f"export-{export_id}" / "export.json").read_text())
    assert meta == {"owner": "sub:owner-a", "file": "folio-folder-docs-20261116T000000Z.tar"}


def test_cex34_an_export_is_claimed_before_it_is_served(as_a, env, monkeypatch):
    """CEX-34 -- even before the copy is deleted, a second request cannot get it."""
    with patch("src.backup.folder.export_folder", side_effect=_fake_export):
        url = as_a.post("/api/admin/manage/folders/1/export").json()["data"]["download_url"]
    monkeypatch.setattr(AR, "BackgroundTask", lambda *a, **k: None)  # keep the served copy
    assert as_a.get(url).status_code == 200
    assert as_a.get(url).status_code == 404
    assert [p.name.startswith("served-") for p in env.work.iterdir()] == [True]
