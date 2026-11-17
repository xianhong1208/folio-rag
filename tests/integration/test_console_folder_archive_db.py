"""Integration test -- the console's folder export / import endpoints on real PostgreSQL.

Person A exports their folder through the API and downloads it; person B cannot
export A's folder, and imports A's archive as a folder of their own; refusals (an
index job running, another embedding model) answer with their codes and leave
nothing behind. Uses the folder built by test_folder_export_db's ``source`` fixture.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.integration.test_folder_export_db import OWNER, RAG, _q, _url, source  # noqa: F401 (fixture)

OWNER_B = "sub:itest-folder-b"
TAR = {"Content-Type": "application/x-tar"}


@pytest.fixture()
def api(source, monkeypatch, as_identity, make_identity):  # noqa: F811
    from src.api.router import admin_folder_archive as AR
    from src.api.router.admin_settings import router
    monkeypatch.setattr(AR, "_storage_root", lambda: source.storage)
    monkeypatch.setattr(AR, "_rag_conf", lambda: RAG)

    def client(owner):
        app = FastAPI()
        app.include_router(router)
        return TestClient(as_identity(app, make_identity(owner)))
    return SimpleNamespace(a=client(OWNER), b=client(OWNER_B), AR=AR,
                           work=source.storage / ".folio-work" / "console")


def _export(c, folder_id):
    r = c.post(f"/api/admin/manage/folders/{folder_id}/export")
    assert r.status_code == 200, r.text
    got = c.get(r.json()["data"]["download_url"])
    assert got.status_code == 200
    return got.content


def test_cex32_export_download_and_import_as_another_person(source, api):  # noqa: F811
    """CEX-32"""
    archive = _export(api.a, source.folder.id)
    assert api.b.post(f"/api/admin/manage/folders/{source.folder.id}/export").status_code == 404
    # The owner is checked again inside the export's own transaction.
    from src.backup import folder as F
    with pytest.raises(F.FolderRefused) as e:
        F.export_folder(db_url=_url(), storage_root=source.storage, folder_id=source.folder.id,
                        out_dir=source.storage.parent / "x", rag_conf=RAG, require_owner=OWNER_B)
    assert e.value.code == "not_found"

    r = api.b.post(f"/api/admin/manage/folders/import?owner={OWNER}", content=archive, headers=TAR)
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert d["name"] == source.folder.name and d["indexed"] and d["chunks"] == 2 and d["files"] == 1
    (owner, uuid_), = _q('SELECT user_token, vector_table_uuid FROM "Folders" WHERE id = :i', i=d["folder_id"])
    assert owner == OWNER_B
    (path,), = _q('SELECT file_path FROM "Files" WHERE folder_id = :i', i=d["folder_id"])
    assert path.startswith(f"storage/sub_itest-folder-b/{source.folder.name}/")
    assert (source.storage.parent / path).read_text(encoding="utf-8") == "Oolong tea from Taiwan"
    old = _q(f'SELECT node_id, text, embedding::text, text_search_tsv::text FROM "{source.table}" ORDER BY id')
    new = _q(f'SELECT node_id, text, embedding::text, text_search_tsv::text FROM '
             f'"data_{d["folder_id"]}_{uuid_}" ORDER BY id')
    assert new == old

    # A imports their own export: a copy named -2, theirs.
    r = api.a.post("/api/admin/manage/folders/import", content=archive, headers=TAR)
    assert r.status_code == 200 and r.json()["data"]["name"] == f"{source.folder.name}-2"
    assert _q('SELECT user_token FROM "Folders" WHERE id = :i', i=r.json()["data"]["folder_id"]) == [(OWNER,)]
    # B sees the imported folder in their list right away (the cached list was dropped).
    from db.cached_folderdb import CachedFolderDB
    assert d["folder_id"] in {f.id for f in CachedFolderDB.get_by_user_token(OWNER_B)}
    assert list(api.work.iterdir()) == []
    assert not list(source.storage.glob(".folio-work/import-*"))
    # Everything temporary lived inside the storage root, under .folio-work (CEX-40).
    assert sorted(p.name for p in (source.storage / ".folio-work").iterdir()) == ["console"]
    assert not list(source.storage.parent.glob(".folio-*"))


def test_cex33_running_job_and_model_mismatch(source, api, monkeypatch):  # noqa: F811
    """CEX-33"""
    from db.db import get_engine
    archive = _export(api.a, source.folder.id)
    with get_engine().begin() as c:
        c.execute(text('INSERT INTO "IndexJobs" (job_id, folder_id, status, last_updated_at) '
                       "VALUES (gen_random_uuid(), :d, 'running', :t)"),
                  {"d": source.folder.id, "t": datetime.now(timezone.utc).isoformat()})
    try:
        r = api.a.post(f"/api/admin/manage/folders/{source.folder.id}/export")
        assert r.status_code == 409 and r.json()["detail"]["error"] == "index_job_running"
    finally:
        with get_engine().begin() as c:
            c.execute(text('DELETE FROM "IndexJobs" WHERE folder_id = :d'), {"d": source.folder.id})

    other = SimpleNamespace(embedding=SimpleNamespace(model="other-embed", dimension=8), profile="cloud",
                            retrieval=SimpleNamespace(segmenter="bigram"))
    monkeypatch.setattr(api.AR, "_rag_conf", lambda: other)
    before = _q('SELECT count(*) FROM "Folders"')
    r = api.b.post("/api/admin/manage/folders/import", content=archive, headers=TAR)
    assert r.status_code == 409 and r.json()["detail"]["error"] == "embedding_mismatch"
    assert "other-embed" in r.json()["detail"]["error_description"]
    assert _q('SELECT count(*) FROM "Folders"') == before
    r = api.b.post("/api/admin/manage/folders/import?files_only=true&name=tea-files", content=archive, headers=TAR)
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert d["name"] == "tea-files" and not d["indexed"] and d["chunks"] == 0
    assert _q('SELECT count(*) FROM "FileIndices" WHERE folder_id = :i', i=d["folder_id"]) == [(0,)]

    monkeypatch.setattr(api.AR, "_rag_conf", lambda: RAG)
    damaged = bytearray(archive)
    i = damaged.index(b"Oolong tea from Taiwan")
    damaged[i:i + 6] = b"Assam!"
    r = api.b.post("/api/admin/manage/folders/import", content=bytes(damaged), headers=TAR)
    assert r.status_code == 400 and r.json()["detail"]["error"] == "damaged_archive"
    assert "checksum" in json.dumps(r.json())
    assert list(api.work.iterdir()) == []
