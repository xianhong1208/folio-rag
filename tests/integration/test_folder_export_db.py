"""Integration test -- export one folder and import it back as a copy, on real PostgreSQL.

The folder has a file, a leaf chunk and a parent chunk with the metadata the indexer
writes (file id and path, folder id and name, a serialized ``_node_content``), a
detached keyword vector with a segmenter stamp, and an HNSW index. The import must
create a new folder with new file ids whose chunks, vectors and keyword vectors equal
the original's, point at the new file / folder, and carry the lookup indexes.
"""

import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

RAG = SimpleNamespace(embedding=SimpleNamespace(model="itest-embed", dimension=3), profile="cloud",
                      retrieval=SimpleNamespace(segmenter="bigram"))
OWNER = "sub:itest-folder-owner"


def _url():
    from src.config.config_manager import Config
    from src.utils.db_url import resolve_db_url
    return resolve_db_url(Config.get_database_config())


def _meta(fid, rel, folder, role, extra):
    base = {"file_id": fid, "file_name": "tea.md", "file_path": rel, "folder_id": folder.id,
            "folder_name": folder.name, "node_role": role, **extra}
    node = {"id_": extra.pop("_id"), "embedding": None, "metadata": dict(base), "relationships": {},
            "text": "", "class_name": "TextNode"}
    return json.dumps({**base, "_node_content": json.dumps(node), "_node_type": "TextNode",
                       "document_id": "None", "doc_id": "None", "ref_doc_id": "None"})


@pytest.fixture()
def source(itest_db, tmp_path):
    from db.db import get_engine
    from db.filedb import FileDB
    from db.folderdb import FolderDB
    folder = FolderDB.create(name=f"fx-{uuid.uuid4().hex[:6]}", user_token=OWNER, description="teas")
    storage = tmp_path / "storage"
    fid = str(uuid.uuid4())
    rel = f"storage/sub_itest-folder-owner/{folder.name}/{fid}"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes("Oolong tea from Taiwan".encode("utf-8"))
    FileDB.create(id=uuid.UUID(fid), folder_id=folder.id, file_name="tea.md", file_path=rel, file_size=22,
                  mime_type="text/markdown", tags=["tea"], content_hash="ab" * 32)
    table = f"data_{folder.id}_{folder.vector_table_uuid}"
    leaf, parent = str(uuid.uuid4()), str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text(
            f'CREATE TABLE "{table}" (id BIGSERIAL PRIMARY KEY, text VARCHAR NOT NULL, metadata_ JSON, '
            f"node_id VARCHAR, embedding vector(3), text_search_tsv TSVECTOR)"))
        c.execute(text(f'INSERT INTO "{table}" (text, metadata_, node_id, embedding, text_search_tsv) VALUES '
                       f"('oolong tea', :m1, :n1, '[0.1,0.2,0.3]', to_tsvector('simple', 'oolong tea')), "
                       f"('oolong tea. dong ding.', :m2, :n2, NULL, NULL)"),
                  {"m1": _meta(fid, rel, folder, "leaf", {"_id": leaf, "chunk_index": 0, "parent_node_id": parent,
                                                          "chunk_hash": "h1"}),
                   "n1": leaf,
                   "m2": _meta(fid, rel, folder, "parent", {"_id": parent, "children_node_ids": [leaf]}),
                   "n2": parent})
        c.execute(text(f'COMMENT ON COLUMN "{table}".text_search_tsv IS \'segmenter=bigram\''))
        c.execute(text('INSERT INTO "FileIndices" (id, file_id, folder_id, index_id, vector_store_table, '
                       'embedding_model, embedding_dimension, num_chunks, status, pipeline_version) VALUES '
                       "(gen_random_uuid(), :f, :d, :i, :t, 'itest-embed', 3, 1, 'indexed', '3')"),
                  {"f": fid, "d": folder.id, "i": f"file_{fid}", "t": table})
    yield SimpleNamespace(folder=folder, storage=storage, fid=fid, table=table, leaf=leaf, parent=parent)
    with get_engine().begin() as c:
        ids = [r[0] for r in c.execute(text('SELECT id FROM "Folders" WHERE user_token LIKE \'sub:itest-folder-%\''))]
        for (t,) in c.execute(text("SELECT tablename FROM pg_tables WHERE tablename LIKE ANY(:p)"),
                              {"p": [f"data\\_{i}\\_%" for i in ids]}):
            c.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
        for t in ("FileIndices", "Files"):
            c.execute(text(f'DELETE FROM "{t}" WHERE folder_id = ANY(:i)'), {"i": ids})
        c.execute(text('DELETE FROM "Folders" WHERE id = ANY(:i)'), {"i": ids})


def _q(sql, **params):
    eng = create_engine(_url(), poolclass=NullPool)
    try:
        with eng.connect() as c:
            return c.execute(text(sql), params).all()
    finally:
        eng.dispose()


def test_export_then_import_as_a_copy(source, tmp_path):
    from src.backup import folder as F
    from src.backup.archive import read_manifest

    exp = F.export_folder(db_url=_url(), storage_root=source.storage, folder_id=source.folder.id,
                          out_dir=tmp_path / "out", rag_conf=RAG)
    m = read_manifest(exp.path, F.FOLDER_EXPORT)
    assert m["folder"] == {"name": source.folder.name, "files": 1, "indexed_files": 1, "chunks": 2,
                           "bytes": 22, "missing_files": 0}
    assert m["indexed_embeddings"] == [{"model": "itest-embed", "dimension": 3, "files": 1}]
    assert m["vector_table"] == {"exists": True, "dimension": 3, "segmenter": "bigram"}
    assert f"files/{source.fid}" in m["checksums"]
    import tarfile
    with tarfile.open(exp.path) as tar:
        doc = json.load(tar.extractfile(F.FOLDER_JSON))
    # where the files sat here (the owner's directory) does not travel
    assert doc["files"][0]["id"] == source.fid and "file_path" not in doc["files"][0]
    assert OWNER not in json.dumps(doc) and "itest-folder-owner" not in json.dumps(doc)

    # Same owner: a copy next to the original, named -2.
    r = F.import_folder(exp.path, db_url=_url(), storage_root=source.storage, owner=OWNER, rag_conf=RAG)
    assert r.imported and r.indexed and r.name == f"{source.folder.name}-2" and r.chunks == 2
    assert r.folder_id != source.folder.id
    (new_id, new_uuid, desc, owner, count), = _q(
        'SELECT id, vector_table_uuid, description, user_token, file_count FROM "Folders" WHERE id = :i',
        i=r.folder_id)
    assert (desc, owner, count) == ("teas", OWNER, 1)
    (nfid, path, tags, chash), = _q('SELECT id::text, file_path, tags, content_hash FROM "Files" WHERE folder_id = :i',
                                     i=new_id)
    assert nfid != source.fid and tags == ["tea"] and chash == "ab" * 32
    assert path == f"storage/sub_itest-folder-owner/{r.name}/{nfid}"
    assert (source.storage.parent / path).read_text(encoding="utf-8") == "Oolong tea from Taiwan"
    assert (source.storage.parent / f"storage/sub_itest-folder-owner/{source.folder.name}/{source.fid}").exists()
    table = f"data_{new_id}_{new_uuid}"
    (idx_fid, idx_table, idx_id, model), = _q(
        'SELECT file_id::text, vector_store_table, index_id, embedding_model FROM "FileIndices" WHERE folder_id = :i',
        i=new_id)
    assert (idx_fid, idx_table, idx_id, model) == (nfid, table, f"file_{nfid}", "itest-embed")

    old = _q(f'SELECT node_id, text, embedding::text, text_search_tsv::text FROM "{source.table}" ORDER BY id')
    new = _q(f'SELECT node_id, text, embedding::text, text_search_tsv::text FROM "{table}" ORDER BY id')
    assert new == old
    for (raw,) in _q(f'SELECT metadata_::text FROM "{table}"'):
        meta = json.loads(raw)
        inner = json.loads(meta["_node_content"])["metadata"]
        for d in (meta, inner):
            assert (d["file_id"], d["file_path"], d["folder_id"], d["folder_name"]) == (nfid, path, new_id, r.name)
        assert source.fid not in raw
    leaf_meta = json.loads(_q(f'SELECT metadata_::text FROM "{table}" WHERE node_id = :n', n=source.leaf)[0][0])
    assert leaf_meta["parent_node_id"] == source.parent

    indexes = {n for (n,) in _q("SELECT indexname FROM pg_indexes WHERE tablename = :t", t=table)}
    assert {f"{table}_hnsw_idx", f"{table}_nid_idx", f"{table}_fid_idx"} <= indexes
    assert any(n.endswith("_idx_1") for n in indexes)
    (stamp,), = _q("SELECT col_description(to_regclass(:t), 6)", t=f'"{table}"')
    assert stamp == "segmenter=bigram"
    # the keyword index answers on the copy
    assert _q(f"SELECT count(*) FROM \"{table}\" WHERE text_search_tsv @@ to_tsquery('simple', 'oolong')") == [(1,)]

    # Another owner keeps the name.
    r2 = F.import_folder(exp.path, db_url=_url(), storage_root=source.storage, owner="itest-folder-other",
                         rag_conf=RAG)
    assert r2.name == source.folder.name and r2.owner == "sub:itest-folder-other"


def test_another_embedding_model_is_refused_or_imported_without_the_index(source, tmp_path):
    from src.backup import folder as F
    exp = F.export_folder(db_url=_url(), storage_root=source.storage, owner=OWNER, name=source.folder.name,
                          out_dir=tmp_path / "out", rag_conf=RAG)
    other = SimpleNamespace(embedding=SimpleNamespace(model="other-embed", dimension=8), profile="cloud",
                            retrieval=SimpleNamespace(segmenter="bigram"))
    before = _q('SELECT count(*) FROM "Folders"')
    with pytest.raises(F.FolderRefused, match="--files-only"):
        F.import_folder(exp.path, db_url=_url(), storage_root=source.storage, owner=OWNER, rag_conf=other)
    assert _q('SELECT count(*) FROM "Folders"') == before
    r = F.import_folder(exp.path, db_url=_url(), storage_root=source.storage, owner=OWNER, rag_conf=other,
                        files_only=True)
    assert r.imported and not r.indexed and r.chunks == 0
    assert _q('SELECT count(*) FROM "Files" WHERE folder_id = :i', i=r.folder_id) == [(1,)]
    assert _q('SELECT count(*) FROM "FileIndices" WHERE folder_id = :i', i=r.folder_id) == [(0,)]
    (u,), = _q('SELECT vector_table_uuid FROM "Folders" WHERE id = :i', i=r.folder_id)
    assert _q("SELECT to_regclass(:t) IS NULL", t=f'"data_{r.folder_id}_{u}"') == [(True,)]


@pytest.mark.parametrize("where", ["chunks", "files"])
def test_a_failed_import_leaves_nothing_behind(source, tmp_path, monkeypatch, where):
    from src.backup import folder as F
    exp = F.export_folder(db_url=_url(), storage_root=source.storage, folder_id=source.folder.id,
                          out_dir=tmp_path / "out", rag_conf=RAG)
    before = _q('SELECT count(*) FROM "Folders"'), _q('SELECT count(*) FROM "Files"')

    def boom(*a, **k):
        boom.n = getattr(boom, "n", 0) + 1
        if boom.n == 2 or where == "files":
            raise RuntimeError("disk full")
        return real(*a, **k)
    if where == "chunks":
        real = F.remap_chunk_metadata
        monkeypatch.setattr(F, "remap_chunk_metadata", boom)
    else:  # while the files are moved into the new folder's directory
        real = F.os.replace
        monkeypatch.setattr(F.os, "replace", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        F.import_folder(exp.path, db_url=_url(), storage_root=source.storage, owner=OWNER, rag_conf=RAG)
    assert (_q('SELECT count(*) FROM "Folders"'), _q('SELECT count(*) FROM "Files"')) == before
    assert not (source.storage / "sub_itest-folder-owner" / f"{source.folder.name}-2").exists()
    assert not list(source.storage.glob(".folio-work/import-*"))
    # the transaction also rolled back the new vector table
    assert _q("SELECT count(*) FROM pg_tables WHERE tablename LIKE 'data\\_%' AND tablename <> :t",
              t=source.table) == [(0,)]


def test_a_running_index_job_blocks_the_export(source, tmp_path):
    from datetime import datetime, timezone

    from db.db import get_engine
    from src.backup import folder as F
    from src.backup.service import JobsRunning
    with get_engine().begin() as c:
        c.execute(text('INSERT INTO "IndexJobs" (job_id, folder_id, status, last_updated_at) '
                       "VALUES (gen_random_uuid(), :d, 'running', :t)"),
                  {"d": source.folder.id, "t": datetime.now(timezone.utc).isoformat()})
    try:
        with pytest.raises(JobsRunning):
            F.export_folder(db_url=_url(), storage_root=source.storage, folder_id=source.folder.id,
                            out_dir=tmp_path / "out", rag_conf=RAG)
        assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())
    finally:
        with get_engine().begin() as c:
            c.execute(text('DELETE FROM "IndexJobs" WHERE folder_id = :d'), {"d": source.folder.id})
