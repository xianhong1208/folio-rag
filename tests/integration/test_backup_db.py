"""Integration test -- a real pg_dump / pg_restore round trip of a Folio database.

A folder with a vector table (one table keeps the STORED GENERATED text_search_tsv,
one has it detached and stamped with a COMMENT, as CKIP folders do) and a stored
file are backed up from the integration database and restored into a second
throwaway database; rows, vectors, sequences and the stored file must come back
unchanged. Skipped when the PostgreSQL client tools are not installed.
"""

import shutil
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

pytestmark = pytest.mark.skipif(not (shutil.which("pg_dump") and shutil.which("pg_restore")),
                                reason="pg_dump / pg_restore are not installed")

RAG = SimpleNamespace(embedding=SimpleNamespace(model="itest-embed", dimension=3), profile="cloud",
                      retrieval=SimpleNamespace(segmenter="bigram"))


def _url():
    from src.config.config_manager import Config
    from src.utils.db_url import resolve_db_url
    return resolve_db_url(Config.get_database_config())


@pytest.fixture()
def second_db(itest_db):
    """An empty database next to the integration one (pgvector created as the admin would)."""
    name = f"{itest_db}_restore"
    url = make_url(_url())
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool)
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{name}"'))
    target = url.set(database=name).render_as_string(hide_password=False)
    eng = create_engine(target, poolclass=NullPool)
    with eng.begin() as c:
        c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    eng.dispose()
    yield target
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture()
def populated(itest_db, tmp_path):
    from db.db import get_engine
    from db.filedb import FileDB
    from db.folderdb import FolderDB
    folder = FolderDB.create(name=f"bk-{uuid.uuid4().hex[:6]}", user_token="sub:itest-user")
    storage = tmp_path / "storage"
    rel = f"storage/sub_itest-user/{folder.name}/{uuid.uuid4()}"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes("Oolong tea from Taiwan".encode("utf-8"))
    f = FileDB.create(folder_id=folder.id, file_name="tea.md", file_path=rel, file_size=22,
                      mime_type="text/markdown")
    gen = f"data_{folder.id}_{folder.vector_table_uuid}"
    detached = f"data_{folder.id}9_{uuid.uuid4()}"
    with get_engine().begin() as c:
        for t in (gen, detached):
            c.execute(text(
                f'CREATE TABLE "{t}" (id BIGSERIAL PRIMARY KEY, text VARCHAR NOT NULL, metadata_ JSON, '
                f"node_id VARCHAR, embedding vector(3), text_search_tsv TSVECTOR GENERATED ALWAYS AS "
                f"(to_tsvector('simple', text)) STORED)"))
            c.execute(text(f'INSERT INTO "{t}" (text, metadata_, node_id, embedding) VALUES '
                           f"('oolong tea', '{{\"file_id\": \"{f.id}\"}}', 'n1', '[0.1,0.2,0.3]'), "
                           f"('dong ding', '{{\"file_id\": \"{f.id}\"}}', 'n2', '[0.3,0.2,0.1]')"))
        c.execute(text(f'ALTER TABLE "{detached}" ALTER COLUMN text_search_tsv DROP EXPRESSION'))
        c.execute(text(f'UPDATE "{detached}" SET text_search_tsv = to_tsvector(\'simple\', \'segmented tokens\')'))
        c.execute(text(f'COMMENT ON COLUMN "{detached}".text_search_tsv IS \'folio-segmenter:bigram\''))
        c.execute(text(f'CREATE INDEX "{gen}_hnsw_idx" ON "{gen}" USING hnsw (embedding vector_cosine_ops)'))
        c.execute(text('INSERT INTO "FileIndices" (id, file_id, folder_id, index_id, embedding_model, '
                       'embedding_dimension, num_chunks, status) VALUES '
                       "(gen_random_uuid(), :f, :d, 'i', 'itest-embed', 3, 2, 'indexed')"),
                  {"f": f.id, "d": folder.id})
    yield SimpleNamespace(folder=folder, storage=storage, rel=rel, tables=(gen, detached))
    with get_engine().begin() as c:
        for t in (gen, detached):
            c.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
        c.execute(text('DELETE FROM "FileIndices" WHERE folder_id = :d'), {"d": folder.id})
        c.execute(text('DELETE FROM "Files" WHERE folder_id = :d'), {"d": folder.id})
        c.execute(text('DELETE FROM "Folders" WHERE id = :d'), {"d": folder.id})


def _rows(url, table):
    eng = create_engine(url, poolclass=NullPool)
    try:
        with eng.connect() as c:
            return c.execute(text(f'SELECT id, text, node_id, embedding::text, text_search_tsv::text '
                                  f'FROM "{table}" ORDER BY id')).all()
    finally:
        eng.dispose()


def test_backup_restore_round_trip(populated, second_db, tmp_path):
    from src.backup import service as S
    from src.backup.archive import read_manifest

    out = tmp_path / "backups"
    result = S.create_backup(db_url=_url(), storage_root=populated.storage, out_dir=out, rag_conf=RAG)
    m = read_manifest(result.path)
    mine = [f for f in m["folders"] if f["id"] == populated.folder.id]
    assert mine and mine[0]["chunks"] == 2 and mine[0]["files"] == 1
    assert {"model": "itest-embed", "dimension": 3, "files": 1} in m["indexed_embeddings"]
    assert 3 in m["vector_dimensions"]
    assert m["alembic_revision"] in S.code_revisions()

    target_storage = tmp_path / "restored" / "storage"
    # Other integration tests leave FileIndices rows of other (fake) models in the shared
    # database; the model check itself is covered by the unit tests.
    mixed = len({e["model"] for e in m["indexed_embeddings"]}) > 1
    r = S.restore_backup(result.path, db_url=second_db, storage_root=target_storage, rag_conf=RAG,
                         allow_mismatch=mixed)
    assert r.restored
    for t in populated.tables:
        assert _rows(second_db, t) == _rows(_url(), t)
    assert (target_storage.parent / populated.rel).read_text(encoding="utf-8") == "Oolong tea from Taiwan"

    eng = create_engine(second_db, poolclass=NullPool)
    try:
        with eng.begin() as c:
            # the segmenter stamp, the HNSW index and the sequences came back
            stamp = c.execute(text("SELECT col_description(to_regclass(:t), attnum) FROM pg_attribute "
                                   "WHERE attrelid = to_regclass(:t) AND attname = 'text_search_tsv'"),
                              {"t": f'"{populated.tables[1]}"'}).scalar()
            assert stamp == "folio-segmenter:bigram"
            assert c.execute(text("SELECT count(*) FROM pg_indexes WHERE indexname = :i"),
                             {"i": f"{populated.tables[0]}_hnsw_idx"}).scalar() == 1
            new_id = c.execute(text('INSERT INTO "Folders" (name, vector_table_uuid) VALUES '
                                    "('after', gen_random_uuid()) RETURNING id")).scalar()
            assert new_id > populated.folder.id
            rev = c.execute(text("SELECT version_num FROM alembic_version")).scalar()
            assert rev == m["alembic_revision"]
    finally:
        eng.dispose()

    # A second restore into the now non-empty database is refused, then forced.
    with pytest.raises(S.BackupRefused, match="already has"):
        S.restore_backup(result.path, db_url=second_db, storage_root=target_storage, rag_conf=RAG,
                         allow_mismatch=mixed)
    r2 = S.restore_backup(result.path, db_url=second_db, storage_root=target_storage, rag_conf=RAG,
                          allow_mismatch=mixed, force=True)
    assert r2.restored and (r2.moved_aside / populated.rel.split("/", 1)[1]).exists()
    eng = create_engine(second_db, poolclass=NullPool)
    try:
        with eng.connect() as c:
            assert c.execute(text('SELECT count(*) FROM "Folders" WHERE name = \'after\'')).scalar() == 0
    finally:
        eng.dispose()


def test_running_index_job_blocks_a_backup(populated, tmp_path):
    from datetime import datetime, timezone

    from db.db import get_engine
    from src.backup import service as S
    job = uuid.uuid4()
    with get_engine().begin() as c:
        c.execute(text('INSERT INTO "IndexJobs" (job_id, folder_id, status, last_updated_at) '
                       "VALUES (:j, :d, 'running', :t)"),
                  {"j": job, "d": populated.folder.id, "t": datetime.now(timezone.utc).isoformat()})
    try:
        with pytest.raises(S.JobsRunning):
            S.create_backup(db_url=_url(), storage_root=populated.storage, out_dir=tmp_path / "b", rag_conf=RAG)
        r = S.create_backup(db_url=_url(), storage_root=populated.storage, out_dir=tmp_path / "b",
                            rag_conf=RAG, allow_running_jobs=True)
        assert [j["job_id"] for j in r.manifest["running_jobs"]] == [str(job)]
    finally:
        with get_engine().begin() as c:
            c.execute(text('DELETE FROM "IndexJobs" WHERE job_id = :j'), {"j": job})
