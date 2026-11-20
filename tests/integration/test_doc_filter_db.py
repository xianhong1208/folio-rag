"""Integration test -- document filters, tags and metadata on real PostgreSQL + pgvector.

Person A has two folders, person B one; their files carry tags (mixed case, and one
legacy row with JSON null tags) and metadata. Checks every filter field, the owner
scope (B's files never match A's filter, whatever it names), the listing helpers,
injection-shaped input, and the filtered dense / BM25 SQL: with a selective filter an
HNSW scan alone comes back short, the iterative scan fills k.
"""

from __future__ import annotations

import random
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text

A, B = "sub:itest-docs-a", "sub:itest-docs-b"


def _file(folder, name, tags=None, meta=None, mime="application/pdf", when="2024-06-01"):
    from datetime import datetime
    from db.filedb import FileDB
    return str(FileDB.create(folder_id=folder.id, file_name=name, file_path=f"/tmp/x/{uuid.uuid4().hex}",
                             file_size=len(name), mime_type=mime, tags=tags or [],
                             doc_metadata=meta or {}, upload_time=datetime.fromisoformat(f"{when}T10:00:00")).id)


@pytest.fixture()
def lib(itest_db):
    from db.db import get_engine
    from db.folderdb import FolderDB
    suffix = uuid.uuid4().hex[:6]
    fa = FolderDB.create(name=f"contracts-{suffix}", user_token=A)
    fa2 = FolderDB.create(name=f"hr-{suffix}", user_token=A)
    fb = FolderDB.create(name=f"contracts-b-{suffix}", user_token=B)
    ids = {
        "acme": _file(fa, "acme-2024.pdf", ["Invoice", "Signed"],
                      {"client": "Acme", "year": 2024, "due": "2024-12-31", "paid": True}),
        "beta": _file(fa, "beta_q3.docx", ["invoice"], {"client": "Beta", "year": 2023,
                                                        "due": "2023-09-30"},
                      mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                      when="2023-10-01"),
        "memo": _file(fa2, "memo.md", ["HR", "draft"], {"year": "2024"}, mime="text/markdown"),
        "scan": _file(fa2, "scan.png", [], {}, mime="image/png", when="2025-01-02"),
        "other": _file(fb, "acme-b.pdf", ["Invoice", "Signed"], {"client": "Acme", "year": 2024}),
    }
    legacy = _file(fa2, "legacy.txt", mime="text/plain")
    with get_engine().begin() as c:     # an upload from before 1.6 stored JSON null
        c.execute(text("UPDATE \"Files\" SET tags = 'null'::jsonb WHERE id = :i"), {"i": legacy})
    ids["legacy"] = legacy
    yield SimpleNamespace(ids=ids, fa=fa, fa2=fa2, fb=fb)
    with get_engine().begin() as c:
        fids = [fa.id, fa2.id, fb.id]
        c.execute(text('DELETE FROM "Files" WHERE folder_id = ANY(:i)'), {"i": fids})
        c.execute(text('DELETE FROM "Folders" WHERE id = ANY(:i)'), {"i": fids})


def _match(lib, owner=A, folder_id=None, **flt):
    from db.db import get_engine
    from src.domain.rag.doc_filter import DocumentFilter, matching_file_ids
    rev = {v: k for k, v in lib.ids.items()}
    with get_engine().connect() as c:
        got = matching_file_ids(c, DocumentFilter.model_validate(flt), owner=owner, folder_id=folder_id)
    return sorted(rev[g] for g in got if g in rev)


def test_schema(itest_db):
    from db.db import get_engine
    with get_engine().connect() as c:
        col = c.execute(text("SELECT data_type, is_nullable, column_default FROM information_schema.columns "
                             "WHERE table_name = 'Files' AND column_name = 'metadata'")).one()
        idx = dict(c.execute(text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'Files' "
                                  "AND indexname LIKE 'idx_files_%_gin'")).all())
    assert col[0] == "jsonb" and col[1] == "NO" and "'{}'" in col[2]
    assert "gin (tags jsonb_path_ops)" in idx["idx_files_tags_gin"]
    assert "gin (metadata jsonb_path_ops)" in idx["idx_files_metadata_gin"]


def test_owner_scope(lib):
    assert _match(lib) == ["acme", "beta", "legacy", "memo", "scan"]
    assert _match(lib, owner=B) == ["other"]
    # naming B's folder (by id or by name) never reaches B's files
    assert _match(lib, folders=[lib.fb.id, lib.fb.name]) == []
    assert _match(lib, folders=[lib.fa2.name]) == ["legacy", "memo", "scan"]
    assert _match(lib, folder_id=lib.fa.id, tags_any=["invoice"]) == ["acme", "beta"]


def test_tags_case_insensitive(lib):
    assert _match(lib, tags_any=["INVOICE"]) == ["acme", "beta"]
    assert _match(lib, tags_any=["signed", "hr"]) == ["acme", "memo"]
    assert _match(lib, tags_all=["invoice", "SIGNED"]) == ["acme"]
    assert _match(lib, tags_all=["invoice", "draft"]) == []


def test_file_type_dates_and_names(lib):
    assert _match(lib, file_type=["pdf", ".DOCX"]) == ["acme", "beta"]
    assert _match(lib, file_type=["image/*"]) == ["scan"]
    assert _match(lib, file_type=["text/markdown"]) == ["memo"]
    assert _match(lib, uploaded_after="2024-01-01", uploaded_before="2025-01-01") == ["acme", "legacy", "memo"]
    assert _match(lib, uploaded_after="2025-01-02") == ["scan"]          # inclusive
    assert _match(lib, name_contains="Q3") == ["beta"]
    assert _match(lib, name_contains="_") == ["beta"]                    # wildcard escaped


def test_metadata(lib):
    assert _match(lib, metadata={"client": "Acme"}) == ["acme"]
    assert _match(lib, metadata={"client": "acme"}) == []                # values are exact
    assert _match(lib, metadata={"year": 2024}) == ["acme"]              # the number, not "2024"
    assert _match(lib, metadata={"year": "2024"}) == ["memo"]
    assert _match(lib, metadata={"client": {"in": ["Beta", "Zeta"]}}) == ["beta"]
    assert _match(lib, metadata={"year": {"gte": 2024}}) == ["acme"]     # memo's "2024" is a string
    assert _match(lib, metadata={"year": {"gte": 2023, "lte": 2023.5}}) == ["beta"]
    assert _match(lib, metadata={"due": {"lte": "2024-01-01"}}) == ["beta"]
    assert _match(lib, metadata={"due": {"gte": "2024-01-01"}}) == ["acme"]
    assert _match(lib, metadata={"paid": True}) == ["acme"]
    assert _match(lib, metadata={"due": {"exists": True}}) == ["acme", "beta"]
    assert _match(lib, metadata={"due": {"exists": False}}) == ["legacy", "memo", "scan"]
    assert _match(lib, tags_any=["invoice"], metadata={"year": {"lte": 2023}}) == ["beta"]


def test_file_ids(lib):
    ids = lib.ids
    assert _match(lib, file_ids=[ids["acme"], ids["memo"]]) == ["acme", "memo"]
    # another person's file id never matches, whatever the filter names
    assert _match(lib, file_ids=[ids["other"]]) == []
    assert _match(lib, owner=B, file_ids=[ids["other"], ids["acme"]]) == ["other"]
    assert _match(lib, file_ids=[ids["acme"], ids["beta"]], tags_all=["signed"]) == ["acme"]
    assert _match(lib, folder_id=lib.fa2.id, file_ids=[ids["acme"]]) == []
    assert _match(lib, file_ids=[str(uuid.uuid4())]) == []


def test_injection_shaped_input_is_just_data(lib):
    from db.db import get_engine
    evil = "x'); DELETE FROM \"Files\"; --"
    assert _match(lib, tags_any=[evil], name_contains=evil, folders=[evil],
                  metadata={"client": evil, "year": {"in": [evil]}}) == []
    with get_engine().connect() as c:
        assert c.execute(text('SELECT count(*) FROM "Files" WHERE folder_id = :f'),
                         {"f": lib.fa.id}).scalar() == 2


def test_listing_helpers(lib):
    from db.db import get_engine
    from src.domain.rag import doc_filter as DF
    with get_engine().connect() as c:
        items, total = DF.search_documents(c, DF.DocumentFilter(tags_any=["invoice"]), owner=A,
                                           sort="name", order="asc", limit=1)
        assert total == 2 and [i["name"] for i in items] == ["acme-2024.pdf"]
        assert items[0]["folder"] == {"id": lib.fa.id, "name": lib.fa.name}
        assert items[0]["status"] == "unindexed" and items[0]["metadata"]["client"] == "Acme"
        page2, _ = DF.search_documents(c, DF.DocumentFilter(tags_any=["invoice"]), owner=A,
                                       sort="name", order="asc", limit=1, offset=1)
        assert [i["name"] for i in page2] == ["beta_q3.docx"]
        items, total = DF.search_documents(c, None, owner=A, sort="size", order="desc", limit=200)
        assert total == 5 and items[0]["name"] == "acme-2024.pdf"          # size = name length, 13
        assert {i["name"]: i["tags"] for i in items}["legacy.txt"] == []
        tags = {t["tag"]: t["count"] for t in DF.tag_counts(c, owner=A)}
        # one group of 2 documents; the 1:1 tie between spellings goes to byte order
        assert tags == {"Invoice": 2, "Signed": 1, "draft": 1, "HR": 1}
        keys = {k["key"]: k for k in DF.metadata_keys(c, owner=A)}
        assert keys["year"]["types"] == {"number": 2, "string": 1}
        assert keys["due"]["types"] == {"date": 2} and keys["paid"]["types"] == {"boolean": 1}
        assert "other" not in {i["name"] for i in items}


# --------------------------------------------------- filtered dense / BM25 SQL

DIM, ROWS, TARGET = 8, 4000, 12


@pytest.fixture()
def vec_table(itest_db):
    from db.db import get_engine
    table = f"data_990201_{uuid.uuid4().hex[:8]}"
    rnd = random.Random(7)
    rows = []
    for i in range(ROWS):
        fid = "target" if i % (ROWS // TARGET) == 0 else f"f{i % 50}"
        vec = "[" + ",".join(f"{rnd.uniform(-1, 1):.4f}" for _ in range(DIM)) + "]"
        word = "needle" if fid == "target" or i % 7 == 0 else "hay"
        rows.append({"t": f"{word} {i}", "m": f'{{"file_id": "{fid}", "node_role": "leaf"}}',
                     "n": f"n{i}", "e": vec, "w": word})
    with get_engine().begin() as c:
        c.execute(text(f'CREATE TABLE "{table}" (id BIGSERIAL PRIMARY KEY, text VARCHAR NOT NULL, '
                       f'metadata_ JSON, node_id VARCHAR, embedding vector({DIM}), text_search_tsv TSVECTOR)'))
        c.execute(text(f'INSERT INTO "{table}" (text, metadata_, node_id, embedding, text_search_tsv) '
                       "VALUES (:t, :m, :n, CAST(:e AS vector), to_tsvector('simple', :w))"), rows)
        c.execute(text(f'CREATE INDEX ON "{table}" USING hnsw (embedding vector_cosine_ops)'))
        c.execute(text(f'ANALYZE "{table}"'))
    yield table
    with get_engine().begin() as c:
        c.execute(text(f'DROP TABLE IF EXISTS "{table}"'))


def _async_engine():
    """A NullPool asyncpg engine of its own: each async test runs on its own event loop."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool
    from src.config.config_manager import Config
    from src.utils.db_url import resolve_db_url
    url = resolve_db_url(Config.get_database_config()).replace(
        "postgresql://", "postgresql+asyncpg://", 1)
    return create_async_engine(url, poolclass=NullPool)


def _searcher(table):
    from src.domain.rag.hybrid_search import HybridSearcher
    s = HybridSearcher(SimpleNamespace(table_name=table), index=SimpleNamespace())
    s.embed_query = AsyncMock(return_value=[0.3] * DIM)
    return s


async def test_hnsw_alone_comes_back_short_iterative_scan_fills_k(vec_table):
    """The baseline proves the test is meaningful (the index is used and the filter is
    selective); the searcher's filtered query returns all k."""
    from src.domain.rag.hybrid_search import _dense_statement, _vector_literal
    eng = _async_engine()
    params = {"emb": _vector_literal([0.3] * DIM), "k": 10, "file_ids": ["target"]}
    async with eng.begin() as conn:
        await conn.execute(text("SET LOCAL hnsw.ef_search = 40"))
        await conn.execute(text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join(r[0] for r in (await conn.execute(
            text("EXPLAIN " + str(_dense_statement(vec_table, True))), params)).fetchall())
        baseline = (await conn.execute(_dense_statement(vec_table, True), params)).fetchall()
    assert "hnsw" in plan.lower() or "Index Scan" in plan
    assert len(baseline) < 10

    with patch("src.domain.rag.hybrid_search.HybridSearcher._dense_engine",
               return_value=eng):
        out = await _searcher(vec_table).dense("q", 10, file_ids=("target",))
    await eng.dispose()
    assert len(out) == 10
    assert {n.node.metadata["file_id"] for n in out} == {"target"}
    scores = [n.score for n in out]
    assert scores == sorted(scores, reverse=True)


async def test_unfiltered_dense_unchanged(vec_table):
    eng = _async_engine()
    with patch("src.domain.rag.hybrid_search.HybridSearcher._dense_engine", return_value=eng):
        out = await _searcher(vec_table).dense("q", 10)
    await eng.dispose()
    assert len(out) == 10 and len({n.node.metadata["file_id"] for n in out}) > 1


async def test_filtered_bm25(vec_table):
    from src.domain.rag.hybrid_search import SearchSettings
    s = _searcher(vec_table)
    s.folder_segmenter = AsyncMock(return_value=None)
    s.segment_query = AsyncMock(return_value="needle")
    st = SearchSettings(text_search_config="simple")
    eng = _async_engine()
    with patch("db.db.get_async_engine", return_value=eng):
        out = await s.sparse("needle", 50, st, file_ids=("target",))
        everyone = await s.sparse("needle", 50, st)
        nobody = await s.sparse("needle", 50, st, file_ids=("nobody",))
    await eng.dispose()
    assert len(out) == len(range(0, ROWS, ROWS // TARGET)) == 13
    assert {n.node.metadata["file_id"] for n in out} == {"target"}
    assert len(everyone) == 50                                          # unfiltered: every file
    assert nobody == []
