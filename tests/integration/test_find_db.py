"""Integration test -- folio_find and folder searches limited to documents, on real
PostgreSQL + pgvector.

Person A has two folders (one with a contract and a tea note, one with a tea note);
person B has one folder whose text matches the query best. Checks folio_find with a
query (documents across A's folders only; B's folder named is "not found"), without a
query (filters, folders, the vocabulary, owner isolation), the scope of a folder
search (ids of another folder ignored, a filter intersected) and that the scope
reaches both the dense and the BM25 SQL of a real search.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text

A, B = "sub:itest-find-a", "sub:itest-find-b"
DIM = 4
TEA = [1.0, 0.0, 0.0, 0.0]


def _vec(v):
    return "[" + ",".join(str(x) for x in v) + "]"


def _file(folder, name, tags=(), meta=None, when="2024-06-01"):
    from datetime import datetime
    from db.filedb import FileDB
    return str(FileDB.create(folder_id=folder.id, file_name=name,
                             file_path=f"/tmp/find/{uuid.uuid4().hex}", file_size=10,
                             mime_type="text/markdown", tags=list(tags), doc_metadata=meta or {},
                             upload_time=datetime.fromisoformat(f"{when}T10:00:00")).id)


def _table(folder, chunks):
    from db.db import get_engine
    table = f"data_{folder.id}_{str(folder.vector_table_uuid).replace('-', '')}"
    with get_engine().begin() as c:
        c.execute(text(f'CREATE TABLE "{table}" (id BIGSERIAL PRIMARY KEY, text VARCHAR NOT NULL, '
                       f'metadata_ JSON, node_id VARCHAR, embedding vector({DIM}), '
                       'text_search_tsv TSVECTOR)'))
        for fid, fname, body, vec in chunks:
            c.execute(text(
                f'INSERT INTO "{table}" (text, metadata_, node_id, embedding, text_search_tsv) '
                "VALUES (:t, :m, :n, CAST(:e AS vector), to_tsvector('simple', :t))"),
                {"t": body, "n": str(uuid.uuid4()), "e": _vec(vec),
                 "m": json.dumps({"file_id": fid, "file_name": fname, "node_role": "leaf"})})
    return table


def _indexed(folder, fid):
    from db.db import get_engine
    with get_engine().begin() as c:
        c.execute(text(
            'INSERT INTO "FileIndices" (id, file_id, folder_id, index_id, status, embedding_model, '
            "embedding_dimension) VALUES (:i, :f, :d, 'x', 'indexed', 'emb-a', :dim)"),
            {"i": str(uuid.uuid4()), "f": fid, "d": folder.id, "dim": DIM})


@pytest.fixture()
def lib(itest_db):
    from db.db import get_engine
    from db.folderdb import FolderDB
    s = uuid.uuid4().hex[:6]
    work = FolderDB.create(name=f"work-{s}", user_token=A)
    home = FolderDB.create(name=f"home-{s}", user_token=A)
    theirs = FolderDB.create(name=f"theirs-{s}", user_token=B)
    ids = {
        "contract": _file(work, "contract.md", ["Signed"], {"year": 2024, "client": "Acme"}),
        "tea_w": _file(work, "tea-work.md", ["drinks"], {"year": 2023}, when="2023-01-01"),
        "tea_h": _file(home, "tea-home.md", ["Drinks"], {}, when="2025-01-01"),
        "tea_b": _file(theirs, "tea-theirs.md", ["drinks", "secret"], {"year": 2024}),
    }
    tables = [
        _table(work, [(ids["contract"], "contract.md", "contract termination clause tea", [0, 0, 1, 0]),
                      (ids["tea_w"], "tea-work.md", "green tea harvest notes", [0.9, 0.1, 0, 0]),
                      (ids["tea_w"], "tea-work.md", "tea storage tips", [0.8, 0.2, 0, 0])]),
        _table(home, [(ids["tea_h"], "tea-home.md", "black tea brewing", [0.95, 0, 0.05, 0])]),
        _table(theirs, [(ids["tea_b"], "tea-theirs.md", "tea tea tea secret", [1, 0, 0, 0])]),
    ]
    for f, k in ((work, "contract"), (work, "tea_w"), (home, "tea_h"), (theirs, "tea_b")):
        _indexed(f, ids[k])
    yield SimpleNamespace(ids=ids, work=work, home=home, theirs=theirs)
    with get_engine().begin() as c:
        for t in tables:
            c.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
        fids = [work.id, home.id, theirs.id]
        c.execute(text('DELETE FROM "FileIndices" WHERE folder_id = ANY(:i)'), {"i": fids})
        c.execute(text('DELETE FROM "Files" WHERE folder_id = ANY(:i)'), {"i": fids})
        c.execute(text('DELETE FROM "Folders" WHERE id = ANY(:i)'), {"i": fids})


def _async_engine():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool
    from src.config.config_manager import Config
    from src.utils.db_url import resolve_db_url
    url = resolve_db_url(Config.get_database_config()).replace(
        "postgresql://", "postgresql+asyncpg://", 1)
    return create_async_engine(url, poolclass=NullPool)


class _Seg:
    name = "ws"

    def segment(self, t):
        return t


def _searcher(folder, embed=TEA):
    from src.domain.rag.hybrid_search import HybridSearcher
    table = f"data_{folder.id}_{str(folder.vector_table_uuid).replace('-', '')}"
    s = HybridSearcher(SimpleNamespace(table_name=table), index=SimpleNamespace())
    s.embed_query = AsyncMock(return_value=embed)
    return s


@pytest.fixture()
def find(lib):
    """``find(owner, **args)``: doc_find.find wired to the real DB and a real
    cross-folder search (stub embedding, no reranker)."""
    from db.db import get_engine
    from src.domain.rag import doc_find as FIND
    from src.domain.rag import doc_search as DS
    from src.domain.rag.hybrid_search import SearchSettings

    eng = _async_engine()
    searchers = {}

    def searcher_for(folder):
        if folder.id not in searchers:
            searchers[folder.id] = _searcher(folder)
        return searchers[folder.id]

    async def search_across(req, *, owner):
        deps = DS.SearchDeps(connect=lambda: get_engine().connect(), searcher_for=searcher_for,
                             query_one=AsyncMock(side_effect=AssertionError("one folder")),
                             settings=SearchSettings(top_k=5, text_search_config="simple"),
                             merge_threshold=1e9, current_embedding=("emb-a", DIM))
        return await DS.search_across(req, owner=owner, deps=deps)

    async def go(owner=A, budget=6000, **args):
        with patch("src.domain.rag.hybrid_search.HybridSearcher._dense_engine", return_value=eng), \
                patch("db.db.get_async_engine", return_value=eng), \
                patch("src.domain.rag.text_segmenter.get_segmenter", return_value=_Seg()):
            return await FIND.find(owner=owner, search_across=search_across,
                                   connect=lambda: get_engine().connect(),
                                   tool_of=lambda n: f"Folio_{n}", budget=budget,
                                   run_sync=asyncio.to_thread, **args)
    return go


def _names(out):
    return [d["name"] for d in out["documents"]]


async def test_find_with_query_is_owner_scoped(lib, find):
    out = await find(query="tea")
    assert set(_names(out)) == {"tea-work.md", "tea-home.md", "contract.md"}
    assert "tea-theirs.md" not in _names(out)
    tea = next(d for d in out["documents"] if d["name"] == "tea-work.md")
    assert tea["hits"] == 2 and tea["tool"] == f"Folio_{lib.work.name}" and tea["tags"] == ["drinks"]
    assert 1 <= len(tea["snippets"]) <= 2
    theirs = await find(owner=B, query="tea")
    assert _names(theirs) == ["tea-theirs.md"]


async def test_find_with_query_filters_and_folders(lib, find):
    from src.domain.rag.doc_find import FindError
    out = await find(query="tea", filters={"tags_any": ["DRINKS"]})
    assert sorted(_names(out)) == ["tea-home.md", "tea-work.md"]
    out = await find(query="tea", folders=[lib.home.name])
    assert _names(out) == ["tea-home.md"]
    with pytest.raises(FindError) as e:
        await find(query="tea", folders=[lib.theirs.name])
    assert e.value.code == "folder_not_found"


async def test_listing(lib, find):
    from src.domain.rag.doc_find import FindError
    out = await find()
    assert _names(out) == ["tea-home.md", "contract.md", "tea-work.md"]     # newest first
    assert out["total"] == 3 and out["vocabulary"]["tags"][0] == {"tag": "Drinks", "count": 2}   # a tie: byte order
    assert {k["key"] for k in out["vocabulary"]["metadata_keys"]} == {"year", "client"}
    out = await find(filters={"metadata": {"year": {"gte": 2024}}})
    assert _names(out) == ["contract.md"] and "vocabulary" not in out
    out = await find(folders=[lib.work.id], limit=1)
    assert out["total"] == 2 and out["next_offset"] == 1 and _names(out) == ["contract.md"]
    out = await find(folders=[lib.work.id], limit=1, offset=1)
    assert _names(out) == ["tea-work.md"] and out["next_offset"] is None
    with pytest.raises(FindError):
        await find(folders=[lib.theirs.id])
    # B's listing and vocabulary are B's only
    out = await find(owner=B)
    assert _names(out) == ["tea-theirs.md"]
    assert {t["tag"] for t in out["vocabulary"]["tags"]} == {"drinks", "secret"}
    # a filter naming B's folder narrows A's listing to nothing
    assert (await find(filters={"folders": [lib.theirs.name]}))["documents"] == []


def test_search_scope(lib):
    from db.db import get_engine
    from src.domain.rag.doc_find import FindError, search_scope
    ids = lib.ids
    with get_engine().connect() as c:
        s = search_scope(c, folder_id=lib.work.id, owner=A,
                         file_ids=[ids["contract"], ids["tea_h"], ids["tea_b"]])
        assert s["file_ids"] == (ids["contract"],)
        assert sorted(s["ignored_file_ids"]) == sorted([ids["tea_h"], ids["tea_b"]])
        with pytest.raises(FindError):
            search_scope(c, folder_id=lib.work.id, owner=A, file_ids=[ids["tea_b"]])
        s = search_scope(c, folder_id=lib.work.id, owner=A, filters={"tags_any": ["signed"]})
        assert s["file_ids"] == (ids["contract"],)
        s = search_scope(c, folder_id=lib.work.id, owner=A,
                         file_ids=[ids["contract"], ids["tea_w"]], filters={"tags_any": ["drinks"]})
        assert s["file_ids"] == (ids["tea_w"],)
        # the filter is scoped to the caller too: B's key matches nothing in A's folder
        s = search_scope(c, folder_id=lib.work.id, owner=B, filters={"tags_any": ["signed"]})
        assert s["file_ids"] == ()


async def test_scope_reaches_dense_and_bm25(lib):
    """The contract is the only document in the scope. Without the scope both lists
    rank the tea notes first (closest vector, more 'tea' text); with it, neither the
    dense nor the BM25 candidates may bring a tea note back."""
    from src.domain.rag.hybrid_search import SearchSettings

    eng = _async_engine()
    s = _searcher(lib.work)
    base = SearchSettings(top_k=10, text_search_config="simple")
    with patch("src.domain.rag.hybrid_search.HybridSearcher._dense_engine", return_value=eng), \
            patch("db.db.get_async_engine", return_value=eng), \
            patch("src.domain.rag.text_segmenter.get_segmenter", return_value=_Seg()):
        everything = await s.search("tea", base)
        scoped = await s.search("tea", base.with_overrides(file_ids=(lib.ids["contract"],)))
        dense = await s.dense("tea", 10, file_ids=(lib.ids["contract"],))
        sparse = await s.sparse("tea", 10, base, file_ids=(lib.ids["contract"],))

    def fids(nodes):
        return {(getattr(n, "node", n).metadata or {}).get("file_id") for n in nodes}

    assert fids(everything.nodes) == {lib.ids["contract"], lib.ids["tea_w"]}
    assert fids(scoped.nodes) == {lib.ids["contract"]}
    assert fids(dense) == {lib.ids["contract"]} and fids(sparse) == {lib.ids["contract"]}
