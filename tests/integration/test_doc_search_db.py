"""Integration test -- search across folders and results by document on real
PostgreSQL + pgvector.

Person A has four folders: two searchable ones, one indexed with another embedding
model and one with nothing indexed; person B has one folder whose text matches the
query best. Checks the owner scope (B's folder is never searched, and naming it is
"not found"), the skip reasons, a filter across folders, MaxP documents with names
read live (a folder rename shows even on a cached result), passages that carry their
folder, and the folder cap. Each folder keeps its own vector table; the real dense and
BM25 SQL run against them.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text

A, B = "sub:itest-xf-a", "sub:itest-xf-b"
DIM = 4
CUR = ("emb-a", DIM)
TEA = [1.0, 0.0, 0.0, 0.0]


def _vec(v):
    return "[" + ",".join(str(x) for x in v) + "]"


def _folder(name, owner, updated):
    from db.folderdb import FolderDB
    f = FolderDB.create(name=name, user_token=owner)
    from db.db import get_engine
    with get_engine().begin() as c:
        c.execute(text('UPDATE "Folders" SET updated_at = :u WHERE id = :i'),
                  {"u": updated, "i": f.id})
    return f


def _file(folder, name, tags=()):
    from db.filedb import FileDB
    return str(FileDB.create(folder_id=folder.id, file_name=name,
                             file_path=f"/tmp/xf/{uuid.uuid4().hex}", file_size=10,
                             mime_type="text/markdown", tags=list(tags)).id)


def _table(folder, chunks):
    """A vector table for the folder: chunks = [(file_id, file_name, text, vector)]."""
    from db.db import get_engine
    table = f"data_{folder.id}_{str(folder.vector_table_uuid).replace('-', '')}"
    with get_engine().begin() as c:
        c.execute(text(f'CREATE TABLE "{table}" (id BIGSERIAL PRIMARY KEY, text VARCHAR NOT NULL, '
                       f'metadata_ JSON, node_id VARCHAR, embedding vector({DIM}), '
                       'text_search_tsv TSVECTOR)'))
        for fid, fname, body, vec in chunks:
            import json
            c.execute(text(
                f'INSERT INTO "{table}" (text, metadata_, node_id, embedding, text_search_tsv) '
                "VALUES (:t, :m, :n, CAST(:e AS vector), to_tsvector('simple', :t))"),
                {"t": body, "n": str(uuid.uuid4()), "e": _vec(vec),
                 "m": json.dumps({"file_id": fid, "file_name": fname, "node_role": "leaf"})})
    return table


def _indexed(folder, fid, model="emb-a", dim=DIM):
    from db.db import get_engine
    with get_engine().begin() as c:
        c.execute(text(
            'INSERT INTO "FileIndices" (id, file_id, folder_id, index_id, status, embedding_model, '
            "embedding_dimension) VALUES (:i, :f, :d, 'x', 'indexed', :m, :dim)"),
            {"i": str(uuid.uuid4()), "f": fid, "d": folder.id, "m": model, "dim": dim})


@pytest.fixture()
def lib(itest_db):
    from db.db import get_engine
    s = uuid.uuid4().hex[:6]
    alpha = _folder(f"alpha-{s}", A, "2026-01-03")
    beta = _folder(f"beta-{s}", A, "2026-01-02")
    gamma = _folder(f"gamma-{s}", A, "2026-01-04")
    empty = _folder(f"empty-{s}", A, "2026-01-05")
    bravo = _folder(f"bravo-{s}", B, "2026-01-06")
    ids = {
        "a_tea": _file(alpha, "alpha-tea.md", ["Drinks"]),
        "a_car": _file(alpha, "alpha-car.md"),
        "b_tea": _file(beta, "beta-tea.md", ["drinks"]),
        "b_law": _file(beta, "beta-law.md"),
        "g_tea": _file(gamma, "gamma-tea.md", ["drinks"]),
        "x_tea": _file(bravo, "bravo-tea.md", ["drinks"]),
    }
    tables = [
        _table(alpha, [(ids["a_tea"], "alpha-tea.md", "green tea harvest notes", [0.9, 0.1, 0, 0]),
                       (ids["a_tea"], "alpha-tea.md", "tea storage tips", [0.8, 0.2, 0, 0]),
                       (ids["a_car"], "alpha-car.md", "engine oil change", [0, 1, 0, 0])]),
        _table(beta, [(ids["b_tea"], "beta-tea.md", "black tea brewing guide", [0.95, 0, 0.05, 0]),
                      (ids["b_law"], "beta-law.md", "contract termination clause", [0, 0, 1, 0])]),
        _table(gamma, [(ids["g_tea"], "gamma-tea.md", "tea tea tea", [1, 0, 0, 0])]),
        _table(bravo, [(ids["x_tea"], "bravo-tea.md", "tea tea tea secret", [1, 0, 0, 0])]),
    ]
    for f, k in ((alpha, "a_tea"), (alpha, "a_car"), (beta, "b_tea"), (beta, "b_law"),
                 (bravo, "x_tea")):
        _indexed(f, ids[k])
    _indexed(gamma, ids["g_tea"], model="emb-old", dim=8)
    yield SimpleNamespace(ids=ids, alpha=alpha, beta=beta, gamma=gamma, empty=empty, bravo=bravo)
    with get_engine().begin() as c:
        for t in tables:
            c.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
        fids = [f.id for f in (alpha, beta, gamma, empty, bravo)]
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


@pytest.fixture()
def run(lib):
    from db.db import get_engine
    from src.domain.rag import doc_search as DS
    from src.domain.rag.hybrid_search import HybridSearcher, SearchSettings
    from src.infrastructure.cache.cache_service import InMemoryCache

    eng = _async_engine()
    searchers = {}

    def searcher_for(folder):
        if folder.id not in searchers:
            table = f"data_{folder.id}_{folder.vector_table_uuid.replace('-', '')}"
            s = HybridSearcher(SimpleNamespace(table_name=table), index=SimpleNamespace())
            s.embed_query = AsyncMock(return_value=TEA)
            searchers[folder.id] = s
        return searchers[folder.id]

    cache = DS.CrossFolderCache(InMemoryCache(500, 300), 300)

    async def go(owner=A, max_folders=20, **req):
        deps = DS.SearchDeps(connect=lambda: get_engine().connect(), searcher_for=searcher_for,
                             query_one=AsyncMock(side_effect=AssertionError("one folder")),
                             settings=SearchSettings(top_k=5, text_search_config="simple"),
                             cache=cache, merge_threshold=1e9, max_folders=max_folders,
                             current_embedding=CUR)
        with patch("src.domain.rag.hybrid_search.HybridSearcher._dense_engine", return_value=eng), \
                patch("db.db.get_async_engine", return_value=eng), \
                patch("src.domain.rag.text_segmenter.get_segmenter", return_value=_Seg()):
            return await DS.search_across(DS.SearchRequest(**req), owner=owner, deps=deps)

    go.searchers = searchers
    return go      # the engine uses NullPool: nothing stays open between tests


def _reasons(data):
    return {s["name"].split("-")[0]: s["reason"] for s in data["skipped_folders"]}


async def test_all_folders_documents_owner_scoped(lib, run):
    data = await run(query="tea", group_by="document")
    names = [d["name"] for d in data["documents"]]
    assert set(names) == {"alpha-tea.md", "beta-tea.md", "alpha-car.md", "beta-law.md"}
    assert names[:2] in (["alpha-tea.md", "beta-tea.md"], ["beta-tea.md", "alpha-tea.md"])
    assert "bravo-tea.md" not in names and "gamma-tea.md" not in names
    assert {f["id"] for f in data["folders"]} == {lib.alpha.id, lib.beta.id}
    assert _reasons(data) == {"gamma": "embedding_model", "empty": "nothing_indexed"}
    tea = next(d for d in data["documents"] if d["name"] == "alpha-tea.md")
    assert tea["hits"] == 2 and tea["folder"] == {"id": lib.alpha.id, "name": lib.alpha.name}
    assert tea["tags"] == ["Drinks"] and len(tea["snippets"]) == 2
    assert tea["score"] == max(s["score"] for s in tea["snippets"])
    # One embedding for the whole search.
    assert sum(s.embed_query.await_count for s in run.searchers.values()) == 1


async def test_someone_elses_folder_is_not_found(lib, run):
    from src.domain.rag.doc_search import FolderNotFound
    for name in (lib.bravo.name, lib.bravo.id):
        with pytest.raises(FolderNotFound):
            await run(query="tea", folders=[name], group_by="document")
    # B searching sees only B's folder.
    data = await run(owner=B, query="tea", group_by="document")
    assert [d["name"] for d in data["documents"]] == ["bravo-tea.md"]


async def test_filter_across_folders(lib, run):
    from src.domain.rag.doc_filter import DocumentFilter
    data = await run(query="tea", group_by="document", filters=DocumentFilter(tags_any=["DRINKS"]))
    assert sorted(d["name"] for d in data["documents"]) == ["alpha-tea.md", "beta-tea.md"]
    # gamma's tea file matches the filter but gamma stays skipped (other embedding model)
    assert _reasons(data) == {"gamma": "embedding_model"}


async def test_names_are_live_even_from_cache(lib, run):
    from db.db import get_engine
    first = await run(query="tea", group_by="document", folders=[lib.alpha.id, lib.beta.id])
    assert first["cached"] is False
    with get_engine().begin() as c:
        c.execute(text('UPDATE "Folders" SET name = :n WHERE id = :i'),
                  {"n": lib.alpha.name + "-renamed", "i": lib.alpha.id})
    again = await run(query="tea", group_by="document", folders=[lib.alpha.id, lib.beta.id])
    assert again["cached"] is True
    tea = next(d for d in again["documents"] if d["name"] == "alpha-tea.md")
    assert tea["folder"]["name"] == lib.alpha.name + "-renamed"


async def test_passages_carry_their_folder(lib, run):
    data = await run(query="tea", group_by="passage", top_k=3)
    assert len(data["results"]) == 3
    folders = {r["folder"]["id"] for r in data["results"]}
    assert folders == {lib.alpha.id, lib.beta.id}
    for r in data["results"]:
        assert r["metadata"]["folder_name"] == r["folder"]["name"]
        assert "tea" in r["text"]


async def test_folder_cap_searches_the_most_recent(lib, run):
    data = await run(query="tea", group_by="document", max_folders=1)
    assert [f["id"] for f in data["folders"]] == [lib.alpha.id]
    assert _reasons(data)["beta"] == "folder_limit"
    assert {d["folder"]["id"] for d in data["documents"]} == {lib.alpha.id}


def test_describe_files_is_owner_scoped(lib):
    from db.db import get_engine
    from src.domain.rag.doc_filter import ALL_FOLDERS
    from src.domain.rag.doc_search import describe_files
    ids = [lib.ids["a_tea"], lib.ids["x_tea"], "not-a-uuid"]
    with get_engine().connect() as c:
        assert set(describe_files(c, ids, owner=A)) == {lib.ids["a_tea"]}
        assert set(describe_files(c, ids, owner=B)) == {lib.ids["x_tea"]}
        assert set(describe_files(c, ids, owner=ALL_FOLDERS)) == {lib.ids["a_tea"], lib.ids["x_tea"]}
        assert describe_files(c, ["not-a-uuid"], owner=A) == {}
