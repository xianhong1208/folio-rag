"""folio_find's logic and the folder-search scope (src/domain/rag/doc_find.py).

Pins: a query goes to the cross-folder document search with the caller's owner and
the limit as top_k; no query is a listing (folders resolved in the owner scope, the
vocabulary only on an overview call); refusals name the bad key; the compact shape
(the folder tool to use next, short snippets); the token-budget trim; and the scope
of a folder search (ids of other folders ignored, none left refused, a filter
intersected).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.rag import doc_filter as DF
from src.domain.rag import doc_find as FIND
from src.domain.rag import doc_search as DS

F1 = "11111111-1111-1111-1111-111111111111"
F2 = "22222222-2222-2222-2222-222222222222"
F3 = "33333333-3333-3333-3333-333333333333"


def tool_of(name):
    return f"Folio_{name}"


async def _sync(fn):
    return fn()


def _doc(fid, name="a.md", folder="docs", score=0.9, snippets=None, **kw):
    return {"file_id": fid, "name": name, "folder": {"id": 1, "name": folder}, "mime": "text/markdown",
            "tags": kw.get("tags", ["x"]), "metadata": kw.get("metadata", {"year": 2024}),
            "score": score, "hits": kw.get("hits", 1),
            "snippets": snippets or [{"text": "some text", "score": score, "page": 2,
                                      "headings": ["H"], "content_type": None, "node_id": "n"}]}


def _found(*docs, skipped=()):
    return {"query": "q", "group_by": "document", "score_type": "rerank",
            "documents": list(docs), "skipped_folders": list(skipped), "total_results": len(docs)}


async def _find(search=None, connect=None, budget=0, **kw):
    search = search or AsyncMock(return_value=_found())
    out = await FIND.find(owner=kw.pop("owner", "sub:a"), search_across=search,
                          connect=connect or MagicMock(), tool_of=tool_of, budget=budget,
                          run_sync=_sync, **kw)
    return out, search


class TestArguments:
    def test_bad_filter_key_is_named(self):
        with pytest.raises(FIND.FindError) as e:
            FIND.parse_filters({"tags_anyy": ["x"]})
        assert e.value.code == "invalid_filter" and "tags_anyy" in e.value.detail
        with pytest.raises(FIND.FindError) as e:
            FIND.parse_filters({"metadata": {"year": {"gt": 1}}})
        assert "gt" in e.value.detail
        with pytest.raises(FIND.FindError):
            FIND.parse_filters(["tags"])

    def test_query_and_folders(self):
        assert FIND.parse_query(None) == "" and FIND.parse_query("  hi ") == "hi"
        for bad in ("x", "y" * 2001, 5):
            with pytest.raises(FIND.FindError):
                FIND.parse_query(bad)
        for bad in ([], "docs", [True], [1.5], list(range(101))):
            with pytest.raises(FIND.FindError):
                FIND.parse_folders(bad)
        assert FIND.parse_folders(["a", 3]) == ["a", 3]

    def test_clamp(self):
        assert FIND.clamp(None, 10, 1, 50) == 10
        assert FIND.clamp(500, 10, 1, 50) == 50 and FIND.clamp(0, 10, 1, 50) == 1
        assert FIND.clamp("x", 10, 1, 50) == 10


class TestFindWithQuery:
    async def test_document_search_with_owner_and_limit(self):
        flt = {"tags_any": ["Signed"]}
        out, search = await _find(query=" payment terms ", filters=flt, folders=["docs"], limit=99)
        req = search.await_args.args[0]
        assert search.await_args.kwargs["owner"] == "sub:a"
        assert req.group_by == DS.GROUP_DOCUMENT and req.top_k == FIND.MAX_LIMIT
        assert req.query == "payment terms" and req.folders == ["docs"]
        assert req.filters.tags_any == ["Signed"] and req.snippets == FIND.FIND_SNIPPETS
        assert out["mode"] == "find"

    async def test_default_limit(self):
        _, search = await _find(query="tea")
        assert search.await_args.args[0].top_k == FIND.DEFAULT_LIMIT

    async def test_foreign_folder_is_not_found(self):
        search = AsyncMock(side_effect=DS.FolderNotFound("theirs"))
        with pytest.raises(FIND.FindError) as e:
            await _find(search=search, query="tea", folders=["theirs"])
        assert e.value.code == "folder_not_found" and "theirs" in e.value.detail

    async def test_bad_filter_never_searches(self):
        search = AsyncMock()
        with pytest.raises(FIND.FindError):
            await _find(search=search, query="tea", filters={"nope": 1})
        assert not search.await_count

    async def test_owner_is_required(self):
        for owner in (None, ""):
            with pytest.raises(ValueError):
                await _find(owner=owner, query="tea")

    async def test_shape(self):
        long = "word " * 2000
        search = AsyncMock(return_value=_found(
            _doc(F1, snippets=[{"text": long, "score": 0.91234, "page": None},
                               {"text": "b", "score": 0.5, "page": 3}]),
            skipped=[{"id": 7, "name": "old", "reason": "embedding_model", "detail": "…"}]))
        out, _ = await _find(search=search, query="tea")
        d = out["documents"][0]
        assert d["tool"] == "Folio_docs" and d["folder"] == "docs" and d["file_id"] == F1
        assert d["snippets"][0]["score"] == 0.9123 and "page" not in d["snippets"][0]
        assert d["snippets"][0]["text"].endswith("…") and len(d["snippets"][0]["text"]) < 2000
        assert d["snippets"][1] == {"text": "b", "page": 3, "score": 0.5}
        assert out["skipped_folders"] == [{"folder": "old", "reason": "embedding_model"}]
        assert "file_ids" in out["_hint"]

    async def test_nothing_found(self):
        out, _ = await _find(query="tea")
        assert out["documents"] == [] and "Rephrase" in out["_hint"]
        assert "skipped_folders" not in out


class TestListing:
    @pytest.fixture()
    def db(self):
        rows = [{"id": F1, "name": "a.pdf", "folder": {"id": 1, "name": "docs"}, "size_bytes": 1,
                 "mime": "application/pdf", "uploaded_at": "2026-01-01T00:00:00",
                 "updated_at": None, "status": "indexed", "tags": ["signed"],
                 "metadata": {"year": 2024}, "description": "d" * 500}]
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        with patch.object(DF, "search_documents", return_value=(rows, 3)) as sd, \
             patch.object(FIND, "vocabulary", return_value={"tags": [{"tag": "signed", "count": 1}],
                                                            "more_tags": 0, "metadata_keys": [],
                                                            "more_metadata_keys": 0}) as voc, \
             patch.object(DS, "owned_folders",
                          side_effect=lambda c, o, w=None: [DS.FolderRef(4, "docs", "u", o)]
                          if w is not None else [DS.FolderRef(4, "docs", "u", o),
                                                 DS.FolderRef(5, "notes", "u", o)]) as owned:
            yield SimpleNamespace(conn=conn, sd=sd, voc=voc, owned=owned)

    async def test_overview_lists_newest_first_with_vocabulary(self, db):
        out, search = await _find(connect=lambda: db.conn)
        assert not search.await_count
        args, kw = db.sd.call_args
        assert kw["owner"] == "sub:a" and kw["sort"] == "uploaded" and kw["order"] == "desc"
        assert kw["limit"] == FIND.DEFAULT_LIMIT and kw["offset"] == 0 and args[1] is None
        assert out["mode"] == "list" and out["total"] == 3 and out["next_offset"] == 1
        d = out["documents"][0]
        assert d["tool"] == "Folio_docs" and d["file_id"] == F1 and d["tags"] == ["signed"]
        assert len(d["description"]) == FIND.DESCRIPTION_MAX_CHARS
        assert out["vocabulary"]["tags"] == [{"tag": "signed", "count": 1}]

    async def test_filtered_listing_has_no_vocabulary(self, db):
        out, _ = await _find(connect=lambda: db.conn, filters={"tags_all": ["signed"]},
                             limit=2, offset=2)
        kw = db.sd.call_args.kwargs
        assert kw["limit"] == 2 and kw["offset"] == 2
        assert db.sd.call_args.args[1].tags_all == ["signed"]
        assert "vocabulary" not in out and out["next_offset"] is None
        assert not db.voc.called

    async def test_folders_become_owned_folder_ids(self, db):
        await _find(connect=lambda: db.conn, folders=["docs"])
        assert db.owned.call_args_list[0].args[1] == "sub:a"
        assert db.sd.call_args.args[1].folders == [4]

    async def test_folders_and_filter_folders_both_narrow(self, db):
        await _find(connect=lambda: db.conn, folders=["docs"], filters={"folders": ["notes"]})
        assert db.sd.call_args.args[1].folders == []
        await _find(connect=lambda: db.conn, folders=["docs"], filters={"folders": [4]})
        assert db.sd.call_args.args[1].folders == [4]

    async def test_foreign_folder_is_not_found(self, db):
        db.owned.side_effect = DS.FolderNotFound("theirs")
        with pytest.raises(FIND.FindError) as e:
            await _find(connect=lambda: db.conn, folders=["theirs"])
        assert e.value.code == "folder_not_found"
        assert not db.sd.called


class TestBudget:
    def _resp(self, n, text="t"):
        docs = [{"file_id": str(i), "name": f"{i}.md", "snippets": [{"text": text}, {"text": text}]}
                for i in range(n)]
        return {"mode": "find", "total_results": n, "documents": docs, "_hint": "h"}

    def test_drops_lowest_ranked(self):
        r = FIND.fit_documents_to_budget(self._resp(40, "x" * 300), 600)
        assert 0 < r["total_results"] < 40 and r["omitted_results"] == 40 - r["total_results"]
        assert [d["file_id"] for d in r["documents"]] == [str(i) for i in range(r["total_results"])]
        assert r["response_tokens_estimated"] <= 600 and "left out" in r["_hint"]
        assert json.loads(json.dumps(r))

    def test_first_document_too_big_keeps_it_without_snippets(self):
        r = FIND.fit_documents_to_budget(self._resp(3, "x" * 4000), 300)
        assert r["total_results"] == 1 and r["documents"][0]["snippets_truncated"] is True
        assert r["documents"][0]["snippets"] == [] and r["omitted_results"] == 2

    def test_fits_or_disabled(self):
        r = FIND.fit_documents_to_budget(self._resp(3), 6000)
        assert r["total_results"] == 3 and "omitted_results" not in r
        r = FIND.fit_documents_to_budget(self._resp(40, "x" * 300), 0)
        assert r["total_results"] == 40

    async def test_find_applies_the_budget(self):
        docs = [_doc(f"{i:08d}-0000-0000-0000-000000000000", name=f"{i}.md",
                     snippets=[{"text": "w " * 200, "score": 0.5}]) for i in range(10)]
        out, _ = await _find(search=AsyncMock(return_value=_found(*docs)), query="tea", budget=500)
        assert out["omitted_results"] > 0 and out["response_tokens_estimated"] <= 500


class TestVocabulary:
    def test_top_n_and_text(self):
        tags = [{"tag": f"t{i}", "count": 30 - i} for i in range(30)]
        keys = [{"key": "year", "count": 3, "types": {"number": 3}},
                {"key": "due", "count": 1, "types": {"date": 1, "string": 1}}]
        with patch.object(DF, "tag_counts", return_value=tags) as tc, \
             patch.object(DF, "metadata_keys", return_value=keys):
            v = FIND.vocabulary("conn", owner="sub:a")
        assert tc.call_args.kwargs["owner"] == "sub:a"
        assert len(v["tags"]) == FIND.VOCAB_TAGS and v["more_tags"] == 5
        assert v["metadata_keys"][1] == {"key": "due", "types": ["date", "string"]}
        text = FIND.vocabulary_text(v)
        assert "t0 (30)" in text and "and 5 more" in text and "due (date/string)" in text

    def test_empty(self):
        assert FIND.vocabulary_text({"tags": [], "metadata_keys": []}) == ""


class TestSearchScope:
    def _conn(self, in_folder=(F1,), matched=(F1, F2)):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [(x,) for x in in_folder]
        patcher = patch.object(DF, "matching_file_ids", return_value=list(matched))
        return conn, patcher

    def test_nothing_asked_is_the_whole_folder(self):
        conn, _ = self._conn()
        assert FIND.search_scope(conn, folder_id=1, owner="sub:a") is None
        assert FIND.search_scope(conn, folder_id=1, owner="sub:a", filters={}) is None
        assert not conn.execute.called

    def test_ids_of_other_folders_are_ignored(self):
        conn, _ = self._conn(in_folder=(F1,))
        s = FIND.search_scope(conn, folder_id=9, owner="sub:a", file_ids=[F1, F2, "junk"])
        assert s == {"file_ids": (F1,), "ignored_file_ids": ["junk", F2]}
        sql, params = conn.execute.call_args.args
        assert "f.folder_id = :fid" in str(sql) and params == {"fid": 9, "ids": [F1, F2]}

    def test_no_id_left_is_refused(self):
        conn, _ = self._conn(in_folder=())
        with pytest.raises(FIND.FindError) as e:
            FIND.search_scope(conn, folder_id=1, owner="sub:a", file_ids=[F3])
        assert "none of file_ids" in e.value.detail
        conn2, _ = self._conn()
        with pytest.raises(FIND.FindError):
            FIND.search_scope(conn2, folder_id=1, owner="sub:a", file_ids=["junk"])
        assert not conn2.execute.called
        for bad in ([], [F1] * 101, {"a": 1}):
            with pytest.raises(FIND.FindError):
                FIND.search_scope(conn2, folder_id=1, owner="sub:a", file_ids=bad)

    def test_filter_is_scoped_and_intersected(self):
        conn, patcher = self._conn(in_folder=(F1, F3))
        with patcher as m:
            s = FIND.search_scope(conn, folder_id=4, owner="sub:a", file_ids=[F1, F3],
                                  filters={"tags_any": ["x"]})
            assert s["file_ids"] == (F1,)
            assert m.call_args.kwargs == {"owner": "sub:a", "folder_id": 4}
            s = FIND.search_scope(conn, folder_id=4, owner="sub:a", filters={"tags_any": ["x"]})
            assert s["file_ids"] == (F1, F2)
            m.return_value = []
            s = FIND.search_scope(conn, folder_id=4, owner="sub:a", filters={"tags_any": ["x"]})
            assert s == {"file_ids": (), "ignored_file_ids": []}

    def test_a_single_id_string_and_a_bad_filter(self):
        conn, _ = self._conn(in_folder=(F1,))
        assert FIND.search_scope(conn, folder_id=1, owner="o", file_ids=F1)["file_ids"] == (F1,)
        with pytest.raises(FIND.FindError) as e:
            FIND.search_scope(conn, folder_id=1, owner="o", filters={"bogus": 1})
        assert e.value.code == "invalid_filter" and "bogus" in e.value.detail

