"""Search across folders and results by document (src/domain/rag/doc_search.py).

Pins: MaxP document scores (not the sum) with ties broken by hit count, the snippet
limit, rank fusion across folders that ignores raw scores, one rerank over the pooled
candidates (RRF when there is no reranker or the rerank fails), the per-folder pool
depths, the plan (owner scope, filter pruning, skip reasons, folder cap), the cache key
parts and the per-folder invalidation markers, and the shared query work (one
embedding, one segmentation per segmenter).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.domain.rag import doc_search as DS
from src.domain.rag import hybrid_search as hs
from src.domain.rag.doc_filter import ALL_FOLDERS, DocumentFilter
from src.domain.rag.hybrid_search import HybridSearcher, SearchSettings
from src.domain.rag.rerank_policy import RerankOutcome
from src.infrastructure.cache.cache_service import InMemoryCache

F1, F2, F3 = (f"00000000-0000-0000-0000-00000000000{i}" for i in (1, 2, 3))
NA, NB = "aaaaaaaa-0000-0000-0000-000000000001", "bbbbbbbb-0000-0000-0000-000000000001"


def nws(nid, score=0.5, file_id=F1, text=None, **md):
    meta = {"node_role": "leaf", "file_id": file_id, "file_name": f"{file_id[-1]}.md", **md}
    return NodeWithScore(node=TextNode(id_=nid, text=text or f"text {nid}", metadata=meta),
                         score=score)


def ids(nodes):
    return [n.node.node_id for n in nodes]


def ref(i, name=None, owner="sub:a"):
    return DS.FolderRef(i, name or f"folder{i}", f"uuid-{i}", owner)


# ------------------------------------------------------------- documents


class TestGroupDocuments:
    def test_maxp_not_sum(self):
        # B has three middling passages (sum 1.5), A one strong one (0.9): MaxP puts A first.
        nodes = [nws("a1", 0.9, F1), nws("b1", 0.5, F2), nws("b2", 0.5, F2), nws("b3", 0.5, F2)]
        docs = DS.group_documents(nodes, {}, top_k=10)
        assert [d.file_id for d in docs] == [F1, F2]
        assert (docs[0].score, docs[0].hits) == (0.9, 1)
        assert (docs[1].score, docs[1].hits) == (0.5, 3)

    def test_tie_goes_to_more_hits_then_rank(self):
        nodes = [nws("a1", 0.7, F1), nws("b1", 0.7, F2), nws("b2", 0.2, F2), nws("c1", 0.7, F3)]
        docs = DS.group_documents(nodes, {}, top_k=10)
        assert [d.file_id for d in docs] == [F2, F1, F3]

    def test_snippets_best_first_and_limited(self):
        nodes = [nws("a1", 0.4, F1), nws("a2", 0.9, F1), nws("a3", 0.6, F1), nws("a4", 0.8, F1)]
        (doc,) = DS.group_documents(nodes, {}, top_k=10, snippets=2)
        assert [s["node_id"] for s in doc.snippets] == ["a2", "a4"]
        assert doc.hits == 4 and doc.score == 0.9
        (doc,) = DS.group_documents(nodes, {}, top_k=10, snippets=99)
        assert len(doc.snippets) == DS.MAX_SNIPPETS

    def test_snippet_fields_and_folder(self):
        n = nws("a1", 0.5, F1, page_no=3, headings=["H"], content_type="table")
        (doc,) = DS.group_documents([n], {"a1": 7}, top_k=1)
        assert doc.folder_id == 7
        assert doc.snippets == [{"text": "text a1", "score": 0.5, "page": 3, "headings": ["H"],
                                 "content_type": "table", "node_id": "a1"}]

    def test_top_k_and_passages_without_file(self):
        n = nws("x", 0.99)
        n.node.metadata.pop("file_id")
        docs = DS.group_documents([n, nws("a", 0.5, F1), nws("b", 0.4, F2)], {}, top_k=1)
        assert [d.file_id for d in docs] == [F1]

    def test_payload_drops_unknown_files(self):
        hits = DS.group_documents([nws("a", 0.5, F1), nws("b", 0.4, F2)], {}, top_k=5)
        described = {F2: {"name": "two.md", "mime": "text/markdown", "tags": ["t"],
                          "metadata": {"y": 1}, "folder": {"id": 2, "name": "f2"}}}
        out = DS.documents_payload(hits, described)
        assert [d["file_id"] for d in out] == [F2]
        assert out[0]["name"] == "two.md" and out[0]["folder"] == {"id": 2, "name": "f2"}
        assert out[0]["hits"] == 1 and out[0]["snippets"][0]["node_id"] == "b"


# ---------------------------------------------------------------- fusion


class TestFuseFolders:
    def test_rank_based_ignores_raw_scores(self):
        a = [nws("a1", 0.99), nws("a2", 0.98)]
        b = [nws("b1", 0.01), nws("b2", 0.001)]
        out = DS.fuse_folders([a, b])
        # Rank 1 of each folder first (ties keep folder order), then rank 2.
        assert ids(out) == ["a1", "b1", "a2", "b2"]
        assert out[0].score == pytest.approx(1 / 61)

    def test_same_node_in_two_lists_adds_up(self):
        out = DS.fuse_folders([[nws("x"), nws("y")], [nws("y")]])
        assert ids(out)[0] == "y"

    def test_depths(self):
        st = SearchSettings(top_k=10)
        assert DS.folder_depths(st, 1) == hs.per_list_depths(st) == (30, 10)
        assert DS.folder_depths(st, 3) == (20, 7)
        assert DS.folder_depths(st, 20) == (DS.MIN_FOLDER_DENSE, DS.MIN_FOLDER_SPARSE)


# ------------------------------------------------------------------ plan


class FakeConn:
    """Answers the plan's three queries from fixtures (by SQL text)."""

    def __init__(self, folders, indexed=(), matches=()):
        self.folders, self.indexed, self.matches = folders, list(indexed), list(matches)
        self.calls = []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.calls.append((sql, dict(params or {})))
        if 'SELECT f.id, f.folder_id' in sql:
            rows = [r for r in self.matches if r[1] in params["_fids"]]
        elif 'FROM "FileIndices"' in sql:
            rows = [r for r in self.indexed if r[0] in params["ids"]]
        elif sql.startswith('SELECT d.id, d.name'):
            owner = (params or {}).get("owner")
            rows = [(f.id, f.name, f.vector_table_uuid, f.owner) for f in self.folders
                    if owner is None or f.owner == owner]
        else:  # pragma: no cover - unexpected query
            raise AssertionError(sql)
        return SimpleNamespace(fetchall=lambda: rows)


CUR = ("emb-a", 1024)


class TestPlan:
    def _conn(self, **kw):
        folders = [ref(1), ref(2), ref(3), ref(9, "theirs", owner="sub:b")]
        indexed = [(1, "emb-a", 1024, 3), (2, "emb-a", 1024, 2), (2, "emb-old", 768, 1),
                   (9, "emb-a", 1024, 5)]
        return FakeConn(folders, kw.get("indexed", indexed), kw.get("matches", ()))

    def test_owner_scope_and_skip_reasons(self):
        plan = DS.plan_search(self._conn(), owner="sub:a", current_embedding=CUR)
        assert [t.folder.id for t in plan.targets] == [1]
        assert {(s["id"], s["reason"]) for s in plan.skipped} == {
            (2, DS.SKIP_EMBEDDING), (3, DS.SKIP_NOT_INDEXED)}
        # The other person's folder is never seen, not even as skipped.
        assert all(s["id"] != 9 for s in plan.skipped)

    def test_named_folder_of_someone_else_is_not_found(self):
        with pytest.raises(DS.FolderNotFound):
            DS.plan_search(self._conn(), owner="sub:a", folders=["theirs"], current_embedding=CUR)
        with pytest.raises(DS.FolderNotFound):
            DS.plan_search(self._conn(), owner="sub:a", folders=[9], current_embedding=CUR)

    def test_named_folders_by_id_and_name(self):
        plan = DS.plan_search(self._conn(), owner="sub:a", folders=["folder1", 1],
                              current_embedding=CUR)
        assert [t.folder.id for t in plan.targets] == [1] and plan.skipped == []

    def test_open_mode_sees_every_folder(self):
        plan = DS.plan_search(self._conn(), owner=ALL_FOLDERS, current_embedding=CUR)
        assert [t.folder.id for t in plan.targets] == [1, 9]

    def test_owner_required(self):
        with pytest.raises(ValueError):
            DS.owned_folders(self._conn(), None)

    def test_filter_prunes_folders_and_passes_file_ids(self):
        conn = self._conn(indexed=[(1, "emb-a", 1024, 3), (3, "emb-a", 1024, 1)],
                          matches=[(F2, 1), (F1, 1)])
        plan = DS.plan_search(conn, owner="sub:a", flt=DocumentFilter(tags_any=["x"]),
                              current_embedding=CUR)
        assert [(t.folder.id, t.file_ids) for t in plan.targets] == [(1, (F1, F2))]
        assert plan.skipped == []     # folders 2 and 3 have no matching document
        sql, params = next(c for c in conn.calls if "SELECT f.id, f.folder_id" in c[0])
        assert params["owner_0"] == "sub:a"     # the owner scope is in the filter SQL too

    def test_cap_keeps_first_and_lists_rest(self):
        conn = self._conn(indexed=[(1, "emb-a", 1024, 1), (2, "emb-a", 1024, 1),
                                   (3, "emb-a", 1024, 1)])
        plan = DS.plan_search(conn, owner="sub:a", max_folders=2, current_embedding=CUR)
        assert [t.folder.id for t in plan.targets] == [1, 2]
        assert [(s["id"], s["reason"]) for s in plan.skipped] == [(3, DS.SKIP_LIMIT)]

    def test_unknown_embedding_flags_nothing(self):
        conn = self._conn(indexed=[(1, None, None, 2)])
        plan = DS.plan_search(conn, owner="sub:a", folders=[1], current_embedding=CUR)
        assert [t.folder.id for t in plan.targets] == [1]


# ----------------------------------------------------------------- cache


class TestCache:
    def _fp(self, **kw):
        args = dict(targets=[DS.FolderTarget(ref(1)), DS.FolderTarget(ref(2))],
                    group_by="document", top_k=10, snippets=3, reranker=None,
                    merge_threshold=None, filter_key=None)
        args.update(kw)
        return DS.CrossFolderCache.fingerprint("q", SearchSettings(), **args)

    def test_every_part_changes_the_key(self):
        base = self._fp()
        variants = [
            self._fp(targets=[DS.FolderTarget(ref(1))]),
            self._fp(targets=[DS.FolderTarget(ref(1), (F1,)), DS.FolderTarget(ref(2))]),
            self._fp(targets=[DS.FolderTarget(DS.FolderRef(1, "folder1", "uuid-x", "sub:a")),
                              DS.FolderTarget(ref(2))]),
            self._fp(group_by="passage"), self._fp(top_k=5), self._fp(snippets=1),
            self._fp(filter_key='{"tags_any": ["a"]}'), self._fp(merge_threshold=0.4),
            DS.CrossFolderCache.fingerprint("other", SearchSettings(), **{
                k: v for k, v in dict(targets=[DS.FolderTarget(ref(1)), DS.FolderTarget(ref(2))],
                                      group_by="document", top_k=10, snippets=3, reranker=None,
                                      merge_threshold=None, filter_key=None).items()}),
            DS.CrossFolderCache.fingerprint("q", SearchSettings(rerank_dense_candidates=5), **{
                k: v for k, v in dict(targets=[DS.FolderTarget(ref(1)), DS.FolderTarget(ref(2))],
                                      group_by="document", top_k=10, snippets=3, reranker=None,
                                      merge_threshold=None, filter_key=None).items()}),
        ]
        assert len({base, *variants}) == len(variants) + 1
        assert self._fp() == base

    def test_folder_invalidation_drops_the_entry(self):
        mem = InMemoryCache(100, 300)
        c = DS.CrossFolderCache(mem, 300)
        c.put("fp", [1, 2], {"v": 1})
        assert c.get("fp", [1, 2]) == {"v": 1}
        mem.clear_pattern("query:2:*")       # what any change of folder 2 does
        assert c.get("fp", [1, 2]) is None
        assert mem.get(c.key("fp")) is None

    def test_global_clear_drops_markers(self):
        mem = InMemoryCache(100, 300)
        c = DS.CrossFolderCache(mem, 300)
        c.put("fp", [1], {"v": 1})
        mem.clear_pattern("query:*")
        assert c.get("fp", [1]) is None


# --------------------------------------------------------------- ranking


def fake_searcher(pool, *, missing=False, error=None, reranker=None):
    s = MagicMock()
    s.embed_query = AsyncMock(return_value=[0.1, 0.2])

    async def folder_pool(q, settings, **kw):
        s.seen.append((settings, kw))
        if error:
            raise error
        return None if missing else [NodeWithScore(node=n.node, score=n.score) for n in pool]

    s.seen = []
    s.folder_pool = folder_pool
    s.vector_store = MagicMock(name="store")
    return s


def _rerank_by(table):
    calls = []

    async def fn(reranker, query, nodes, top_n):
        calls.append((query, ids(nodes), top_n))
        ranked = sorted(nodes, key=lambda n: table.get(n.node.node_id, 0.0), reverse=True)
        for n in ranked:
            n.score = table.get(n.node.node_id, 0.0)
        return RerankOutcome(nodes=ranked[:top_n], top_score=None)

    fn.calls = calls
    return fn


class TestRankAcross:
    def _targets(self, *specs):
        return [DS.FolderTarget(ref(i), fids) for i, fids in specs]

    async def test_one_rerank_over_the_pooled_candidates(self):
        s1 = fake_searcher([nws("a1"), nws("a2")])
        s2 = fake_searcher([nws("b1"), nws("b2")])
        searchers = {1: s1, 2: s2}
        rr = _rerank_by({"b2": 0.9, "a2": 0.8, "a1": 0.1, "b1": 0.2})
        out = await DS.rank_across("q", self._targets((1, None), (2, (F2,))),
                                   settings=SearchSettings(top_k=10),
                                   searcher_for=lambda f: searchers[f.id], reranker=object(),
                                   keep=3, rerank_fn=rr)
        assert len(rr.calls) == 1 and sorted(rr.calls[0][1]) == ["a1", "a2", "b1", "b2"]
        assert ids(out.nodes) == ["b2", "a2", "b1"] and out.score_type == "rerank"
        assert out.folder_of == {"a1": 1, "a2": 1, "b1": 2, "b2": 2}
        # Embedded once, the vector handed to both folders; each folder got its filter.
        assert s1.embed_query.await_count == 1 and s2.embed_query.await_count == 0
        assert s2.seen[0][1]["query_embedding"] == [0.1, 0.2]
        assert s1.seen[0][0].file_ids is None and s2.seen[0][0].file_ids == (F2,)
        assert (s1.seen[0][1]["dense_k"], s1.seen[0][1]["sparse_k"]) == (30, 10)

    async def test_without_reranker_rrf_across_folders(self):
        s1 = fake_searcher([nws("a1", 0.99), nws("a2", 0.98)])
        s2 = fake_searcher([nws("b1", 0.01)])
        out = await DS.rank_across("q", self._targets((1, None), (2, None)),
                                   settings=SearchSettings(), searcher_for=lambda f: {1: s1, 2: s2}[f.id],
                                   reranker=None, keep=10)
        assert ids(out.nodes) == ["a1", "b1", "a2"] and out.score_type == "rrf"

    async def test_rerank_failure_falls_back_to_rrf(self):
        s1 = fake_searcher([nws("a1"), nws("a2")])
        s2 = fake_searcher([nws("b1")])

        async def failing(reranker, query, nodes, top_n):
            for n in nodes:
                n.score = 0.0
            return RerankOutcome(nodes=list(nodes)[:top_n], failed=True)

        out = await DS.rank_across("q", self._targets((1, None), (2, None)),
                                   settings=SearchSettings(), searcher_for=lambda f: {1: s1, 2: s2}[f.id],
                                   reranker=object(), keep=10, rerank_fn=failing)
        assert out.rerank_failed and out.score_type == "rrf"
        assert ids(out.nodes) == ["a1", "b1", "a2"]
        assert out.nodes[0].score == pytest.approx(1 / 61)

    async def test_failed_and_missing_folders_are_reported(self):
        s1 = fake_searcher([nws("a1")])
        s2 = fake_searcher([], error=RuntimeError("boom"))
        s3 = fake_searcher([], missing=True)
        out = await DS.rank_across("q", self._targets((1, None), (2, None), (3, None)),
                                   settings=SearchSettings(),
                                   searcher_for=lambda f: {1: s1, 2: s2, 3: s3}[f.id],
                                   reranker=None, keep=5)
        assert ids(out.nodes) == ["a1"] and out.errors == [2] and out.not_indexed == [3]

    async def test_every_folder_failing_raises(self):
        s = fake_searcher([], error=RuntimeError("db down"))
        with pytest.raises(RuntimeError, match="db down"):
            await DS.rank_across("q", self._targets((1, None)), settings=SearchSettings(),
                                 searcher_for=lambda f: s, reranker=None, keep=5)

    async def test_no_targets(self):
        out = await DS.rank_across("q", [], settings=SearchSettings(), searcher_for=None,
                                   reranker=None, keep=5)
        assert out.nodes == [] and out.score_type == "rrf"

    async def test_pool_cap_and_gate(self):
        many = [nws(f"n{i}") for i in range(DS.MAX_POOLED_CANDIDATES + 20)]
        s = fake_searcher(many)
        rr = _rerank_by({})
        gate = asyncio.Semaphore(1)
        await DS.rank_across("q", self._targets((1, None)), settings=SearchSettings(),
                             searcher_for=lambda f: s, reranker=object(), keep=5,
                             rerank_fn=rr, gate=gate)
        assert len(rr.calls[0][1]) == DS.MAX_POOLED_CANDIDATES
        assert gate._value == 1

    async def test_fanout_is_bounded(self):
        live, peak = [0], [0]

        def make(i):
            s = fake_searcher([nws(f"x{i}")])

            async def pool(q, settings, **kw):
                live[0] += 1
                peak[0] = max(peak[0], live[0])
                await asyncio.sleep(0.01)
                live[0] -= 1
                return [nws(f"x{i}")]

            s.folder_pool = pool
            return s

        searchers = {i: make(i) for i in range(1, 11)}
        await DS.rank_across("q", self._targets(*[(i, None) for i in range(1, 11)]),
                             settings=SearchSettings(), searcher_for=lambda f: searchers[f.id],
                             reranker=None, keep=5)
        assert peak[0] == DS.FANOUT_CONCURRENCY


class TestMergePassages:
    async def test_disabled_keeps_leaves(self):
        nodes = [nws("a"), nws("b")]
        assert await DS.merge_passages(nodes, {}, {}, 1e9) == nodes

    async def test_merges_per_folder_store(self):
        from src.domain.rag.merge_plan import MergeGroup

        a = nws("a", parent_node_id="p1")
        b = nws("b")
        parent = TextNode(id_="p1", text="parent", metadata={"node_role": "parent"})
        calls = []

        async def plan(store, leaves, thr):
            calls.append((store, ids(leaves)))
            return {"p1": MergeGroup(parent=parent, hits=[a])} if store == "s1" else {}

        with patch("src.domain.rag.merge_plan.plan_merges", plan):
            out = await DS.merge_passages([a, b], {"a": 1, "b": 2}, {1: "s1", 2: "s2"}, 0.4)
        assert calls == [("s1", ["a"]), ("s2", ["b"])]
        assert isinstance(out[0], MergeGroup) and out[1] is b
        assert DS.item_folder(out[0], {"a": 1, "b": 2}) == 1
        assert DS.item_folder(out[1], {"a": 1, "b": 2}) == 2


# ------------------------------------------------------------ the service


class _ConnCM:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *a):
        return False


def _deps(plan, searchers, *, reranker=None, cache=None, query_one=None, described=None):
    deps = DS.SearchDeps(
        connect=lambda: _ConnCM("conn"), searcher_for=lambda f: searchers[f.id],
        query_one=query_one or AsyncMock(), settings=SearchSettings(top_k=5),
        reranker=reranker, cache=cache, merge_threshold=1e9)
    return deps


class TestSearchAcross:
    @pytest.fixture()
    def patched(self):
        state = {"plan": None, "described": {}}

        def describe(conn, fids, owner):
            state["describe_owner"] = owner
            return {f: v for f, v in state["described"].items() if f in fids}

        with patch.object(DS, "plan_search", side_effect=lambda conn, **kw: state["plan"]), \
                patch.object(DS, "describe_files", side_effect=describe):
            yield state

    async def test_documents_across_folders_with_live_names(self, patched):
        patched["plan"] = DS.SearchPlan(
            targets=[DS.FolderTarget(ref(1)), DS.FolderTarget(ref(2))],
            skipped=[DS._skip(ref(3), DS.SKIP_EMBEDDING)])
        patched["described"] = {
            F1: {"name": "renamed.md", "mime": "text/markdown", "tags": [], "metadata": {},
                 "folder": {"id": 1, "name": "folder1-renamed"}},
            F2: {"name": "two.md", "mime": "text/markdown", "tags": ["x"], "metadata": {},
                 "folder": {"id": 2, "name": "folder2"}}}
        searchers = {1: fake_searcher([nws("a1", file_id=F1), nws("a2", file_id=F1)]),
                     2: fake_searcher([nws("b1", file_id=F2), nws("gone", file_id=F3)])}
        mem = InMemoryCache(100, 300)
        deps = _deps(None, searchers, cache=DS.CrossFolderCache(mem, 300))
        req = DS.SearchRequest(query="q", group_by="document")
        data = await DS.search_across(req, owner="sub:a", deps=deps)
        assert [d["file_id"] for d in data["documents"]] == [F1, F2]     # F3: no file row
        assert data["documents"][0]["name"] == "renamed.md"
        assert data["documents"][0]["hits"] == 2
        assert data["folders"] == [{"id": 1, "name": "folder1"}, {"id": 2, "name": "folder2"}]
        assert [s["reason"] for s in data["skipped_folders"]] == [DS.SKIP_EMBEDDING]
        assert data["score_type"] == "rrf" and data["total_results"] == 2
        assert data["cached"] is False and patched["describe_owner"] == "sub:a"
        # Second call: from the cache, names read again.
        patched["described"][F1]["name"] = "renamed-again.md"
        again = await DS.search_across(req, owner="sub:a", deps=deps)
        assert again["cached"] is True and again["documents"][0]["name"] == "renamed-again.md"
        assert searchers[1].embed_query.await_count == 1

    async def test_passages_across_folders_carry_their_folder(self, patched):
        patched["plan"] = DS.SearchPlan(targets=[DS.FolderTarget(ref(1)), DS.FolderTarget(ref(2))])
        searchers = {1: fake_searcher([nws(NA, file_id=F1)]),
                     2: fake_searcher([nws(NB, file_id=F2)])}
        data = await DS.search_across(DS.SearchRequest(query="q", top_k=1), owner="sub:a",
                                      deps=_deps(None, searchers))
        assert len(data["results"]) == 1
        r = data["results"][0]
        assert r["folder"] == {"id": 1, "name": "folder1"}
        assert r["metadata"]["folder_name"] == "folder1"
        assert r["metadata"]["node_id"] == NA

    async def test_one_folder_passages_use_the_folder_search(self, patched):
        from src.adapter.model import RAGQueryResult
        from src.domain.rag.dto import RAGChunkMetadata, RAGSearchResult

        patched["plan"] = DS.SearchPlan(targets=[DS.FolderTarget(ref(4, "solo"))])
        res = RAGQueryResult(query="q", results=[RAGSearchResult(
            text="t", score=0.7, metadata=RAGChunkMetadata(node_id=NA, mcp_file_id=F1,
                                                           file_name="x.md"))],
            total_results=1)
        one = AsyncMock(return_value=res)
        data = await DS.search_across(DS.SearchRequest(query="q"), owner="sub:a",
                                      deps=_deps(None, {}, query_one=one))
        one.assert_awaited_once()
        assert one.await_args.args[0].id == 4 and one.await_args.args[1] == 5
        assert data["results"][0]["folder"] == {"id": 4, "name": "solo"}
        assert data["results"][0]["metadata"]["folder_name"] == "solo"

    async def test_one_folder_not_indexed_is_reported(self, patched):
        from src.adapter.model import RAGQueryResult

        patched["plan"] = DS.SearchPlan(targets=[DS.FolderTarget(ref(4))])
        one = AsyncMock(return_value=RAGQueryResult(query="q", results=[], total_results=0,
                                                    not_indexed=True))
        data = await DS.search_across(DS.SearchRequest(query="q"), owner="sub:a",
                                      deps=_deps(None, {}, query_one=one))
        assert data["folders"] == [] and data["skipped_folders"][0]["reason"] == DS.SKIP_NOT_INDEXED

    async def test_nothing_to_search(self, patched):
        patched["plan"] = DS.SearchPlan(skipped=[DS._skip(ref(1), DS.SKIP_NOT_INDEXED)])
        data = await DS.search_across(DS.SearchRequest(query="q", group_by="document"),
                                      owner="sub:a", deps=_deps(None, {}))
        assert data["documents"] == [] and data["total_results"] == 0

    async def test_degraded_results_are_not_cached(self, patched):
        patched["plan"] = DS.SearchPlan(targets=[DS.FolderTarget(ref(1)), DS.FolderTarget(ref(2))])
        searchers = {1: fake_searcher([nws(NA, file_id=F1)]),
                     2: fake_searcher([], error=RuntimeError("x"))}
        mem = InMemoryCache(100, 300)
        data = await DS.search_across(
            DS.SearchRequest(query="q", group_by="passage"), owner="sub:a",
            deps=_deps(None, searchers, cache=DS.CrossFolderCache(mem, 300)))
        assert [s["reason"] for s in data["skipped_folders"]] == [DS.SKIP_ERROR]
        assert data["folders"] == [{"id": 1, "name": "folder1"}]
        assert not any(k.startswith("search:") for k in mem._cache)

    async def test_top_k_bounds(self, patched):
        patched["plan"] = DS.SearchPlan()
        data = await DS.search_across(DS.SearchRequest(query="q", group_by="document", top_k=999),
                                      owner="sub:a", deps=_deps(None, {}))
        assert data["top_k"] == DS.MAX_TOP_K
        data = await DS.search_across(DS.SearchRequest(query="q", group_by="bogus"),
                                      owner="sub:a", deps=_deps(None, {}))
        assert data["group_by"] == "passage" and data["top_k"] == 5


# ------------------------------------------------------------ the searcher


class TestFolderPool:
    def _searcher(self, dense, sparse, reranker=None):
        s = HybridSearcher(MagicMock(), reranker=reranker, index=MagicMock())
        s.dense_calls, s.sparse_calls = [], []

        async def dense_fn(q, k, **kw):
            s.dense_calls.append((k, kw))
            return [NodeWithScore(node=n.node, score=n.score) for n in dense][:k]

        async def sparse_fn(q, k, st, **kw):
            s.sparse_calls.append((k, kw))
            return [NodeWithScore(node=n.node, score=n.score) for n in sparse][:k]

        s.dense, s.sparse = dense_fn, sparse_fn
        return s

    async def test_reranker_gets_per_list_pool_with_given_depths(self):
        dense = [nws(f"d{i}", 1 - i / 100) for i in range(20)]
        sparse = [nws(f"s{i}") for i in range(10)]
        s = self._searcher(dense, sparse, reranker=object())
        pool = await s.folder_pool("q", SearchSettings(top_k=10), dense_k=5, sparse_k=2,
                                   query_embedding=[1.0])
        assert sorted(ids(pool)) == sorted(["d0", "d1", "d2", "d3", "d4", "s0", "s1"])
        assert s.dense_calls[0] == (5, {"query_embedding": [1.0]})

    async def test_without_reranker_fused_top(self):
        s = self._searcher([nws("d0"), nws("d1"), nws("d2")], [nws("s0")])
        pool = await s.folder_pool("q", SearchSettings(), dense_k=2, sparse_k=1)
        assert len(pool) == 2 and ids(pool)[0] == "d0"

    async def test_missing_table_is_none(self):
        s = self._searcher([], [])

        class Missing(Exception):
            sqlstate = "42P01"

        async def boom(q, k, **kw):
            raise Missing('relation "data_1_x" does not exist')

        s.dense = boom
        s._table_name = lambda: "data_1_x"
        assert await s.folder_pool("q", SearchSettings(), dense_k=2, sparse_k=1) is None

    async def test_other_errors_raise(self):
        s = self._searcher([], [])

        async def boom(q, k, **kw):
            raise RuntimeError("down")

        s.dense = boom
        with pytest.raises(RuntimeError):
            await s.folder_pool("q", SearchSettings(), dense_k=2, sparse_k=1)


class TestSharedQueryWork:
    async def test_segmentation_once_per_segmenter(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        calls = []

        async def seg(text, name):
            calls.append((text, name))
            await asyncio.sleep(0)
            return "a b"

        s._segment = seg
        st = SearchSettings(text_search_config="simple")
        with hs.shared_query_work():
            out = await asyncio.gather(s.segment_query("q", st, "ckip"), s.segment_query("q", st, "ckip"),
                                       s.segment_query("q", st, "bigram"))
        assert out == ["a b"] * 3 and sorted(calls) == [("q", "bigram"), ("q", "ckip")]
        # Outside the context nothing is shared.
        await s.segment_query("q", st, "ckip")
        await s.segment_query("q", st, "ckip")
        assert len(calls) == 4

    async def test_dense_uses_a_given_embedding(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        s.embed_query = AsyncMock(return_value=[9.0])

        class Conn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def execute(self, stmt, params=None):
                Conn.params = params
                return SimpleNamespace(fetchall=lambda: [])

        eng = MagicMock()
        eng.begin.return_value = Conn()
        s._dense_engine = MagicMock(return_value=eng)
        with patch("src.domain.rag.chunk_lookup.ChunkLookup._get_table_name", return_value="t"):
            await s.dense("q", 3, query_embedding=[0.5, 0.25])
        s.embed_query.assert_not_awaited()
        assert Conn.params["emb"] == "[0.5,0.25]"
