"""The MCP tool folio_find (src/fastmcp_tools/find_tool.py) and the folder tools'
file_ids / filters, as a client sees them.

Pins: folio_find is registered once with its schema; the owner comes from the
caller's token (``sub``), never from the arguments, and open mode searches every
folder like REST; a token without ``sub`` is refused, a token whose ``sub`` owns
nothing gets an empty answer; refusals come back as ``{"error": ...}``; the tool is
static — every caller sees it in tools/list, with their own tags and metadata keys
in its description — and is never mistaken for a folder tool. The folder tools pass
file_ids / filters to the backend, and a search scope limits the retriever's
settings (an empty scope searches nothing).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client, FastMCP

from src.domain.rag import doc_filter as DF
from src.domain.rag import doc_find as FIND
from src.domain.rag import doc_search as DS
from src.fastmcp_tools import find_tool as FT

USER = "5d1c9f0e-7a3b-4c1d-9e8f-0123456789ab"
OTHER = "a0b1c2d3-e4f5-4a6b-8c7d-0123456789ff"
F1 = "11111111-1111-1111-1111-111111111111"
_CFG = "src.config.config_manager.Config.get_config_model"


def _cfg(enabled=True, prefix="Folio", budget=6000):
    return SimpleNamespace(auth=SimpleNamespace(enabled=enabled),
                           app=SimpleNamespace(tool_prefix=prefix),
                           rag=SimpleNamespace(retrieval=SimpleNamespace(
                               mcp_response_token_budget=budget)))


def _found(*names, folder="docs"):
    return {"query": "q", "score_type": "rrf", "skipped_folders": [], "documents": [
        {"file_id": F1, "name": n, "folder": {"id": 1, "name": folder}, "tags": [], "metadata": {},
         "score": 0.5, "hits": 1, "snippets": [{"text": "x", "score": 0.5}]} for n in names]}


@pytest.fixture()
def server():
    mcp = FastMCP("t")
    FT.register_find_tool(mcp)
    return mcp


async def _call(server, args, *, search=None, cfg=None, header=None, verified=None):
    """Call folio_find through a real FastMCP client, with the request's Authorization
    header and the auth provider's verified claims stubbed."""
    search = search or AsyncMock(return_value=_found("a.md"))
    req = SimpleNamespace(headers={"Authorization": header} if header else {})
    with patch(_CFG, return_value=cfg or _cfg()), \
            patch.object(FT, "_search_across", new=search), \
            patch("src.fastmcp_tools.acl.get_http_request", return_value=req), \
            patch("src.fastmcp_tools.acl.get_access_token", return_value=verified), \
            patch("src.adapter.folder_claim.claim_for", new=AsyncMock(return_value=[])):
        async with Client(server) as c:
            res = await c.call_tool("folio_find", args)
    return json.loads(res.content[0].text), search


def _verified(token, sub):
    claims = {"jti": "j"} if sub is None else {"sub": sub, "jti": "j"}
    return SimpleNamespace(token=token, claims=claims)


class TestSchema:
    async def test_registered_once_with_its_parameters(self, server):
        async with Client(server) as c:
            (tool,) = await c.list_tools()
        assert tool.name == FT.FIND_TOOL_NAME == "folio_find"
        assert set(tool.input_schema["properties"]) == {"query", "filters", "folders", "limit", "offset"}
        assert not tool.input_schema.get("required")
        d = tool.description
        assert "file_ids" in d and "tags_any" in d and "metadata" in d and "next_offset" in d


class TestOwner:
    async def test_owner_is_the_tokens_person(self, server):
        out, search = await _call(server, {"query": "tea", "folders": ["docs"]},
                                  header="Bearer tok", verified=_verified("tok", USER))
        assert search.await_args.kwargs["owner"] == f"sub:{USER}"
        assert out["mode"] == "find" and out["documents"][0]["tool"] == "Folio_docs"

    async def test_two_people_get_their_own_scope(self, server):
        _, s1 = await _call(server, {"query": "tea"}, header="Bearer t1", verified=_verified("t1", USER))
        _, s2 = await _call(server, {"query": "tea"}, header="Bearer t2", verified=_verified("t2", OTHER))
        assert s1.await_args.kwargs["owner"] != s2.await_args.kwargs["owner"] == f"sub:{OTHER}"

    async def test_token_without_sub_is_refused(self, server):
        out, search = await _call(server, {"query": "tea"}, header="Bearer t",
                                  verified=_verified("t", None))
        assert out["error"] == "unauthorized" and "sub" in out["detail"]
        assert not search.await_count

    async def test_missing_token_is_refused(self, server):
        out, search = await _call(server, {"query": "tea"})
        assert out["error"] == "unauthorized" and not search.await_count

    async def test_a_sub_that_owns_nothing_gets_an_empty_answer(self, server):
        out, search = await _call(server, {"query": "tea"}, header="Bearer t",
                                  verified=_verified("t", "mcp-gate-scanner"),
                                  search=AsyncMock(return_value=_found()))
        assert search.await_args.kwargs["owner"] == "sub:mcp-gate-scanner"
        assert out["documents"] == [] and "error" not in out

    async def test_open_mode_is_every_folder(self, server):
        _, search = await _call(server, {"query": "tea"}, cfg=_cfg(enabled=False))
        assert search.await_args.kwargs["owner"] is DF.ALL_FOLDERS

    async def test_unreadable_config_needs_a_token(self, server):
        with patch(_CFG, side_effect=RuntimeError):
            assert FT.auth_enabled() is True


class TestRefusals:
    async def test_bad_filter_names_the_key(self, server):
        out, search = await _call(server, {"query": "tea", "filters": {"tagz": ["a"]}},
                                  cfg=_cfg(enabled=False))
        assert out["error"] == "invalid_filter" and "tagz" in out["detail"]
        assert not search.await_count

    async def test_foreign_folder(self, server):
        out, _ = await _call(server, {"query": "tea", "folders": ["theirs"]}, cfg=_cfg(enabled=False),
                             search=AsyncMock(side_effect=DS.FolderNotFound("theirs")))
        assert out == {"error": "folder_not_found", "detail": "Folder not found: theirs"}

    async def test_unexpected_failure(self, server):
        out, _ = await _call(server, {"query": "tea"}, cfg=_cfg(enabled=False),
                             search=AsyncMock(side_effect=RuntimeError("boom")))
        assert out["error"] == "internal_error" and "boom" in out["detail"]

    async def test_query_too_long_for_the_model(self, server):
        out, _ = await _call(server, {"query": "tea"}, cfg=_cfg(enabled=False),
                             search=AsyncMock(side_effect=RuntimeError("maximum context length")))
        assert out["error"] == "invalid_request"


class TestBudgetAndListing:
    async def test_budget_from_config(self, server):
        names = [f"{i}.md" for i in range(30)]
        found = _found(*names)
        for d in found["documents"]:
            d["snippets"] = [{"text": "word " * 150, "score": 0.5}]
        out, _ = await _call(server, {"query": "tea", "limit": 30}, cfg=_cfg(enabled=False, budget=800),
                             search=AsyncMock(return_value=found))
        assert out["omitted_results"] > 0 and out["response_tokens_estimated"] <= 800

    async def test_listing_without_query(self, server):
        calls = {}

        async def fake_find(**kw):
            calls.update(kw)
            return {"mode": "list", "documents": []}

        with patch.object(FIND, "find", new=fake_find):
            out, search = await _call(server, {"filters": {"tags_any": ["x"]}, "offset": 5},
                                      cfg=_cfg(enabled=False))
        assert out["mode"] == "list" and calls["offset"] == 5 and calls["query"] is None
        assert calls["owner"] is DF.ALL_FOLDERS and calls["tool_of"]("a b") == FT.tool_for_folder("a b")


# -- tools/list: visibility and the per-person description ---------------------------


@pytest.fixture()
def tool_filter(gate_verifier):
    from src.auth.dependencies import set_token_verifier
    from src.fastmcp_tools import auth_filtered_mcp as afm
    from src.fastmcp_tools.dynamic_tool_manager import DynamicToolManager

    folders = {f"sub:{USER}": [SimpleNamespace(name="docs")]}
    DynamicToolManager._instance = None
    m = DynamicToolManager.initialize(MagicMock())
    m._registered_tools, m._token_folder_to_tool = {}, {}
    m._token_folder_descriptions, m._tool_to_folder = {}, {}
    set_token_verifier(gate_verifier)
    afm._filter_cache.clear()
    get = lambda user_token=None, **kw: folders.get(user_token, [])  # noqa: E731
    descs = {}

    def describe(owner):
        descs.setdefault("owners", []).append(owner)
        return f"FIND for {owner}" if owner == f"sub:{USER}" else None

    with patch("src.fastmcp_tools.dynamic_tool_manager.Config.get_config_model",
               return_value=_cfg()), \
         patch("src.fastmcp_tools.dynamic_tool_manager.FolderDB.get", side_effect=get), \
         patch.object(afm.FolderDB, "get", side_effect=get), \
         patch.object(FT, "find_description_for", side_effect=describe), \
         patch("src.adapter.folder_claim.claim_for", new=AsyncMock(return_value=[])):
        m.register_folder_tool("docs", MagicMock(), user_token=f"sub:{USER}")
        srv = afm.AuthFilteredFastMCP("t")
        all_tools = [SimpleNamespace(name=n, description="d", model_copy=lambda update, n=n:
                                     SimpleNamespace(name=n, **update))
                     for n in ("Folio_docs", "folio_find")]

        async def run(token):
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            with patch.object(afm, "get_http_request", return_value=SimpleNamespace(headers=headers)):
                return {t.name: t.description for t in await srv._apply_token_filter(list(all_tools))}
        yield SimpleNamespace(run=run, descs=descs, srv=srv)
    set_token_verifier(None)
    afm._filter_cache.clear()
    DynamicToolManager._instance = None


class TestToolList:
    async def test_every_caller_sees_folio_find(self, tool_filter, tokens):
        mine = await tool_filter.run(tokens.make(sub=USER))
        assert set(mine) == {"Folio_docs", "folio_find"}
        assert mine["folio_find"] == f"FIND for sub:{USER}"
        theirs = await tool_filter.run(tokens.make(sub=OTHER))
        assert set(theirs) == {"folio_find"} and theirs["folio_find"] == "d"
        assert set(await tool_filter.run(tokens.make(drop=("sub",)))) == {"folio_find"}
        assert set(await tool_filter.run(None)) == {"folio_find"}
        # each person's vocabulary is read with their own key only
        assert tool_filter.descs["owners"] == [f"sub:{USER}", f"sub:{OTHER}"]

    async def test_vocabulary_failure_keeps_the_plain_tool(self, tool_filter, tokens):
        with patch.object(FT, "find_description_for", side_effect=RuntimeError("db down")):
            mine = await tool_filter.run(tokens.make(sub=USER))
        assert mine == {"Folio_docs": "d", "folio_find": "d"}

    def test_never_a_folder_tool_even_with_a_lowercase_prefix(self, tool_filter):
        with patch("src.fastmcp_tools.dynamic_tool_manager.Config.get_config_model",
                   return_value=_cfg(prefix="folio")):
            assert tool_filter.srv._is_dynamic_folder_tool("folio_find") is False
            assert tool_filter.srv._is_dynamic_folder_tool("folio_other") is True


def test_find_description_for_adds_the_vocabulary():
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    with patch.object(FT, "_connect", return_value=conn), \
            patch.object(FIND, "vocabulary", return_value={
                "tags": [{"tag": "signed", "count": 2}], "metadata_keys": []}) as voc:
        d = FT.find_description_for("sub:x")
    assert voc.call_args.kwargs["owner"] == "sub:x"
    assert d.startswith(FT.FIND_DESCRIPTION) and "signed (2)" in d
    with patch.object(FT, "_connect", return_value=conn), \
            patch.object(FIND, "vocabulary", return_value={"tags": [], "metadata_keys": []}):
        assert FT.find_description_for("sub:x") is None


# -- folder tools: file_ids / filters -------------------------------------------------


class TestFolderToolScope:
    @pytest.fixture()
    def folder_tool(self):
        from src.fastmcp_tools.dynamic_tool_manager import DynamicToolManager
        DynamicToolManager._instance = None
        mcp = FastMCP("t")
        m = DynamicToolManager.initialize(mcp)
        m._registered_tools, m._token_folder_to_tool = {}, {}
        m._token_folder_descriptions, m._tool_to_folder = {}, {}
        calls = []

        async def backend(**kw):
            calls.append(kw)
            return [json.dumps({"ok": True})]

        with patch("src.fastmcp_tools.dynamic_tool_manager.FolderDB.get",
                   return_value=[SimpleNamespace(name="docs")]), \
             patch("src.fastmcp_tools.dynamic_tool_manager.Config.get_config_model",
                   return_value=_cfg()):
            assert m.register_folder_tool("docs", backend, user_token="sub:a")
        yield mcp, calls
        DynamicToolManager._instance = None

    async def test_file_ids_and_filters_reach_the_backend(self, folder_tool):
        mcp, calls = folder_tool
        async with Client(mcp) as c:
            (tool,) = await c.list_tools()
            assert {"file_ids", "filters"} <= set(tool.input_schema["properties"])
            assert "folio_find" in tool.description
            await c.call_tool("Folio_docs", {"query": "q", "file_ids": [F1],
                                             "filters": {"tags_any": ["x"]}})
            await c.call_tool("Folio_docs", {"query": "q"})
        assert calls[0]["file_ids"] == [F1] and calls[0]["filters"] == {"tags_any": ["x"]}
        assert "file_ids" not in calls[1] and "filters" not in calls[1]

    async def test_query_passes_them_to_the_adapter(self):
        from src.fastmcp_tools import agentic_tools as at
        adapter = MagicMock()
        adapter.query = AsyncMock(return_value={"mode": "search", "results": []})
        with patch(_CFG, return_value=SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(
                default_top_k=5, expand_context_default=True, return_resource_files=False)))), \
                patch.object(at, "extract_owner_key", new=AsyncMock(return_value="sub:a")), \
                patch.object(at, "get_rag_adapter", return_value=adapter):
            await at._query(folder_name="docs", mode="search", query="q", file_id=None, top_k=None,
                            similarity_cutoff=None, expand_context=None, file_ids=[F1],
                            filters={"tags_any": ["x"]})
        kw = adapter.query.await_args.kwargs
        assert kw["file_ids"] == [F1] and kw["filters"] == {"tags_any": ["x"]}


class TestHandleSearchScope:
    def _retriever(self, results=()):
        r = MagicMock()
        r.aquery = AsyncMock(return_value=list(results))
        r.last_outcome = SimpleNamespace(score_type="rrf", top_score=None, not_indexed=False)
        return r

    async def _run(self, scope, retriever):
        from src.domain.rag import agentic_handlers as H
        with patch.object(H, "configured_min_relevance", return_value=0.0), \
                patch.object(H, "configured_response_budget", return_value=6000):
            return await H.handle_search(query="tea time", folder=SimpleNamespace(id=1, name="docs"),
                                         retriever=retriever, top_k=5, expand_context=False,
                                         scope=scope)

    async def test_scope_ids_go_into_the_search_settings(self):
        r = self._retriever()
        out = await self._run({"file_ids": (F1,), "ignored_file_ids": ["x"]}, r)
        settings = r.aquery.await_args.kwargs["settings"]
        assert settings.file_ids == (F1,)
        assert out["scope"] == {"file_ids": 1, "ignored_file_ids": ["x"]}
        assert "ignored" in out["_hint"]

    async def test_no_scope_searches_the_whole_folder(self):
        r = self._retriever()
        out = await self._run(None, r)
        assert r.aquery.await_args.kwargs["settings"] is None and "scope" not in out

    async def test_empty_scope_searches_nothing(self):
        r = self._retriever()
        out = await self._run({"file_ids": (), "ignored_file_ids": []}, r)
        assert not r.aquery.await_count
        assert out["results"] == [] and out["scope"] == {"file_ids": 0}
        assert "matches the filters" in out["_hint"]

    async def test_query_agentic_returns_a_scope_refusal(self):
        from src.adapter.rag_query import RAGQueryService
        svc = RAGQueryService(MagicMock())
        folder = SimpleNamespace(id=3, name="docs", vector_table_uuid="u")
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        with patch("src.domain.rag.folder_acl.averify_folder_access",
                   new=AsyncMock(return_value=folder)), \
                patch("db.db.get_engine", return_value=SimpleNamespace(connect=lambda: conn)), \
                patch.object(FIND, "search_scope",
                             side_effect=FIND.FindError("invalid_request", "none of file_ids")) as sc:
            out = await svc.query_agentic(folder_name="docs", mode="search", query="tea",
                                          file_id=None, top_k=5, similarity_cutoff=None,
                                          expand_context=False, token="sub:a", file_ids=["x"])
        assert out["error"] == "none of file_ids" and out["error_code"] == "invalid_request"
        assert sc.call_args.kwargs == {"folder_id": 3, "owner": "sub:a", "file_ids": ["x"],
                                       "filters": None}
