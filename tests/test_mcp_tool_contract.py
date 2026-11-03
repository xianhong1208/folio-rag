"""The per-folder MCP tool as a client sees it: schema, live defaults, the
deprecated similarity_cutoff, read paging and the resource-file policy."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client, FastMCP

from src.fastmcp_tools import agentic_tools as at
from src.fastmcp_tools.dynamic_tool_manager import DynamicToolManager

_CFG = "src.config.config_manager.Config.get_config_model"


def _cfg(**retrieval):
    base = dict(default_top_k=7, default_similarity_cutoff=0.3, expand_context_default=False,
                return_resource_files=False, resource_file_max_bytes=100)
    base.update(retrieval)
    return SimpleNamespace(app=SimpleNamespace(tool_prefix="Folio"),
                           rag=SimpleNamespace(retrieval=SimpleNamespace(**base)))


@pytest.fixture()
def registered():
    """A real FastMCP server with one folder tool whose backend records its kwargs."""
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
        assert m.register_folder_tool("docs", backend, user_token="jti-a")
    yield mcp, calls
    DynamicToolManager._instance = None


class TestSchema:
    async def test_similarity_cutoff_hidden_but_accepted(self, registered):
        mcp, calls = registered
        async with Client(mcp) as c:
            (tool,) = await c.list_tools()
            props = tool.input_schema["properties"]
            assert "similarity_cutoff" not in props
            assert {"query", "mode", "file_id", "top_k", "expand_context", "offset"} <= set(props)
            # an old client still sending it is not rejected
            await c.call_tool("Folio_docs", {"query": "q", "similarity_cutoff": 0.4})
        assert calls[-1]["similarity_cutoff"] == 0.4

    async def test_omitted_knobs_reach_the_backend_as_none(self, registered):
        mcp, calls = registered
        async with Client(mcp) as c:
            await c.call_tool("Folio_docs", {"query": "q"})
            await c.call_tool("Folio_docs", {"mode": "read", "file_id": "f", "offset": 12})
        assert calls[0]["top_k"] is None and calls[0]["expand_context"] is None
        assert calls[1]["offset"] == 12 and calls[1]["mode"] == "read"

    async def test_description_has_no_baked_defaults(self, registered):
        mcp, _ = registered
        async with Client(mcp) as c:
            (tool,) = await c.list_tools()
        assert "default 7" not in tool.description
        assert "similarity_cutoff" not in tool.description
        assert "lower the cutoff" not in tool.description
        assert "score_type" in tool.description and "next_offset" in tool.description


class TestLiveDefaults:
    def test_defaults_follow_the_current_config(self):
        with patch(_CFG, return_value=_cfg(default_top_k=3, expand_context_default=True)):
            assert at.resolve_search_defaults(None, None) == (3, True)
        with patch(_CFG, return_value=_cfg(default_top_k=9, expand_context_default=False)):
            assert at.resolve_search_defaults(None, None) == (9, False)

    def test_explicit_values_win_and_top_k_is_bounded(self):
        with patch(_CFG, return_value=_cfg()):
            assert at.resolve_search_defaults(50, True) == (20, True)
            assert at.resolve_search_defaults(0, False) == (1, False)

    def test_unreadable_config(self):
        with patch(_CFG, side_effect=RuntimeError):
            assert at.resolve_search_defaults(None, None) == (5, True)


class TestQueryBackend:
    async def _call(self, cfg, response, **kw):
        adapter = MagicMock()
        adapter.query = AsyncMock(return_value=response)
        args = dict(folder_name="docs", mode="search", query="q", file_id=None, top_k=None,
                    similarity_cutoff=None, expand_context=None)
        args.update(kw)
        with patch(_CFG, return_value=cfg), \
             patch.object(at, "extract_token", return_value="tok"), \
             patch.object(at, "get_rag_adapter", return_value=adapter):
            blocks = await at._query(**args)
        return blocks, adapter.query.await_args.kwargs

    async def test_resolves_defaults_and_passes_offset(self):
        blocks, kw = await self._call(_cfg(default_top_k=4), {"mode": "search", "results": []},
                                      mode="read", offset=30)
        assert kw["top_k"] == 4 and kw["expand_context"] is False and kw["offset"] == 30
        assert json.loads(blocks[0])["mode"] == "search" and len(blocks) == 1

    async def test_no_resource_files_by_default(self):
        resp = {"results": [{"file_id": "f1"}]}
        with patch.object(at, "_build_resource_files") as build:
            blocks, _ = await self._call(_cfg(), resp)
        build.assert_not_called()
        assert len(blocks) == 1

    async def test_resource_file_attached_when_enabled(self):
        resp = {"results": [{"file_id": "f1"}]}
        f = at.RagFile(data=b"abc", format="txt", name="a", mime_type="text/plain")
        with patch.object(at, "_build_resource_files", return_value=[f]) as build:
            blocks, _ = await self._call(_cfg(return_resource_files=True), resp)
        assert build.call_args.args[1] == 100 and blocks[1] is f

    async def test_errors_become_json(self):
        from src.domain.exceptions import UnauthorizedAccessError

        adapter = MagicMock()
        adapter.query = AsyncMock(side_effect=UnauthorizedAccessError(
            resource_type="folder", resource_id="docs", reason="no"))
        with patch(_CFG, return_value=_cfg()), \
             patch.object(at, "extract_token", return_value="tok"), \
             patch.object(at, "get_rag_adapter", return_value=adapter):
            blocks = await at._query(folder_name="docs", mode="search", query="q", file_id=None,
                                     top_k=None, similarity_cutoff=None, expand_context=None)
        assert json.loads(blocks[0])["error"] == "forbidden"


class TestResourceFiles:
    def _record(self, size=10, name="Report.PDF"):
        return SimpleNamespace(file_name=name, file_path="p", file_size=size)

    def test_only_the_top_hit_file(self):
        resp = {"results": [{"file_id": "f2"}, {"file_id": "f1"}, {"file_id": "f1"}]}
        with patch.object(at.FileDB, "get_by_ids", return_value={"f2": self._record()}) as get, \
             patch.object(at.FileStorage, "read_file", return_value=b"%PDF"):
            files = at._build_resource_files(resp, max_bytes=1000)
        assert get.call_args.args[0] == ["f2"]
        assert len(files) == 1 and files[0].data == b"%PDF"

    def test_over_the_cap_is_not_attached(self):
        resp = {"results": [{"file_id": "f1"}]}
        with patch.object(at.FileDB, "get_by_ids", return_value={"f1": self._record(size=5000)}), \
             patch.object(at.FileStorage, "read_file") as read:
            assert at._build_resource_files(resp, max_bytes=1000) == []
        read.assert_not_called()
        # a record without a size is checked on the bytes read
        with patch.object(at.FileDB, "get_by_ids", return_value={"f1": self._record(size=None)}), \
             patch.object(at.FileStorage, "read_file", return_value=b"x" * 2000):
            assert at._build_resource_files(resp, max_bytes=1000) == []

    def test_missing_record_unreadable_file_and_no_results(self):
        assert at._build_resource_files({"results": []}) == []
        resp = {"results": [{"file_id": "f1"}]}
        with patch.object(at.FileDB, "get_by_ids", return_value={}):
            assert at._build_resource_files(resp) == []
        with patch.object(at.FileDB, "get_by_ids", return_value={"f1": self._record()}), \
             patch.object(at.FileStorage, "read_file", side_effect=OSError("gone")):
            assert at._build_resource_files(resp) == []
