"""handle_search surfaces content_type (table/picture) for MCP agents.

Parity with the REST /query path: a table/figure hit must carry content_type so an
agent knows its text is a serialized table, not prose. Absent content_type stays
omitted (older indexed data lacks it).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.domain.rag.agentic_handlers import handle_search


def _result(**kw):
    base = dict(node_id="n1", file_id="f1", file_name="a.xlsx", score=0.9,
                node_role="leaf", chunk_index_range=[0, 0], text="Row, Col = 1",
                page=None, headings=None, content_type=None, merged_from_leaves=[])
    base.update(kw)
    return SimpleNamespace(**base)


async def _first_result(result):
    retriever = SimpleNamespace(aquery=AsyncMock(return_value=[result]))
    folder = SimpleNamespace(id=1, name="f")
    resp = await handle_search(query="hello world", folder=folder, retriever=retriever,
                               top_k=5, similarity_cutoff=0.0, expand_context=False)
    return resp["results"][0]


async def test_table_result_includes_content_type():
    r = await _first_result(_result(content_type="table"))
    assert r["content_type"] == "table"


async def test_picture_result_includes_content_type():
    r = await _first_result(_result(content_type="picture"))
    assert r["content_type"] == "picture"


async def test_plain_result_omits_content_type():
    r = await _first_result(_result(content_type=None))
    assert "content_type" not in r
