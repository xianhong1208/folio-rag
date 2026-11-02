"""RagFile.to_resource_content: the standard MIME type must reach the client.

The MCP SDK's resource-content models expose ``mimeType`` as a read-only alias
property, so the MIME type cannot be patched onto an already-built object; it
has to go in when the object is constructed. Covers a text file (embedded as
TextResourceContents) and a binary file (BlobResourceContents).
"""
import base64

from mcp import types as mcp_types

from src.fastmcp_tools.agentic_tools import RagFile


def test_text_file_gets_standard_mime_and_text_content():
    f = RagFile(data="hello 世界".encode("utf-8"), format="md", name="notes",
                mime_type="text/markdown")
    out = f.to_resource_content()
    assert isinstance(out, mcp_types.EmbeddedResource)
    assert isinstance(out.resource, mcp_types.TextResourceContents)
    assert out.resource.mime_type == "text/markdown"
    assert out.resource.text == "hello 世界"
    assert str(out.resource.uri).endswith("notes.md")


def test_plain_text_file():
    out = RagFile(data=b"plain", format="txt", name="a", mime_type="text/plain").to_resource_content()
    assert isinstance(out.resource, mcp_types.TextResourceContents)
    assert out.resource.mime_type == "text/plain"
    assert out.resource.text == "plain"


def test_binary_file_gets_standard_mime_and_blob_content():
    payload = b"%PDF-1.4\x00\xff\x10"
    f = RagFile(data=payload, format="docx", name="report",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    out = f.to_resource_content()
    assert isinstance(out.resource, mcp_types.BlobResourceContents)
    assert out.resource.mime_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert base64.b64decode(out.resource.blob) == payload
    assert str(out.resource.uri).endswith("report.docx")


def test_pdf_file():
    out = RagFile(data=b"%PDF", format="pdf", name="p", mime_type="application/pdf").to_resource_content()
    assert isinstance(out.resource, mcp_types.BlobResourceContents)
    assert out.resource.mime_type == "application/pdf"


def test_explicit_mime_argument_wins():
    out = RagFile(data=b"x", format="txt", name="a", mime_type="text/plain").to_resource_content(
        mime_type="text/csv")
    assert out.resource.mime_type == "text/csv"


def test_without_standard_mime_falls_back_to_format_mime():
    out = RagFile(data=b"\x00\x01", format="bin", name="blob").to_resource_content()
    assert isinstance(out.resource, mcp_types.BlobResourceContents)
    assert out.resource.mime_type  # FastMCP's format-derived type


async def test_tool_returning_rag_files_round_trips_through_fastmcp():
    """The path that failed in production: a tool result list holding a JSON
    string plus RagFile objects, serialised by FastMCP and read by a client."""
    from fastmcp import Client, FastMCP

    mcp = FastMCP("t")

    @mcp.tool
    def search() -> list:
        return [
            '{"results": []}',
            RagFile(data=b"line one\n", format="txt", name="notes", mime_type="text/plain"),
            RagFile(data=b"%PDF-1.4\xff", format="pdf", name="doc", mime_type="application/pdf"),
        ]

    async with Client(mcp) as client:
        result = await client.call_tool("search", {})
    kinds = [type(c).__name__ for c in result.content]
    assert kinds == ["TextContent", "EmbeddedResource", "EmbeddedResource"]
    text_res, blob_res = result.content[1].resource, result.content[2].resource
    assert text_res.mime_type == "text/plain" and text_res.text == "line one\n"
    assert blob_res.mime_type == "application/pdf"
    assert base64.b64decode(blob_res.blob) == b"%PDF-1.4\xff"
