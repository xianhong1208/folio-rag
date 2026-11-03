"""Per-folder MCP tool names must match ^[a-zA-Z0-9_-]{1,64}$.

Claude and OpenAI clients reject other names, and one bad tool can break tool
listing for the whole server. Folder names may be CJK, contain spaces, or be long.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.fastmcp_tools.dynamic_tool_manager import DynamicToolManager, tool_name_for

_VALID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@pytest.mark.parametrize("folder", [
    "docs", "產品手冊", "Product Docs 2026", "café", "a" * 200, "東京 tokyo/../x", "-", "   ",
    "報告" * 40,
])
def test_names_are_always_valid(folder):
    assert _VALID.match(tool_name_for("Folio", folder)), tool_name_for("Folio", folder)


def test_plain_ascii_names_are_unchanged():
    assert tool_name_for("Folio", "product-docs_v2") == "Folio_product-docs_v2"


def test_deterministic():
    assert tool_name_for("Folio", "產品手冊") == tool_name_for("Folio", "產品手冊")


def test_readable_part_kept_and_hash_added_when_changed():
    name = tool_name_for("Folio", "Product Docs")
    assert name.startswith("Folio_Product_Docs_") and len(name.split("_")[-1]) == 8


@pytest.mark.parametrize("a, b", [
    ("產品手冊", "使用手冊"),            # both slug to nothing readable
    ("a b", "a_b"),                     # same slug, one needed changing
    ("a" * 100 + "x", "a" * 100 + "y"),  # differ only after truncation
    ("café", "cafe"),
])
def test_distinct_folders_get_distinct_names(a, b):
    assert tool_name_for("Folio", a) != tool_name_for("Folio", b)


def test_prefix_is_sanitised_too():
    assert _VALID.match(tool_name_for("My RAG!", "docs"))


@pytest.fixture()
def manager():
    DynamicToolManager._instance = None
    m = DynamicToolManager.initialize(MagicMock())
    m._registered_tools = {}
    m._token_folder_to_tool = {}
    m._token_folder_descriptions = {}
    m._tool_to_folder = {}
    cfg = SimpleNamespace(app=SimpleNamespace(tool_prefix="Folio"),
                          rag=SimpleNamespace(retrieval=SimpleNamespace(
                              default_top_k=5, default_similarity_cutoff=0.3,
                              expand_context_default=True)))
    with patch("src.fastmcp_tools.dynamic_tool_manager.Config.get_config_model", return_value=cfg):
        yield m
    DynamicToolManager._instance = None


def test_register_uses_safe_name_and_reverse_maps(manager):
    with patch("src.fastmcp_tools.dynamic_tool_manager.FolderDB.get",
               return_value=[SimpleNamespace(name="產品 手冊")]):
        assert manager.register_folder_tool("產品 手冊", MagicMock(), user_token="jti-a")
    (tool_name,) = manager.get_tools_by_token("jti-a")
    assert _VALID.match(tool_name)
    assert manager._mcp.add_tool.call_args.args[0].name == tool_name
    assert manager.get_folder_name_from_tool(tool_name) == "產品 手冊"


def test_unregister_clears_reverse_map(manager):
    with patch("src.fastmcp_tools.dynamic_tool_manager.FolderDB.get",
               return_value=[SimpleNamespace(name="docs x")]):
        manager.register_folder_tool("docs x", MagicMock(), user_token="jti-a")
    (tool_name,) = manager.get_tools_by_token("jti-a")
    manager.unregister_folder_tool("docs x", "jti-a")
    assert manager.get_folder_name_from_tool(tool_name) is None


def test_tool_description_is_english(manager):
    with patch("src.fastmcp_tools.dynamic_tool_manager.FolderDB.get",
               return_value=[SimpleNamespace(name="docs")]):
        manager.register_folder_tool("docs", MagicMock(), folder_description="Specs",
                                     user_token="jti-a")
    desc = manager._mcp.add_tool.call_args.args[0].description
    assert not re.search(r"[一-鿿]", desc), desc
    assert "mode" in desc and "Specs" in desc


def test_default_prefix_is_folio(manager):
    with patch("src.fastmcp_tools.dynamic_tool_manager.FolderDB.get",
               return_value=[SimpleNamespace(name="docs")]):
        manager.register_folder_tool("docs", MagicMock(), user_token="jti-a")
    assert manager.get_tools_by_token("jti-a") == ["Folio_docs"]


def test_empty_prefix_falls_back_to_folio():
    assert tool_name_for("", "docs") == "Folio_docs"
    assert tool_name_for("!!!", "docs") == "Folio_docs"


def test_configured_prefix_is_used():
    from src.fastmcp_tools.dynamic_tool_manager import configured_tool_prefix
    cfg = SimpleNamespace(app=SimpleNamespace(tool_prefix="KB Tools"))
    with patch("src.fastmcp_tools.dynamic_tool_manager.Config.get_config_model", return_value=cfg):
        assert configured_tool_prefix() == "KB_Tools"
    with patch("src.fastmcp_tools.dynamic_tool_manager.Config.get_config_model", return_value=None):
        assert configured_tool_prefix() == "Folio"


def test_pre_rename_config_prefix_becomes_folio():
    """A config.yaml from before the rename (tool_prefix: Agentic) loads as Folio."""
    from src.config.model import AppConfig
    app = AppConfig.model_validate({"name": "Agentic RAG", "tool_prefix": "Agentic"})
    assert (app.name, app.tool_prefix) == ("Folio", "Folio")
    assert len(app.renamed_defaults) == 2
    custom = AppConfig.model_validate({"name": "Team KB", "tool_prefix": "KB"})
    assert (custom.name, custom.tool_prefix, custom.renamed_defaults) == ("Team KB", "KB", [])


def test_auth_filter_prefix_fallback_follows_config():
    """A folder tool unknown to the manager is still recognised by the configured prefix."""
    from src.fastmcp_tools.auth_filtered_mcp import AuthFilteredFastMCP
    DynamicToolManager._instance = None
    srv = AuthFilteredFastMCP("t")
    cfg = SimpleNamespace(app=SimpleNamespace(tool_prefix="Folio"))
    with patch("src.fastmcp_tools.dynamic_tool_manager.Config.get_config_model", return_value=cfg):
        assert srv._is_dynamic_folder_tool("Folio_docs")
        assert not srv._is_dynamic_folder_tool("Agentic_docs")
        assert not srv._is_dynamic_folder_tool("health_check")
