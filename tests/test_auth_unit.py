"""Unit tests for src/auth/mcp_gate_auth.py (the FastMCP auth provider)."""

from types import SimpleNamespace


def test_protected_resource_metadata_advertises_issuer_unchanged():
    """mcp 2 clients compare the advertised authorization server with the issuer as an exact string
    (RFC 8414 3.3): "http://gate.test:4568/" would be rejected as a mismatch."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from src.auth.mcp_gate_auth import build_auth_provider

    config = SimpleNamespace(
        auth=SimpleNamespace(enabled=True, issuer="http://gate.test:4568", audience=None, required_scopes=None),
        server=SimpleNamespace(host="127.0.0.1", port=5039),
    )
    routes = [r for r in build_auth_provider(config).get_routes("/mcp") if "protected-resource" in r.path]
    metadata = TestClient(Starlette(routes=routes)).get(routes[0].path).json()
    assert metadata["authorization_servers"] == ["http://gate.test:4568"]
