
"""Authentication system initialization.

Builds the offline MCP Center token verifier and registers it via
``set_remote_verifier`` — the SAME registration contract the legacy remote Token
Server flow used, so the middleware, dependencies and MCP tool filter keep working
unchanged. Only the verifier implementation changed (offline RS256 JWKS instead of an
HTTP call to a token server).
"""

from src.log import get_api_logger
from src.auth.dependencies import set_remote_verifier
from src.auth.mcp_center_auth import McpCenterTokenVerifier, settings_from_config

api_logger = get_api_logger()


def initialize_auth_system(config):
    """Initialize authentication — verify MCP Center OAuth 2.1 access tokens offline."""

    auth_config = getattr(config, "auth", None)
    auth_enabled = getattr(auth_config, "enabled", False) if auth_config else False

    if not auth_enabled:
        api_logger.info("Authentication system is disabled")
        return

    # Resolve settings (issuer/audience/scopes). Raises clearly if issuer is missing.
    settings = settings_from_config(config)
    if settings is None:
        # settings_from_config only returns None when auth is disabled, already handled.
        return

    api_logger.info(
        f"Initializing MCP Center authentication "
        f"(issuer: {settings.issuer}, audience: {settings.audience}, "
        f"jwks: {settings.jwks_uri})"
    )

    verifier = McpCenterTokenVerifier(settings)
    set_remote_verifier(verifier)
    api_logger.success("MCP Center authentication system initialized (offline RS256 JWT)")
