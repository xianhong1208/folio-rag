
"""Authentication system initialization.

Builds the offline MCP Gate token verifier (RS256 JWT checked against the issuer's
JWKS) and registers it via ``set_token_verifier``, where the REST auth dependency and
the MCP tool filter pick it up. The admin console reuses the same verifier.
"""

from src.log import get_api_logger
from src.auth.console_session import build_console_auth, set_console_auth
from src.auth.dependencies import set_token_verifier
from src.auth.mcp_gate_auth import McpGateTokenVerifier, settings_from_config

api_logger = get_api_logger()


def initialize_auth_system(config):
    """Initialize authentication — verify MCP Gate OAuth 2.1 access tokens offline."""

    auth_config = getattr(config, "auth", None)
    auth_enabled = getattr(auth_config, "enabled", False) if auth_config else False

    if not auth_enabled:
        api_logger.info("Authentication system is disabled")
        # Console runs in open (dev) mode; build_console_auth logs the warning.
        set_console_auth(build_console_auth(config))
        return

    # Resolve settings (issuer/audience/scopes). Raises clearly if issuer is missing.
    settings = settings_from_config(config)
    if settings is None:
        # settings_from_config only returns None when auth is disabled, already handled.
        return

    api_logger.info(
        f"Initializing MCP Gate authentication "
        f"(issuer: {settings.issuer}, audience: {settings.audience}, "
        f"jwks: {settings.jwks_uri})"
    )

    verifier = McpGateTokenVerifier(settings)
    set_token_verifier(verifier)
    # Admin console login reuses the same offline verifier for the pasted PAT.
    set_console_auth(build_console_auth(config, verifier=verifier))
    api_logger.success("MCP Gate authentication system initialized (offline RS256 JWT)")
