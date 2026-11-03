
"""Authentication system initialization.

Builds the offline MCP Gate token verifier (RS256 JWT checked against the issuer's
JWKS) and registers it via ``set_token_verifier``, where the REST auth dependency and
the MCP tool filter pick it up. The admin console reuses the same verifier.

Also builds MCP Gate's revocation list (``auth.revocation``); app.py's lifespan runs
its poller via ``revocation_lifespan``.
"""

from src.log import get_api_logger
from src.auth.console_session import build_console_auth, set_console_auth
from src.auth.dependencies import set_token_verifier
from src.auth.mcp_gate_auth import McpGateTokenVerifier, settings_from_config
from src.auth.revocation import build_revocation_list, set_revocation_list

api_logger = get_api_logger()


def initialize_auth_system(config):
    """Initialize authentication — verify MCP Gate OAuth 2.1 access tokens offline."""

    auth_config = getattr(config, "auth", None)
    auth_enabled = getattr(auth_config, "enabled", False) if auth_config else False

    if not auth_enabled:
        api_logger.info("Authentication system is disabled")
        set_revocation_list(None)
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

    _init_revocation(config, settings.issuer)

    verifier = McpGateTokenVerifier(settings)
    set_token_verifier(verifier)
    # Admin console login reuses the same offline verifier for the pasted PAT.
    set_console_auth(build_console_auth(config, verifier=verifier))
    api_logger.success("MCP Gate authentication system initialized (offline RS256 JWT)")


def _init_revocation(config, issuer: str) -> None:
    """Build (or clear) the revocation list and say once whether it is on."""
    from src.config.config_manager import Config

    rl = build_revocation_list(config, issuer=issuer, env_refs=Config.get_env_refs())
    set_revocation_list(rl)
    if rl is None:
        api_logger.warning(
            "MCP Gate revocation feed: DISABLED (auth.revocation.enabled: false) - a token "
            "revoked in MCP Gate stays valid here until it expires")
        return
    s = rl.settings
    api_logger.info(
        f"MCP Gate revocation feed: ENABLED (feed: {s.feed_url}, client: {s.client_id}, "
        f"poll every {s.poll_interval:g}s, fail_mode: {s.fail_mode})")
