"""MCP Gate revocation feed: reject revoked tokens that still verify offline.

Folio verifies MCP Gate's RS256 tokens offline, so on its own a token revoked in
MCP Gate keeps working until it expires (a personal access token can live a year).
This module closes that gap without introspecting every request:

* a background task polls ``POST <issuer>/oauth/revoked`` every
  ``poll_interval_sec`` (with jitter) and keeps an in-memory map ``jti -> exp`` of
  tokens that were revoked but have not expired yet;
* every verification path (the FastMCP ``/mcp`` transport, the REST/MCP-filter
  verifier, console sign-in and every console request) calls ``check_revoked(jti)``
  and rejects a listed token.

The first poll is a full fetch; later polls pass ``since`` (MCP Gate's own clock,
minus an overlap) so only new revocations come back. A full fetch is repeated every
hour as a safety net. Entries are dropped once the token would have expired anyway.

The feed is only served to a *resource server*: a confidential client registered in
the MCP Gate console and bound to this server. Its ``client_id`` / ``client_secret``
come from ``auth.revocation`` (the secret only as an ``${ENV_VAR}`` reference).

``fail_mode`` decides what happens when the feed cannot be reached:

* ``open`` (default) keeps accepting tokens and logs loudly — a revocation made
  while the feed is down takes effect once it is reachable again;
* ``closed`` rejects every token once the last successful poll is older than
  ``stale_after_intervals`` x ``poll_interval_sec`` (and until the first poll
  succeeds after startup).
"""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

import httpx

from src.log import get_auth_logger

logger = get_auth_logger()

FAIL_OPEN = "open"
FAIL_CLOSED = "closed"

# Rejection reasons returned by check_revoked (also used as verifier error codes).
REVOKED = "token_revoked"
UNAVAILABLE = "revocation_unavailable"

# `since` overlaps the previous window so a revocation committed while a poll was in
# flight is not missed; duplicates are harmless (it is a map).
SINCE_OVERLAP_SEC = 30
# A full fetch every hour replaces the map, so nothing can drift for long.
FULL_RESYNC_SEC = 3600
# Keep an entry this long past its exp: covers clock skew between MCP Gate and us.
PRUNE_GRACE_SEC = 60
# Error backoff doubles from poll_interval up to this cap (or poll_interval if larger).
MAX_BACKOFF_SEC = 300
# +/- fraction of jitter applied to every delay so a fleet does not poll in lockstep.
JITTER = 0.1
REQUEST_TIMEOUT_SEC = 10.0


class RevocationFeedError(Exception):
    """The revocation feed answered with an error or an unexpected body."""


@dataclass(frozen=True)
class RevocationSettings:
    """Resolved ``auth.revocation`` settings."""

    issuer: str
    client_id: str
    client_secret: str = field(repr=False)
    poll_interval: float = 30.0
    fail_mode: str = FAIL_OPEN
    stale_after_intervals: int = 4

    @property
    def feed_url(self) -> str:
        return f"{self.issuer.rstrip('/')}/oauth/revoked"

    @property
    def stale_after(self) -> float:
        """Seconds without a successful poll after which the list counts as stale."""
        return self.poll_interval * self.stale_after_intervals


def _iso(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


class RevocationList:
    """In-memory list of revoked-but-unexpired jtis, kept fresh by polling MCP Gate."""

    def __init__(
        self,
        settings: RevocationSettings,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Any] = asyncio.sleep,
        rng: Callable[[float, float], float] = random.uniform,
    ):
        self.settings = settings
        self._own_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SEC)
        self._clock = clock
        self._sleep = sleep
        self._rng = rng
        self._revoked: dict[str, int] = {}  # jti -> exp (unix seconds)
        self._since: Optional[int] = None  # MCP Gate clock; None -> next poll is a full fetch
        self._last_full: Optional[float] = None
        self._last_success: Optional[float] = None
        self._last_attempt: Optional[float] = None
        self._failures = 0
        self._last_error: Optional[str] = None
        self._stale_logged = False
        self._task: Optional[asyncio.Task] = None

    # -- lookups ------------------------------------------------------------------

    def __contains__(self, jti: object) -> bool:
        return isinstance(jti, str) and jti in self._revoked

    def __len__(self) -> int:
        return len(self._revoked)

    def is_stale(self) -> bool:
        """True when no poll succeeded within ``stale_after`` (or none succeeded yet)."""
        if self._last_success is None:
            return True
        return self._clock() - self._last_success > self.settings.stale_after

    def check(self, jti: Optional[str]) -> Optional[str]:
        """Return a rejection reason for a verified token's jti, or None to accept it."""
        if jti and jti in self._revoked:
            return REVOKED
        if self.settings.fail_mode == FAIL_CLOSED and self.is_stale():
            return UNAVAILABLE
        return None

    # -- polling ------------------------------------------------------------------

    def _needs_full_fetch(self) -> bool:
        return (self._since is None or self._last_full is None
                or self._clock() - self._last_full >= FULL_RESYNC_SEC)

    async def poll_once(self) -> int:
        """Fetch the feed once and update the list. Returns how many entries came back.

        Raises on any failure (network, HTTP status, malformed body); the caller
        records it and backs off.
        """
        full = self._needs_full_fetch()
        form = {"client_id": self.settings.client_id, "client_secret": self.settings.client_secret}
        if not full:
            form["since"] = str(self._since)
        self._last_attempt = self._clock()
        resp = await self._http.post(self.settings.feed_url, data=form)
        if resp.status_code != 200:
            raise RevocationFeedError(f"HTTP {resp.status_code}: {_error_text(resp)}")
        try:
            data = resp.json()
        except ValueError as e:
            raise RevocationFeedError(f"response is not JSON: {e}") from e
        items = data.get("revoked") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise RevocationFeedError("response has no 'revoked' list")
        try:
            server_now = int(data.get("now") or self._clock())
        except (TypeError, ValueError):
            server_now = int(self._clock())

        entries: dict[str, int] = {}
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(item.get("jti"), str):
                continue
            try:
                exp = int(item.get("exp"))
            except (TypeError, ValueError):
                # No usable exp: keep it for the longest a token can live.
                exp = server_now + 366 * 86400
            entries[item["jti"]] = exp

        if full:
            self._revoked = entries
            self._last_full = self._clock()
        else:
            self._revoked.update(entries)
        self._prune(max(server_now, int(self._clock())))
        self._since = max(0, server_now - SINCE_OVERLAP_SEC)

        if self._failures:
            logger.info(f"Revocation feed reachable again after {self._failures} failed poll(s)")
        self._failures = 0
        self._last_error = None
        self._stale_logged = False
        self._last_success = self._clock()
        return len(entries)

    def _prune(self, now: int) -> None:
        for jti in [j for j, exp in self._revoked.items() if exp + PRUNE_GRACE_SEC < now]:
            del self._revoked[jti]

    def _record_failure(self, exc: BaseException) -> None:
        self._failures += 1
        self._last_error = f"{type(exc).__name__}: {exc}"[:300]
        since = _iso(self._last_success) or "startup"
        if self.settings.fail_mode == FAIL_CLOSED:
            consequence = (f"tokens will be rejected once no poll has succeeded for "
                           f"{self.settings.stale_after:.0f}s")
        else:
            consequence = "fail_mode=open: tokens revoked in MCP Gate since then are still accepted"
        logger.warning(
            f"Revocation feed poll failed ({self._failures} in a row; last success: {since}): "
            f"{self._last_error} - {consequence}")
        if self.is_stale() and not self._stale_logged:
            self._stale_logged = True
            if self.settings.fail_mode == FAIL_CLOSED:
                logger.error("Revocation list is STALE and fail_mode=closed: REJECTING EVERY TOKEN "
                             "until MCP Gate's revocation feed is reachable again")
            else:
                logger.error("Revocation list is STALE (fail_mode=open): revoked tokens may be "
                             "accepted until MCP Gate's revocation feed is reachable again")

    def _jittered(self, delay: float) -> float:
        return max(1.0, delay * (1 + self._rng(-JITTER, JITTER)))

    def next_delay(self) -> float:
        """Seconds until the next poll: the interval, or an exponential backoff after errors."""
        interval = self.settings.poll_interval
        if not self._failures:
            return self._jittered(interval)
        cap = max(interval, MAX_BACKOFF_SEC)
        return self._jittered(min(interval * 2 ** min(self._failures - 1, 16), cap))

    async def run(self) -> None:
        """Poll forever (until cancelled)."""
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep the loop alive; the next tick retries
                self._record_failure(e)
            await self._sleep(self.next_delay())

    # -- lifecycle ----------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the poller on the running loop (idempotent)."""
        if self.running:
            return
        if self._own_client and self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SEC)
        self._task = asyncio.get_running_loop().create_task(
            self.run(), name="mcp-gate-revocation-poller")

    async def stop(self) -> None:
        """Cancel the poller and close the HTTP client it owns."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if self._own_client:
            await self._http.aclose()

    # -- observability ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        stale = self.is_stale()
        return {
            "status": "degraded" if stale else "healthy",
            "enabled": True,
            "running": self.running,
            "fail_mode": self.settings.fail_mode,
            "poll_interval_sec": self.settings.poll_interval,
            "revoked_count": len(self._revoked),
            "last_success": _iso(self._last_success),
            "last_attempt": _iso(self._last_attempt),
            "consecutive_failures": self._failures,
            "last_error": self._last_error,
            "stale": stale,
        }


def _error_text(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("error"):
            desc = body.get("error_description")
            return f"{body['error']}" + (f" ({desc})" if desc else "")
    except ValueError:
        pass
    return (resp.text or "")[:200]


# -- configuration ------------------------------------------------------------------


def build_revocation_list(config, *, issuer: Optional[str],
                          env_refs: Optional[Mapping[str, str]] = None,
                          http_client: Optional[httpx.AsyncClient] = None) -> Optional[RevocationList]:
    """Build the revocation list from ``config.auth.revocation``; None when disabled.

    Raises RuntimeError on a misconfiguration (enabled without credentials, a literal
    client secret, an unknown fail_mode) so the server refuses to start rather than
    silently running without revocation.
    """
    auth = getattr(config, "auth", None)
    rev = getattr(auth, "revocation", None)
    if rev is None or not getattr(rev, "enabled", False):
        return None
    if not getattr(auth, "enabled", False) or not issuer:
        logger.warning("auth.revocation.enabled is ignored: authentication is disabled")
        return None
    client_id = (getattr(rev, "client_id", None) or "").strip()
    client_secret = (getattr(rev, "client_secret", None) or "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "auth.revocation.enabled is true but client_id / client_secret are empty: register a "
            "confidential client in MCP Gate, bind it to this server, and set its credentials")
    if env_refs is not None and "auth.revocation.client_secret" not in env_refs:
        raise RuntimeError(
            "auth.revocation.client_secret must be an ${ENV_VAR} reference, not a literal value "
            "(e.g. client_secret: \"${FOLIO_REVOCATION_CLIENT_SECRET}\")")
    fail_mode = str(getattr(rev, "fail_mode", FAIL_OPEN) or FAIL_OPEN)
    if fail_mode not in (FAIL_OPEN, FAIL_CLOSED):
        raise RuntimeError(f"auth.revocation.fail_mode must be 'open' or 'closed', got {fail_mode!r}")
    settings = RevocationSettings(
        issuer=str(issuer).rstrip("/"),
        client_id=client_id,
        client_secret=client_secret,
        poll_interval=float(getattr(rev, "poll_interval_sec", 30) or 30),
        fail_mode=fail_mode,
        stale_after_intervals=int(getattr(rev, "stale_after_intervals", 4) or 4),
    )
    return RevocationList(settings, http_client=http_client)


# -- module state ---------------------------------------------------------------------

_revocation: Optional[RevocationList] = None


def set_revocation_list(rl: Optional[RevocationList]) -> None:
    global _revocation
    _revocation = rl


def get_revocation_list() -> Optional[RevocationList]:
    return _revocation


def check_revoked(jti: Optional[str]) -> Optional[str]:
    """Rejection reason for a verified token's jti, or None (also when revocation is off)."""
    rl = _revocation
    return rl.check(jti) if rl is not None else None


def revocation_status() -> dict[str, Any]:
    """Health view of the revocation feed (``{"status": "disabled"}`` when off)."""
    rl = _revocation
    if rl is None:
        return {"status": "disabled", "enabled": False}
    return rl.status()


@asynccontextmanager
async def revocation_lifespan():
    """Run the revocation poller for the lifetime of the app (no-op when disabled)."""
    rl = _revocation
    if rl is not None:
        rl.start()
    try:
        yield
    finally:
        if rl is not None:
            await rl.stop()
