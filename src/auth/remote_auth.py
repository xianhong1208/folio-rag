
"""Remote token verification client.

Used by the MCP Server to authenticate via HTTP calls to the Token Server.
"""

import time
import asyncio
import hashlib
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass

import httpx

from src.log import get_auth_logger

logger = get_auth_logger()


@dataclass
class TokenVerifyResult:
    """Token verification result."""
    valid: bool
    user_name: Optional[str] = None
    scopes: Optional[list] = None
    expires_at: Optional[int] = None
    error: Optional[str] = None


class RemoteTokenVerifier:
    """Remote token verifier.

    Verifies via HTTP calls to the Token Server's /auth/verify API.

    Usage:
        verifier = RemoteTokenVerifier("http://localhost:8000")
        result = await verifier.verify("token_string")
        if result.valid:
            print(f"User: {result.user_name}")
    """

    def __init__(
        self,
        token_server_url: str,
        timeout: float = 5.0,
        retry_count: int = 2,
        cache_ttl: int = 60,
        service_host: Optional[str] = None,
        service_port: Optional[int] = None
    ):
        """Initialize the remote verifier.

        Args:
            token_server_url: Base URL of the Token Server (e.g. http://localhost:8000)
            timeout: HTTP request timeout (seconds)
            retry_count: Number of retries on failure
            cache_ttl: Verification-result cache lifetime (seconds)
            service_host: The host this service is registered under in the Token Server
            service_port: The port this service is registered under in the Token Server
        """
        self.token_server_url = token_server_url.rstrip('/')
        self.verify_url = f"{self.token_server_url}/auth/verify"
        self.timeout = timeout
        self.retry_count = retry_count
        self.service_host = service_host
        self.service_port = service_port

        # Simple in-memory cache (reduces requests to the Token Server)
        self._cache: Dict[str, tuple[TokenVerifyResult, float]] = {}
        self._cache_ttl = cache_ttl
        self._cache_lock = threading.Lock()

        # Persistent HTTP client (avoids Nuitka + anyio cancel-scope issues).
        # Note: only the sync client is used; async methods call it via asyncio.to_thread.
        self._sync_client: Optional[httpx.Client] = None

        logger.info(f"RemoteTokenVerifier initialized: {self.verify_url} (cache_ttl: {cache_ttl}s)")

    def _get_cache_key(self, token: str) -> str:
        """Generate a cache key (uses SHA-256 so the raw token doesn't stay in memory)."""
        return hashlib.sha256(token.encode()).hexdigest()

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.timeout)
        return self._sync_client

    async def verify(self, token: str) -> TokenVerifyResult:
        """Verify a token.

        Args:
            token: The token string to verify.

        Returns:
            TokenVerifyResult: the verification result.
        """
        cache_key = self._get_cache_key(token)

        cached = self._get_from_cache(cache_key)
        if cached:
            logger.debug(f"Token verification cache hit: {token[:8]}...")
            return cached

        result = await self._call_verify_api(token)

        if result.valid:
            self._set_cache(cache_key, result)

        return result

    def verify_sync(self, token: str) -> TokenVerifyResult:
        """Synchronous version of token verification (for non-async contexts)."""
        cache_key = self._get_cache_key(token)

        cached = self._get_from_cache(cache_key)
        if cached:
            logger.debug(f"Token verification cache hit: {token[:8]}...")
            return cached

        result = self._call_verify_api_sync(token)

        if result.valid:
            self._set_cache(cache_key, result)

        return result

    async def _call_verify_api(self, token: str) -> TokenVerifyResult:
        """Asynchronously call the Token Server API (wraps sync httpx via asyncio.to_thread).

        After Nuitka compilation, anyio's cancel scope (happy eyeballs) misbehaves, so a sync client is used.

        Args:
            token: The token to verify.

        Returns:
            TokenVerifyResult (valid + user_name + scopes, or error).
        """
        # Run the sync version in a thread, avoiding anyio entirely
        return await asyncio.to_thread(self._call_verify_api_sync, token)

    def _call_verify_api_sync(self, token: str) -> TokenVerifyResult:
        """Synchronously call the Token Server API (persistent httpx.Client + retries).

        Args:
            token: The token to verify.

        Returns:
            TokenVerifyResult; on exhausted retries or a non-retryable error, valid=False with error details.
        """
        last_error = None

        # Build the request payload (include service_host + service_port so the Token Server can verify ownership)
        payload = {"token": token}
        if self.service_host and self.service_port:
            payload["service_host"] = self.service_host
            payload["service_port"] = self.service_port

        client = self._get_sync_client()

        for attempt in range(self.retry_count + 1):
            try:
                response = client.post(
                    self.verify_url,
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()
                    return TokenVerifyResult(
                        valid=data.get("valid", False),
                        user_name=data.get("user_name"),
                        scopes=data.get("scopes"),
                        expires_at=data.get("expires_at"),
                        error=data.get("error")
                    )
                elif 400 <= response.status_code < 500:
                    # Client errors are not retryable
                    logger.warning(f"Token Server returned {response.status_code}")
                    return TokenVerifyResult(
                        valid=False,
                        error=f"token_server_error_{response.status_code}"
                    )
                else:
                    # 5xx server errors — retry
                    last_error = f"token_server_error_{response.status_code}"
                    logger.warning(f"Token Server returned {response.status_code} (attempt {attempt + 1})")

            except httpx.TimeoutException:
                last_error = "timeout"
                logger.warning(f"Token Server timeout (attempt {attempt + 1})")
            except httpx.ConnectError:
                last_error = "connection_error"
                logger.warning(f"Cannot connect to Token Server (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error calling Token Server: {e}")

        logger.error(f"Token verification failed after {self.retry_count + 1} attempts")
        return TokenVerifyResult(valid=False, error=last_error)

    def _get_from_cache(self, cache_key: str) -> Optional[TokenVerifyResult]:
        with self._cache_lock:
            if cache_key in self._cache:
                result, cached_time = self._cache[cache_key]
                if time.time() - cached_time < self._cache_ttl:
                    return result
                else:
                    del self._cache[cache_key]
            return None

    def _set_cache(self, cache_key: str, result: TokenVerifyResult):
        with self._cache_lock:
            self._cache[cache_key] = (result, time.time())

            # Clean up expired cache entries (simple LRU)
            if len(self._cache) > 1000:
                current_time = time.time()
                expired_keys = [
                    k for k, (_, t) in self._cache.items()
                    if current_time - t >= self._cache_ttl
                ]
                for k in expired_keys:
                    del self._cache[k]

    def clear_cache(self):
        """Clear the entire cache (thread-safe)."""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Remote token verifier cache cleared")

    def _health_check_sync(self) -> bool:
        try:
            client = self._get_sync_client()
            response = client.get(f"{self.token_server_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Token Server health check failed: {e}")
            return False

    async def health_check(self) -> bool:
        """Check whether the Token Server is available.

        Uses asyncio.to_thread to avoid Nuitka + anyio compatibility issues.
        """
        return await asyncio.to_thread(self._health_check_sync)

    def close(self):
        """Close the HTTP client connection."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        logger.debug("RemoteTokenVerifier HTTP client closed")


_verifier_instance: Optional[RemoteTokenVerifier] = None


def get_remote_verifier(
    token_server_url: str = None,
    timeout: float = 5.0,
    retry_count: int = 2,
    cache_ttl: int = 60,
    service_host: Optional[str] = None,
    service_port: Optional[int] = None
) -> RemoteTokenVerifier:
    """Get or create the remote verifier instance.

    Args:
        token_server_url: Token Server URL (required on the first call)
        timeout: HTTP request timeout (seconds)
        retry_count: Number of retries on failure
        cache_ttl: Verification-result cache lifetime (seconds)
        service_host: The host this service is registered under in the Token Server
        service_port: The port this service is registered under in the Token Server

    Returns:
        The RemoteTokenVerifier instance.
    """
    global _verifier_instance

    if _verifier_instance is None:
        if token_server_url is None:
            raise ValueError("token_server_url is required for first initialization")
        _verifier_instance = RemoteTokenVerifier(
            token_server_url,
            timeout=timeout,
            retry_count=retry_count,
            cache_ttl=cache_ttl,
            service_host=service_host,
            service_port=service_port
        )

    return _verifier_instance


def reset_remote_verifier():
    """Reset the remote verifier instance (closes the HTTP client connection)."""
    global _verifier_instance
    if _verifier_instance is not None:
        _verifier_instance.close()
    _verifier_instance = None
