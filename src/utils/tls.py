"""TLS trust for model endpoints served behind a private CA (e.g. Forgeyard's own CA).

``rag.asr.ca_file`` and ``rag.parsing.vision.ca_file`` name a PEM file that is *added* to
the default trust store (httpx honours ``SSL_CERT_FILE`` for the default), so every public CA
keeps working. A missing or unreadable file is an error, never a silent fallback to
unverified TLS.
"""

from __future__ import annotations

import ssl
from typing import Optional, Union

import httpx


def tls_verify(ca_file: Optional[str], setting: str = "ca_file") -> Union[bool, ssl.SSLContext]:
    """httpx ``verify`` for an endpoint: True (the default bundle), or a context that also
    trusts ``ca_file``. ``setting`` names the config field in the error message."""
    if not ca_file:
        return True
    ctx = httpx.create_ssl_context()
    try:
        ctx.load_verify_locations(cafile=ca_file)
    except (OSError, ssl.SSLError) as e:
        raise RuntimeError(f"{setting} {ca_file!r} could not be loaded: {e}") from e
    return ctx


def is_certificate_error(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything in its cause / context chain) is a failed TLS
    certificate verification: the server's CA is not trusted here."""
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(cur):
            return True
        cur = cur.__cause__ or cur.__context__
    return False
