"""Query input limits shared by REST, the console playground and MCP search.

An over-long query used to reach the embedding endpoint, which rejected it for
exceeding the model's context window; the error surfaced as HTTP 500. A character
cap catches the common case up front (422), and is_input_too_long_error() maps the
provider's own error to a 4xx when a query under the cap still exceeds a small
model's window (e.g. 512-token e5 models with dense CJK text).
"""

from __future__ import annotations

MIN_QUERY_CHARS = 2
# Generous for natural-language questions, far below the 8k-token windows of the
# common embedding models; tighter windows are covered by the error mapping below.
MAX_QUERY_CHARS = 2000

QUERY_TOO_LONG_DETAIL = (
    "Query is too long for the embedding model's context window — shorten it"
)

# Lower-cased fragments that OpenAI-compatible servers (OpenAI, vLLM, TEI, Ollama,
# Azure) use when an input exceeds the model's context window.
_CONTEXT_ERROR_FRAGMENTS = (
    "maximum context length",
    "context_length_exceeded",
    "context length",
    "input is too long",
    "too many tokens",
    "maximum sequence length",
    "exceeds the model's maximum",
    "input validation error: `inputs` must have less than",
)


def is_input_too_long_error(exc: BaseException) -> bool:
    """True when ``exc`` (or anything in its cause/context chain) is a context-window error."""
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur).lower()
        if any(f in msg for f in _CONTEXT_ERROR_FRAGMENTS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False
