
"""Tool Handlers — re-export shim.

The three-mode handlers (handle_search / handle_list / handle_read) now live in
the domain layer: src/domain/rag/agentic_handlers.py. They have no fastmcp
dependency, and keeping them in the delivery layer created a reverse adapter
import (a fastmcp -> adapter -> fastmcp layering cycle).

This file preserves the original import path for backward compatibility; new
code should import the domain module directly.
"""

from src.domain.rag.agentic_handlers import (  # noqa: F401
    handle_list,
    handle_read,
    handle_search,
)
