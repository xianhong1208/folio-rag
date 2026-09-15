
"""Agentic RAG Control Center — console HTML loader.

Reads the standalone `web/admin_console.html` into `ADMIN_PAGE_HTML` at import time. The file lives
under top-level `web/` (checked into git, shipped with the build) because the console HTML is code
and must travel with the app; Nuitka's `data_dirs` must include `web`, and path resolution goes
through the onefile-aware `resolve_external_dir("web", …)` so it works in dev / Docker / onefile.
"""

from pathlib import Path

from src.log import get_api_logger

logger = get_api_logger()

_HTML_NAME = "admin_console.html"


def _load_admin_html() -> str:
    """Read the console HTML (runs once at import). Returns a minimal fallback page if not found, so import never crashes."""
    dev_root = Path(__file__).resolve().parents[3]  # repo root (src/api/router -> three levels up)
    candidates = []
    try:
        from src.utils.runtime_paths import resolve_external_dir
        web = resolve_external_dir("web", dev_root)
        if web:
            candidates.append(web / _HTML_NAME)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ADMIN_HTML] resolve_external_dir unavailable: {e}")
    candidates.append(dev_root / "web" / _HTML_NAME)  # dev / onefile bundled fallback
    for p in candidates:
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ADMIN_HTML] read failed {p}: {e}")
    logger.error(f"[ADMIN_HTML] admin_console.html not found (looked: {[str(c) for c in candidates]})")
    return ("<!DOCTYPE html><html><body style='font-family:sans-serif;padding:40px'>"
            "<h2>Agentic RAG Control Center</h2><p>admin_console.html not found — "
            "confirm <code>web/admin_console.html</code> ships with the build "
            "(the packaging config data_dirs must include <code>web</code>).</p>"
            "</body></html>")


ADMIN_PAGE_HTML = _load_admin_html()
