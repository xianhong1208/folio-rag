
"""Control Center data-plane API (read-only aggregates; mounted on the admin module router).

- GET /api/admin/overview            — overview KPIs (folder/file/index/job)
- GET /api/admin/folders             — your folders + index aggregates
- GET /api/admin/folders/{id}/files  — per-file index status for one folder
- GET /api/admin/jobs                — recent index jobs
- /api/admin/eval/*                  — retrieval evaluation for one of your folders

Auth: every endpoint needs a console session (or a bearer PAT). Per-folder data is
limited to folders owned by the session's person (sub), and so are cross-folder
aggregates (overview, jobs, analytics) — for admins too: ``rag:admin`` never reveals
other owners' folders or queries. The settings audit log and system resources
(storage totals, no folder content) are admin-only. See src/api/router/admin_scope.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from src.api.router.admin_scope import aggregate_owner, owned_folder, owned_folder_ids
from src.auth.console_session import (
    ConsoleIdentity,
    require_console_admin,
    require_console_session,
)
from src.log import get_api_logger

logger = get_api_logger()

router = APIRouter(tags=["Admin: Overview"], dependencies=[Depends(require_console_session)])

# Retrieval evaluation (harness) background run state — a single in-memory slot (evaluation is infrequent).
# "owner" is the owner key of the folder being evaluated; it is kept out of status responses.
_EVAL: dict = {"running": False, "folder": None, "done": 0, "total": 0,
               "result": None, "error": None, "started_at": None, "owner": None}


def _load_eval_mod():
    """Load scripts/rag_eval.py (not a package; loaded directly via importlib)."""
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    path = os.path.join(root, "scripts", "rag_eval.py")
    spec = importlib.util.spec_from_file_location("rag_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EVAL_TASK = None  # Keep a task reference so it isn't garbage-collected (the Task object stays out of _EVAL, which must remain JSON-serializable for status)


async def _run_eval_bg(folder_ref: str, n: int, k: int, regenerate: bool, judge: bool = False) -> None:
    """Evaluation body: runs on the main event loop (same loop as the cached PGVectorStore
    async engine, avoiding asyncpg's "attached to a different loop"). Results/errors are written back to _EVAL."""
    try:
        mod = _load_eval_mod()
        _EVAL["result"] = await mod.run_eval_async(
            folder_ref, n=n, k=k, regenerate=regenerate, judge=judge,
            progress=lambda d, t: _EVAL.update(done=d, total=t))
    except (Exception, SystemExit) as e:  # noqa: BLE001 — background run; store the error for status to report
        # SystemExit too: the eval module is a script loaded in-process, and an exit
        # escaping this task would stop the whole server.
        _EVAL["error"] = str(e)
        logger.warning(f"[EVAL] run failed: {e}")
    finally:
        _EVAL["running"] = False


@router.post("/api/admin/eval/run")
async def eval_run(
    body: dict = Body(default={}, examples=[{"folder_id": 1, "n": 15}]),
    ident: ConsoleIdentity = Depends(require_console_session),
):
    """Start one retrieval evaluation (background task on the main loop; generate synthetic
    questions -> three-way retrieval -> nDCG/Recall/MRR).

    After a reindex, pass regenerate=true (node_ids change, invalidating the old labeled set)."""
    import asyncio
    import time
    global _EVAL_TASK
    if _EVAL["running"]:
        raise HTTPException(409, "An evaluation is already running")
    try:
        folder_id = int(body.get("folder_id"))
    except (TypeError, ValueError):
        raise HTTPException(422, "folder_id is required")
    # Only your own folders; the run is then addressed by id (names are not unique
    # across owners).
    folder = owned_folder(folder_id, ident)
    folder_ref = str(folder.id)
    # Bound n/k: each question is one LLM generation + retrieval across four modes,
    # and there is a single global run slot, so an unbounded n (e.g. 5000) would
    # monopolize evaluation for every folder. Reject non-numeric / out-of-range
    # input with a clean 422 instead of a 500 or a runaway run.
    try:
        n = int(body.get("n", 15))
        k = int(body.get("k", 10))
    except (TypeError, ValueError):
        raise HTTPException(422, "n and k must be integers")
    if not (1 <= n <= 100):
        raise HTTPException(422, "n must be between 1 and 100")
    if not (1 <= k <= 50):
        raise HTTPException(422, "k must be between 1 and 50")
    regen = bool(body.get("regenerate", False))
    judge = bool(body.get("judge", False))
    # Check the folder before scheduling, so a bad request gets an HTTP error rather
    # than a background run that fails. 404: no such folder. 409: the folder exists
    # and the request is well-formed, but its current state (nothing indexed yet)
    # prevents the run; the same request succeeds once files are indexed.
    mod = _load_eval_mod()
    try:
        await asyncio.to_thread(mod.check_ready, folder_ref)
    except mod.FolderNotFound as e:
        raise HTTPException(404, str(e))
    except mod.NoIndexedChunks as e:
        raise HTTPException(409, str(e))
    if _EVAL["running"]:  # another request started a run while we were checking
        raise HTTPException(409, "An evaluation is already running")
    _EVAL.update(running=True, folder=folder_ref, done=0, total=n,
                 result=None, error=None, started_at=time.time(),
                 owner=folder.user_token)
    # Run on the main loop (no new thread/loop) so the shared asyncpg connection pool stays on one loop
    _EVAL_TASK = asyncio.create_task(_run_eval_bg(folder_ref, n, k, regen, judge))
    return {"data": {"started": True}, "message": "evaluation started"}


@router.get("/api/admin/eval/status")
async def eval_status(ident: ConsoleIdentity = Depends(require_console_session)):
    """Evaluation progress / results (polled by the frontend).

    There is one global run slot. Someone else's run is reported only as "busy":
    its folder, progress and results stay private to the folder's owner.
    """
    data = {k: v for k, v in _EVAL.items() if k != "owner"}
    if _EVAL["folder"] is not None and not ident.owns(_EVAL["owner"]):
        data = {"running": _EVAL["running"], "folder": None, "done": 0, "total": 0,
                "result": None, "error": None, "started_at": None, "busy": True}
    return {"data": data, "message": "ok"}


@router.get("/api/admin/eval/report")
async def eval_report(folder_id: int = Query(...),
                      ident: ConsoleIdentity = Depends(require_console_session)):
    """Read the folder's most recent evaluation report (reports/rag_eval_<id>_<name>.json); returns null if none."""
    import json
    import os
    from src.domain.rag.eval_paths import report_path, report_slug
    folder = owned_folder(folder_id, ident)
    path = report_path(report_slug(folder))
    if not os.path.exists(path):
        return {"data": None, "message": "no report yet"}
    with open(path, encoding="utf-8") as f:
        return {"data": json.load(f), "message": "ok"}


@router.get("/api/admin/eval/history")
async def eval_history(folder_id: int = Query(...), limit: int = Query(30, ge=1, le=200),
                       ident: ConsoleIdentity = Depends(require_console_session)):
    """The folder's past evaluation runs (oldest first) for baseline comparison / trend."""
    import json
    import os
    from src.domain.rag.eval_paths import history_path, report_slug
    folder = owned_folder(folder_id, ident)
    path = history_path(report_slug(folder))
    entries = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    return {"data": {"entries": entries[-limit:]}, "message": "ok"}


@router.get("/api/admin/overview")
async def get_overview(ident: ConsoleIdentity = Depends(require_console_session)):
    """Overview KPIs + in-progress jobs + a summary of the current model configuration.

    KPIs cover your own folders (``scope: own``); ``all`` only in open mode.
    Admins also get ``legacy_owned_folders``: how many folders server-wide still belong
    to a pre-1.4.1 per-token key, waiting for their token to be used again.
    """
    from db import admin_stats
    from src.config import runtime_overrides as ro
    from src.config.config_manager import Config

    owner = aggregate_owner(ident)
    data = admin_stats.overview(owner=owner)
    data["scope"] = "own" if owner is not None else "all"
    if ident.is_admin:
        from src.adapter.folder_claim import count_legacy_folders
        data["legacy_owned_folders"] = count_legacy_folders()
    try:
        data["trend"] = admin_stats.indexing_trend(7, owner=owner)
    except Exception:
        data["trend"] = []
    # Model configuration summary (for the overview page's service cards; reuses the settings whitelist's masking semantics)
    cfg = Config.get_config_model()
    eff = ro.get_effective(cfg) if cfg else {}
    try:
        from src.version import get_version
        data["version"] = get_version()
    except Exception:
        data["version"] = None
    data["models"] = {
        "embedding": {"model": eff.get("rag.embedding.model"),
                      "dimension": eff.get("rag.embedding.dimension"),
                      "base_url": eff.get("rag.embedding.base_url")},
        "llm": {"model": eff.get("rag.llm.model"), "base_url": eff.get("rag.llm.base_url")},
        "rerank": {"enabled": eff.get("rag.rerank.enabled"),
                   "model": eff.get("rag.rerank.model"),
                   "base_url": eff.get("rag.rerank.base_url")},
        "asr": {"enabled": eff.get("rag.asr.enabled"),
                "provider": eff.get("rag.asr.provider"),
                "available": _asr_available(cfg)},
    }
    # How retrieval fuses vector and BM25 results (the overview used to claim a fixed
    # ".75/.25" weighting that is not used).
    try:
        from src.domain.rag.hybrid_search import SearchSettings
        data["retrieval"] = {"fusion": SearchSettings.from_config().fusion}
    except Exception:
        data["retrieval"] = {"fusion": None}
    return {"data": data, "message": "ok"}


def _asr_available(cfg) -> bool:
    """Whether the ASR provider is ready (fireredasr = weights present; docling-whisper = .pt present;
    cloud = configuration complete). A cheap file-existence check that does not load any model."""
    try:
        from src.domain.rag.asr_provider import create_asr_provider
        asr_cfg = getattr(getattr(cfg, "rag", None), "asr", None)
        return bool(create_asr_provider(asr_cfg).available())
    except Exception:
        return False


@router.get("/api/admin/folders")
async def list_folders(ident: ConsoleIdentity = Depends(require_console_session)):
    """Your folders + per-folder index aggregates (file count / indexed / failed / chunks / vector table)."""
    from db import admin_stats
    return {"data": {"folders": admin_stats.folders_with_index_stats(owner=ident.owner_key)},
            "message": "ok"}


@router.get("/api/admin/folders/{folder_id}/files")
async def list_folder_files(folder_id: int,
                            ident: ConsoleIdentity = Depends(require_console_session)):
    """Per-file index status for a single folder (with truncated failure reasons)."""
    from db import admin_stats
    owned_folder(folder_id, ident)
    files = admin_stats.files_with_index_status(folder_id)
    return {"data": {"folder_id": folder_id, "files": files}, "message": "ok"}


@router.get("/api/admin/active-stages")
async def active_stages(ident: ConsoleIdentity = Depends(require_console_session)):
    """Which stage each currently-indexing file is at ({file_id: {stage,done,total,eta}}) —
    used by the file list to show "parse / generate context / vectorize / write" in real time.
    Limited to files in your own folders."""
    from src.domain.rag.index_job_manager import IndexingJobManager
    stages = IndexingJobManager.get_instance().active_file_stages()
    allowed = owned_folder_ids(ident)
    if allowed is not None:
        stages = {fid: st for fid, st in stages.items() if st.get("folder_id") in allowed}
    return {"data": {"stages": stages}, "message": "ok"}


@router.get("/api/admin/folders/{folder_id}/files/{file_id}/chunks")
async def list_file_chunks(folder_id: int, file_id: str,
                           limit: int = Query(500, ge=1, le=2000),
                           ident: ConsoleIdentity = Depends(require_console_session)):
    """The leaf chunks a file was split into (lets the customer inspect what the index looks like)."""
    from db import admin_stats
    owned_folder(folder_id, ident)
    return {"data": admin_stats.chunks_for_file(folder_id, file_id, limit), "message": "ok"}


@router.get("/api/admin/audit", dependencies=[Depends(require_console_admin)])
async def settings_audit(limit: int = Query(100, ge=1, le=500)):
    """Settings-change audit log (who changed what to what, and when). Admin only."""
    from src.adapter import runtime_settings_service as svc
    return {"data": {"entries": svc.audit_log(limit)}, "message": "ok"}


@router.get("/api/admin/resources", dependencies=[Depends(require_console_admin)])
async def system_resources_endpoint():
    """System resources: DB size / vector table size / disk. Admin only (server-wide)."""
    from db import admin_stats
    return {"data": admin_stats.system_resources(), "message": "ok"}


@router.get("/api/admin/jobs")
async def list_jobs(limit: int = Query(20, ge=1, le=100),
                    ident: ConsoleIdentity = Depends(require_console_session)):
    """Recent index jobs on your folders (ordered by last update), admins included.

    Each job carries ``owned`` (whether you can act on it, e.g. cancel)."""
    from db import admin_stats
    jobs = admin_stats.recent_jobs(limit, owner=aggregate_owner(ident), viewer=ident.owner_key)
    return {"data": {"jobs": jobs}, "message": "ok"}


@router.get("/api/admin/analytics")
async def query_analytics(
    folder_id: int | None = Query(None),
    days: int = Query(30, ge=1, le=365),
    ident: ConsoleIdentity = Depends(require_console_session),
):
    """Query analytics: query volume / average latency / zero-result rate / top queries /
    knowledge gaps / daily trend.

    folder_id must be one of your folders; omitting it = all your folders (admins
    included); days = the statistics window. The data source is QueryLog
    (best-effort one record per REST / console / MCP search). Knowledge gaps use
    rag.rerank.gap_threshold on each question's latest search."""
    from db.query_log_db import QueryLogDB
    scope_ids = None
    if folder_id is not None:
        owned_folder(folder_id, ident)
    elif aggregate_owner(ident) is not None:
        scope_ids = owned_folder_ids(ident)
    return {"data": QueryLogDB.analytics(folder_id=folder_id, days=days, folder_ids=scope_ids),
            "message": "ok"}


async def _probe_endpoint_health(base_url: str, api_key: str | None):
    """Probe an OpenAI-compatible endpoint (GET {base_url}/models); returns (status, detail, latency_ms).

    SSRF semantics match admin_settings.probe_endpoint: the intent is to probe internal model
    services, so no subnet filtering is applied; always GET /models, no redirect following, and
    errors report only the exception class name.
    """
    import time

    import httpx
    base = (base_url or "").rstrip("/")
    if not base:
        return "unset", "No endpoint configured", None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=headers, follow_redirects=False) as c:
            resp = await c.get(f"{base}/models")
            # Some OpenAI-compatible services (e.g. a vLLM reranker) expose the model
            # list only under /v1/ and are configured with a base_url that omits it,
            # so /models 404s even though the service is healthy. Fall back to
            # /v1/models before reporting, so the check reflects the real state.
            if resp.status_code == 404 and not base.endswith("/v1"):
                alt = await c.get(f"{base}/v1/models")
                if alt.status_code < resp.status_code:
                    resp = alt
        ms = int((time.perf_counter() - t0) * 1000)
        if resp.status_code < 500:
            return "ok", f"HTTP {resp.status_code}", ms
        return "down", f"HTTP {resp.status_code}", ms
    except Exception as e:  # noqa: BLE001 — return only the class name, never internal details
        return "down", type(e).__name__, int((time.perf_counter() - t0) * 1000)


def _probe_db_health():
    """Synchronously probe the DB (SELECT 1); returns (status, detail, latency_ms). Called via to_thread."""
    import time
    from sqlalchemy import text
    from db.db import get_engine
    t0 = time.perf_counter()
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return "ok", "Connection verified", int((time.perf_counter() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        return "down", type(e).__name__, int((time.perf_counter() - t0) * 1000)


@router.get("/api/admin/health")
async def health_check():  # read-only status; any console session may view it
    """Live health of external dependencies: DB / Embedding / LLM / Reranker / Speech-to-Text.

    Lets operators spot connection/endpoint problems at a glance (instead of digging through logs
    after the fact). Each entry returns status (ok/down/disabled/unset), detail, latency_ms, endpoint."""
    import asyncio

    from src.config import runtime_overrides as ro
    from src.config.config_manager import Config

    cfg = Config.get_config_model()
    # An unmasked api_key is needed to actually probe with Authorization (the masked value •••
    # is non-latin-1 and would blow up httpx header encoding with UnicodeEncodeError); the key is
    # only used to send the request, never returned.
    eff = ro.get_effective(cfg, mask_secrets=False) if cfg else {}
    checks = []

    # DB (sync engine; offloaded to a threadpool so it doesn't occupy the loop)
    db_status, db_detail, db_ms = await asyncio.to_thread(_probe_db_health)
    checks.append({"name": "Database", "kind": "db", "status": db_status,
                   "detail": db_detail, "latency_ms": db_ms,
                   "endpoint": (getattr(getattr(cfg, "database", None), "url", "") or "").split("@")[-1]})

    # Endpoint checks (embedding / llm / rerank) — probed in parallel
    endpoint_specs = [
        ("Embedding", "layers", eff.get("rag.embedding.base_url"), eff.get("rag.embedding.api_key"), True),
        ("LLM", "spark", eff.get("rag.llm.base_url"), eff.get("rag.llm.api_key"), True),
        ("Reranker", "sort", eff.get("rag.rerank.base_url"), eff.get("rag.rerank.api_key"),
         bool(eff.get("rag.rerank.enabled"))),
    ]
    # Cloud document parsing (vision model) — only listed when PDFs/images go to the cloud
    from src.config.roles import uses_cloud_parsing
    rag_cfg = getattr(cfg, "rag", None)
    if rag_cfg is not None and uses_cloud_parsing(rag_cfg):
        endpoint_specs.append(("Document Parsing", "scan", eff.get("rag.parsing.vision.base_url"),
                               eff.get("rag.parsing.vision.api_key"), True))
    probe_tasks = [
        _probe_endpoint_health(url, key) if enabled else None
        for _, _, url, key, enabled in endpoint_specs
    ]
    results = await asyncio.gather(*[t for t in probe_tasks if t is not None])
    ri = 0
    for (name, icon, url, _key, enabled) in endpoint_specs:
        if not enabled:
            checks.append({"name": name, "kind": icon, "status": "disabled",
                           "detail": "Disabled in settings", "latency_ms": None, "endpoint": url or ""})
            continue
        status, detail, ms = results[ri]
        ri += 1
        checks.append({"name": name, "kind": icon, "status": status,
                       "detail": detail, "latency_ms": ms, "endpoint": url or ""})

    # ASR (whether local weights or cloud configuration are ready; not an HTTP probe)
    asr_enabled = bool(eff.get("rag.asr.enabled"))
    asr_provider = eff.get("rag.asr.provider") or "—"
    if not asr_enabled:
        checks.append({"name": "Speech-to-Text", "kind": "mic", "status": "disabled",
                       "detail": "Disabled in settings", "latency_ms": None, "endpoint": asr_provider})
    else:
        avail = _asr_available(cfg)
        checks.append({"name": "Speech-to-Text", "kind": "mic",
                       "status": "ok" if avail else "down",
                       "detail": "Ready" if avail else "Model/weights not found",
                       "latency_ms": None, "endpoint": asr_provider})

    active = [c for c in checks if c["status"] not in ("disabled", "unset")]
    overall = "ok" if active and all(c["status"] == "ok" for c in active) else (
        "down" if any(c["status"] == "down" for c in active) else "unset")
    return {"data": {"checks": checks, "overall": overall}, "message": "ok"}
