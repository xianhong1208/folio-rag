
"""Control Center data-plane API (read-only aggregates; mounted on the admin module router).

- GET /api/admin/overview            — overview KPIs (folder/file/index/job)
- GET /api/admin/folders             — all folders + index aggregates
- GET /api/admin/folders/{id}/files  — per-file index status for one folder
- GET /api/admin/jobs                — recent index jobs

Auth semantics match the settings API (unauthenticated by product decision — see admin_settings.py).
Everything is read-only and triggers no indexing/deletion actions.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query

from src.log import get_api_logger

logger = get_api_logger()

router = APIRouter(tags=["Admin: Overview"])

# Retrieval evaluation (harness) background run state — a single in-memory slot (evaluation is infrequent).
_EVAL: dict = {"running": False, "folder": None, "done": 0, "total": 0,
               "result": None, "error": None, "started_at": None}


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
    except Exception as e:  # noqa: BLE001 — background run; store the error for status to report
        _EVAL["error"] = str(e)
        logger.warning(f"[EVAL] run failed: {e}")
    finally:
        _EVAL["running"] = False


@router.post("/api/admin/eval/run")
async def eval_run(body: dict = Body(default={}, example={"folder_id": 1, "n": 15})):
    """Start one retrieval evaluation (background task on the main loop; generate synthetic
    questions -> three-way retrieval -> nDCG/Recall/MRR).

    After a reindex, pass regenerate=true (node_ids change, invalidating the old labeled set)."""
    import asyncio
    import time
    global _EVAL_TASK
    if _EVAL["running"]:
        raise HTTPException(409, "An evaluation is already running")
    folder_ref = str(body.get("folder_id") or body.get("folder") or "").strip()
    if not folder_ref:
        raise HTTPException(422, "folder_id (or folder name) is required")
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
    _EVAL.update(running=True, folder=folder_ref, done=0, total=n,
                 result=None, error=None, started_at=time.time())
    # Run on the main loop (no new thread/loop) so the shared asyncpg connection pool stays on one loop
    _EVAL_TASK = asyncio.create_task(_run_eval_bg(folder_ref, n, k, regen, judge))
    return {"data": {"started": True}, "message": "evaluation started"}


@router.get("/api/admin/eval/status")
async def eval_status():
    """Evaluation progress / results (polled by the frontend)."""
    return {"data": dict(_EVAL), "message": "ok"}


@router.get("/api/admin/eval/report")
async def eval_report(folder_id: int = Query(...)):
    """Read the folder's most recent evaluation report (reports/rag_eval_<name>.json); returns null if none."""
    import json
    import os
    from db.cached_folderdb import CachedFolderDB
    folder = CachedFolderDB.get_by_id(folder_id)
    if not folder:
        raise HTTPException(404, f"Folder {folder_id} not found")
    slug = str(folder.name).replace("/", "_")
    path = os.path.join("reports", f"rag_eval_{slug}.json")
    if not os.path.exists(path):
        return {"data": None, "message": "no report yet"}
    with open(path, encoding="utf-8") as f:
        return {"data": json.load(f), "message": "ok"}


@router.get("/api/admin/eval/history")
async def eval_history(folder_id: int = Query(...), limit: int = Query(30, ge=1, le=200)):
    """The folder's past evaluation runs (oldest first) for baseline comparison / trend."""
    import json
    import os
    from db.cached_folderdb import CachedFolderDB
    folder = CachedFolderDB.get_by_id(folder_id)
    if not folder:
        raise HTTPException(404, f"Folder {folder_id} not found")
    slug = str(folder.name).replace("/", "_")
    path = os.path.join("reports", f"rag_eval_history_{slug}.jsonl")
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
async def get_overview():
    """Overview KPIs + in-progress jobs + a summary of the current model configuration."""
    from db import admin_stats
    from src.config import runtime_overrides as ro
    from src.config.config_manager import Config

    data = admin_stats.overview()
    try:
        data["trend"] = admin_stats.indexing_trend(7)
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
async def list_folders():
    """All folders + per-folder index aggregates (file count / indexed / failed / chunks / vector table)."""
    from db import admin_stats
    return {"data": {"folders": admin_stats.folders_with_index_stats()}, "message": "ok"}


@router.get("/api/admin/folders/{folder_id}/files")
async def list_folder_files(folder_id: int):
    """Per-file index status for a single folder (with truncated failure reasons)."""
    from db import admin_stats
    files = admin_stats.files_with_index_status(folder_id)
    return {"data": {"folder_id": folder_id, "files": files}, "message": "ok"}


@router.get("/api/admin/active-stages")
async def active_stages():
    """Which stage each currently-indexing file is at ({file_id: {stage,done,total,eta}}) —
    used by the file list to show "parse / generate context / vectorize / write" in real time."""
    from src.domain.rag.index_job_manager import IndexingJobManager
    return {"data": {"stages": IndexingJobManager.get_instance().active_file_stages()},
            "message": "ok"}


@router.get("/api/admin/folders/{folder_id}/files/{file_id}/chunks")
async def list_file_chunks(folder_id: int, file_id: str,
                           limit: int = Query(500, ge=1, le=2000)):
    """The leaf chunks a file was split into (lets the customer inspect what the index looks like)."""
    from db import admin_stats
    return {"data": admin_stats.chunks_for_file(folder_id, file_id, limit), "message": "ok"}


@router.get("/api/admin/audit")
async def settings_audit(limit: int = Query(100, ge=1, le=500)):
    """Settings-change audit log (who changed what to what, and when)."""
    from src.adapter import runtime_settings_service as svc
    return {"data": {"entries": svc.audit_log(limit)}, "message": "ok"}


@router.get("/api/admin/resources")
async def system_resources_endpoint():
    """System resources: DB size / vector table size / disk."""
    from db import admin_stats
    return {"data": admin_stats.system_resources(), "message": "ok"}


@router.get("/api/admin/jobs")
async def list_jobs(limit: int = Query(20, ge=1, le=100)):
    """Recent index jobs (ordered by last update)."""
    from db import admin_stats
    return {"data": {"jobs": admin_stats.recent_jobs(limit)}, "message": "ok"}


@router.get("/api/admin/analytics")
async def query_analytics(
    folder_id: int | None = Query(None),
    days: int = Query(30, ge=1, le=365),
):
    """Query analytics: query volume / average latency / zero-result rate / top queries / daily trend.

    Omitting folder_id = all folders; days = the statistics window. The data source is QueryLog
    (best-effort one record per /query retrieval)."""
    from db.query_log_db import QueryLogDB
    return {"data": QueryLogDB.analytics(folder_id=folder_id, days=days), "message": "ok"}


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
async def health_check():
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
        status, detail, ms = results[ri]; ri += 1
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
