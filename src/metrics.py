"""Prometheus metrics for ``GET /metrics`` (docs/metrics.md).

Every metric lives in one private registry, ``REGISTRY``, so tests and the endpoint see
exactly Folio's metrics plus the process/Python collectors, never a library's.

Labels are bounded on purpose: route *templates* (never raw paths), fixed stage, role,
mode and outcome names, job kinds and statuses. Nothing carries a folder name or id, a
file name, a user, a token or a query - those would grow the series count without limit
and expose what people search for. Unknown values fall into ``other``.

Recording helpers never raise: a metrics bug must not fail a search or an index job.
Gauges that describe live state (running jobs, the revocation feed, DB pools, build
info) are read at scrape time by ``_StateCollector`` instead of being kept up to date.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    disable_created_metrics,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily
from prometheus_client.registry import Collector

# No ``*_created`` series: they double the output and Prometheus does not need them.
disable_created_metrics()

REGISTRY = CollectorRegistry(auto_describe=True)
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)

OTHER = "other"

# -- label vocabularies ------------------------------------------------------------------

HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
# The per-request stages recorded by hybrid_search.record_stage and auto_merging._timing.
RETRIEVAL_STAGES = frozenset({"embed", "dense", "segment", "bm25", "fuse", "rerank",
                              "prefetch", "merge", "expand"})
MCP_MODES = frozenset({"search", "list", "read", "find", "find_list"})
MCP_OUTCOMES = frozenset({"ok", "invalid_request", "unauthorized", "forbidden", "error"})
MODEL_ROLES = frozenset({"embedding", "llm", "contextual_llm", "rerank", "asr", "vision"})
MODEL_OUTCOMES = frozenset({"ok", "client_error", "server_error", "error"})
JOB_KINDS = frozenset({"index", "rebuild_fts"})
JOB_STATUSES = frozenset({"pending", "running", "succeeded", "partial_success", "failed",
                          "cancelled", "interrupted"})
FILE_STATUSES = frozenset({"success", "failed", "skipped", "cancelled", "timeout"})


def bounded(value: Optional[str], allowed: Iterable[str]) -> str:
    """``value`` when it is one of ``allowed``, else ``other`` (keeps label sets finite)."""
    return value if isinstance(value, str) and value in allowed else OTHER


# -- buckets -----------------------------------------------------------------------------

# Seconds. HTTP and MCP answers: a few ms (health) up to a minute (uploads, answers).
_FAST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
# Model calls: embeddings in ms, LLM calls up to minutes.
_MODEL_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
# Index jobs: seconds for one small file, hours (days at most) for a large folder.
_JOB_BUCKETS = (1.0, 5.0, 15.0, 30.0, 60.0, 300.0, 900.0, 1800.0, 3600.0, 3 * 3600.0,
                12 * 3600.0, 24 * 3600.0)

# -- metrics -----------------------------------------------------------------------------

HTTP_REQUESTS = Counter(
    "folio_http_requests_total", "HTTP requests by route template, method and status code.",
    ["route", "method", "status"], registry=REGISTRY)
HTTP_DURATION = Histogram(
    "folio_http_request_duration_seconds",
    "Time until the response started (headers sent), by route template and method.",
    ["route", "method"], buckets=_FAST_BUCKETS, registry=REGISTRY)

RETRIEVAL_STAGE = Histogram(
    "folio_retrieval_stage_duration_seconds",
    "Wall time of each retrieval stage per search (dense and BM25 run concurrently).",
    ["stage"], buckets=_FAST_BUCKETS, registry=REGISTRY)
RETRIEVAL_SEARCH = Histogram(
    "folio_retrieval_search_duration_seconds",
    "Wall time of one hybrid search: candidates, fusion and rerank.",
    buckets=_FAST_BUCKETS, registry=REGISTRY)

MCP_TOOL_CALLS = Counter(
    "folio_mcp_tool_calls_total", "MCP tool calls by mode and outcome.",
    ["mode", "outcome"], registry=REGISTRY)
MCP_TOOL_DURATION = Histogram(
    "folio_mcp_tool_call_duration_seconds", "MCP tool call duration by mode.",
    ["mode"], buckets=_FAST_BUCKETS, registry=REGISTRY)

INDEX_JOBS_SUBMITTED = Counter(
    "folio_index_jobs_submitted_total", "Index jobs created (started at once or queued).",
    ["kind"], registry=REGISTRY)
INDEX_JOBS_STARTED = Counter(
    "folio_index_jobs_started_total", "Index jobs that began running.",
    ["kind"], registry=REGISTRY)
INDEX_JOBS_FINISHED = Counter(
    "folio_index_jobs_finished_total", "Index jobs that reached a final status.",
    ["kind", "status"], registry=REGISTRY)
INDEX_JOB_DURATION = Histogram(
    "folio_index_job_duration_seconds",
    "Time from a job's creation to its final status (includes time queued).",
    ["kind", "status"], buckets=_JOB_BUCKETS, registry=REGISTRY)
INDEX_FILES = Counter(
    "folio_index_files_total", "Files processed by index jobs, by result.",
    ["status"], registry=REGISTRY)

MODEL_REQUESTS = Counter(
    "folio_model_requests_total",
    "HTTP requests to model endpoints by role and outcome (each retry counts).",
    ["role", "outcome"], registry=REGISTRY)
MODEL_DURATION = Histogram(
    "folio_model_request_duration_seconds", "Model endpoint request duration by role.",
    ["role"], buckets=_MODEL_BUCKETS, registry=REGISTRY)


# -- recording helpers (never raise) -----------------------------------------------------

def observe_http(route: str, method: str, status: int, seconds: float) -> None:
    try:
        m = method.upper() if isinstance(method, str) else ""
        m = m if m in HTTP_METHODS else OTHER
        HTTP_REQUESTS.labels(route, m, str(int(status))).inc()
        HTTP_DURATION.labels(route, m).observe(max(seconds, 0.0))
    except Exception:  # noqa: BLE001
        pass


def observe_search(timings_ms: Mapping[str, float], seconds: float) -> None:
    """One hybrid search: its per-stage timings (ms, as in the query trace) and total."""
    try:
        for stage, ms in (timings_ms or {}).items():
            RETRIEVAL_STAGE.labels(bounded(stage, RETRIEVAL_STAGES)).observe(max(float(ms), 0.0) / 1000)
        RETRIEVAL_SEARCH.observe(max(seconds, 0.0))
    except Exception:  # noqa: BLE001
        pass


def observe_stage(stage: str, seconds: float) -> None:
    """A stage measured outside the hybrid search (auto-merge's merge / expand)."""
    try:
        RETRIEVAL_STAGE.labels(bounded(stage, RETRIEVAL_STAGES)).observe(max(seconds, 0.0))
    except Exception:  # noqa: BLE001
        pass


def observe_mcp_call(mode: str, outcome: str, seconds: float) -> None:
    try:
        m = bounded(mode, MCP_MODES)
        MCP_TOOL_CALLS.labels(m, bounded(outcome, MCP_OUTCOMES)).inc()
        MCP_TOOL_DURATION.labels(m).observe(max(seconds, 0.0))
    except Exception:  # noqa: BLE001
        pass


def _kind(kind) -> str:
    return bounded(getattr(kind, "value", kind), JOB_KINDS)


def _status(status) -> str:
    return bounded(getattr(status, "value", status), JOB_STATUSES)


def job_submitted(kind) -> None:
    try:
        INDEX_JOBS_SUBMITTED.labels(_kind(kind)).inc()
    except Exception:  # noqa: BLE001
        pass


def job_started(kind) -> None:
    try:
        INDEX_JOBS_STARTED.labels(_kind(kind)).inc()
    except Exception:  # noqa: BLE001
        pass


def job_finished(kind, status, started_at: Optional[str], finished_at: Optional[str] = None) -> None:
    """A job reached a final status. ``started_at`` / ``finished_at`` are ISO timestamps
    (the job record's ``started_at`` is its creation time); no duration when unparsable."""
    try:
        k, s = _kind(kind), _status(status)
        INDEX_JOBS_FINISHED.labels(k, s).inc()
        if started_at:
            begin = datetime.fromisoformat(started_at)
            end = datetime.fromisoformat(finished_at) if finished_at else datetime.now(begin.tzinfo)
            INDEX_JOB_DURATION.labels(k, s).observe(max((end - begin).total_seconds(), 0.0))
    except Exception:  # noqa: BLE001
        pass


def file_finished(status) -> None:
    try:
        INDEX_FILES.labels(bounded(getattr(status, "value", status), FILE_STATUSES)).inc()
    except Exception:  # noqa: BLE001
        pass


def model_outcome(status_code: Optional[int]) -> str:
    """``ok`` / ``client_error`` / ``server_error`` from a status code; ``error`` for None
    (no response: timeout, refused or reset connection)."""
    if status_code is None:
        return "error"
    if status_code >= 500:
        return "server_error"
    if status_code >= 400:
        return "client_error"
    return "ok"


def observe_model_call(role: str, status_code: Optional[int], seconds: float) -> None:
    try:
        r = bounded(role, MODEL_ROLES)
        MODEL_REQUESTS.labels(r, model_outcome(status_code)).inc()
        MODEL_DURATION.labels(r).observe(max(seconds, 0.0))
    except Exception:  # noqa: BLE001
        pass


# -- scrape-time state -------------------------------------------------------------------

def _job_counts() -> dict:
    from src.domain.rag.index_job_manager import IndexingJobManager

    manager = IndexingJobManager._instance  # never created by a scrape
    if manager is None:
        return {}
    return manager.live_job_counts()


class _StateCollector(Collector):
    """Gauges read from live state on every scrape."""

    def collect(self):
        yield from self._build_info()
        yield from self._jobs()
        yield from self._revocation()
        yield from self._db_pools()

    @staticmethod
    def _build_info():
        from src.version import get_version

        labels = {"version": str(get_version())}
        try:
            from src.config.config_manager import Config

            rag = getattr(Config.get_config_model(), "rag", None)
            profile = getattr(getattr(rag, "profile", None), "value", getattr(rag, "profile", None))
            if profile:
                labels["profile"] = str(profile)
        except Exception:  # noqa: BLE001
            pass
        info = InfoMetricFamily("folio_build", "Folio version and model profile.")
        info.add_metric([], labels)
        yield info

    @staticmethod
    def _jobs():
        g = GaugeMetricFamily("folio_index_jobs_active", "Index jobs waiting or running now.",
                              labels=["kind", "status"])
        try:
            counts = _job_counts()
        except Exception:  # noqa: BLE001
            counts = {}
        for kind in sorted(JOB_KINDS):
            for status in ("pending", "running"):
                g.add_metric([kind, status], counts.get((kind, status), 0))
        yield g

    @staticmethod
    def _revocation():
        from src.auth.revocation import get_revocation_list

        rl = get_revocation_list()
        enabled = GaugeMetricFamily("folio_revocation_feed_enabled",
                                    "1 when the MCP Gate revocation feed is configured.")
        enabled.add_metric([], 1 if rl is not None else 0)
        yield enabled
        if rl is None:
            return
        snap = rl.metrics_snapshot()
        last = GaugeMetricFamily("folio_revocation_feed_last_success_timestamp_seconds",
                                 "Unix time of the last successful poll (0 = none yet).")
        last.add_metric([], snap["last_success"] or 0)
        yield last
        fails = GaugeMetricFamily("folio_revocation_feed_consecutive_failures",
                                  "Failed polls since the last success.")
        fails.add_metric([], snap["consecutive_failures"])
        yield fails
        stale = GaugeMetricFamily("folio_revocation_feed_stale",
                                  "1 when the list is older than its stale limit.")
        stale.add_metric([], 1 if snap["stale"] else 0)
        yield stale
        held = GaugeMetricFamily("folio_revocation_revoked_tokens",
                                 "Revoked, unexpired token ids held in memory.")
        held.add_metric([], snap["revoked"])
        yield held

    @staticmethod
    def _db_pools():
        try:
            from db.db import existing_engine_pools

            pools = existing_engine_pools()
        except Exception:  # noqa: BLE001
            pools = {}
        size = GaugeMetricFamily("folio_db_pool_size", "Configured pool size.", labels=["pool"])
        out = GaugeMetricFamily("folio_db_pool_checked_out", "Connections in use.", labels=["pool"])
        idle = GaugeMetricFamily("folio_db_pool_checked_in", "Idle pooled connections.", labels=["pool"])
        over = GaugeMetricFamily("folio_db_pool_overflow",
                                 "Connections open beyond the pool size.", labels=["pool"])
        for name, pool in sorted(pools.items()):
            try:
                values = (pool.size(), pool.checkedout(), pool.checkedin(), max(pool.overflow(), 0))
            except Exception:  # noqa: BLE001 - e.g. NullPool has no counters
                continue
            for family, value in zip((size, out, idle, over), values):
                family.add_metric([name], value)
        yield from (size, out, idle, over)


REGISTRY.register(_StateCollector())


def render() -> tuple[bytes, str]:
    """The exposition body and its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST

