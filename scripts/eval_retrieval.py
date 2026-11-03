"""Golden-set retrieval eval against a running Folio server — HTTP only.

    python scripts/eval_retrieval.py --token-file <json> [--runs 2] [--out evals/baselines]

It (1) indexes evals/rag/documents/ into one dedicated folder through the console API
(reused when every file's sha256 matches), (2) asks every question in
evals/rag/golden_qa.yaml through

    mcp        — the MCP search tool with its defaults (what an agent sees: top_k,
                 auto-merge and neighbour expansion as configured);
    mcp_k20    — the same tool with top_k=20, for Recall@20;
    trace      — the console retrieval trace (top_k=20): the production fusion + rerank
                 ranking over leaf chunks, before auto-merge and expansion. Each hit
                 carries its rank in the dense, BM25 and fused candidate lists.

and (3) scores chunk-level Recall@5/10/20, MRR@10, nDCG@10, file-level Recall@5, the
false-accept rate on unanswerable questions, MCP response size and latency, broken
down by question type and language. Gold evidence is file + quoted spans (see
evals/retrieval_metrics.py), so the set survives chunking changes.

The token file is JSON {"token": ..., "jti": ...} for an MCP Gate PAT issued for the
Folio service (rag:admin is needed only for the settings snapshot). The token is
never printed or written anywhere.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evals import retrieval_metrics as rm  # noqa: E402

DOCS_DIR = REPO / "evals" / "rag" / "documents"
QA_FILE = REPO / "evals" / "rag" / "golden_qa.yaml"
OUT_DIR = REPO / "evals" / "baselines"
FOLDER_NAME = "folio_eval_golden_v1"
FOLDER_DESC = ("Folio retrieval eval corpus (evals/rag/documents): Taiwanese labour and "
               "personal-data laws, U.S. government pages, fictional company documents.")
HARNESS_VERSION = 1
THRESHOLDS = (0.25, 0.5)
METRICS = ("recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10", "file_recall@5")


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------------ HTTP


class Folio:
    """Thin client for the console API (/api/admin/*) and the MCP endpoint."""

    def __init__(self, base_url: str, token: str):
        self.base = base_url.rstrip("/")
        self._token = token
        self.http = httpx.Client(base_url=self.base, timeout=300,
                                 headers={"Authorization": f"Bearer {token}"})

    def get(self, path: str, **kw):
        r = self.http.get(path, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, **kw):
        r = self.http.post(path, **kw)
        r.raise_for_status()
        return r.json()

    def delete(self, path: str):
        r = self.http.delete(path)
        r.raise_for_status()
        return r.json()

    # -- folder / files
    def folder_by_name(self, name: str) -> Optional[dict]:
        for f in self.get("/api/admin/folders")["data"]["folders"]:
            if f.get("name") == name:
                return f
        return None

    def files(self, folder_id: int) -> List[dict]:
        return self.get(f"/api/admin/folders/{folder_id}/files")["data"]["files"]

    def file_sha256(self, folder_id: int, file_id: str) -> str:
        r = self.http.get(f"/api/admin/manage/folders/{folder_id}/files/{file_id}/download")
        r.raise_for_status()
        return hashlib.sha256(r.content).hexdigest()

    def chunks(self, folder_id: int, file_id: str) -> List[dict]:
        return self.get(f"/api/admin/folders/{folder_id}/files/{file_id}/chunks",
                        params={"limit": 2000})["data"]["chunks"]

    def trace(self, folder_id: int, query: str, top_k: int) -> dict:
        return self.post(f"/api/admin/manage/folders/{folder_id}/query",
                         json={"query": query, "top_k": top_k, "trace": True})["data"]

    def settings(self) -> dict:
        return self.get("/api/admin/settings")["data"]

    # -- MCP
    def mcp_client(self):
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport
        return Client(StreamableHttpTransport(
            f"{self.base}/mcp", headers={"Authorization": f"Bearer {self._token}"}),
            timeout=300)


# -------------------------------------------------------------------- indexing


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def corpus_files() -> List[Path]:
    return sorted(p for p in DOCS_DIR.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.name != "SOURCES.md")


def _folder_job_running(api: Folio, folder_id: int) -> bool:
    try:
        jobs = api.get("/api/admin/jobs", params={"limit": 20})["data"]["jobs"]
    except httpx.HTTPError:
        return True
    return any(j.get("folder_id") == folder_id and j.get("status") in ("running", "queued", "pending")
               for j in jobs)


def ensure_indexed(api: Folio, poll_s: float = 5.0, timeout_s: float = 3600) -> dict:
    """Create / reconcile the eval folder and wait until every file is indexed."""
    folder = api.folder_by_name(FOLDER_NAME)
    if folder is None:
        api.post("/api/admin/manage/folders",
                 json={"name": FOLDER_NAME, "description": FOLDER_DESC})
        folder = api.folder_by_name(FOLDER_NAME)
        log(f"created folder {FOLDER_NAME} (id={folder['id']})")
    fid = folder["id"]
    local = {p.name: p for p in corpus_files()}
    remote = {f["name"]: f for f in api.files(fid)}
    changed = False
    for name, f in remote.items():
        if name not in local:
            log(f"  removing {name} (not in the corpus)")
            api.delete(f"/api/admin/manage/folders/{fid}/files/{f['id']}")
            changed = True
        elif api.file_sha256(fid, f["id"]) != sha256_file(local[name]):
            log(f"  replacing {name} (content changed)")
            api.delete(f"/api/admin/manage/folders/{fid}/files/{f['id']}")
            del remote[name]
            changed = True
    for name, p in local.items():
        if name not in remote:
            log(f"  uploading {name}")
            with open(p, "rb") as fh:
                api.post(f"/api/admin/manage/folders/{fid}/files",
                         files={"file": (name, fh)}, data={"auto_index": "false"})
            changed = True
    files = api.files(fid)
    if (changed or any(f["status"] == "unindexed" for f in files)) and not _folder_job_running(api, fid):
        job = api.post(f"/api/admin/manage/folders/{fid}/index",
                       json={"skip_existing": True})["data"]
        log(f"  indexing job {job.get('job_id')} started")
    t0 = time.time()
    retried = False
    while True:
        try:
            files = api.files(fid)
        except httpx.HTTPError as e:  # transient (server or DB restart): keep polling
            log(f"  status poll failed ({e.__class__.__name__}); retrying")
            time.sleep(poll_s)
            continue
        states: Dict[str, int] = {}
        for f in files:
            states[f["status"]] = states.get(f["status"], 0) + 1
        busy = _folder_job_running(api, fid)
        if not busy and all(f["status"] in ("indexed", "failed", "cancelled", "interrupted")
                            for f in files):
            bad = [f for f in files if f["status"] != "indexed"]
            if not bad:
                break
            if retried:
                raise SystemExit(f"indexing failed for: {[f['name'] for f in bad]}")
            log(f"  retrying {len(bad)} file(s) that did not index: {[f['name'] for f in bad]}")
            api.post(f"/api/admin/manage/folders/{fid}/retry",
                     json={"file_ids": [f["id"] for f in bad]})
            retried = True
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"indexing did not finish in {timeout_s}s: {states}")
        log(f"  waiting for indexing: {states}")
        time.sleep(poll_s)
    return {"id": fid, "name": FOLDER_NAME, "files": {f["name"]: f for f in files},
            "index_seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------- gold localisation


def localise_gold(api: Folio, folder: dict, questions: List[dict]) -> dict:
    """Find, for every evidence item, the leaf chunk range that holds its spans.

    Returns {"leaves": {file: [chunk...]}, "gold": {qid: {ev_idx: (lo, hi)}},
    "unreachable": [{qid, file, spans}]}. A span split by a chunk boundary is found in
    two consecutive leaves (range of 2). Unreachable evidence means the indexed text no
    longer contains the span (parser dropped or rewrote it) — no ranking can score it.
    """
    leaves = {name: api.chunks(folder["id"], f["id"]) for name, f in folder["files"].items()}
    gold: Dict[str, dict] = {}
    unreachable = []
    for q in questions:
        for i, ev in enumerate(q.get("evidence") or []):
            ch = leaves.get(ev["file"], [])
            rng = None
            for c in ch:
                if rm.covers_spans(ev["spans"], c["text"]):
                    rng = (c["index"], c["index"])
                    break
            if rng is None:
                for a, b in zip(ch, ch[1:]):
                    if rm.covers_spans(ev["spans"], a["text"] + "\n" + b["text"]):
                        rng = (a["index"], b["index"])
                        break
            if rng is None:
                unreachable.append({"id": q["id"], "file": ev["file"], "spans": ev["spans"]})
            else:
                gold.setdefault(q["id"], {})[i] = rng
    return {"leaves": leaves, "gold": gold, "unreachable": unreachable}


def leaf_index_by_text(leaves: Dict[str, List[dict]]) -> Dict[tuple, int]:
    return {(name, rm.normalize(c["text"])): c["index"]
            for name, ch in leaves.items() for c in ch}


# ------------------------------------------------------------------- querying


async def mcp_tool_name(client) -> str:
    tools = await client.list_tools()
    for t in tools:
        if t.name.endswith("_" + FOLDER_NAME) or f"'{FOLDER_NAME}'" in (t.description or ""):
            return t.name
    raise SystemExit(f"no MCP tool for folder {FOLDER_NAME} (tools: {[t.name for t in tools]})")


async def mcp_search(client, tool: str, query: str, top_k: Optional[int] = None) -> dict:
    args = {"query": query, "mode": "search"}
    if top_k:
        args["top_k"] = top_k
    t0 = time.perf_counter()
    res = await client.call_tool(tool, args, raise_on_error=False)
    ms = (time.perf_counter() - t0) * 1000
    text = next((b.text for b in res.content if getattr(b, "type", "") == "text"), "{}")
    data = json.loads(text)
    return {"ms": ms, "raw": text, "data": data}


def score_results(results: List[dict], q: dict, gold: dict, ks=(5, 10, 20)) -> dict:
    ev = q.get("evidence") or []
    hit_lists = [rm.result_hits(r, ev, gold) for r in results]
    files = []
    for r in results:
        if r.get("file_name") and r["file_name"] not in files:
            files.append(r["file_name"])
    out = {"first_rank": rm.first_rank(hit_lists),
           "top_score": max((r.get("score") or 0.0 for r in results), default=None)}
    if ev:
        out.update(rm.ranking_metrics(hit_lists, len(ev), ks=ks))
        out["file_recall@5"] = rm.file_recall(
            [r.get("file_name") for r in results[:5]], [e["file"] for e in ev], 5)
    return out


def mcp_results(data: dict) -> List[dict]:
    out = []
    for r in data.get("results") or []:
        rng = r.get("chunk_index_range") or None
        out.append({"text": r.get("text") or "", "file_name": r.get("file_name"),
                    "score": r.get("score"), "role": r.get("node_role"),
                    "chunk_range": tuple(rng) if rng and len(rng) == 2 else None})
    return out


def trace_results(data: dict, idx: Dict[tuple, int]) -> List[dict]:
    out = []
    for r in (data.get("trace") or {}).get("results") or []:
        md = r.get("metadata") or {}
        ci = idx.get((md.get("file_name"), rm.normalize(r.get("text") or "")))
        out.append({"text": r.get("text") or "", "file_name": md.get("file_name"),
                    "score": r.get("final_score"),
                    "chunk_range": (ci, ci) if ci is not None else None,
                    "vector_rank": r.get("vector_rank"), "bm25_rank": r.get("bm25_rank"),
                    "hybrid_rank": r.get("hybrid_rank")})
    return out


async def run_once(api: Folio, folder: dict, questions: List[dict], loc: dict,
                   run_no: int) -> List[dict]:
    idx = leaf_index_by_text(loc["leaves"])
    rows = []
    async with api.mcp_client() as client:
        tool = await mcp_tool_name(client)
        for n, q in enumerate(questions, 1):
            gold = loc["gold"].get(q["id"], {})
            row = {"id": q["id"], "type": q["type"], "lang": q["lang"],
                   "answerable": bool(q.get("evidence"))}
            m = await mcp_search(client, tool, q["question"])
            res = mcp_results(m["data"])
            # The default call returns top_k (10) results: Recall@20 comes from mcp_k20.
            row["mcp"] = score_results(res, q, gold, ks=(5, 10))
            row["mcp"].update({
                "ms": round(m["ms"], 1), "n_results": len(res),
                "tokens": round(rm.estimate_tokens(m["raw"]), 1),
                "chars": len(m["raw"]),
                "dup_ratio": round(rm.duplicate_ratio([r["text"] for r in res]), 4),
                "roles": [r["role"] for r in res],
                "files": [r["file_name"] for r in res[:5]]})
            if m["data"].get("error"):
                row["mcp"]["error"] = m["data"]["error"]
            m20 = await mcp_search(client, tool, q["question"], top_k=20)
            row["mcp_k20"] = score_results(mcp_results(m20["data"]), q, gold)
            t0 = time.perf_counter()
            tr = trace_results(await asyncio.to_thread(api.trace, folder["id"], q["question"], 20),
                               idx)
            row["trace"] = score_results(tr, q, gold)
            row["trace"]["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            # Candidate-stage ranks of the best-ranked hit (only visible for final hits).
            ev = q.get("evidence") or []
            stage = {}
            for r in tr:
                if ev and rm.result_hits(r, ev, gold):
                    for s in ("vector_rank", "bm25_rank", "hybrid_rank"):
                        if r.get(s) is not None:
                            stage[s] = min(stage.get(s, 10**6), r[s])
            row["trace"]["stage_ranks"] = stage
            rows.append(row)
            if n % 20 == 0 or n == len(questions):
                log(f"  run {run_no}: {n}/{len(questions)}")
    return rows


# -------------------------------------------------------------------- reporting


def summarise(rows: List[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    out = {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una)}
    for mode in ("mcp", "mcp_k20", "trace"):
        flat = [dict(r[mode], type=r["type"], lang=r["lang"]) for r in ans]
        ks = [m for m in METRICS if any(m in f for f in flat)]
        s = {"overall": {m: round(rm.mean(f[m] for f in flat if m in f), 4) for m in ks},
             "by_type": rm.group_means(flat, lambda f: f["type"], ks),
             "by_lang": rm.group_means(flat, lambda f: f["lang"], ks)}
        fa = {}
        for t in THRESHOLDS:
            fa[f"false_accept@{t}"] = (round(sum(1 for r in una if (r[mode]["top_score"] or 0) >= t)
                                             / len(una), 4) if una else None)
            fa[f"answerable_accept@{t}"] = round(
                sum(1 for r in ans if (r[mode]["top_score"] or 0) >= t) / len(ans), 4) if ans else None
        s["thresholds"] = fa
        out[mode] = s
    allm = [r["mcp"] for r in rows]
    out["mcp_response"] = {
        "tokens_mean": round(rm.mean(m["tokens"] for m in allm), 1),
        "tokens_p50": round(rm.percentile([m["tokens"] for m in allm], 50), 1),
        "tokens_p95": round(rm.percentile([m["tokens"] for m in allm], 95), 1),
        "results_mean": round(rm.mean(m["n_results"] for m in allm), 2),
        "dup_ratio_mean": round(rm.mean(m["dup_ratio"] for m in allm), 4),
        "latency_ms_p50": round(rm.percentile([m["ms"] for m in allm], 50), 1),
        "latency_ms_p95": round(rm.percentile([m["ms"] for m in allm], 95), 1),
        "trace_latency_ms_p50": round(rm.percentile([r["trace"]["ms"] for r in rows], 50), 1),
        "trace_latency_ms_p95": round(rm.percentile([r["trace"]["ms"] for r in rows], 95), 1),
        "by_lang": rm.group_means([dict(m, lang=r["lang"]) for m, r in zip(allm, rows)],
                                  lambda f: f["lang"], ["tokens", "dup_ratio", "ms"]),
    }
    roles: Dict[str, int] = {}
    for m in allm:
        for ro in m["roles"]:
            roles[ro] = roles.get(ro, 0) + 1
    out["mcp_response"]["roles"] = roles
    # Candidate stages, from the trace: of the answerable questions, how often the
    # best-ranked gold hit in the final top-20 came from each candidate list. A lower
    # bound on each list's recall: the trace only reports ranks for final hits.
    st = [r["trace"]["stage_ranks"] for r in ans]
    stages = {}
    for s, label in (("vector_rank", "dense"), ("bm25_rank", "bm25"), ("hybrid_rank", "fused")):
        for k in (10, 20):
            stages[f"{label}_in_top{k}_lb"] = round(
                sum(1 for x in st if x.get(s) is not None and x[s] <= k) / len(st), 4) if st else None
    stages["bm25_only"] = round(sum(1 for x in st if "bm25_rank" in x and "vector_rank" not in x)
                                / len(st), 4) if st else None
    stages["dense_only"] = round(sum(1 for x in st if "vector_rank" in x and "bm25_rank" not in x)
                                 / len(st), 4) if st else None
    out["trace_stages"] = stages
    return out


def variance(summaries: List[dict]) -> dict:
    """Max |difference| between runs for the headline metrics (0 = deterministic)."""
    def flat(s):
        d = {}
        for mode in ("mcp", "mcp_k20", "trace"):
            for m, v in s[mode]["overall"].items():
                d[f"{mode}.{m}"] = v
            for m, v in s[mode]["thresholds"].items():
                d[f"{mode}.{m}"] = v
        for m in ("tokens_mean", "dup_ratio_mean", "latency_ms_p50", "latency_ms_p95"):
            d[f"mcp_response.{m}"] = s["mcp_response"][m]
        return d
    fl = [flat(s) for s in summaries]
    out = {}
    for k in fl[0]:
        vals = [f[k] for f in fl if f.get(k) is not None]
        if vals:
            out[k] = {"runs": vals, "max_abs_diff": round(max(vals) - min(vals), 4)}
    return out


def rank_flips(runs: List[List[dict]]) -> dict:
    """Questions whose first-hit rank differs between runs, per mode."""
    out = {}
    for mode in ("mcp", "mcp_k20", "trace"):
        ids = [r["id"] for r in runs[0]
               if len({tuple([x[mode]["first_rank"]]) for x in
                       [next(rr for rr in run if rr["id"] == r["id"]) for run in runs]}) > 1]
        out[mode] = ids
    return out


def settings_snapshot(api: Folio) -> dict:
    """GET /api/admin/settings flattened to {path: value}; secrets stay redacted."""
    try:
        data = api.settings()
    except httpx.HTTPStatusError as e:
        return {"error": f"settings unavailable: HTTP {e.response.status_code}"}
    snap = {}
    for s in data.get("settings", []):
        v = s.get("value")
        leaf = s["path"].rsplit(".", 1)[-1]
        if s.get("secret") or leaf in ("api_key", "password", "secret", "token", "session_secret"):
            v = "<redacted>" if s.get("is_set", v not in (None, "")) else None
        snap[s["path"]] = v
    return snap


def git_short_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def fmt(v) -> str:
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def markdown_report(rep: dict) -> str:
    s = rep["summary"]
    L = [f"# Retrieval baseline — {rep['date']} — `{rep['label']}`", "",
         f"Folio at {rep['server']} (code `{rep['code_ref']}`), folder `{rep['folder']['name']}` "
         f"(id {rep['folder']['id']}), {rep['corpus']['n_files']} documents, "
         f"{s['n']} questions ({s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable), "
         f"{rep['runs']} runs. Harness v{rep['harness']}; scores below are the mean of the runs.", "",
         "Models: " + ", ".join(f"{k} `{v}`" for k, v in rep["models"].items())
         + ("; environment: " + ", ".join(f"{k} {v}" for k, v in rep.get("environment", {}).items())
            if rep.get("environment") else ""), "",
         "## Overall", "",
         "| path | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | file R@5 | FA@0.25 | FA@0.5 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for mode, label in (("mcp", "MCP search (defaults)"), ("mcp_k20", "MCP search, top_k=20"),
                        ("trace", "Trace: fused+rerank leaves, k=20")):
        o, t = s[mode]["overall"], s[mode]["thresholds"]
        L.append(f"| {label} | {fmt(o.get('recall@5'))} | {fmt(o.get('recall@10'))} | "
                 f"{fmt(o.get('recall@20'))} | {fmt(o.get('mrr@10'))} | {fmt(o.get('ndcg@10'))} | "
                 f"{fmt(o.get('file_recall@5'))} | {fmt(t.get('false_accept@0.25'))} | "
                 f"{fmt(t.get('false_accept@0.5'))} |")
    L += ["", "FA@t = share of unanswerable questions whose top result scores ≥ t "
          "(answerable questions clearing the same bar: "
          + ", ".join(f"{t} → {fmt(s['mcp']['thresholds'][f'answerable_accept@{t}'])}" for t in THRESHOLDS)
          + " on the MCP path).", ""]
    for dim, title in (("by_type", "By question type"), ("by_lang", "By question language")):
        L += [f"## {title}", "",
              "| group | n | MCP R@5 | MCP R@10 | MCP MRR@10 | MCP nDCG@10 | file R@5 | MCP k20 R@20 "
              "| trace R@10 | trace R@20 | trace MRR@10 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for g, v in s["mcp"][dim].items():
            tv = s["trace"][dim].get(g, {})
            k20 = s["mcp_k20"][dim].get(g, {})
            L.append(f"| {g} | {v['n']} | {fmt(v.get('recall@5'))} | {fmt(v.get('recall@10'))} | "
                     f"{fmt(v.get('mrr@10'))} | {fmt(v.get('ndcg@10'))} | {fmt(v.get('file_recall@5'))} | "
                     f"{fmt(k20.get('recall@20'))} | {fmt(tv.get('recall@10'))} | "
                     f"{fmt(tv.get('recall@20'))} | {fmt(tv.get('mrr@10'))} |")
        L.append("")
    r = s["mcp_response"]
    L += ["## MCP response size and latency", "",
          f"- estimated tokens per call: mean {r['tokens_mean']}, p50 {r['tokens_p50']}, p95 {r['tokens_p95']} "
          "(CJK 0.7 tokens/char, other 0.25/char, measured on the JSON text block)",
          f"- results per call: {r['results_mean']}; result roles: {r['roles']}",
          f"- duplicate-text ratio (10-char shingles repeated within a response): {r['dup_ratio_mean']}",
          f"- latency (client wall time): MCP p50 {r['latency_ms_p50']} ms, p95 {r['latency_ms_p95']} ms; "
          f"trace p50 {r['trace_latency_ms_p50']} ms, p95 {r['trace_latency_ms_p95']} ms",
          "- by question language: " + "; ".join(
              f"{g}: {v.get('tokens', 0):.0f} tokens, dup {v.get('dup_ratio', 0):.3f}, {v.get('ms', 0):.0f} ms"
              for g, v in r["by_lang"].items()), "",
          "## Candidate stages (from the trace, lower bounds)", "",
          "Share of answerable questions whose best gold hit in the final top-20 ranked within "
          "the top 10 / 20 of each candidate list. The trace reports stage ranks only for final "
          "hits, so these are lower bounds on each list's recall.", ""]
    for k, v in s["trace_stages"].items():
        L.append(f"- {k}: {fmt(v)}")
    gr = rep["gold_reachability"]
    L += ["", "## Run-to-run variance", "",
          "Largest |run1 − run2| over the headline metrics: " + ", ".join(
              f"{k} {v['max_abs_diff']}" for k, v in rep["variance"].items()
              if k.split(".")[-1] in ("recall@10", "mrr@10", "latency_ms_p50", "tokens_mean")), "",
          "Questions whose first-hit rank changed between runs: " + ", ".join(
              f"{m} {len(v)}" for m, v in rep["rank_flips"].items()), "",
          "## Gold reachability", "",
          f"{gr['n_evidence'] - gr['n_unreachable']}/{gr['n_evidence']} evidence items are found in "
          f"the indexed leaf chunks ({gr['n_split']} only across two adjacent chunks)."
          + (" Unreachable items can never be retrieved and cap recall:" if rep["unreachable"] else ""),
          ""]
    for u in rep["unreachable"][:20]:
        L.append(f"- {u['id']} `{u['file']}`: {', '.join(repr(x[:60]) for x in u['spans'])}")
    L += ["", "## Effective configuration", "", "```json",
          json.dumps({k: v for k, v in rep["config"].items()
                      if k.startswith(("rag.retrieval", "rag.rerank", "rag.embedding.model",
                                       "rag.contextual_retrieval.enabled", "rag.chunk",
                                       "rag.index"))}, ensure_ascii=False, indent=1),
          "```", "", "Full snapshot (secrets redacted) in the JSON file.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------ main


def load_questions(path: Path, limit: Optional[int] = None) -> List[dict]:
    qs = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    qs = [q for q in qs if isinstance(q, dict) and q.get("question")]
    return qs[:limit] if limit else qs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:5039")
    ap.add_argument("--token-file", help='JSON {"token": ..., "jti": ...} (required unless --report)')
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--limit", type=int, help="score only the first N questions (smoke test)")
    ap.add_argument("--label", default=None, help="baseline label (default dev-<shortsha>)")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--report", metavar="JSON", help="only re-render the Markdown of a results JSON")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="environment facts the API cannot report (e.g. pgvector=0.8.7); "
                         "recorded in the baseline")
    args = ap.parse_args()

    if args.report:
        rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
        Path(args.report).with_suffix(".md").write_text(markdown_report(rep), encoding="utf-8")
        return 0
    token = json.loads(Path(args.token_file).read_text())["token"]
    api = Folio(args.server, token)
    t_index = time.time()
    folder = ensure_indexed(api)
    log(f"folder {folder['name']} id={folder['id']}: {len(folder['files'])} files indexed "
        f"({time.time() - t_index:.0f}s)")
    if args.index_only:
        return 0

    questions = load_questions(QA_FILE, args.limit)
    loc = localise_gold(api, folder, questions)
    n_ev = sum(len(q.get("evidence") or []) for q in questions)
    n_split = sum(1 for g in loc["gold"].values() for lo_hi in g.values() if lo_hi[0] != lo_hi[1])
    log(f"{len(questions)} questions; {n_ev - len(loc['unreachable'])}/{n_ev} evidence items "
        f"located in the index")

    config = settings_snapshot(api)
    runs = []
    for i in range(1, args.runs + 1):
        runs.append(asyncio.run(run_once(api, folder, questions, loc, i)))
    summaries = [summarise(r) for r in runs]

    # Mean of the runs is the reported number.
    def avg(objs):
        if isinstance(objs[0], dict):
            return {k: avg([o[k] for o in objs]) for k in objs[0]}
        if isinstance(objs[0], (int, float)) and not isinstance(objs[0], bool):
            if all(isinstance(o, int) for o in objs) and len(set(objs)) == 1:
                return objs[0]  # counts
            return round(sum(objs) / len(objs), 4)
        return objs[0]
    summary = avg(summaries)

    sha = git_short_sha()
    label = args.label or f"dev-{sha}"
    date = datetime.date.today().isoformat()
    models = {k.split(".")[1]: config.get(k) for k in
              ("rag.embedding.model", "rag.rerank.model", "rag.llm.model") if config.get(k)}
    rep = {
        "date": date, "label": label, "code_ref": sha, "server": args.server,
        "harness": HARNESS_VERSION, "runs": args.runs,
        "folder": {"id": folder["id"], "name": folder["name"]},
        "corpus": {"n_files": len(folder["files"]),
                   "files": {n: {"chunks": f.get("chunks"), "status": f.get("status")}
                             for n, f in sorted(folder["files"].items())}},
        "golden_set": {"file": str(QA_FILE.relative_to(REPO)), "n": len(questions),
                       "sha256": sha256_file(QA_FILE)},
        "models": models, "config": config,
        "environment": dict(e.split("=", 1) for e in args.env if "=" in e),
        "gold_reachability": {"n_evidence": n_ev, "n_unreachable": len(loc["unreachable"]),
                              "n_split": n_split},
        "unreachable": loc["unreachable"],
        "summary": summary, "summary_per_run": summaries,
        "variance": variance(summaries), "rank_flips": rank_flips(runs),
        "per_question": runs[0],
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"rag-{date}-{label}"
    stem.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown_report(rep), encoding="utf-8")
    o = summary["mcp"]["overall"]
    log(f"\nMCP R@5 {o['recall@5']:.3f} R@10 {o['recall@10']:.3f} MRR@10 {o['mrr@10']:.3f} "
        f"nDCG@10 {o['ndcg@10']:.3f} | tokens/call {summary['mcp_response']['tokens_mean']}")
    log(f"wrote {stem}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
