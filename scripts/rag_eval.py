
"""RAG retrieval evaluation harness — turn subjective accuracy into comparable metrics.

Usage:
    uv run --no-sync python scripts/rag_eval.py --folder <name or ID> --n 20

Samples N indexed leaf chunks, uses the configured LLM to generate one question
per chunk (gold = that chunk's node_id), then scores three retrieval modes with
Recall@k, MRR and nDCG@k (see MODES). The eval set is cached under reports/ and
reused so scores are comparable across parameter changes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EvalError(Exception):
    """The evaluation cannot run as requested.

    This module is also loaded in-process by the server (POST /api/admin/eval/run),
    so it must never raise SystemExit: inside the event loop that would stop the
    whole server. Only the CLI entry point (main) turns these into a process exit.
    """


class FolderNotFound(EvalError):
    """No folder matches the given name or id."""


class NoIndexedChunks(EvalError):
    """The folder has no indexed leaf chunks to build questions from."""


def _resolve_folder(folder_arg: str):
    from db.folderdb import FolderDB
    if folder_arg.isdigit():
        rows = FolderDB.get(id=int(folder_arg))
    else:
        rows = FolderDB.get(name=folder_arg)
    if not rows:
        raise FolderNotFound(f"folder not found: {folder_arg}")
    return rows[0]


_NO_CHUNKS_MSG = "no indexed leaf chunks found — index some files first"

# Leaf chunks long enough to generate a meaningful question from.
_LEAF_FILTER = "COALESCE(metadata_->>'node_role','leaf')='leaf' AND length(text) > 80"

# PostgreSQL SQLSTATE for "undefined_table".
_PG_UNDEFINED_TABLE = "42P01"


def _query_vector_table(folder, sql: str, params: dict) -> list:
    """Run a statement against the folder's vector table and return the rows.

    ``sql`` names the table as ``{table}``. A folder that was never indexed has no
    vector table yet: that is reported as "no rows" rather than as a raw SQL error.
    """
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError
    from db.db import get_engine
    from src.domain.rag.vector_store_manager import VectorStoreManager
    table = VectorStoreManager.physical_table_name(folder.id, folder.vector_table_uuid)
    try:
        with get_engine().connect() as c:
            return c.execute(text(sql.format(table=f'"{table}"')), params).fetchall()
    except ProgrammingError as e:
        if getattr(getattr(e, "orig", None), "pgcode", None) == _PG_UNDEFINED_TABLE:
            return []
        raise


def _sample_chunks(folder, n: int):
    """Randomly sample n leaf chunks (node_id, text) from the folder's pgvector physical table."""
    rows = _query_vector_table(
        folder,
        f"SELECT node_id, text FROM {{table}} WHERE {_LEAF_FILTER} "
        "ORDER BY random() LIMIT :n", {"n": n})
    return [(r[0], r[1]) for r in rows]


def has_indexed_chunks(folder) -> bool:
    """True when the folder has at least one leaf chunk the eval can sample."""
    return bool(_query_vector_table(
        folder, f"SELECT 1 FROM {{table}} WHERE {_LEAF_FILTER} LIMIT 1", {}))


def check_ready(folder_ref: str):
    """Resolve the folder and confirm it can be evaluated; return the folder.

    Raises FolderNotFound or NoIndexedChunks. The server calls this before it
    schedules a background run, so those cases become an HTTP error instead of a
    failed run.
    """
    folder = _resolve_folder(folder_ref)
    if not has_indexed_chunks(folder):
        raise NoIndexedChunks(_NO_CHUNKS_MSG)
    return folder


def _gen_question(client, model: str, chunk_text: str) -> str:
    prompt = (
        "Below is a passage from a document. Write one question a user might ask whose answer is clearly in this passage, "
        "in the passage's own language. Make it specific (proper nouns are fine); do not refer to 'this passage'."
        "Output only the question itself, with no prefix.\n\nPassage:\n" + chunk_text[:1500] + "\n\nQuestion:")
    # Reasoning models (e.g. gpt-oss) spend reasoning tokens first; too small a
    # max_tokens leaves content None (no budget left for the answer) -> allow 1024.
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=1024, temperature=0.3)
    return (r.choices[0].message.content or "").strip().strip('"').replace("\n", " ")


def generate_eval_set(folder, n: int, out_path: str):
    from src.config.config_manager import Config
    from src.domain.rag.context_generator import ContextGenerator
    llm = Config.get_config_model().rag.llm
    client = ContextGenerator._create_sync_client(llm)
    model = getattr(llm, "azure_deployment", None) or llm.model
    samples = _sample_chunks(folder, n)
    if not samples:
        raise NoIndexedChunks(_NO_CHUNKS_MSG)
    items = []
    for i, (nid, txt) in enumerate(samples):
        try:
            q = _gen_question(client, model, txt)
            if q:
                items.append({"query": q, "gold_node_id": nid})
                print(f"  [{i+1}/{len(samples)}] {q[:60]}")
        except Exception as e:
            print(f"  [{i+1}] gen failed: {e}")
    try:
        client.close()
    except Exception:
        pass
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"→ wrote {len(items)} eval items to {out_path}")
    return items


# Modes scored for every question. "production" is THE number: the same adapter call
# POST /api/rag/query makes, with the live settings (fusion, rerank, thresholds), so a
# settings change shows up here exactly as users see it. "vector" and "bm25" isolate
# the two signals production fuses, via the production QueryEngine's own per-path
# retrieval (BM25 gets the same CKIP OR-query as production). The old "hybrid" mode
# (llama-index concat fusion) was dropped: pgvector's concat ranked dense first, so it
# always equalled "vector" and suggested BM25 added nothing.
MODES = ("vector", "bm25", "production")
MODE_LABELS = {
    "vector": "Vector only (dense embeddings)",
    "bm25": "Keyword only (BM25)",
    "production": "Production (/api/rag/query: fusion + rerank when enabled)",
}
BEST_MODE = "production"
# Bumped when modes or scoring change, so history deltas never compare across harnesses.
HARNESS_VERSION = 2


class _Retrievers:
    """The retrieval calls behind each mode, bound to one folder."""

    def __init__(self, folder, k: int):
        from src.adapter.rag import get_rag_adapter
        from src.adapter.rag_query import _fusion_config
        from src.api.router.rag_query import _resolve_retrieval_defaults
        from src.domain.rag.query_engine import QueryEngine

        self.folder, self.k = folder, k
        self.adapter = get_rag_adapter()
        self.defaults = _resolve_retrieval_defaults()
        fusion, rrf_k, tsc = _fusion_config()
        ctx = self.adapter._query_svc._ctx
        # Diagnostics engine for the single-signal modes (no reranker: raw path ranks).
        self.engine = QueryEngine(
            vector_store=ctx.get_vector_store(folder.id, folder.user_token),
            top_k=k, similarity_cutoff=self.defaults["similarity_cutoff"],
            sparse_top_k=max(k, self.defaults["sparse_top_k"] or 0),
            hybrid_alpha=self.defaults["hybrid_alpha"], reranker=None,
            hybrid_fusion=fusion, rrf_k=rrf_k, text_search_config=tsc,
        )

    async def nodes(self, query: str, mode: str) -> list:
        """[(node_id, text)] of this mode's top-k, best first."""
        k = self.k
        if mode == "vector":
            got = await self.engine._aretrieve_mode(query, "default", k)
            return [(n.node.node_id, n.node.text) for n in got[:k]]
        if mode == "bm25":
            got = await self.engine._aretrieve_mode(query, "sparse", k)
            return [(n.node.node_id, n.node.text) for n in got[:k]]
        if mode == "production":
            d = self.defaults
            res = await self.adapter.query_rag(
                query=query, folder_id=self.folder.id, token=self.folder.user_token,
                top_k=k, similarity_cutoff=d["similarity_cutoff"], use_cache=False,
                sparse_top_k=d["sparse_top_k"], hybrid_alpha=d["hybrid_alpha"])
            return [(r.metadata.node_id, r.text) for r in res.results[:k]]
        raise ValueError(f"unknown eval mode: {mode}")


def _rank_in(node_ids: list, gold: str):
    """1-based rank of gold among node_ids, or None."""
    for i, nid in enumerate(node_ids):
        if nid == gold:
            return i + 1
    return None


def question_set_id(items: list) -> str:
    """Stable fingerprint of an eval set (gold ids + questions) — history rows are only
    comparable when it matches."""
    import hashlib
    h = hashlib.sha1()
    for line in sorted(f"{it.get('gold_node_id')}\t{it.get('query')}" for it in items):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def comparable_previous(history: list, rep: dict):
    """The latest history row scored on the same question set, k and harness, else None.

    A delta against a run on different questions, a different k or the old mode set is
    noise, not a regression or a gain — the console hides it instead.
    """
    for h in reversed(history):
        if (h.get("harness") == rep.get("harness")
                and h.get("question_set") == rep.get("question_set")
                and h.get("k") == rep.get("k")):
            return h
    return None


def _answer_and_judge(client, model: str, query: str, contexts: list) -> dict:
    """Generate an answer from the retrieved contexts, then LLM-judge it (blocking).

    Two calls: (1) answer using ONLY the contexts; (2) grade faithfulness (is every
    claim grounded in the context?) and relevance (does it answer the question?) on a
    0.0-1.0 scale. Runs in an executor from the async evaluate loop.
    """
    import re
    ctx = "\n\n".join(f"[{i+1}] {(t or '')[:1200]}" for i, t in enumerate(contexts))
    aprompt = (
        "Answer the question using ONLY the context below. Cite sources inline as "
        "[n]. If the context lacks the answer, say so.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:")
    ar = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": aprompt}],
        max_tokens=1024, temperature=0.2)
    answer = (ar.choices[0].message.content or "").strip()

    jprompt = (
        "You are grading a RAG answer. Given the QUESTION, CONTEXT and ANSWER, rate "
        "two scores from 0.0 to 1.0:\n"
        "- faithfulness: is every claim in the answer supported by the context? "
        "(1.0 = fully grounded, 0.0 = fabricated)\n"
        "- relevance: does the answer actually address the question?\n"
        'Reply ONLY compact JSON: {"faithfulness": <0-1>, "relevance": <0-1>}.\n\n'
        f"QUESTION: {query}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nJSON:")
    jr = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": jprompt}],
        max_tokens=512, temperature=0)
    faith = rel = None
    try:
        m = re.search(r"\{.*\}", jr.choices[0].message.content or "", re.S)
        if m:
            d = json.loads(m.group(0))
            faith = max(0.0, min(1.0, float(d.get("faithfulness"))))
            rel = max(0.0, min(1.0, float(d.get("relevance"))))
    except Exception:
        pass
    return {"answer": answer, "faithfulness": faith, "relevance": rel}


def _config_snapshot() -> dict:
    """Key retrieval settings, recorded per run so history rows say what produced each score."""
    try:
        from src.config.config_manager import Config
        c = Config.get_config_model().rag
        ret = getattr(c, "retrieval", None)
        return {
            "embedding_model": getattr(getattr(c, "embedding", None), "model", None),
            "rerank_enabled": bool(getattr(getattr(c, "rerank", None), "enabled", False)),
            "rerank_model": getattr(getattr(c, "rerank", None), "model", None),
            "fusion": getattr(ret, "hybrid_fusion", None),
            "contextual": bool(getattr(getattr(c, "contextual_retrieval", None), "enabled", False)),
        }
    except Exception:
        return {}


def _history_path(slug: str) -> str:
    from src.domain.rag.eval_paths import history_path
    return history_path(slug)


def _read_history(slug: str) -> list:
    """All past run summaries for this folder (oldest first); [] when none."""
    p = _history_path(slug)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _append_history(slug: str, rep: dict) -> None:
    """Append one compact run record so scores are comparable across parameter changes."""
    entry = {
        "generated_at": rep.get("generated_at"), "n": rep.get("n"), "k": rep.get("k"),
        "summary": rep.get("summary"), "answer_quality": rep.get("answer_quality"),
        "config": rep.get("config"), "modes": rep.get("modes"),
        "harness": rep.get("harness"), "question_set": rep.get("question_set"),
    }
    p = _history_path(slug)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


_STALE_REASON = ("stale eval set — none of its gold node_ids exist in the index "
                 "(folder reindexed?); question set regenerated")


def _existing_gold_ids(folder, items: list) -> set:
    """The eval set's gold node_ids that still exist in the folder's vector table.

    A reindex rewrites node_ids: a full reindex silently turns a cached eval set
    into an all-zero run; a single-file reindex quietly drags scores down. Callers
    drop stale items (recording how many) and regenerate only when none survive.
    """
    ids = [it["gold_node_id"] for it in items if it.get("gold_node_id")]
    if not ids:
        return set()
    rows = _query_vector_table(
        folder, "SELECT node_id FROM {table} WHERE node_id = ANY(:ids)", {"ids": ids})
    return {r[0] for r in rows}


# Regenerate (rather than score a shrunk, non-comparable sample) once more than
# this fraction of a cached eval set has gone stale after a reindex.
_STALE_REGEN_FRACTION = 0.30


def _prune_stale(folder, items: list, alive: set):
    """Return (items_kept, regen_reason, stale_dropped) for a loaded eval set.

    Fully stale, or stale beyond _STALE_REGEN_FRACTION of the set, → regenerate
    (return None): scoring on a materially smaller sample makes nDCG/Recall/MRR
    non-comparable to prior history (which recorded a larger n), and small eval sets
    are already noisy. Light staleness keeps the survivors and reports the drop
    count (surfaced in the UI) rather than forcing a full regen.
    """
    kept = [it for it in items if it.get("gold_node_id") in alive]
    dropped = len(items) - len(kept)
    if not kept:
        return None, _STALE_REASON, 0
    if items and dropped / len(items) > _STALE_REGEN_FRACTION:
        pctv = round(dropped / len(items) * 100)
        return None, (f"stale eval set — {dropped}/{len(items)} gold chunks ({pctv}%) "
                      "no longer exist (folder reindexed?); question set regenerated"), 0
    return kept, None, dropped


async def evaluate(folder, items, k: int, progress=None, judge: bool = False) -> dict:
    """Run the multi-mode retrieval eval and return a report dict (no printing or file writing — left to the caller).

    progress(done, total) is optional, for a UI to report progress.
    judge=True also generates an answer per question over the best retrieval mode and
    LLM-grades faithfulness + relevance (2 extra LLM calls/item — opt-in, slower).
    """
    retr = _Retrievers(folder, k)
    modes = list(MODES)

    jclient = jmodel = None
    best_mode = BEST_MODE
    loop = asyncio.get_event_loop()
    faith_sum = rel_sum = 0.0
    faith_n = rel_n = 0
    if judge:
        from src.config.config_manager import Config
        from src.domain.rag.context_generator import ContextGenerator
        llm = Config.get_config_model().rag.llm
        jclient = ContextGenerator._create_sync_client(llm)
        jmodel = getattr(llm, "azure_deployment", None) or llm.model

    agg = {m: {"recall": 0, "mrr": 0.0, "ndcg": 0.0} for m in modes}
    per_item = []
    for j, it in enumerate(items):
        row = {"query": it["query"], "ranks": {}}
        best_nodes = []
        for m in modes:
            got = await retr.nodes(it["query"], m)
            if m == best_mode:
                best_nodes = got
            rank = _rank_in([nid for nid, _ in got], it["gold_node_id"])
            row["ranks"][m] = rank
            if rank is not None and rank <= k:
                agg[m]["recall"] += 1
                agg[m]["mrr"] += 1.0 / rank
                agg[m]["ndcg"] += 1.0 / math.log2(rank + 1)
        if judge:
            try:
                contexts = [t for _, t in best_nodes[:6]]
                res = await loop.run_in_executor(
                    None, _answer_and_judge, jclient, jmodel, it["query"], contexts)
                row["answer"] = res["answer"]
                row["faithfulness"] = res["faithfulness"]
                row["relevance"] = res["relevance"]
                if res["faithfulness"] is not None:
                    faith_sum += res["faithfulness"]
                    faith_n += 1
                if res["relevance"] is not None:
                    rel_sum += res["relevance"]
                    rel_n += 1
            except Exception as e:  # noqa: BLE001 — one bad item shouldn't fail the run
                row["answer_error"] = str(e)
        per_item.append(row)
        if progress:
            progress(j + 1, len(items))

    if jclient is not None:
        try:
            jclient.close()
        except Exception:
            pass

    n = len(items) or 1
    summary = {m: {"recall@%d" % k: round(agg[m]["recall"] / n, 3),
                   "mrr": round(agg[m]["mrr"] / n, 3),
                   "ndcg@%d" % k: round(agg[m]["ndcg"] / n, 3)} for m in modes}
    answer_quality = None
    if judge:
        answer_quality = {
            "mode": best_mode,
            "faithfulness": round(faith_sum / faith_n, 3) if faith_n else None,
            "relevance": round(rel_sum / rel_n, 3) if rel_n else None,
            "n_judged": max(faith_n, rel_n),
        }
    return {"folder": folder.name, "n": len(items), "k": k,
            "modes": modes, "mode_labels": {m: MODE_LABELS[m] for m in modes},
            "summary": summary, "per_item": per_item,
            "answer_quality": answer_quality, "config": _config_snapshot(),
            "harness": HARNESS_VERSION, "question_set": question_set_id(items)}


def _print_report(rep: dict):
    k = rep["k"]
    print("\n" + "=" * 56)
    print(f"RAG retrieval eval — folder='{rep['folder']}'  n={rep['n']}  k={k}")
    print("=" * 56)
    print(f"{'mode':<11} {'Recall@%d'%k:>10} {'MRR':>8} {'nDCG@%d'%k:>9}")
    for m in rep["modes"]:
        s = rep["summary"][m]
        print(f"{m:<11} {s['recall@%d'%k]:>10} {s['mrr']:>8} {s['ndcg@%d'%k]:>9}")
    for m in rep["modes"]:
        print(f"  {m}: {rep.get('mode_labels', {}).get(m, '')}")
    aq = rep.get("answer_quality")
    if aq:
        print("-" * 56)
        print(f"answer quality ({aq['mode']}, n={aq['n_judged']}): "
              f"faithfulness={aq['faithfulness']}  relevance={aq['relevance']}")
    print("=" * 56)


def run_eval(folder_name: str, n: int = 15, k: int = 10,
             regenerate: bool = False, progress=None, judge: bool = False) -> dict:
    """One-stop: load/generate eval set -> evaluate -> write report + history -> return dict (shared by CLI and API).

    regenerate=True forces regenerating the eval set (use when node_ids have
    changed after a reindex and the old set is stale). judge=True adds LLM-graded
    answer quality.
    """
    folder = _resolve_folder(folder_name)
    from src.domain.rag.eval_paths import eval_set_path, report_path, report_slug
    slug = report_slug(folder)
    eval_path = eval_set_path(slug)
    items, regen_reason, stale_dropped = None, None, 0
    if not regenerate and os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        if items:
            items, regen_reason, stale_dropped = _prune_stale(
                folder, items, _existing_gold_ids(folder, items))
            if regen_reason:
                print(f"  ! {regen_reason}")
            elif stale_dropped:
                print(f"  ! dropped {stale_dropped} stale item(s) whose gold chunk no longer exists")
    if not items:  # None, an empty cached file, or all-stale (pruned to None)
        items = generate_eval_set(folder, n, eval_path)
    if not items:
        # Every question-gen LLM call failed. Do NOT evaluate/append: an empty set
        # scores 0.000 across the board (n = len(items) or 1) and would poison the
        # baseline history permanently. Fail loudly instead.
        raise RuntimeError(
            "evaluation aborted: no eval questions could be generated "
            "(LLM question generation failed for every sampled chunk). "
            "Check the LLM endpoint and retry — nothing was written to history.")
    prev = _read_history(slug)
    rep = asyncio.run(evaluate(folder, items, k, progress=progress, judge=judge))
    rep["regenerate_reason"] = regen_reason
    rep["stale_dropped"] = stale_dropped
    import datetime
    rep["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    rep["previous"] = comparable_previous(prev, rep)
    if prev and rep["previous"] is None:
        rep["previous_note"] = ("no earlier run on the same question set, k and modes — "
                                "deltas are hidden")
    out_path = report_path(slug)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    _append_history(slug, rep)
    rep["report_path"] = out_path  # for the CLI message; not part of the saved report
    return rep


async def run_eval_async(folder_name: str, n: int = 15, k: int = 10,
                         regenerate: bool = False, progress=None, judge: bool = False) -> dict:
    """Async version of run_eval — runs evaluate on the caller's event loop.

    For server use: it must run on the main loop (the same loop as the cached
    PGVectorStore async engine), otherwise asyncpg raises "attached to a
    different loop". The blocking synchronous question generation is offloaded to
    an executor so it does not occupy the loop. The CLI still uses run_eval
    (asyncio.run brings its own loop, and a separate process avoids this issue).
    """
    import datetime
    folder = _resolve_folder(folder_name)
    from src.domain.rag.eval_paths import eval_set_path, report_path, report_slug
    slug = report_slug(folder)
    eval_path = eval_set_path(slug)
    loop = asyncio.get_event_loop()
    items, regen_reason, stale_dropped = None, None, 0
    if not regenerate and os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        if items:
            alive = await loop.run_in_executor(None, _existing_gold_ids, folder, items)
            items, regen_reason, stale_dropped = _prune_stale(folder, items, alive)
    if not items:  # None, an empty cached file, or all-stale (pruned to None)
        items = await loop.run_in_executor(None, generate_eval_set, folder, n, eval_path)
    if not items:
        # See run_eval: refuse to evaluate/append an empty set (would write an
        # all-zero run to baseline history and skew every later delta).
        raise RuntimeError(
            "evaluation aborted: no eval questions could be generated "
            "(LLM question generation failed for every sampled chunk). "
            "Check the LLM endpoint and retry — nothing was written to history.")
    prev = _read_history(slug)
    rep = await evaluate(folder, items, k, progress=progress, judge=judge)
    rep["regenerate_reason"] = regen_reason
    rep["stale_dropped"] = stale_dropped
    rep["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    rep["previous"] = comparable_previous(prev, rep)
    if prev and rep["previous"] is None:
        rep["previous_note"] = ("no earlier run on the same question set, k and modes — "
                                "deltas are hidden")
    out_path = report_path(slug)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    _append_history(slug, rep)
    return rep


def main():
    ap = argparse.ArgumentParser(description="RAG retrieval eval harness")
    ap.add_argument("--folder", required=True, help="folder name or id")
    ap.add_argument("--n", type=int, default=20, help="number of synthetic eval questions to generate")
    ap.add_argument("--k", type=int, default=10, help="top-k cutoff for metrics")
    ap.add_argument("--regenerate", action="store_true",
                    help="force-regenerate the eval set (use after reindex — node_ids change)")
    ap.add_argument("--judge", action="store_true",
                    help="also LLM-grade answer quality (faithfulness + relevance; 2 extra LLM calls/item)")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    from src.config.config_manager import Config
    Config.set_config(args.config)
    from db.runtime_settings_db import RuntimeSettingsDB
    RuntimeSettingsDB.ensure_table()

    try:
        rep = run_eval(args.folder, n=args.n, k=args.k, regenerate=args.regenerate,
                       judge=args.judge,
                       progress=lambda d, t: print(f"  [{d}/{t}]", end="\r", flush=True))
    except EvalError as e:
        # CLI only: exit non-zero with the message on stderr.
        raise SystemExit(f"error: {e}")
    _print_report(rep)
    print(f"→ wrote report to {rep.get('report_path')}")


if __name__ == "__main__":
    main()
