
"""RAG retrieval evaluation harness — turn subjective accuracy into comparable metrics.

Usage:
    uv run --no-sync python scripts/rag_eval.py --folder <name or ID> --n 20

Samples N indexed leaf chunks, uses the configured LLM to generate one question
per chunk (gold = that chunk's node_id), then runs vector / hybrid / rrf / rerank
retrieval and reports Recall@k, MRR, and nDCG@k for each mode. The eval set is
cached under reports/ and reused so scores are comparable across parameter changes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_folder(folder_arg: str):
    from db.folderdb import FolderDB
    if folder_arg.isdigit():
        rows = FolderDB.get(id=int(folder_arg))
    else:
        rows = FolderDB.get(name=folder_arg)
    if not rows:
        raise SystemExit(f"folder not found: {folder_arg}")
    return rows[0]


def _sample_chunks(folder, n: int):
    """Randomly sample n leaf chunks (node_id, text) from the folder's pgvector physical table."""
    from sqlalchemy import text
    from db.db import get_engine
    from src.domain.rag.vector_store_manager import VectorStoreManager
    table = VectorStoreManager.physical_table_name(folder.id, folder.vector_table_uuid)
    with get_engine().connect() as c:
        rows = c.execute(text(
            f"SELECT node_id, text FROM \"{table}\" "
            f"WHERE COALESCE(metadata_->>'node_role','leaf')='leaf' "
            f"AND length(text) > 80 ORDER BY random() LIMIT :n"), {"n": n}).fetchall()
    return [(r[0], r[1]) for r in rows]


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
        raise SystemExit("no indexed leaf chunks found — index some files first")
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


async def _retr(index, query: str, vmode: str, k: int):
    from llama_index.core.retrievers import VectorIndexRetriever
    kwargs = {"sparse_top_k": max(k, 20)} if vmode in ("hybrid", "sparse") else {}
    r = VectorIndexRetriever(index=index, similarity_top_k=k,
                             vector_store_query_mode=vmode, vector_store_kwargs=kwargs)
    nodes = await r.aretrieve(query)
    return [n for n in nodes if n.node.metadata.get("node_role", "leaf") == "leaf"]


def _rrf(lists, k, rrf_k=60):
    scores, node_map = {}, {}
    for lst in lists:
        for rank, n in enumerate(lst):
            nid = n.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (rrf_k + rank + 1)
            node_map.setdefault(nid, n)
    return sorted(node_map.values(), key=lambda n: scores[n.node.node_id], reverse=True)[:k]


async def _rank_of_gold(index, reranker, query: str, gold: str, mode: str, k: int):
    """Return the gold node_id's rank (1-based) in this mode's top-k, or None if absent.

    mode: vector (dense) / hybrid (llama_index concat) / rrf (dense+sparse RRF) /
          rerank (cross-encoder rerank over the RRF candidates — matches
          production aquery).
    """
    retrieve_k = k * 3 if mode == "rerank" else k
    if mode == "vector":
        nodes = (await _retr(index, query, "default", k))[:k]
    elif mode == "hybrid":
        nodes = (await _retr(index, query, "hybrid", k))[:k]
    elif mode in ("rrf", "rerank"):
        dense = await _retr(index, query, "default", retrieve_k)
        sparse = await _retr(index, query, "sparse", retrieve_k)
        nodes = _rrf([dense, sparse], retrieve_k)
        if mode == "rerank" and reranker and nodes:
            nodes = await reranker.arerank(query, nodes, top_n=k)
        else:
            nodes = nodes[:k]
    else:
        nodes = []
    for i, n in enumerate(nodes):
        if n.node.node_id == gold:
            return i + 1
    return None


async def _retrieve_nodes(index, reranker, query: str, mode: str, k: int):
    """Return this mode's top-k nodes (same retrieval as _rank_of_gold, but the nodes
    themselves — used to build answer-generation context for the judge)."""
    retrieve_k = k * 3 if mode == "rerank" else k
    if mode == "vector":
        return (await _retr(index, query, "default", k))[:k]
    if mode == "hybrid":
        return (await _retr(index, query, "hybrid", k))[:k]
    if mode in ("rrf", "rerank"):
        dense = await _retr(index, query, "default", retrieve_k)
        sparse = await _retr(index, query, "sparse", retrieve_k)
        nodes = _rrf([dense, sparse], retrieve_k)
        if mode == "rerank" and reranker and nodes:
            return await reranker.arerank(query, nodes, top_n=k)
        return nodes[:k]
    return []


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
    return f"reports/rag_eval_history_{slug}.jsonl"


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
        "config": rep.get("config"),
    }
    p = _history_path(slug)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def evaluate(folder, items, k: int, progress=None, judge: bool = False) -> dict:
    """Run the multi-mode retrieval eval and return a report dict (no printing or file writing — left to the caller).

    progress(done, total) is optional, for a UI to report progress.
    judge=True also generates an answer per question over the best retrieval mode and
    LLM-grades faithfulness + relevance (2 extra LLM calls/item — opt-in, slower).
    """
    from llama_index.core import VectorStoreIndex
    from src.adapter.rag import get_rag_adapter
    ctx = get_rag_adapter()._query_svc._ctx
    vs = ctx.get_vector_store(folder.id, folder.user_token)
    index = VectorStoreIndex.from_vector_store(vs)
    reranker = ctx.reranker
    modes = ["vector", "hybrid", "rrf"] + (["rerank"] if reranker else [])

    jclient = jmodel = None
    best_mode = "rerank" if reranker else "rrf"
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
        for m in modes:
            rank = await _rank_of_gold(index, reranker, it["query"], it["gold_node_id"], m, k)
            row["ranks"][m] = rank
            if rank is not None and rank <= k:
                agg[m]["recall"] += 1
                agg[m]["mrr"] += 1.0 / rank
                agg[m]["ndcg"] += 1.0 / math.log2(rank + 1)
        if judge:
            try:
                nodes = await _retrieve_nodes(index, reranker, it["query"], best_mode, k)
                contexts = [n.node.text for n in nodes[:6]]
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
            "modes": modes, "summary": summary, "per_item": per_item,
            "answer_quality": answer_quality, "config": _config_snapshot()}


def _print_report(rep: dict):
    k = rep["k"]
    print("\n" + "=" * 56)
    print(f"RAG retrieval eval — folder='{rep['folder']}'  n={rep['n']}  k={k}")
    print("=" * 56)
    print(f"{'mode':<10} {'Recall@%d'%k:>10} {'MRR':>8} {'nDCG@%d'%k:>9}")
    for m in rep["modes"]:
        s = rep["summary"][m]
        print(f"{m:<10} {s['recall@%d'%k]:>10} {s['mrr']:>8} {s['ndcg@%d'%k]:>9}")
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
    slug = str(folder.name).replace("/", "_")
    eval_path = f"reports/eval_set_{slug}.jsonl"
    if regenerate or not os.path.exists(eval_path):
        items = generate_eval_set(folder, n, eval_path)
    else:
        with open(eval_path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    prev = _read_history(slug)
    rep = asyncio.run(evaluate(folder, items, k, progress=progress, judge=judge))
    import datetime
    rep["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    rep["previous"] = prev[-1] if prev else None
    out_path = f"reports/rag_eval_{slug}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    _append_history(slug, rep)
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
    slug = str(folder.name).replace("/", "_")
    eval_path = f"reports/eval_set_{slug}.jsonl"
    loop = asyncio.get_event_loop()
    if regenerate or not os.path.exists(eval_path):
        items = await loop.run_in_executor(None, generate_eval_set, folder, n, eval_path)
    else:
        with open(eval_path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    prev = _read_history(slug)
    rep = await evaluate(folder, items, k, progress=progress, judge=judge)
    rep["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    rep["previous"] = prev[-1] if prev else None
    out_path = f"reports/rag_eval_{slug}.json"
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

    rep = run_eval(args.folder, n=args.n, k=args.k, regenerate=args.regenerate,
                   judge=args.judge,
                   progress=lambda d, t: print(f"  [{d}/{t}]", end="\r", flush=True))
    _print_report(rep)
    print(f"→ wrote report to reports/rag_eval_{str(rep['folder']).replace('/', '_')}.json")


if __name__ == "__main__":
    main()
