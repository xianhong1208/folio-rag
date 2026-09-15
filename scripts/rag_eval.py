
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


async def evaluate(folder, items, k: int, progress=None) -> dict:
    """Run the multi-mode retrieval eval and return a report dict (no printing or file writing — left to the caller).

    progress(done, total) is optional, for a UI to report progress.
    """
    from llama_index.core import VectorStoreIndex
    from src.adapter.rag import get_rag_adapter
    ctx = get_rag_adapter()._query_svc._ctx
    vs = ctx.get_vector_store(folder.id, folder.user_token)
    index = VectorStoreIndex.from_vector_store(vs)
    reranker = ctx.reranker
    modes = ["vector", "hybrid", "rrf"] + (["rerank"] if reranker else [])

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
        per_item.append(row)
        if progress:
            progress(j + 1, len(items))

    n = len(items) or 1
    summary = {m: {"recall@%d" % k: round(agg[m]["recall"] / n, 3),
                   "mrr": round(agg[m]["mrr"] / n, 3),
                   "ndcg@%d" % k: round(agg[m]["ndcg"] / n, 3)} for m in modes}
    return {"folder": folder.name, "n": len(items), "k": k,
            "modes": modes, "summary": summary, "per_item": per_item}


def _print_report(rep: dict):
    k = rep["k"]
    print("\n" + "=" * 56)
    print(f"RAG retrieval eval — folder='{rep['folder']}'  n={rep['n']}  k={k}")
    print("=" * 56)
    print(f"{'mode':<10} {'Recall@%d'%k:>10} {'MRR':>8} {'nDCG@%d'%k:>9}")
    for m in rep["modes"]:
        s = rep["summary"][m]
        print(f"{m:<10} {s['recall@%d'%k]:>10} {s['mrr']:>8} {s['ndcg@%d'%k]:>9}")
    print("=" * 56)


def run_eval(folder_name: str, n: int = 15, k: int = 10,
             regenerate: bool = False, progress=None) -> dict:
    """One-stop: load/generate eval set -> evaluate -> write report -> return dict (shared by CLI and API).

    regenerate=True forces regenerating the eval set (use when node_ids have
    changed after a reindex and the old set is stale).
    """
    folder = _resolve_folder(folder_name)
    slug = str(folder.name).replace("/", "_")
    eval_path = f"reports/eval_set_{slug}.jsonl"
    if regenerate or not os.path.exists(eval_path):
        items = generate_eval_set(folder, n, eval_path)
    else:
        with open(eval_path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    rep = asyncio.run(evaluate(folder, items, k, progress=progress))
    import datetime
    rep["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    out_path = f"reports/rag_eval_{slug}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return rep


async def run_eval_async(folder_name: str, n: int = 15, k: int = 10,
                         regenerate: bool = False, progress=None) -> dict:
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
    rep = await evaluate(folder, items, k, progress=progress)
    rep["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    out_path = f"reports/rag_eval_{slug}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return rep


def main():
    ap = argparse.ArgumentParser(description="RAG retrieval eval harness")
    ap.add_argument("--folder", required=True, help="folder name or id")
    ap.add_argument("--n", type=int, default=20, help="number of synthetic eval questions to generate")
    ap.add_argument("--k", type=int, default=10, help="top-k cutoff for metrics")
    ap.add_argument("--regenerate", action="store_true",
                    help="force-regenerate the eval set (use after reindex — node_ids change)")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    from src.config.config_manager import Config
    Config.set_config(args.config)
    from db.runtime_settings_db import RuntimeSettingsDB
    RuntimeSettingsDB.ensure_table()

    rep = run_eval(args.folder, n=args.n, k=args.k, regenerate=args.regenerate,
                   progress=lambda d, t: print(f"  [{d}/{t}]", end="\r", flush=True))
    _print_report(rep)
    print(f"→ wrote report to reports/rag_eval_{str(rep['folder']).replace('/', '_')}.json")


if __name__ == "__main__":
    main()
