
"""Score RAG retrieval evaluation: index a golden document set -> retrieve per QA -> recall@k / MRR.

Usage:
    uv run --no-sync python scripts/run_rag_eval.py                # index + query + score
    uv run --no-sync python scripts/run_rag_eval.py --keep         # keep the eval folder (debugging)
    uv run --no-sync python scripts/run_rag_eval.py --baseline evals/baselines/rag_xxx.json

Material (see evals/README.md):
    evals/rag/documents/*                # representative document set (any whitelisted format)
    evals/rag/golden_qa.yaml             # QA set (question / expected_files)

Prerequisite: the PostgreSQL and embedding services referenced by config must be reachable.
The eval folder uses a dedicated token 'rag-eval' and is cleaned up automatically
when done (--keep to retain it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import date
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DOCS_DIR = REPO / "evals" / "rag" / "documents"
QA_FILE = REPO / "evals" / "rag" / "golden_qa.yaml"
BASELINE_DIR = REPO / "evals" / "baselines"
EVAL_TOKEN = "rag-eval"
TOP_K = 10


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the eval folder")
    ap.add_argument("--baseline", help="compare against an existing baseline JSON")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    docs = [f for f in sorted(DOCS_DIR.iterdir()) if f.is_file()] if DOCS_DIR.is_dir() else []
    qa_items = yaml.safe_load(QA_FILE.read_text(encoding="utf-8")) if QA_FILE.is_file() else None
    qa_items = [q for q in (qa_items or []) if isinstance(q, dict) and q.get("question")]
    if not docs or not qa_items:
        print("No eval material. Please provide:")
        print(f"   {DOCS_DIR.relative_to(REPO)}/ — representative document set")
        print(f"   {QA_FILE.relative_to(REPO)} — QA set (format described in the file)")
        return 2

    from src.config.config_manager import Config
    Config.set_config(args.config)

    # Preflight the embedding service (needed for both indexing and querying); fail fast with a clear message
    from src.adapter.rag_context import RAGContext
    try:
        ctx = RAGContext.from_config()
    except Exception as e:
        print(f"RAGContext init failed (embedding/DB service not ready?): {e}")
        return 2

    from db.folderdb import FolderDB
    from db.db import get_engine
    from src.domain.rag.vector_store_manager import VectorStoreManager

    folder = FolderDB.create(
        name=f"eval-{date.today().isoformat()}-{uuid.uuid4().hex[:6]}",
        description="RAG eval run (auto-created)", user_token=EVAL_TOKEN,
    )
    print(f"eval folder: {folder.name} (id={folder.id})")
    store = ctx.vector_store_manager.get_or_create(
        folder_id=folder.id, vector_table_uuid=str(folder.vector_table_uuid))

    try:
        t0 = time.perf_counter()
        for f in docs:
            doc = ctx.indexer.load_document_from_file(
                file_path=str(f), file_id=str(uuid.uuid4()), file_name=f.name)
            # Retrieval hit detection uses file_name — overwrite metadata to keep it consistent
            doc.metadata["file_name"] = f.name
            r = await ctx.indexer.index_document(doc, store)
            print(f"  indexed {f.name}: {r['leaves']} leaves / {r['parents']} parents")
        index_s = time.perf_counter() - t0

        from src.domain.rag.auto_merging import AutoMergingRetriever
        from evals.metrics import mrr, recall_at_k
        retr_cfg = Config.get_config_model().rag.retrieval
        retriever = AutoMergingRetriever(
            vector_store=store, reranker=ctx.reranker,
            merge_threshold=ctx.merge_threshold if ctx.auto_merging_enabled else 1.1,
            expand_context_neighbors=ctx.expand_neighbors,
            sparse_top_k=retr_cfg.default_sparse_top_k,
            hybrid_alpha=retr_cfg.default_hybrid_alpha,
        )
        rows = []
        for qa in qa_items:
            results = await retriever.aquery(
                query_text=qa["question"], top_k=TOP_K, similarity_cutoff=0.0)
            retrieved_files = []
            for res in results:  # order-preserving dedup
                if res.file_name not in retrieved_files:
                    retrieved_files.append(res.file_name)
            expected = qa.get("expected_files", [])
            rows.append({
                "id": qa.get("id", qa["question"][:20]),
                "recall_at_5": recall_at_k(retrieved_files, expected, 5),
                "recall_at_10": recall_at_k(retrieved_files, expected, 10),
                "mrr": round(mrr(retrieved_files, expected), 4),
                "retrieved_top5": retrieved_files[:5],
            })
            print(f"  {rows[-1]['id']}: R@5={rows[-1]['recall_at_5']:.2f} MRR={rows[-1]['mrr']:.2f}")

        summary = {
            "date": date.today().isoformat(),
            "embedding_model": ctx.model_name,
            "n_documents": len(docs), "n_queries": len(rows),
            "index_seconds": round(index_s, 1),
            "avg_recall_at_5": round(sum(r["recall_at_5"] for r in rows) / len(rows), 4),
            "avg_recall_at_10": round(sum(r["recall_at_10"] for r in rows) / len(rows), 4),
            "avg_mrr": round(sum(r["mrr"] for r in rows) / len(rows), 4),
            "queries": rows,
        }
        out = BASELINE_DIR / f"rag_{date.today().isoformat()}.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n== avg R@5 {summary['avg_recall_at_5']:.2%} | R@10 "
              f"{summary['avg_recall_at_10']:.2%} | MRR {summary['avg_mrr']:.3f} ==")
        print(f"baseline written: {out.relative_to(REPO)}")

        if args.baseline:
            base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            print(f"vs {base['date']} (emb={base.get('embedding_model')}): "
                  f"R@5 {base['avg_recall_at_5']:.2%} → {summary['avg_recall_at_5']:.2%}, "
                  f"MRR {base['avg_mrr']:.3f} → {summary['avg_mrr']:.3f}")
        return 0
    finally:
        if args.keep:
            print(f"--keep: retaining folder {folder.id} and its vector table")
        else:
            ctx.vector_store_manager.drop_table(folder.id, str(folder.vector_table_uuid))
            FolderDB.delete(id=folder.id)
            print("eval folder cleaned up")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
