# Retrieval baseline — 2026-11-04 — `search-latency`

Folio at http://127.0.0.1:5139 (code `dbec758`), folder `folio_eval_golden_v1` (id 4), 19 documents, 205 questions (185 answerable, 20 unanswerable) + 20 out-of-scope, 1 runs. Harness v2; scores below are the mean of the runs.

Models: embedding `intfloat/multilingual-e5-large`, rerank `mixedbread-ai/mxbai-rerank-base-v2`, llm `openai/gpt-oss-120b`; environment: pgvector 0.8.7

## Overall

| path | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | file R@5 | FA@0.25 | FA@0.5 |
|---|---|---|---|---|---|---|---|---|
| MCP search (defaults) | 0.976 | 0.981 | — | 0.939 | 0.938 | 0.987 | 1.000 | 1.000 |
| MCP search, top_k=20 | 0.976 | 0.987 | 0.987 | 0.939 | 0.939 | 0.987 | 1.000 | 1.000 |
| Trace: fused+rerank leaves, k=20 | 0.968 | 0.976 | 0.981 | 0.926 | 0.927 | 0.984 | 1.000 | 1.000 |

FA@t = share of unanswerable questions whose top result scores ≥ t (answerable questions clearing the same bar: 0.25 → 1.000, 0.5 → 1.000 on the MCP path).

## Relevance cutoff (`rag.retrieval.min_relevance`)

What each cutoff would report as "nothing relevant" on the default MCP search: a search is rejected when it returned nothing or its best rerank score (`top_relevance`, before `score_threshold`) is below the cutoff. Recall columns score a rejected answerable question as 0 (as if the results were dropped); lost@5 = answerable questions with a top-5 hit that would be rejected. out_of_scope: n=20, unanswerable: n=20.

| min_relevance | out_of_scope rejected | unanswerable rejected | all unanswerable rejected | answerable wrongly rejected | R@5 | R@10 | ΔR@5 | lost@5 |
|---|---|---|---|---|---|---|---|---|
| 0 | 55.0% | 0.0% | 27.5% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.25 | 55.0% | 0.0% | 27.5% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.5 | 90.0% | 0.0% | 45.0% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.75 | 95.0% | 5.0% | 50.0% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.8 | 100.0% | 5.0% | 52.5% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.85 | 100.0% | 5.0% | 52.5% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.9 | 100.0% | 5.0% | 52.5% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.95 | 100.0% | 10.0% | 55.0% | 0.5% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.98 | 100.0% | 10.0% | 55.0% | 0.5% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.99 | 100.0% | 20.0% | 60.0% | 1.1% | 0.970 | 0.976 | -0.5 pt | 1 |

As served (min_relevance = 0.900): `low_relevance: true` on answerable 0.0%, unanswerable 5.0%, out_of_scope 100.0%.

## By question type

| group | n | MCP R@5 | MCP R@10 | MCP MRR@10 | MCP nDCG@10 | file R@5 | MCP k20 R@20 | trace R@10 | trace R@20 | trace MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| en_factual | 25 | 1.000 | 1.000 | 0.960 | 0.971 | 1.000 | 1.000 | 1.000 | 1.000 | 0.960 |
| keyword | 25 | 0.960 | 0.960 | 0.900 | 0.916 | 1.000 | 1.000 | 0.960 | 1.000 | 0.900 |
| multihop | 20 | 0.975 | 0.975 | 1.000 | 0.888 | 0.975 | 0.975 | 0.975 | 0.975 | 1.000 |
| paraphrase | 25 | 0.960 | 1.000 | 0.925 | 0.943 | 1.000 | 1.000 | 0.960 | 0.960 | 0.833 |
| table | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| xling_en2zh | 20 | 1.000 | 1.000 | 0.883 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.883 |
| xling_zh2en | 20 | 0.900 | 0.900 | 0.842 | 0.857 | 0.900 | 0.900 | 0.900 | 0.900 | 0.842 |
| zh_factual | 30 | 1.000 | 1.000 | 0.983 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |

## By question language

| group | n | MCP R@5 | MCP R@10 | MCP MRR@10 | MCP nDCG@10 | file R@5 | MCP k20 R@20 | trace R@10 | trace R@20 | trace MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 83 | 1.000 | 1.000 | 0.954 | 0.959 | 1.000 | 1.000 | 1.000 | 1.000 | 0.936 |
| zh | 102 | 0.956 | 0.966 | 0.926 | 0.921 | 0.976 | 0.976 | 0.956 | 0.966 | 0.918 |

## MCP response size and latency

- estimated tokens per call: mean 2672.0, p50 2573.7, p95 3845.0 (CJK 0.7 tokens/char, other 0.25/char, measured on the JSON text block)
- results per call: 7.38; result roles: {'expanded': 1299, 'parent': 213}
- duplicate-text ratio (10-char shingles repeated within a response): 0.0728
- latency (client wall time): MCP p50 329.4 ms, p95 376.4 ms; trace p50 411.6 ms, p95 511.8 ms
- by question language: en: 2618 tokens, dup 0.100, 318 ms; zh: 2717 tokens, dup 0.051, 345 ms

## Candidate stages (from the trace, lower bounds)

Share of answerable questions whose best gold hit in the final top-20 ranked within the top 10 / 20 of each candidate list. The trace reports stage ranks only for final hits, so these are lower bounds on each list's recall.

- dense_in_top10_lb: 0.919
- dense_in_top20_lb: 0.957
- bm25_in_top10_lb: 0.784
- bm25_in_top20_lb: 0.843
- fused_in_top10_lb: 0.854
- fused_in_top20_lb: 0.924
- bm25_only: 0.005
- dense_only: 0.081

## Run-to-run variance

Largest |run1 − run2| over the headline metrics: mcp.recall@10 0.0, mcp.mrr@10 0.0, mcp_k20.recall@10 0.0, mcp_k20.mrr@10 0.0, trace.recall@10 0.0, trace.mrr@10 0.0, mcp_response.tokens_mean 0.0, mcp_response.latency_ms_p50 0.0

Questions whose first-hit rank changed between runs: mcp 0, mcp_k20 0, trace 0

## Gold reachability

205/205 evidence items are found in the indexed leaf chunks (1 only across two adjacent chunks).


## Effective configuration

```json
{
 "rag.embedding.model": "intfloat/multilingual-e5-large",
 "rag.contextual_retrieval.enabled": true,
 "rag.rerank.enabled": true,
 "rag.rerank.model": "mixedbread-ai/mxbai-rerank-base-v2",
 "rag.rerank.base_url": "http://localhost:8787",
 "rag.rerank.api_key": "<redacted>",
 "rag.rerank.top_n": null,
 "rag.rerank.score_threshold": 0.25,
 "rag.rerank.query_template": "query: {text}",
 "rag.rerank.document_template": "document: {text}\nYou are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:",
 "rag.rerank.auto_template": true,
 "rag.rerank.gap_threshold": 0.5,
 "rag.rerank.api": "score",
 "rag.rerank.endpoint_path": null,
 "rag.rerank.timeout": 10.0,
 "rag.rerank.max_retries": 1,
 "rag.retrieval.default_top_k": 10,
 "rag.retrieval.default_similarity_cutoff": 0.25,
 "rag.retrieval.default_sparse_top_k": 12,
 "rag.retrieval.rerank_candidates": 60,
 "rag.retrieval.rerank_pool": "per_list",
 "rag.retrieval.rerank_dense_candidates": 30,
 "rag.retrieval.rerank_sparse_candidates": 10,
 "rag.retrieval.min_relevance": 0.9,
 "rag.retrieval.default_hybrid_alpha": 0.75,
 "rag.retrieval.hybrid_fusion": "rrf",
 "rag.retrieval.segmenter": null,
 "rag.retrieval.expand_context_default": true,
 "rag.retrieval.expand_context_neighbors": 1,
 "rag.retrieval.auto_merging.enabled": true,
 "rag.retrieval.auto_merging.merge_threshold": 0.4
}
```

Full snapshot (secrets redacted) in the JSON file.

## Search latency: before and after

Two scratch instances of Folio on the same host, the same index (a copy of the eval
folder per instance) and the same shared model servers: **A** = `dev` at `dbec758`, **B**
= this change. Harness runs were interleaved (A1, B1, A2, B2, then C1, F1, A3), one run
each; C and F are B with live settings changed. Latency is the harness client wall
time of the default MCP search (205 golden + 20 out-of-scope questions).

| run | variant | R@5 | R@10 | MRR@10 | nDCG@10 | zh→en R@5 | k20 R@20 | trace R@5 | MCP p50 | MCP p95 | trace p50 | trace p95 | low_relevance (ans / unans / oos) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | before | 0.976 | 0.981 | 0.938 | 0.938 | 0.900 | 0.987 | 0.968 | 502 | 580 | 413 | 507 | 0 / 0 / 1.0 |
| B1 | after (per_list 30/10) | 0.976 | 0.981 | 0.939 | 0.938 | 0.900 | 0.987 | 0.968 | 329 | 376 | 412 | 512 | 0 / 0.05 / 1.0 |
| A2 | before | 0.976 | 0.981 | 0.938 | 0.938 | 0.900 | 0.987 | 0.968 | 499 | 596 | 412 | 500 | 0 / 0 / 1.0 |
| B2 | after (per_list 30/10) | 0.976 | 0.981 | 0.939 | 0.938 | 0.900 | 0.987 | 0.968 | 335 | 431 | 410 | 531 | 0 / 0.05 / 1.0 |
| C1 | after, per_list 40/15 | 0.976 | 0.981 | 0.939 | 0.938 | 0.900 | 0.987 | 0.968 | 410 | 557 | 428 | 586 | 0 / 0 / 1.0 |
| F1 | after, `rerank_pool: fused` (old pool) | 0.976 | 0.981 | 0.938 | 0.938 | 0.900 | 0.987 | 0.968 | 459 | 534 | 389 | 478 | 0 / 0 / 1.0 |
| A3 | before | 0.976 | 0.981 | 0.938 | 0.938 | 0.900 | 0.987 | 0.968 | 503 | 649 | 416 | 573 | 0 / 0 / 1.0 |

- **Before (A, mean of 3): p50 501 ms, p95 608 ms. After (B, mean of 2): p50 332 ms,
  p95 404 ms** (−34% / −34%). Quality identical; MRR@10 +0.001 (one tie broken the
  other way). One near-miss unanswerable question is now flagged `low_relevance`: its
  best-scoring candidate was outside the smaller pool (answerable questions wrongly
  flagged: still 0).
- F1 isolates the changes that leave the pool alone (merge/expand prefetch, dedicated
  CKIP thread, background query log): ~−40 ms p50. The pool change is the rest (~−130 ms).
- The trace path (`top_k=20`) is unchanged: its dense side is 3 × 20 = 60 deep, so it
  scores ~65 pairs, about what the fused pool did.
- C1 (40/15, ~45 pairs) costs ~80 ms p50 more than 30/10 for no measured gain; its
  p95 is noisier than the rest.

### Per stage (server log, default MCP search over the 205 golden questions, probed back to back)

| stage | before p50 | before p95 | after p50 | after p95 |
|---|---|---|---|---|
| query embedding (HTTP, e5-large) | 57.7 | 65.7 | 55.1 | 66.0 |
| dense SQL (HNSW, 60 rows) | 18.2 | 29.5 | 20.0 | 29.8 |
| CKIP query segmentation (CPU) | 39.4 | 60.8 | 40.2 | 83.5 |
| BM25 SQL | 17.0 | 38.2 | 18.7 | 40.3 |
| fusion | 0.3 | 0.6 | 0.3 | 0.5 |
| **pre-rerank critical path** (embed → dense ‖ stamp → segment → BM25) | 82.8 | 100.7 | 82.3 | 118.2 |
| **rerank HTTP call** (60 → ~35 pairs) | **360.2** | **416.9** | **223.0** | **283.7** |
| merge lookups (parent counts + parents) | 7.9 | 13.5 | 0.3 | 0.4 |
| neighbour-window lookup | 11.6 | 18.7 | 0.3 | 0.4 |
| prefetch of both (runs during rerank, not on the path) | — | — | 25.2 | 38.5 |
| response building + token budget | 3.0 | 5.0 | 3.0 | 5.0 |
| handle_search total | 468.0 | 537.0 | 313.0 | 401.0 |
| folder ACL + query log + JSON block (tool − handle_search) | 12.0 | 16.0 | 8.0 | 11.0 |
| server tool total | 481.0 | 552.0 | 321.0 | 411.0 |
| transport + JWT verify + MCP framing (client − server) | 16.0 | 18.1 | 16.0 | 18.9 |
| **client wall** | **495.6** | **570.1** | **336.8** | **425.8** |

Dense and BM25 run concurrently, so the stage times do not add up; "critical path" is
the time from the start of the search to the start of the rerank call.

## Why the rerank pool, and why per list

**The reranker's cost is per pair, not per token.** Direct calls to the vLLM 0.17
`/v1/score` endpoint serving mxbai-rerank-base-v2 (60 documents of ~70 tokens vs ~400
tokens: 330–390 ms either way; 10 / 30 / 60 two-word documents: 50 / 120 / 275 ms) cost
~5–7 ms per pair. `/classify`, `/pooling` and `/v1/rerank` cost the same (~390 ms for 60
pairs). Splitting the pool into concurrent requests does not help (real pools, 40
questions × 2: one call of 60 → 427 ms p50; 2 × 30 → 416; 4 × 15 → 420; 6 × 10 → 415;
12 × 5 → 426), so the only lever on the client side is the number of pairs: 30 → 236 ms,
20 → 146 ms, 10 → 102 ms.

**Which pairs.** Offline, on the same index, every question's dense top 60, BM25 top 60
and fused top 60 were scored once with the reranker, and pool policies were simulated
(leaf-level, the same matching as the trace path; the 60-deep fused pool reproduces the
trace numbers exactly: R@5 0.968, R@10 0.976, MRR 0.926, zh→en 0.90):

| pool | pairs (mean) | R@5 | R@10 | MRR | zh→en R@5 | en→zh R@5 |
|---|---|---|---|---|---|---|
| fused top 20 | 20 | 0.905 | 0.914 | 0.877 | 0.55 | 0.80 |
| fused top 30 (old 3 × top_k) | 30 | 0.927 | 0.935 | 0.892 | 0.65 | 0.90 |
| fused top 40 | 40 | 0.957 | 0.965 | 0.915 | 0.80 | 1.00 |
| fused top 50 | 50 | 0.962 | 0.970 | 0.921 | 0.85 | 1.00 |
| **fused top 60 (old default)** | 60 | **0.968** | **0.976** | **0.926** | **0.90** | 1.00 |
| dense top 20 + BM25 top 5 | 21.4 | 0.949 | 0.957 | 0.914 | 0.75 | 1.00 |
| dense top 30 only | 30 | 0.962 | 0.970 | 0.921 | 0.85 | 1.00 |
| dense top 30 + BM25 top 5 | 31.1 | 0.968 | 0.976 | 0.926 | 0.90 | 1.00 |
| **dense top 30 + BM25 top 10 (new default)** | 33.1 (max 40) | **0.968** | **0.976** | **0.926** | **0.90** | 1.00 |
| dense top 40 + BM25 top 10 | 42.7 | 0.968 | 0.976 | 0.926 | 0.90 | 1.00 |

The gold passages the reranker lifts into the top 5 sit at dense rank ≤ 27 (one at 41,
also BM25 rank 15; three are BM25-only at BM25 rank 2, 4 and 24, each covered by another
hit as well). The 60-deep fused pool reached only dense rank 27–60 (median 41) and BM25
rank 27–60, because passages that both lists found take its first places; the per-list
pool always reaches dense rank 30. The default 30/10 has no recall margin left on this
set at dense rank 27 — raise `rerank_dense_candidates` if a corpus shows lower dense
ranks for cross-lingual gold (40/15 measured above: same recall, +80 ms).

## Rejected (with numbers)

- **Adaptive pool** (rerank the top N first, extend only if the best score is below τ):
  the reranker scores topically close passages ≥ 0.99, so the first batch looks
  confident while the cross-lingual gold sits below it. Simulated on the fused top 60
  (same leaf-level proxy, where the full pool scores R@5 0.968 / zh→en 0.90): top 20,
  then all 60 below τ = 0.999 → 25.4 pairs mean, R@5 0.938, zh→en 0.80; top 30 / 0.9995
  → 36.5 pairs, R@5 0.954, zh→en 0.85; top 40 / 0.999 → 42.1 pairs, R@5 0.962, zh→en
  0.85. All below the bar, and an escalated query pays for both calls.
- **Smaller fused pool** (40): R@5 0.957, zh→en 0.80. Below the bar.
- **Concurrent rerank batches**: no gain (above).
- **HTTP keep-alive / client reuse**: already in place (the reranker's `httpx.AsyncClient`
  and llama-index's OpenAI client are reused); the in-app embedding call (62.6 ms p50)
  costs the same as a bare `httpx` POST to the endpoint (60.2 ms).
- **Query-embedding LRU cache / segmentation cache**: no effect on unique queries, and
  on this harness it would only speed up the second run of each question; embedding
  (~55 ms) is the endpoint's own time and overlaps segmentation + BM25 anyway.
- **HNSW `ef_search`**: the folder has 606 rows; the dense SQL is ~18 ms including
  row decoding, with nothing measurable to gain from a lower `ef_search` (and a recall
  risk on larger folders).
- **Fewer torch threads for CKIP**: one thread is 5× slower (102 ms vs 19 ms p50 with 64
  threads on an idle-ish host); 8 → 31 ms, 16 → 24 ms. The dedicated thread fixed the
  contention without giving up speed (60 queries, CKIP alone: one thread reused → 21 ms
  p50; four threads in rotation → 101 ms).

## Not verified

- Concurrent load: all numbers are one query at a time. With up to
  `max_concurrent_queries` searches in flight, query segmentation now queues on its one
  thread (~40 ms each) instead of running in parallel teams that slowed each other down.
- The host is shared (load average 25–60 during the runs); run-to-run p50 spread was
  ±5 ms for A and ±3 ms for B, p95 spread up to ±35 ms.
