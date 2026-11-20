# Retrieval baseline — 2026-11-20 — `search-documents`

Folio at http://127.0.0.1:5139 (code `d2b2612`), folder `folio_eval_golden_v1` (id 1), 19 documents, 205 questions (185 answerable, 20 unanswerable) + 20 out-of-scope, 2 runs. Harness v3; scores below are the mean of the runs.

Models: embedding `intfloat/multilingual-e5-large`, rerank `mixedbread-ai/mxbai-rerank-base-v2`, llm `openai/gpt-oss-120b`; environment: pgvector 0.8.x, postgres 17

## Overall

| path | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | file R@5 | FA@0.25 | FA@0.5 |
|---|---|---|---|---|---|---|---|---|
| MCP search (defaults) | 0.970 | 0.981 | — | 0.924 | 0.927 | 0.987 | 1.000 | 0.950 |
| MCP search, top_k=20 | 0.973 | 0.987 | 0.987 | 0.926 | 0.929 | 0.984 | 1.000 | 1.000 |
| Trace: fused+rerank leaves, k=20 | 0.968 | 0.973 | 0.987 | 0.905 | 0.911 | 0.984 | 1.000 | 1.000 |

FA@t = share of unanswerable questions whose top result scores ≥ t (answerable questions clearing the same bar: 0.25 → 1.000, 0.5 → 1.000 on the MCP path).

## Relevance cutoff (`rag.retrieval.min_relevance`)

What each cutoff would report as "nothing relevant" on the default MCP search: a search is rejected when it returned nothing or its best rerank score (`top_relevance`, before `score_threshold`) is below the cutoff. Recall columns score a rejected answerable question as 0 (as if the results were dropped); lost@5 = answerable questions with a top-5 hit that would be rejected. out_of_scope: n=20, unanswerable: n=20.

| min_relevance | out_of_scope rejected | unanswerable rejected | all unanswerable rejected | answerable wrongly rejected | R@5 | R@10 | ΔR@5 | lost@5 |
|---|---|---|---|---|---|---|---|---|
| 0 | 60.0% | 0.0% | 30.0% | 0.0% | 0.970 | 0.981 | +0.0 pt | 0 |
| 0.25 | 60.0% | 0.0% | 30.0% | 0.0% | 0.970 | 0.981 | +0.0 pt | 0 |
| 0.5 | 90.0% | 5.0% | 47.5% | 0.0% | 0.970 | 0.981 | +0.0 pt | 0 |
| 0.75 | 100.0% | 5.0% | 52.5% | 0.0% | 0.970 | 0.981 | +0.0 pt | 0 |
| 0.8 | 100.0% | 5.0% | 52.5% | 0.0% | 0.970 | 0.981 | +0.0 pt | 0 |
| 0.85 | 100.0% | 5.0% | 52.5% | 0.0% | 0.970 | 0.981 | +0.0 pt | 0 |
| 0.9 | 100.0% | 5.0% | 52.5% | 0.0% | 0.970 | 0.981 | +0.0 pt | 0 |
| 0.95 | 100.0% | 10.0% | 55.0% | 0.5% | 0.970 | 0.981 | +0.0 pt | 0 |
| 0.98 | 100.0% | 10.0% | 55.0% | 0.5% | 0.970 | 0.981 | +0.0 pt | 0 |
| 0.99 | 100.0% | 20.0% | 60.0% | 0.5% | 0.970 | 0.981 | +0.0 pt | 0 |

As served (min_relevance = 0.900): `low_relevance: true` on answerable 0.0%, unanswerable 5.0%, out_of_scope 100.0%.

## Results by document

POST /api/search with group_by=document, top 10 documents. A question's gold documents are the source files of its evidence; doc R@k = share of them among the top k documents, MRR@10 = 1 / rank of the first one. Documents per answer: 4.79; latency p50 360.3 ms, p95 435.85 ms.

| group | n | doc R@1 | doc R@3 | doc R@5 | doc R@10 | doc MRR@10 |
|---|---|---|---|---|---|---|
| all | — | 0.905 | 0.987 | 0.987 | 0.987 | 0.965 |
| en_factual | 25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| keyword | 25 | 0.800 | 1.000 | 1.000 | 1.000 | 0.900 |
| multihop | 20 | 0.675 | 0.975 | 0.975 | 0.975 | 1.000 |
| paraphrase | 25 | 0.960 | 1.000 | 1.000 | 1.000 | 0.980 |
| table | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| xling_en2zh | 20 | 0.850 | 1.000 | 1.000 | 1.000 | 0.925 |
| xling_zh2en | 20 | 0.900 | 0.900 | 0.900 | 0.900 | 0.900 |
| zh_factual | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## By question type

| group | n | MCP R@5 | MCP R@10 | MCP MRR@10 | MCP nDCG@10 | file R@5 | MCP k20 R@20 | trace R@10 | trace R@20 | trace MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| en_factual | 25 | 1.000 | 1.000 | 0.953 | 0.965 | 1.000 | 1.000 | 1.000 | 1.000 | 0.930 |
| keyword | 25 | 0.960 | 1.000 | 0.865 | 0.899 | 1.000 | 1.000 | 1.000 | 1.000 | 0.864 |
| multihop | 20 | 0.975 | 0.975 | 1.000 | 0.888 | 0.975 | 0.975 | 0.950 | 0.975 | 1.000 |
| paraphrase | 25 | 0.920 | 0.960 | 0.873 | 0.893 | 1.000 | 1.000 | 0.920 | 1.000 | 0.798 |
| table | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| xling_en2zh | 20 | 1.000 | 1.000 | 0.842 | 0.883 | 1.000 | 1.000 | 1.000 | 1.000 | 0.842 |
| xling_zh2en | 20 | 0.900 | 0.900 | 0.842 | 0.857 | 0.900 | 0.900 | 0.900 | 0.900 | 0.842 |
| zh_factual | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |

## By question language

| group | n | MCP R@5 | MCP R@10 | MCP MRR@10 | MCP nDCG@10 | file R@5 | MCP k20 R@20 | trace R@10 | trace R@20 | trace MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 83 | 0.988 | 0.988 | 0.924 | 0.935 | 1.000 | 1.000 | 0.988 | 1.000 | 0.902 |
| zh | 102 | 0.956 | 0.976 | 0.924 | 0.921 | 0.976 | 0.976 | 0.961 | 0.976 | 0.908 |

## MCP response size and latency

- estimated tokens per call: mean 2663.35, p50 2540.95, p95 3843.65 (CJK 0.7 tokens/char, other 0.25/char, measured on the JSON text block)
- results per call: 6.99; result roles: {'expanded': 1209.5, 'parent': 223}
- duplicate-text ratio (10-char shingles repeated within a response): 0.0682
- latency (client wall time): MCP p50 381.5 ms, p95 436.0 ms; trace p50 458.25 ms, p95 550.1 ms
- by question language: en: 2591 tokens, dup 0.091, 365 ms; zh: 2723 tokens, dup 0.050, 430 ms

## Candidate stages (from the trace, lower bounds)

Share of answerable questions whose best gold hit in the final top-20 ranked within the top 10 / 20 of each candidate list. The trace reports stage ranks only for final hits, so these are lower bounds on each list's recall.

- dense_in_top10_lb: 0.924
- dense_in_top20_lb: 0.968
- bm25_in_top10_lb: 0.800
- bm25_in_top20_lb: 0.860
- fused_in_top10_lb: 0.854
- fused_in_top20_lb: 0.919
- bm25_only: 0.005
- dense_only: 0.081

## Run-to-run variance

Largest |run1 − run2| over the headline metrics: mcp.recall@10 0.0, mcp.mrr@10 0.0, mcp_k20.recall@10 0.0, mcp_k20.mrr@10 0.0, trace.recall@10 0.0, trace.mrr@10 0.0001, mcp_response.tokens_mean 1.5, mcp_response.latency_ms_p50 1.4

Questions whose first-hit rank changed between runs: mcp 0, mcp_k20 0, trace 1

## Gold reachability

205/205 evidence items are found in the indexed leaf chunks (5 only across two adjacent chunks).


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
 "rag.retrieval.max_search_folders": 20,
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
