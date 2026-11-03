# Retrieval baseline — 2026-11-03 — `phase1-integration-rerank60`

Folio at http://127.0.0.1:5039 (code `ba8718f`), folder `folio_eval_golden_v1` (id 4), 19 documents, 205 questions (185 answerable, 20 unanswerable), 2 runs. Harness v1; scores below are the mean of the runs.

Models: embedding `intfloat/multilingual-e5-large`, rerank `mixedbread-ai/mxbai-rerank-base-v2`, llm `openai/gpt-oss-120b`; environment: pgvector 0.8.7

## Overall

| path | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | file R@5 | FA@0.25 | FA@0.5 |
|---|---|---|---|---|---|---|---|---|
| MCP search (defaults) | 0.976 | 0.981 | — | 0.938 | 0.938 | 0.987 | 1.000 | 1.000 |
| MCP search, top_k=20 | 0.976 | 0.987 | 0.987 | 0.939 | 0.939 | 0.987 | 1.000 | 1.000 |
| Trace: fused+rerank leaves, k=20 | 0.968 | 0.976 | 0.981 | 0.926 | 0.927 | 0.984 | 1.000 | 1.000 |

FA@t = share of unanswerable questions whose top result scores ≥ t (answerable questions clearing the same bar: 0.25 → 1.000, 0.5 → 1.000 on the MCP path).

## By question type

| group | n | MCP R@5 | MCP R@10 | MCP MRR@10 | MCP nDCG@10 | file R@5 | MCP k20 R@20 | trace R@10 | trace R@20 | trace MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| en_factual | 25 | 1.000 | 1.000 | 0.960 | 0.971 | 1.000 | 1.000 | 1.000 | 1.000 | 0.960 |
| keyword | 25 | 0.960 | 0.960 | 0.900 | 0.916 | 1.000 | 1.000 | 0.960 | 1.000 | 0.900 |
| multihop | 20 | 0.975 | 0.975 | 1.000 | 0.884 | 0.975 | 0.975 | 0.975 | 0.975 | 1.000 |
| paraphrase | 25 | 0.960 | 1.000 | 0.924 | 0.943 | 1.000 | 1.000 | 0.960 | 0.960 | 0.833 |
| table | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| xling_en2zh | 20 | 1.000 | 1.000 | 0.883 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.883 |
| xling_zh2en | 20 | 0.900 | 0.900 | 0.842 | 0.857 | 0.900 | 0.900 | 0.900 | 0.900 | 0.842 |
| zh_factual | 30 | 1.000 | 1.000 | 0.983 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |

## By question language

| group | n | MCP R@5 | MCP R@10 | MCP MRR@10 | MCP nDCG@10 | file R@5 | MCP k20 R@20 | trace R@10 | trace R@20 | trace MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 83 | 1.000 | 1.000 | 0.954 | 0.959 | 1.000 | 1.000 | 1.000 | 1.000 | 0.936 |
| zh | 102 | 0.956 | 0.966 | 0.926 | 0.920 | 0.976 | 0.976 | 0.956 | 0.966 | 0.918 |

## MCP response size and latency

- estimated tokens per call: mean 2632.0, p50 2546.3, p95 3875.1 (CJK 0.7 tokens/char, other 0.25/char, measured on the JSON text block)
- results per call: 7.395; result roles: {'parent': 210.5, 'expanded': 1305.0}
- duplicate-text ratio (10-char shingles repeated within a response): 0.0745
- latency (client wall time): MCP p50 684.9 ms, p95 770.1 ms; trace p50 580.0 ms, p95 676.4 ms
- by question language: en: 2600 tokens, dup 0.101, 686 ms; zh: 2658 tokens, dup 0.053, 689 ms

## Candidate stages (from the trace, lower bounds)

Share of answerable questions whose best gold hit in the final top-20 ranked within the top 10 / 20 of each candidate list. The trace reports stage ranks only for final hits, so these are lower bounds on each list's recall.

- dense_in_top10_lb: 0.919
- dense_in_top20_lb: 0.957
- bm25_in_top10_lb: 0.789
- bm25_in_top20_lb: 0.849
- fused_in_top10_lb: 0.860
- fused_in_top20_lb: 0.924
- bm25_only: 0.005
- dense_only: 0.081

## Run-to-run variance

Largest |run1 − run2| over the headline metrics: mcp.recall@10 0.0, mcp.mrr@10 0.0, mcp_k20.recall@10 0.0, mcp_k20.mrr@10 0.0, trace.recall@10 0.0, trace.mrr@10 0.0, mcp_response.tokens_mean 2.0, mcp_response.latency_ms_p50 1.0

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
