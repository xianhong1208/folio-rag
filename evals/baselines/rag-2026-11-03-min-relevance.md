# Retrieval baseline — 2026-11-03 — `min-relevance`

Folio at http://127.0.0.1:5139 (code `e10fa5f`), folder `folio_eval_golden_v1` (id 4), 19 documents, 205 questions (185 answerable, 20 unanswerable) + 20 out-of-scope, 2 runs. Harness v2; scores below are the mean of the runs.

Models: embedding `intfloat/multilingual-e5-large`, rerank `mixedbread-ai/mxbai-rerank-base-v2`, llm `openai/gpt-oss-120b`; environment: pgvector 0.8.7

## Overall

| path | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | file R@5 | FA@0.25 | FA@0.5 |
|---|---|---|---|---|---|---|---|---|
| MCP search (defaults) | 0.976 | 0.981 | — | 0.938 | 0.938 | 0.987 | 1.000 | 1.000 |
| MCP search, top_k=20 | 0.976 | 0.987 | 0.987 | 0.939 | 0.939 | 0.987 | 1.000 | 1.000 |
| Trace: fused+rerank leaves, k=20 | 0.968 | 0.976 | 0.981 | 0.926 | 0.927 | 0.984 | 1.000 | 1.000 |

FA@t = share of unanswerable questions whose top result scores ≥ t (answerable questions clearing the same bar: 0.25 → 1.000, 0.5 → 1.000 on the MCP path).

## Relevance cutoff (`rag.retrieval.min_relevance`)

What each cutoff would report as "nothing relevant" on the default MCP search: a search is rejected when it returned nothing or its best rerank score (`top_relevance`, before `score_threshold`) is below the cutoff. Recall columns score a rejected answerable question as 0 (as if the results were dropped); lost@5 = answerable questions with a top-5 hit that would be rejected. out_of_scope: n=20, unanswerable: n=20.

| min_relevance | out_of_scope rejected | unanswerable rejected | all unanswerable rejected | answerable wrongly rejected | R@5 | R@10 | ΔR@5 | lost@5 |
|---|---|---|---|---|---|---|---|---|
| 0 | 45.0% | 0.0% | 22.5% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.25 | 45.0% | 0.0% | 22.5% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.5 | 87.5% | 0.0% | 43.8% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.75 | 95.0% | 0.0% | 47.5% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.8 | 100.0% | 0.0% | 50.0% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.85 | 100.0% | 0.0% | 50.0% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.9 | 100.0% | 0.0% | 50.0% | 0.0% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.95 | 100.0% | 5.0% | 52.5% | 0.5% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.98 | 100.0% | 5.0% | 52.5% | 0.5% | 0.976 | 0.981 | +0.0 pt | 0 |
| 0.99 | 100.0% | 15.0% | 57.5% | 1.1% | 0.970 | 0.976 | -0.5 pt | 1 |

As served (min_relevance = 0.900): `low_relevance: true` on answerable 0.0%, unanswerable 0.0%, out_of_scope 100.0%.

## Choosing the signal (offline analysis, same index)

Per question, all 60 fused candidates were rescored with the reranker's raw logits
(vLLM 0.17 `/v1/score` with `use_activation: false`; the default response is
`sigmoid(logit)`, so the probability Folio already receives carries the same ranking,
just compressed near 1). AUROC of each candidate signal for telling unanswerable from
answerable questions (185 answerable; 20 near misses from the golden set; 20
out-of-scope questions):

| signal | near miss | out of scope |
|---|---|---|
| best rerank score (logit / probability) | **0.792** | **1.000** |
| mean of the top-3 rerank scores | 0.679 | 1.000 |
| rerank gap top-1 − top-5 | 0.721 | 0.716 |
| dense cosine of the dense top hit | 0.564 | 0.978 |
| dense cosine of the rerank top hit | 0.652 | 0.970 |
| BM25 ts_rank of the BM25 top hit | 0.637 | 0.930 |
| query terms found in the rerank top hit | 0.713 | 0.942 |
| dense ∩ BM25 overlap in the top 10 | 0.477 | 0.914 |
| dense rank of the rerank top hit (lower = answerable) | 0.774 | 0.862 |

The best rerank score is the best single signal for both kinds, and the only one that
separates out-of-scope questions completely: their best logit is at most 1.10 (p 0.75),
the lowest answerable question scores 2.91 (p 0.948, and its gold passage is not among
the candidates at all), the lowest answerable question with a top-5 hit 3.95 (p 0.981).
Near misses are not separable: their best logits run 2.86–7.1 (p 0.946–0.9992),
inside the answerable range, because the reranker judges topical relevance, not
answerability (e.g. *paid sick leave for part-time workers under the Labor Standards
Act* matches the Act's leave articles). Combining signals did not change that at any
operating point within the recall budget, so Folio ships the one explainable signal.
`top_relevance` is the probability (same scale as `rag.rerank.score_threshold`).

**Default 0.9**: rejects 100% of out-of-scope questions and wrongly rejects 0% of
answerable ones (R@5 unchanged), with margin on both sides (logit 2.20 vs 1.10 for the
highest out-of-scope question and 2.91 for the lowest answerable). 0.95 would also catch
one near miss (q203) but sits on top of the answerable tail (0.948); 0.99 costs 0.5 pt
R@5. Results are flagged, not dropped.

Before this change no flag existed: every one of these 40 questions came back as
ordinary results (the 0.25 / 0.5 false-accept rates above are unchanged at 1.000 for
near misses; 55% of out-of-scope questions returned results above `score_threshold`).

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

- estimated tokens per call: mean 2693.85, p50 2613.9, p95 3939.15 (CJK 0.7 tokens/char, other 0.25/char, measured on the JSON text block)
- results per call: 7.39; result roles: {'parent': 212, 'expanded': 1302}
- duplicate-text ratio (10-char shingles repeated within a response): 0.0745
- latency (client wall time): MCP p50 585.0 ms, p95 657.35 ms; trace p50 462.1 ms, p95 570.65 ms
- by question language: en: 2667 tokens, dup 0.101, 573 ms; zh: 2716 tokens, dup 0.053, 584 ms

## Candidate stages (from the trace, lower bounds)

Share of answerable questions whose best gold hit in the final top-20 ranked within the top 10 / 20 of each candidate list. The trace reports stage ranks only for final hits, so these are lower bounds on each list's recall.

- dense_in_top10_lb: 0.919
- dense_in_top20_lb: 0.957
- bm25_in_top10_lb: 0.784
- bm25_in_top20_lb: 0.838
- fused_in_top10_lb: 0.854
- fused_in_top20_lb: 0.924
- bm25_only: 0.005
- dense_only: 0.081

## Run-to-run variance

Largest |run1 − run2| over the headline metrics: mcp.recall@10 0.0, mcp.mrr@10 0.0, mcp_k20.recall@10 0.0, mcp_k20.mrr@10 0.0, trace.recall@10 0.0, trace.mrr@10 0.0, mcp_response.tokens_mean 0.5, mcp_response.latency_ms_p50 26.2

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
