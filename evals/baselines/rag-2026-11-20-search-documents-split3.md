# Retrieval across folders — 2026-11-20 — `search-documents`

Folio at http://127.0.0.1:5139 (code `d2b2612`): the eval corpus split over 3 folders (`folio_eval_golden_v1_part1of3` 7 files, `folio_eval_golden_v1_part2of3` 6 files, `folio_eval_golden_v1_part3of3` 6 files), 205 questions (185 answerable), 1 run(s). Every question is asked once across all the folders through POST /api/search; each folder is searched with its own index, the candidates are pooled and reranked once. Harness v3.

Models: embedding `intfloat/multilingual-e5-large`, rerank `mixedbread-ai/mxbai-rerank-base-v2`, llm `openai/gpt-oss-120b`

## Passages across folders (top 20, no auto-merge; compare with the trace path of a one-folder baseline)

| R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | file R@5 | latency p50 | p95 |
|---|---|---|---|---|---|---|---|
| 0.968 | 0.976 | 0.987 | 0.920 | 0.923 | 0.984 | 1077.4 ms | 1268.0 ms |

Folders represented in a question's top 20: 1 folder(s): 10, 2 folder(s): 92, 3 folder(s): 103

| type | n | R@5 | R@10 | R@20 | MRR@10 |
|---|---|---|---|---|---|
| en_factual | 25 | 1.000 | 1.000 | 1.000 | 1.000 |
| keyword | 25 | 0.960 | 1.000 | 1.000 | 0.885 |
| multihop | 20 | 0.950 | 0.975 | 0.975 | 1.000 |
| paraphrase | 25 | 0.920 | 0.920 | 1.000 | 0.777 |
| table | 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| xling_en2zh | 20 | 1.000 | 1.000 | 1.000 | 0.867 |
| xling_zh2en | 20 | 0.900 | 0.900 | 0.900 | 0.842 |
| zh_factual | 30 | 1.000 | 1.000 | 1.000 | 0.983 |

## Documents across folders

POST /api/search with group_by=document, top 10 documents. A question's gold documents are the source files of its evidence; doc R@k = share of them among the top k documents, MRR@10 = 1 / rank of the first one. Documents per answer: 6.66; latency p50 639.6 ms, p95 853.7 ms.

| group | n | doc R@1 | doc R@3 | doc R@5 | doc R@10 | doc MRR@10 |
|---|---|---|---|---|---|---|
| all | — | 0.895 | 0.987 | 0.987 | 0.987 | 0.960 |
| en_factual | 25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| keyword | 25 | 0.840 | 1.000 | 1.000 | 1.000 | 0.920 |
| multihop | 20 | 0.675 | 0.975 | 0.975 | 0.975 | 1.000 |
| paraphrase | 25 | 0.880 | 1.000 | 1.000 | 1.000 | 0.940 |
| table | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| xling_en2zh | 20 | 0.850 | 1.000 | 1.000 | 1.000 | 0.925 |
| xling_zh2en | 20 | 0.900 | 0.900 | 0.900 | 0.900 | 0.900 |
| zh_factual | 30 | 0.967 | 1.000 | 1.000 | 1.000 | 0.983 |
