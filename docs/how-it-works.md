# How Folio works

[繁體中文](how-it-works.zh-TW.md)

```
┌────────────────┐     Folio_<folder> tool call (Bearer JWT)    ┌───────────────────────────────┐
│   MCP client   │ ───────────────────────────────────────────▶ │  Folio (resource server)       │
│  Claude / IDE  │ ◀─────────────────────────────────────────── │  /mcp · per-user folder tools  │
└───────┬────────┘        auto-merged passages + scores         └───────────────┬───────────────┘
        │ OAuth flow                                                             │ verify JWT offline
        ▼                                                                        ▼
  ┌─────────────┐                              retrieval pipeline (per query)
  │  MCP Gate │  RS256 JWKS                  ingest ─▶ chunk ─▶ embed ─▶ pgvector ─▶ hybrid retrieve
  │ OAuth 2.1 AS│  (offline verify)            (Docling) (leaf+parent) (e5·1024d)   (vector + BM25/CKIP)
  └─────────────┘                                   └▶ auto-merge + expand ─▶ rerank ─▶ gpt-oss-120b
```

Your folder is indexed once, then every query runs through the pipeline:

| Stage | What happens |
|-------|--------------|
| **Ingest** | Docling parses PDF / DOCX / PPTX / XLSX / images (OCR); audio is transcribed by FireRedASR. |
| **Chunk** | Section-aligned splitting into small **leaves** + large **parents**; each leaf gets a breadcrumb (file › headings) and an LLM context line. Tables stay whole when they fit. |
| **Embed** | Each leaf's retrieval text (context line + breadcrumb + original text) is embedded (default `intfloat/multilingual-e5-large`, 1024-d) and written to **pgvector** together with its BM25 `text_search_tsv` (Chinese CKIP-segmented), in one transaction per file. |
| **Retrieve** | **Hybrid**: dense vector + sparse BM25, fused with Reciprocal Rank Fusion. |
| **Merge** | Leaves from the same passage **auto-merge** into their parent; single hits **expand** to neighbours. |
| **Rerank** | A cross-encoder (`bge-reranker-v2-m3` / `mxbai-rerank-base-v2`) re-scores for final Top-K precision. |
| **Answer** | For the `answer` path, `gpt-oss-120b` synthesises a response from the retrieved context, gated by a corrective-RAG confidence check. |

## Retrieval in detail

**One retrieval path.** REST `/api/rag/query`, the MCP search tool, the console Search Playground (and its trace view) and the evaluation harness all retrieve through the same code (`src/domain/rag/hybrid_search.py`): dense + BM25, fused with `rrf` (default) or `concat` (dense hits, then BM25-only hits, left for the reranker to order), reranked, then auto-merged with one shared rule. REST accepts `auto_merge` per request (`null` = follow `rag.retrieval.auto_merging.enabled`); `hybrid_alpha` is accepted and ignored. Per-request settings are passed with each call, never stored on the shared per-folder engine, so concurrent requests with different `top_k` cannot affect each other.

**Keyword search (BM25).** The query is tokenised the way the index was (segmented for Chinese), question and function words (`什麼`, `嗎`, `what`, `the`, …) are dropped, and Postgres full-text search runs over **leaf chunks only** — first requiring every term (AND), then filling up with any term (OR). Its depth is `max(top_k candidates, rag.retrieval.default_sparse_top_k)`, and it runs concurrently with the vector search. The `simple` text-search config does no English stemming (`policy` ≠ `policies`); set `text_search_config: english` for an English-only corpus.

**Chinese segmentation.** With the default `text_search_config: simple`, Chinese is segmented in Python before it reaches Postgres, by the segmenter in `rag.retrieval.segmenter` — the same function for indexing, *Rebuild FTS* and queries:

| `segmenter` | Notes |
|---|---|
| `ckip` *(default)* | Academia Sinica CKIP BERT model — most accurate on Traditional Chinese; needs torch (`uv sync --extra local`) and a model download on first use. Not allowed under `profile: cloud`. |
| `jieba` | Dictionary-based, pure Python, no GPU. Optional extra: `uv sync --extra jieba`. |
| `bigram` | Overlapping CJK character pairs + Latin/digit words. No model, no dependency; bigger index. The default under `profile: cloud`. |

Left unset, `segmenter` follows the profile: `ckip`, or `bigram` under `profile: cloud`.

Each folder's table records which segmenter built its keyword index. After changing `segmenter` (live, from Settings), the console marks folders whose index was built with another one (**FTS rebuild needed**); *Rebuild FTS* re-segments them with the configured segmenter — no re-embedding.

**Reranking.** The reranker scores a candidate pool (below) and keeps the best `top_k` above `rag.rerank.score_threshold`. `mxbai-rerank-*-v2` is a Qwen model trained on a chat-formatted prompt; with `rag.rerank.auto_template: true` (default) that format is applied automatically when the configured templates are the default or the old chat-less form (turn it off if your rerank server already applies a chat template). With it, scores are calibrated probabilities and **`score_threshold: 0.5`** is meaningful for that model; without it unrelated chunks score 0.8+ and no threshold filters anything. If the reranker fails, the query returns the top `top_k` candidates in retrieval order (not the whole pool), and that degraded result is not cached.

**Candidate pool.** A deep pool matters most for cross-lingual questions: a Chinese question about an English passage often ranks that passage low in both dense and BM25 search, and only the cross-encoder lifts it. Those hits are found by the dense search alone (around dense rank 15–30), and RRF ranks every passage both searches found above them, so a cut of the *fused* list has to go 60 deep to reach them. With `rag.retrieval.rerank_pool: per_list` (default) the reranker instead scores the dense top `rerank_dense_candidates` (default 30, at least 3 × `top_k`) plus the BM25 top `rerank_sparse_candidates` (default 10) — about 35 pairs. On the golden set that keeps Recall@5 0.976 and zh→en Recall@5 0.90 exactly where the 60-deep fused pool had them (from 0.957 / 0.80 with a 3 × `top_k` pool), and since a cross-encoder call costs about 6 ms per pair whatever the text length, the rerank step dropped from ~360 ms to ~225 ms. `rerank_pool: fused` restores the previous behaviour (`max(3 × top_k, rerank_candidates)` fused candidates). All four settings are live from Settings; deeper pools buy recall margin for latency. Without a reranker the pool is just `top_k`.

**Latency.** A default MCP search on the golden set takes ~330 ms p50 / ~400 ms p95 at the client (was ~500 / ~600 ms; `evals/baselines/rag-2026-11-04-search-latency.md` has the per-stage table). Most of it is the rerank call (~225 ms for ~35 pairs). Before it, the query embedding (~55 ms, then the dense query) runs concurrently with segmentation and BM25 (~40 + ~20 ms), so that part costs ~80 ms. The auto-merge and neighbour-window lookups for every candidate are read while the reranker runs, so they add nothing after it; the query-log row is written after the response is sent. CKIP segments queries on one dedicated thread: on the CPU each thread that calls torch gets its own pool of worker threads, and several of them fighting over the cores made segmentation 3× slower (seconds on a busy host).

**Response size.** An MCP search response is fitted to `rag.retrieval.mcp_response_token_budget` (estimated tokens, default 6000): the lowest-ranked results are left out, the response reports `omitted_results` and `response_tokens_estimated`, and the `_hint` tells the agent to ask a narrower query. `0` turns the budget off.

**Saying "nothing relevant found".** Every MCP search reports how well the folder matched: `top_relevance` (the reranker's best score over all candidates, before `score_threshold`), `min_relevance` (the configured cutoff) and `low_relevance` — `true` when nothing came back or `top_relevance` is below `rag.retrieval.min_relevance`; the `_hint` then tells the agent the folder most likely does not answer the question. Each reranked result carries `relevance`, and results below the cutoff are marked `low_relevance: true`. Results are **flagged, never dropped** (dropping is `rag.rerank.score_threshold`'s job): on the golden set the cutoff rejects every out-of-scope question, but questions that are near misses of the corpus (the right law, a detail it does not state) score as high as answerable ones, so the final call stays with the agent. The shipped `0.9` is calibrated for `mxbai-rerank-v2` (sweep in `evals/baselines/rag-2026-11-03-min-relevance.md`); `0` turns the check off, and without a reranker only an empty result is flagged.

**Knowledge gaps.** Every REST, console and MCP search is logged (masked token prefix only) with its result count and the reranker's best score. *Query Analytics* lists a question as a knowledge gap when its **latest** search in a folder returned nothing or its best score was below `rag.rerank.gap_threshold` (default `0.5`) — so a gap disappears once documents that answer it are added and the question is asked again.

**Answers (console).** With *generate answer* on, the top chunks are graded for relevance in one LLM call, irrelevant ones are dropped (nothing relevant → no answer), and the answer is generated from the rest in the **question's language**. Each chunk keeps its result-card number, so `[3]` always means card #3 (citations are links in the console); citations of chunks the model was not given are removed, `【n†…】` markers become `[n]` and Markdown bold markers are stripped — on the streaming path too, where the final `done` frame carries the cleaned answer. If the grader's reply cannot be parsed the answer is labelled **unverified** instead of `high`.

**Auth.** Folio is a *resource server*. FastMCP publishes `/.well-known/oauth-protected-resource/mcp`, which points clients at MCP Gate; the server fetches MCP Gate's JWKS once and verifies every bearer token offline (RS256 — signature, issuer, audience, expiry, scopes). MCP Gate is never on the request path.

## Measuring retrieval

Retrieval and chunking changes are judged by numbers, not by eye. `evals/` holds a public golden set — 205 questions over a 19-document corpus (Taiwanese laws, U.S. government pages, fictional company documents, PDFs with tables, synthetic zh-TW meeting transcripts), stratified by question type and language, with ~10% unanswerable — plus 20 out-of-scope questions for the relevance cutoff. The harness scores a running server through the same MCP search an agent uses:

```bash
.venv/bin/python scripts/eval_retrieval.py --token-file <pat.json> --runs 2
```

It reports Recall@5/10/20, MRR, nDCG, response tokens, duplicate ratio and latency per question type, the false-accept rate on unanswerable questions and a `min_relevance` sweep, and writes `evals/baselines/rag-<date>-<label>.{json,md}`. Committed baselines record each change's before and after. See [`evals/README.md`](../evals/README.md) for the token file, how the corpus is built and what each metric means.

## Features in detail

- **Section-aligned hierarchical chunking + auto-merging retrieval.** Documents are split into small **leaves** (precise matching) and large **parents** (full context) at index time, and neither ever spans two sections. Each leaf carries its **breadcrumb** (file › heading path, xlsx sheet names included), so two documents with the same structure — two product manuals, two price sheets — stay distinguishable. When several leaves from the same passage are hit, the whole parent is returned instead of fragments; a single-leaf hit expands to its neighbours so the answer never loses narrative continuity.
- **Hybrid search, tuned for Chinese.** Dense vector retrieval (pgvector) is fused with sparse **BM25** using Reciprocal Rank Fusion, so keyword signal genuinely affects ranking. Chinese text is word-segmented before Postgres full-text indexing — with **CKIP** (Academia Sinica) by default, or jieba / a dependency-free bigram tokenizer — so space-less Chinese actually gets tokenized instead of collapsing into one term.
- **Cross-encoder reranking.** A `bge-reranker-v2-m3` / `mxbai-rerank-base-v2` model re-scores the fused candidates for fine-grained Top-K precision.
- **Contextual retrieval.** At index time an LLM writes a short context line for each chunk (Anthropic's recommended technique) from the document outline and the chunk's own section, in the document's language. The context and breadcrumb are what the chunk is *found* by (embedding, BM25, rerank); results always show the original document text.
- **Incremental re-indexing.** Re-indexing a file reuses the context line and embedding of every unchanged chunk, so editing one section costs about one LLM call and one embedding. The file's chunks are swapped in a single transaction — it never disappears from search mid-way.
- **Per-folder MCP tools, scoped per user.** Every folder is registered as a `Folio_<folder>` tool; the tool list a caller sees is filtered by who their token belongs to — you only see folders you own. Ownership is keyed on the token's `sub` (the MCP Gate user), so every token of one user — personal access tokens and OAuth sign-ins alike — sees the same folders (see [Folder ownership](auth.md#folder-ownership)). Tool names stay within what every MCP client accepts (`[A-Za-z0-9_-]`, 64 chars): a folder name with spaces, CJK or other symbols becomes `Folio_<ascii part>_<8-char hash>`, the same name every time.
- **OAuth 2.1 resource server.** One `RemoteAuthProvider` + `JWTVerifier` protects `/mcp`; tokens are verified locally against MCP Gate's JWKS (RS256, offline), and are bound to this server's audience so a token for another server never works here.
- **Document + audio ingestion.** [Docling](https://github.com/DS4SD/docling) parses PDF, Office (including legacy `.doc` / `.xls` / `.ppt`, RTF and Excel templates), OpenDocument, HTML and MHTML web archives, Markdown, EPUB, email and images (with OCR); legacy Office and RTF go through LibreOffice, so `soffice` must be installed. **FireRedASR** transcribes audio locally in Traditional Chinese (or any OpenAI-compatible `/v1/transcriptions` endpoint).
- **Built-in evaluation harness.** Auto-generates questions from your own chunks and scores the production pipeline (the same call as `/api/rag/query`) against `vector` / `bm25` / `rrf` / `rerank` — all through the production search path — on **Recall@k · MRR · nDCG@k**, so you can tune retrieval against your real corpus.
- **Retrieval Terminal console.** A React single-page console at `/admin`: pipeline overview, folders & files, a **Search Playground** that shows the per-signal retrieval trace, live index-job stream, query analytics, the evaluation harness, live settings, health and an audit timeline.
- **Runs on your GPU.** PyTorch is selected per platform with PEP 735 dependency groups (`rocm-r714` / `cuda` / `cpu`) — AMD ROCm, NVIDIA CUDA or pure CPU.
