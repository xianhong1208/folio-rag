# Roadmap

Priorities run top to bottom: **Now** and **Next** are the focus; **Later** is
tracked but not started. Items are grounded in the current codebase and its known gaps.

## 🔨 Now — short term

- **Console i18n.** *Dropped by decision* — the console stays English-only. (A
  full English + 繁體中文 implementation with a sidebar toggle was built and then
  removed; it lives in git history at `e4f0c8d` / `d7b50be` if ever wanted.)
- **Rebuild-FTS action.** *Shipped.* One-click `POST …/rebuild-fts` re-segments
  stored chunks with CKIP and rewrites `text_search_tsv` in place (no
  re-embedding), backfilling Chinese BM25 recall for folders indexed before the
  CKIP upgrade. Runs as a background job with progress, like indexing (1.2.0).
- **Retire the legacy console.** *Shipped.* The root URL redirects to the React
  console, and the single-file `admin_console.html`, its loader and the
  `/admin-classic` route have been deleted (recoverable from git history).
- **Frontend tests / CI.** *Shipped.* Vitest pure-function suites (format +
  settings-config) and a GitHub Actions workflow that builds and tests the SPA.

## 🚀 Next — mid term

- **Streaming answers.** *Shipped.* SSE token streaming for the Search Playground
  `answer` path — retrieval + CRAG grading stay non-stream; only the final
  generation streams (blocking OpenAI stream bridged to the loop via a threadpool
  producer + asyncio queue).
- **Revocation.** *Shipped (MCP Gate revocation feed, off by default).* With
  `auth.revocation` enabled, a background poller mirrors MCP Gate's
  `POST /oauth/revoked` (revoked, unexpired `jti`s) and every verify path rejects
  them — the FastMCP `/mcp` verifier, the REST / tool-filter
  `McpGateTokenVerifier`, console sign-in and every console-session request — so a
  revocation takes effect within one poll interval (30 s by default). Fail-open by
  default; `fail_mode: closed` rejects all tokens once the list is stale.
  Truly instant revocation (RFC 7662 introspection on each request) stays deferred.
- **Multi-modal retrieval.** *Shipped (v1 — tables).* A `content_type` signal is
  detected at index time and threaded through to the UI, which badges table chunks
  and rebuilds them as real HTML tables. Verified live: Docling's *chunker*
  serializes a table as row/column triplets (`row, Column = value.`), not Markdown,
  so the renderer parses both forms. Remaining: extract picture items to disk and
  show figure thumbnails (currently flagged but not extracted); consider storing
  the original Markdown table in chunk metadata at index time for lossless display.
- **Evaluation depth.** *Shipped.* Opt-in answer-quality scoring (LLM-graded
  faithfulness + relevance over the best retrieval mode) plus per-run baseline
  history (`rag_eval_history_<slug>.jsonl`), a vs-previous nDCG delta, and a
  baseline trend chart in the console.

## 🔭 Later — watching / larger efforts

Each item here is deliberately *not started*; this section is the design notebook
for when one graduates to **Next**.

### Knowledge structuring: Graph RAG & "LLM Wiki"

Reviewed and **paused** — adopt only if real-world queries turn out to be
relationship- or global-summary-heavy. The current engine is already an *advanced
hybrid RAG* (hierarchical chunking + auto-merge, vector + BM25/CKIP, cross-encoder
rerank, contextual retrieval), which covers the two classic RAG weaknesses of
fragmented chunks and weak precision.

- **Graph RAG** (RAG + knowledge graph). Wins at multi-hop *relational* queries,
  *global / aggregative* summaries, and node-level explainability — the one thing
  this engine lacks. Expensive to build (an LLM extracts entities and relations
  over the whole corpus). Adoption path: a Neo4j or Microsoft GraphRAG layer plus
  a **query router** that sends relational / "summarise across everything"
  questions to the graph and keeps local look-ups on the existing hybrid pipeline.
  Open questions before starting: which extraction model, how to keep the graph in
  sync as folders change, and how to cite graph paths back to source chunks.
  Ref: 資策會 MIC, "Graph RAG" (MOEA 產業技術司, it_id=563).
- **LLM Wiki** (Andrej Karpathy — a *methodology*, not a product). "Compile"
  sources once into cross-referenced Markdown concept-pages (one per concept, YAML
  frontmatter, `[[slug]]` links), then query the organised pages instead of the
  raw sources. Stateful, compounding, human-auditable via Git. Its own stated
  limit — it needs an ANN index past ~500 pages — is exactly what this engine
  provides. Two integration shapes:
  1. *Serve*: compile elsewhere (Claude Code / Obsidian) → upload the `.md` pages
     into a folder → serve via hybrid retrieval + rerank + MCP.
  2. *Synthesise*: add a "concept-page synthesis" step inside Folio that
     reuses the contextual-retrieval LLM to emit wiki pages from a folder.
  Ref: blog.104.com.tw/andrej-karpathy-llm-wiki.
- **Decision gate:** instrument real usage first — measure the share of
  relationship / global-summary questions before investing. Both are complementary
  to, not replacements for, the current engine.

### Retrieval quality experiments

- **Query understanding**: light query rewriting / decomposition for multi-part
  questions before retrieval; HyDE-style hypothetical-answer embedding as an
  optional recall booster.
- **Adaptive fusion**: tune the RRF constant and dense/sparse weighting per folder
  from click / feedback signals instead of one global setting.
- **Chunk feedback loop**: capture thumbs-up/down on answers and retrieved chunks,
  and feed it into rerank thresholds and the evaluation baselines above.

### Operability & deployment

- **Deploy guide**: reverse proxy + HTTPS, and a full `docker-compose` for the
  stack (PostgreSQL + pgvector and the model services), aligned with MCP Gate's
  Deploy chapter.
- **Observability**: structured request tracing across the retrieval pipeline
  (vector → BM25 → fusion → rerank) exposed in the console beyond the current
  single-query trace; Prometheus-friendly metrics.
- **Backup / restore**: a documented path to snapshot and restore folders, their
  files, and their vector tables together.

### Access & multi-tenancy

- **Role tiers**: today the console is unauthenticated by product decision
  (internal-network deployment). A future option: an admin auth mode plus
  read-only vs. manage roles, without breaking the current single-tenant flow.
- **Per-folder sharing**: let more than one person (`sub`) access a folder, for
  team-shared corpora, while keeping the default per-person scoping.

## Known limitations

- **Requires pgvector** — no SQLite fallback (hybrid retrieval + hierarchical
  tables depend on it).
- **Synchronous** SQLAlchemy / psycopg2 — do not use `+asyncpg`.
- **GPU selection is manual** — PyTorch is chosen per platform via a
  `uv sync --extra local --group` (rocm / cuda / cpu).
- **Revocation is not instant, and is off by default.** Tokens are verified
  offline. Without `auth.revocation` a revoked token (and console sessions made
  from it) keeps working until it expires; with it, until the next poll of MCP
  Gate's revocation feed (`poll_interval_sec`, default 30 s). With the default
  `fail_mode: open`, revocations made while MCP Gate is unreachable apply only
  once it is reachable again.
- **The server aborts at shutdown.** Every restart, the old `python3 main.py`
  dies with SIGABRT raised from inside the process (`si_code=SI_TKILL`, i.e. a
  native library's `abort()` during interpreter teardown — torch/CUDA-style), which
  left a ~16 GB `core.<pid>` in the repo root each time. The process now disables
  its own core dumps by default (`FOLIO_CORE_DUMPS=1` re-enables them for
  debugging). Root-causing it needs stderr captured by the launcher and a native
  backtrace (gdb) — tracked, not yet done.

## ✅ Recently shipped

- **1.4.0**: ruff (correctness rules) with a lint job; a dependency-audit workflow (pip-audit + npm audit,
  weekly); ROCm builds on torch 2.13 with ROCm 7.13 dropped, so setuptools moves 81 → 84.2.1 everywhere.
- **1.3.0**: dependency security updates (pip-audit 168 → 12: pyjwt, pypdf, docling, pillow 13, …);
  new upload formats RTF, MHTML, GIF, `.markdown`, Excel templates; MHTML decoded with its declared charset.
- **1.2.0**: renamed to Folio; retrieval golden set + eval harness (205 questions, public corpus);
  chunk pipeline v3, a deeper rerank pool and an MCP response token budget (Recall@10 0.973 → 0.981,
  zh→en 0.80 → 0.90, tokens per search 8.2k → 2.6k); MCP search reports `low_relevance`; search
  latency p50 ~500 → ~330 ms; MCP Gate revocation feed; console aggregates scoped to the token's own
  folders; folder-level re-index of stale files; background Rebuild-FTS; backend CI.
- **Streaming RAG answers** over SSE in the Search Playground.
- **Multi-modal retrieval v1**: Docling tables surfaced (badge + rendered HTML table).
- **Evaluation depth**: LLM-graded answer quality + baseline history and trend.
- One-click **Rebuild-FTS** to backfill CKIP full-text search on existing folders.
- **Frontend tests + CI** (Vitest + GitHub Actions) for the console SPA.
- **React console is the front door** (root URL redirects to `/admin`); legacy
  console kept at `/admin-classic` (since deleted).
- Admin console migrated from a single-file HTML app to a **React SPA** (Vite +
  React Router + Tailwind), "Retrieval Terminal" design.
- Chinese full-text search via **CKIP** word segmentation (replaces pg_jieba).
- Folder ownership keyed on the token's **`jti`** (per-token scoping) — replaced by
  per-person (`sub`) ownership after 1.4.0.
