# Roadmap

Priorities run top to bottom: **Now** and **Next** are the focus; **Later** is
tracked but not started. Items are grounded in the current codebase and its known gaps.

## 🔨 Now — short term

- **Console i18n (English + 繁體中文).** *In progress.* Foundation shipped — an
  `I18nProvider` (browser-detected default, `localStorage`-persisted, English
  fallback with `{var}` interpolation), a sidebar language toggle, and full
  translation of the shell and the Folders page as the reference implementation.
  Remaining: translate the other eight pages (mechanical — reuse the same keys).
- **Rebuild-FTS action.** *Shipped.* One-click `POST …/rebuild-fts` re-segments
  stored chunks with CKIP and rewrites `text_search_tsv` in place (no
  re-embedding), backfilling Chinese BM25 recall for folders indexed before the
  CKIP upgrade.
- **Retire the legacy console.** *Partly shipped.* The root URL now redirects to
  the React console; the single-file `admin_console.html` is retired from the
  front door and kept only at `/admin-classic` as a temporary escape hatch. Hard
  deletion (file + loader + packaging entry) is deferred until the SPA is
  validated in real use.
- **Frontend tests / CI.** *Shipped.* Vitest pure-function suites (format +
  settings-config) and a GitHub Actions workflow that builds and tests the SPA.

## 🚀 Next — mid term

- **Streaming answers.** SSE token streaming for the Search Playground `answer`
  path and the MCP answer flow, so long generations render progressively instead
  of blocking until the final token. Retrieval + CRAG grading stay non-stream;
  only the final generation streams.
- **Instant revocation.** *Deferred by decision* — revisit when needed. A revoked
  token should stop working immediately, not at expiry. Two candidate mechanisms:
  - *jti deny-list* (self-contained): a console-managed revocation list checked on
    every verify, keyed on the `jti` that already scopes ownership. Needs no MCP
    Center change and is fully testable locally.
  - *RFC 7662 introspection* (`IntrospectionTokenVerifier`): the resource server
    asks MCP Center whether the token is still active on each request. Needs MCP
    Center to expose an introspection endpoint.
  Whichever is chosen must cover **both** verify paths — the REST / tool-filter
  `McpCenterTokenVerifier` and the FastMCP `/mcp` transport verifier.
- **Multi-modal retrieval.** Docling already parses tables and images at index
  time; surface them in retrieval results and in the returned context (not just
  text chunks), with a way to preview a table/figure from a result card.
- **Evaluation depth.** Beyond Recall@k · MRR · nDCG@k, add an answer-quality
  score (LLM-graded faithfulness / relevance against the retrieved context) and
  persist run baselines so tuning changes are comparable over time.

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
  2. *Synthesise*: add a "concept-page synthesis" step inside agentic_rag that
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
  stack (PostgreSQL + pgvector and the model services), aligned with MCP Center's
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
- **Per-folder sharing**: allow more than one `jti` to own / access a folder, for
  team-shared corpora, while keeping the default per-token scoping.

## Known limitations

- **Requires pgvector** — no SQLite fallback (hybrid retrieval + hierarchical
  tables depend on it).
- **Synchronous** SQLAlchemy / psycopg2 — do not use `+asyncpg`.
- **GPU selection is manual** — PyTorch is chosen per platform via a
  `uv sync --group` (rocm / cuda / cpu).
- **Console is unauthenticated** — by design, for internal-network use; lock down
  `_AUTH` before exposing it publicly.

## ✅ Recently shipped

- One-click **Rebuild-FTS** to backfill CKIP full-text search on existing folders.
- **Frontend tests + CI** (Vitest + GitHub Actions) for the console SPA.
- **React console is the front door** (root URL redirects to `/admin`); legacy
  console kept at `/admin-classic`.
- **Console i18n foundation** (English + 繁體中文): provider, language toggle,
  translated shell + Folders page.
- Admin console migrated from a single-file HTML app to a **React SPA** (Vite +
  React Router + Tailwind), "Retrieval Terminal" design.
- Chinese full-text search via **CKIP** word segmentation (replaces pg_jieba).
- Folder ownership keyed on the token's **`jti`** (per-token scoping).
