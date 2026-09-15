# Roadmap

Priorities run top to bottom: **Now** and **Next** are the focus; **Later** is
tracked but not started. Items are grounded in the current codebase and its known gaps.

## 🔨 Now — short term

- **Console i18n (English + 繁體中文).** The React console is English-only; add a
  language switch with a full Traditional-Chinese translation (parity with MCP Center).
- **Rebuild-FTS action.** The CKIP full-text vector is only correct for folders indexed
  *after* the CKIP migration. Add a one-click "Rebuild FTS" (folder + global) so existing
  folders get the CKIP-segmented `text_search_tsv` without a full reindex.
- **Retire the legacy console.** Once the React SPA is validated in real use, remove the
  single-file `admin_console.html` served at `/admin-classic`.
- **Frontend tests / CI.** The SPA has no tests yet; add build + smoke checks in CI.

## 🚀 Next — mid term

- **Streaming answers.** SSE token streaming for the Search Playground `answer` path and
  the MCP `read`/answer flows.
- **Instant revocation option.** Offer FastMCP `IntrospectionTokenVerifier` as an
  alternative to the offline JWT verifier, so a revoked token stops working immediately
  instead of at expiry.
- **Multi-modal retrieval.** Docling already parses tables and images — surface them in
  retrieval and in the returned context, not just text chunks.
- **Evaluation depth.** Beyond Recall@k · MRR · nDCG@k, add an answer-quality score and
  persist baselines so runs are comparable over time.

## 🔭 Later — watching / larger efforts

### Knowledge structuring: Graph RAG & "LLM Wiki"

Reviewed and **paused** — adopt only if real-world queries turn out to be relationship- or
global-summary-heavy. The current engine is already an *advanced hybrid RAG* (hierarchical
chunking + auto-merge, vector + BM25/CKIP, cross-encoder rerank, contextual retrieval),
which covers the two classic RAG weaknesses of fragmented chunks and weak precision.

- **Graph RAG** (RAG + knowledge graph). Wins at multi-hop *relational* queries, *global /
  aggregative* summaries, and node-level explainability — the one thing this engine lacks.
  Expensive to build. Adoption path: a Neo4j or Microsoft GraphRAG layer + query routing
  (relational → graph, local → existing hybrid pipeline).
  Ref: 資策會 MIC, "Graph RAG" (MOEA 產業技術司).
- **LLM Wiki** (Andrej Karpathy — a *methodology*, not a product). "Compile" sources once
  into cross-referenced Markdown concept-pages, then query the organized pages. Stateful /
  compounding, human-auditable. Its own stated limit — needs an ANN index past ~500 pages —
  is exactly what this engine provides. Integration path: compile → upload the `.md` into a
  folder → serve via hybrid retrieval + rerank + MCP. Or add a "concept-page synthesis" step.
  Ref: blog.104.com.tw/andrej-karpathy-llm-wiki.
- **Decision gate:** measure the share of relationship / global-summary questions in real
  usage before investing; both are complementary to (not replacements for) the engine.

### Deployment

- Deploy guide (reverse proxy + HTTPS) and a full `docker-compose` for the stack
  (PostgreSQL + pgvector and the model services), aligned with MCP Center's Deploy chapter.

## Known limitations

- **Requires pgvector** — no SQLite fallback (hybrid retrieval + hierarchical tables depend on it).
- **Synchronous** SQLAlchemy / psycopg2 — do not use `+asyncpg`.
- **GPU selection is manual** — PyTorch is chosen per platform via a `uv sync --group` (rocm / cuda / cpu).

## ✅ Recently shipped

- Admin console migrated from a single-file HTML app to a **React SPA** (Vite + React Router
  + Tailwind), "Retrieval Terminal" design, served at `/admin` (legacy at `/admin-classic`).
- Chinese full-text search via **CKIP** word segmentation (replaces pg_jieba).
- Folder ownership keyed on the token's **`jti`** (per-token scoping).
