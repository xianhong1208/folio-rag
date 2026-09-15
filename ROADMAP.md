# Roadmap

## Under evaluation — watching

### Knowledge structuring: Graph RAG & "LLM Wiki"

Two approaches to structuring knowledge (beyond plain retrieval) were reviewed. **Status:
paused / watching** — adopt only if real-world queries turn out to be relationship- or
global-summary-heavy. The current engine is already an *advanced hybrid RAG* (hierarchical
chunking + auto-merge, vector + BM25/CKIP, cross-encoder rerank, contextual retrieval), which
covers the two classic RAG weaknesses of fragmented chunks and weak precision.

- **Graph RAG** (RAG + knowledge graph). Wins at multi-hop *relational* queries, *global /
  aggregative* summaries, and node-level explainability — the one thing this engine lacks.
  Expensive to build (LLM entity/relation extraction over the whole corpus). Adoption path:
  a Neo4j or Microsoft GraphRAG (open source) layer + query routing (relational → graph,
  local → existing hybrid pipeline).
  Ref: 資策會 MIC, "Graph RAG" (MOEA 產業技術司).

- **LLM Wiki** (Andrej Karpathy — a *methodology*, not a product). "Compile" source material
  once into cross-referenced Markdown concept-pages (one per concept, `[[slug]]` links),
  then query the organized pages instead of raw sources. Stateful / compounding, human-
  auditable (Git), source-traceable. Its own stated limit — needs an ANN index past ~500
  pages — is **exactly what this engine provides** (pgvector + hybrid + rerank + CKIP).
  Integration path: compile → upload the `.md` pages into a folder → serve via this engine's
  hybrid retrieval + rerank + MCP. Alternatively, add a "concept-page synthesis" step here
  (reuse the contextual-retrieval LLM).
  Ref: blog.104.com.tw/andrej-karpathy-llm-wiki.

- **Decision gate:** measure the share of relationship / global-summary questions in real
  usage before investing; both approaches are complementary to (not replacements for) the
  current retrieval engine.

## Recently shipped

- Admin console migrated from a single-file HTML app to a **React SPA** (Vite + React Router
  + Tailwind), "Retrieval Terminal" design system, served at `/admin` (legacy single-file at
  `/admin-classic`).
- Chinese full-text search via **CKIP** word segmentation (replaces pg_jieba).
- Folder ownership keyed on the token's **`jti`** (per-token scoping).
