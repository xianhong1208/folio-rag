# Roadmap

[繁體中文](ROADMAP.zh-TW.md)

Where Folio is going after 1.4.2. The plan was agreed on 2026-11-13, after looking at how RAGFlow,
Onyx, Kotaemon, R2R, Dify, Open WebUI and Khoj solve the same problems. Each line says what you
get, not how it is built. Every item has an issue in the GitHub milestone of its version.

## Shipped

**1.0.0 (2026-09), as Agentic RAG.** An MCP server that turns a folder of documents into a
retrieval tool: hierarchical chunking with auto-merge, hybrid vector + BM25 search with CKIP
Chinese segmentation, a cross-encoder reranker, contextual retrieval, pgvector storage, folders
owned per token, and the first React console.

**1.1.0 (2026-09-18).** Answers stream token by token in the Search Playground. Tables found by
Docling are shown as real tables, and MCP results say when a hit is a table. Evaluation grades
answer quality and keeps a baseline history. *Rebuild FTS* re-segments existing folders without
re-embedding. The React console became the front door.

**1.2.0 (2026-11-04), renamed Folio.** A public golden set of 205 questions with a harness that
scores the real MCP search. Chunk pipeline v3, a deeper rerank pool and a response token budget
(Recall@10 0.973 → 0.981, Chinese questions about English text 0.80 → 0.90, tokens per search
8.2k → 2.6k). MCP search says when nothing relevant was found. Search is faster (p50 ~500 → ~330
ms). Tokens revoked in MCP Gate stop working within about 30 seconds. Console figures cover only
your own folders. A folder's stale files re-index in one click.

**1.3.0 (2026-11-05).** Dependency security updates (pip-audit 168 → 12 advisories). New upload
formats: RTF, MHTML web archives, GIF, `.markdown` and Excel templates.

**1.4.0 (2026-11-09).** A lint job and a weekly dependency audit in CI. AMD ROCm builds move to
torch 2.13 with ROCm 7.14.

**1.4.1 (2026-11-09).** Folders belong to the person, not to one token: every personal access
token and OAuth sign-in of one MCP Gate user sees the same folders, and older folders move over on
first use. Signing out ends the console session on the server. Answers come in the question's
language. Duplicate folder and file names are refused. No sideways scrolling in the console on a
phone.

**1.4.2 (2026-11-13).** Known risks fixed: model key settings can no longer be used to send the server's environment variables elsewhere; a large upload is refused before it fills memory; changing the embedding model marks the files that need re-indexing; searching a folder that was never indexed returns an empty result. Dependencies updated for every open advisory (transformers 5, torch 2.14 for CUDA 13). Advance notice that 1.5 removes the built-in OCR, speech-to-text and the `/v1` media endpoints.

Details: [CHANGELOG.md](CHANGELOG.md).

## Next

| Version | What you get |
|---|---|
| **1.5: install in one command, never lose data** | **A lightweight Docker image** and a compose file with PostgreSQL + pgvector, plus an optional MCP Gate profile. The built-in OCR and speech models leave the image: scanned PDFs and images are read by a **vision model over an OpenAI-compatible API** (for example Qwen3-VL served by [Forgeyard](https://github.com/xianhong1208/forgeyard)), and audio goes to Forgeyard's transcription. The `/v1` OCR and speech endpoints go too; a **migration guide** shows how to point Folio at Forgeyard. **A deploy guide.** **`folio backup` / `folio restore`** and scheduled backups. **Export and import one folder.** Prometheus metrics at **`/metrics`**. |
| **1.6: find the document** | **Metadata and tags** on documents, with filters in REST, the console and MCP. **Results by document**, not only by passage. **Search across all your folders.** A **Library** page. An MCP **`find`** mode that returns documents. |
| **1.7: better answers** | **Query rewriting and multi-query**, with HyDE and translation as opt-ins. Each one is kept only if the golden set improves. **Figures** described by a vision model, so they can be found. **Citations open the PDF** at the highlighted passage. **Thumbs up / down** on answers. |
| **1.8: folders that stay current** | **Sources** for a folder: a URL or sitemap, a Git repository, a directory on the server, S3 or WebDAV. They **re-sync on a schedule**, only what changed, and files removed at the source are removed from the folder. |
| **1.9: conversations** | **Chat history**, **follow-up questions** that keep the context, and **export** of a conversation. |
| **1.10: Notion and Google Drive** | **Notion** and **Google Drive** connectors, built on the 1.8 sources. |

Not planned for now: sharing a folder with other people (see Later).

## Later

- **Confluence** connector.
- **Helm chart** for Kubernetes.
- **Python SDK.**
- **A chat widget** to embed in other pages.
- **Folder sharing.** Two shapes are on the table: (A) share a folder with named accounts; (B) team
  folders, which need groups in MCP Gate first. Folders stay per person until then.
- **GraphRAG**, still paused (below).

### GraphRAG and "LLM Wiki": paused, with a decision gate

Folio is already an advanced hybrid RAG (hierarchical chunking with auto-merge, vector + BM25/CKIP,
a cross-encoder reranker, contextual retrieval), which covers the two classic weaknesses of RAG:
fragmented chunks and weak precision. Both ideas below complement it rather than replace it.

- **GraphRAG** (RAG plus a knowledge graph) wins at multi-hop *relational* questions, *global*
  summaries ("what do all of these say about…") and node-level explainability. It is expensive to
  build: an LLM extracts entities and relations from the whole corpus. The path, if adopted: a
  Neo4j or Microsoft GraphRAG layer plus a **query router** that sends relational and
  "summarise everything" questions to the graph and keeps look-ups on the hybrid pipeline. Open
  questions first: which extraction model, how to keep the graph in sync as folders change, and how
  to cite graph paths back to source chunks. Reference: 資策會 MIC, "Graph RAG" (MOEA 產業技術司,
  it_id=563).
- **"LLM Wiki"** (Andrej Karpathy; a method, not a product): compile sources once into
  cross-linked Markdown concept pages (one per concept, YAML front matter, `[[slug]]` links), then
  query those pages instead of the raw sources. Its own stated limit, needing an ANN index past
  ~500 pages, is what Folio provides. Two ways to fit it in: *serve* (compile elsewhere, upload the
  pages to a folder, serve them through hybrid retrieval and MCP), or *synthesise* (a step inside
  Folio that writes concept pages from a folder with the contextual-retrieval LLM). Reference:
  blog.104.com.tw/andrej-karpathy-llm-wiki.
- **Decision gate:** measure real usage first. Only if a meaningful share of questions are about
  relationships or whole-corpus summaries is either one worth building.

### Older ideas, not scheduled

Kept here so they are not lost; none of them is planned.

- Tune the fusion weights per folder from the 1.7 thumbs up / down, instead of one global setting.
- Read-only and manage roles in the console, beyond today's two tiers (any token / `rag:admin`).
- Instant revocation through RFC 7662 introspection on every request (today: the revocation feed,
  within one poll interval).
- Tracing of each query across the pipeline (vector → BM25 → fusion → rerank) beyond the single
  query trace in the Search Playground.
- Keep a table's original Markdown in the chunk metadata, so tables display without loss.

## Known limitations

- **PostgreSQL with pgvector is required.** There is no SQLite fallback: hybrid retrieval and the
  per-folder tables depend on it.
- **Synchronous SQLAlchemy** (psycopg2) for the configured DSN; never `+asyncpg`.
- **The GPU is chosen by hand,** with `uv sync --extra local --group rocm-r714 | cuda | cpu`.
- **Revocation is not instant, and is off by default.** Tokens are verified offline. Without
  `auth.revocation` a revoked token keeps working until it expires; with it, until the next poll of
  MCP Gate (30 s by default). With `fail_mode: open`, revocations made while MCP Gate is unreachable
  apply once it is reachable again.
- **The `/v1` media endpoints have no authentication,** by decision; keep them on a trusted network.
  They are removed in 1.5.
- **The console is English only,** by decision. (A full English + 繁體中文 version was built and then
  removed; it is in git history at `5103cbd` / `53aa71a`, reverted by `be480d6`.)
- **The server aborts at shutdown.** On every restart the old process dies with SIGABRT from a
  native library during interpreter teardown, which used to leave a ~16 GB core file. Folio now
  turns off its own core dumps (`FOLIO_CORE_DUMPS=1` turns them back on for debugging).
