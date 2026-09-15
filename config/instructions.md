# Agentic RAG — Intelligent Document Knowledge Base (Agentic Edition)

You have access to a set of `Agentic_{folder_name}` tools for querying the user's document
knowledge base. Each tool corresponds to one folder, and the tool list is filtered
automatically by your token (you only see folders you have permission for).

This service is built on a **Hierarchical Chunking + AutoMerging Retrieval** architecture,
which adds the following over traditional RAG:

- **Hierarchical chunking**: at index time, documents are split into small leaves (precise
  matching) plus large parents (complete context)
- **Auto-merge**: when multiple leaves from the same passage are hit, the whole parent passage
  is returned automatically instead of just fragments
- **Context Expansion**: when a single leaf is hit, its neighbors are expanded automatically to
  avoid narrative discontinuity
- **Cross-encoder Rerank**: `bge-reranker-v2-m3` performs fine-grained ranking to improve Top-K precision
- **Contextual Retrieval**: at index time, an LLM adds a context prefix to each chunk
  (Anthropic's recommended approach)

## Priority Order of the Three Modes

| Priority | mode | When to use |
|------|------|--------|
| **🔍 Primary (default)** | `search` | Any need to "find answers/clues/fragments from a folder" |
| **📋 Occasional** | `list` | When the user asks "what files are there" or you need a file_id |
| **📖 Strictly limited** | `read` | **Only when the user explicitly asks for the full content or transcript** |

## `read` Mode — Strict Usage Rules

`read` loads an entire file into your context (up to ~30K tokens). **Do not use it by default**, unless:

### ✅ When to use `read`

- The user **explicitly says** "give me the full content / transcript / whole document"
- The user asks to "translate the whole / summarize the whole / rewrite the whole" of file XX
- The file is an **audio transcript** (Whisper output) and the user wants to read the full conversation

### ❌ When **not** to use `read`

- The user is just asking a specific question → use `search` and let retrieval find the relevant passages
- A vague request like "I want to understand this file" → `search` for the relevant fragments is enough
- Search results are not good enough → refine the query or adjust `similarity_cutoff`, **do not fall back to read**
- Exploratory browsing → use `list` to view the directory

**Why**: `read` stuffs ~3000-30000 tokens into context at once, most of which is irrelevant to
the user's specific question — wasting the token budget and diluting your attention. `search`
already includes auto-merge + context expansion and usually returns enough content to answer
90% of questions.

## Recommended Query Strategy

1. **Default to search** — search first for any question
2. **Weak search → tune parameters, do not switch to read** — try: lower `similarity_cutoff` / rewrite the query
3. **Multiple file hits → check _hint** — the system suggests a next step
4. **Explicitly need the whole file → list to get file_id → read** — a two-step flow

## Tool Response Structure

`search` mode:
- `results[]`: hit fragments (text, score, file_name, chunk_index, node_role)
- `node_role`: `leaf` / `parent` (auto-merge triggered) / `expanded` (neighbor expansion)
- `_hint`: suggested next step

`list` mode:
- `files[]`: contains file_id, file_name, estimated_tokens, indexed_status

`read` mode:
- `content`: full file content (markdown format)
- `truncated`: whether it was truncated by max_tokens (30000)
