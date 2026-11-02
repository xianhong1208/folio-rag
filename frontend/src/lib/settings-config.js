// Settings model (ported from the standalone console). Section metadata + per-path
// field definitions drive the Settings page; values come from GET /api/admin/settings.
// `probe` names the model role POST /api/admin/probe/{role} tests for that section.

const INHERIT = 'Empty = inherit from the LLM section'
const LLM_PROVIDERS = ['', 'openai', 'openai-compatible', 'vllm', 'ollama', 'azure']

export const META = {
  'rag.embedding': { title: 'Embedding', icon: 'layers', probe: 'embedding', desc: 'Vectorization for documents and queries', warn: 'Changing any field makes existing folder vectors incompatible with the new model — those folders must be re-indexed. New uploads use the new settings immediately.' },
  'rag.llm': { title: 'LLM', icon: 'spark', probe: 'llm', desc: 'Default model for every LLM role (contextual prefixes, answers, eval)' },
  'rag.contextual_llm': { title: 'Contextual-Prefix LLM', icon: 'spark', probe: 'contextual_llm', desc: 'Override for contextual retrieval prefixes — usually a small, cheap model' },
  'rag.answer_llm': { title: 'Answer LLM', icon: 'message', probe: 'answer_llm', desc: 'Override for answers in the Search Playground' },
  'rag.eval_llm': { title: 'Eval LLM', icon: 'flask', probe: 'eval_llm', desc: 'Override for evaluation question generation and judging' },
  'rag.contextual_retrieval': { title: 'Contextual Retrieval', icon: 'list', desc: 'LLM-generated context prefix per chunk at index time' },
  'rag.rerank': { title: 'Reranker', icon: 'sort', probe: 'rerank', desc: 'Cross-encoder re-scoring of retrieval results' },
  'rag.parsing': { title: 'Document Parsing', icon: 'scan', probe: 'vision', desc: 'Where PDFs and images are parsed: local docling models or a cloud vision model' },
  'rag.asr': { title: 'Speech-to-Text', icon: 'mic', probe: 'asr', desc: 'Transcription source for audio indexing and /v1/transcriptions' },
  'rag.retrieval': { title: 'Retrieval', icon: 'sliders', desc: 'Hybrid search and auto-merging behavior' },
}

const roleLlm = (role) => ({
  [`rag.${role}.provider`]: { label: 'Provider', type: 'select', options: LLM_PROVIDERS, nullable: true, hint: INHERIT },
  [`rag.${role}.model`]: { label: 'Model', type: 'text', nullable: true, hint: INHERIT },
  [`rag.${role}.base_url`]: { label: 'Endpoint', type: 'text', nullable: true, hint: INHERIT },
  [`rag.${role}.api_key`]: { label: 'API Key', type: 'secret', hint: INHERIT },
})

export const FIELDS = {
  'rag.embedding.provider': { label: 'Provider', type: 'select', options: ['vllm', 'openai', 'openai-compatible', 'azure', 'ollama', 'huggingface'] },
  'rag.embedding.model': { label: 'Model', type: 'text', hint: 'e.g. intfloat/multilingual-e5-large / BAAI/bge-m3' },
  'rag.embedding.dimension': { label: 'Dimension', type: 'number', hint: "Must equal the model's actual output dimension (e5-large / bge-m3 = 1024)" },
  'rag.embedding.base_url': { label: 'Endpoint', type: 'text', hint: 'OpenAI-compatible base URL (usually ends in /v1); empty = api.openai.com for provider openai' },
  'rag.embedding.api_key': { label: 'API Key', type: 'secret' },
  'rag.embedding.query_prefix': { label: 'Query Prefix', type: 'text', hint: 'Required "query: " for e5 family; leave empty for bge' },
  'rag.embedding.passage_prefix': { label: 'Passage Prefix', type: 'text', hint: 'Required "passage: " for e5 family; leave empty for bge' },
  'rag.embedding.request_dimensions': { label: 'Send Dimensions', type: 'bool', hint: 'Ask the API for vectors of exactly this size (Matryoshka models such as text-embedding-3-*)' },
  'rag.embedding.timeout': { label: 'Timeout (s)', type: 'number' },
  'rag.embedding.max_retries': { label: 'Retries', type: 'number', hint: 'On 429 / 5xx / timeouts, with backoff' },
  'rag.llm.provider': { label: 'Provider', type: 'select', options: ['vllm', 'openai', 'openai-compatible', 'azure', 'ollama'] },
  'rag.llm.model': { label: 'Model', type: 'text' },
  'rag.llm.base_url': { label: 'Endpoint', type: 'text', hint: 'OpenAI-compatible base URL; empty = api.openai.com for provider openai' },
  'rag.llm.api_key': { label: 'API Key', type: 'secret' },
  'rag.llm.timeout': { label: 'Timeout (s)', type: 'number' },
  'rag.llm.max_retries': { label: 'Retries', type: 'number', hint: 'On 429 / 5xx / timeouts, honoring Retry-After' },
  ...roleLlm('contextual_llm'),
  ...roleLlm('answer_llm'),
  ...roleLlm('eval_llm'),
  'rag.contextual_retrieval.enabled': { label: 'Enabled', type: 'bool', hint: 'Disabling speeds up indexing significantly at the cost of retrieval quality' },
  'rag.contextual_retrieval.max_context_length': { label: 'Max Prefix Length (chars)', type: 'number' },
  'rag.contextual_retrieval.max_concurrent': { label: 'Concurrent LLM Requests', type: 'number' },
  'rag.contextual_retrieval.max_doc_chars': { label: 'Max Document Chars', type: 'number' },
  'rag.contextual_retrieval.max_tokens': { label: 'Max Generation Tokens', type: 'number' },
  'rag.contextual_retrieval.reasoning_effort': { label: 'Reasoning Effort', type: 'select', options: ['low', 'medium', 'high'] },
  'rag.contextual_retrieval.max_calls_per_file': { label: 'Max LLM Calls per File', type: 'number', hint: '0 = unlimited; chunks past the cap get the file-name prefix' },
  'rag.contextual_retrieval.max_tokens_per_day': { label: 'Max Tokens per Day', type: 'number', hint: '0 = unlimited; estimated, process-wide, resets at UTC midnight' },
  'rag.rerank.enabled': { label: 'Enabled', type: 'bool' },
  'rag.rerank.api': { label: 'API Style', type: 'select', options: ['score', 'cohere', 'jina', 'generic'], hint: 'score = vLLM /v1/score; cohere / jina = /v1/rerank; generic = OpenAI-compatible /rerank servers' },
  'rag.rerank.model': { label: 'Model', type: 'text', hint: 'Must match the name served by the rerank service, otherwise results silently fall back to original order' },
  'rag.rerank.base_url': { label: 'Endpoint', type: 'text' },
  'rag.rerank.endpoint_path': { label: 'Endpoint Path', type: 'text', nullable: true, hint: "Empty = the style's default, e.g. /v2/rerank for Cohere v2" },
  'rag.rerank.api_key': { label: 'API Key', type: 'secret' },
  'rag.rerank.top_n': { label: 'Top N', type: 'number', nullable: true, hint: 'Empty = determined by query top_k' },
  'rag.rerank.score_threshold': { label: 'Score Threshold', type: 'number', step: 0.05, range: [0, 1] },
  'rag.rerank.timeout': { label: 'Timeout (s)', type: 'number' },
  'rag.rerank.max_retries': { label: 'Retries', type: 'number', hint: 'Then results fall back to retrieval order' },
  'rag.rerank.query_template': { label: 'Query Template', type: 'textarea' },
  'rag.rerank.document_template': { label: 'Document Template', type: 'textarea' },
  'rag.parsing.pdf': { label: 'PDF Parsing', type: 'select', options: ['', 'local', 'cloud'], nullable: true, hint: 'Empty = follow the profile (cloud profile → cloud)' },
  'rag.parsing.images': { label: 'Image Parsing', type: 'select', options: ['', 'local', 'cloud'], nullable: true, hint: 'Empty = follow the profile' },
  'rag.parsing.vision.provider': { label: 'Vision Provider', type: 'select', options: ['openai-compatible', 'openai', 'vllm', 'ollama', 'azure'] },
  'rag.parsing.vision.model': { label: 'Vision Model', type: 'text', hint: 'A chat model that accepts images, e.g. gpt-4o-mini / Qwen2.5-VL' },
  'rag.parsing.vision.base_url': { label: 'Vision Endpoint', type: 'text' },
  'rag.parsing.vision.api_key': { label: 'API Key', type: 'secret' },
  'rag.parsing.max_concurrency': { label: 'Pages in Flight', type: 'number', hint: 'Process-wide; pages of one file run in this many ordered blocks' },
  'rag.parsing.max_pages_per_file': { label: 'Max Pages per File', type: 'number', hint: '0 = unlimited; larger files are refused before any call' },
  'rag.parsing.max_pages_per_day': { label: 'Max Pages per Day', type: 'number', hint: '0 = unlimited; process-wide, resets at UTC midnight' },
  'rag.parsing.on_page_failure': { label: 'On Page Failure', type: 'select', options: ['mark', 'fail'], hint: 'mark = keep the file and mark the page; fail = fail the whole file' },
  'rag.asr.enabled': { label: 'Enable Audio Transcription', type: 'bool' },
  'rag.asr.provider': { label: 'Provider', type: 'select', options: ['fireredasr', 'docling-whisper', 'openai-compatible'], hint: 'fireredasr = local Traditional Chinese (requires weights); openai-compatible = cloud / self-hosted' },
  'rag.asr.base_url': { label: 'Cloud Endpoint', type: 'text', hint: 'openai-compatible only' },
  'rag.asr.api_key': { label: 'API Key', type: 'secret' },
  'rag.asr.model': { label: 'Cloud Model', type: 'text', hint: 'openai-compatible only, e.g. whisper-1' },
  'rag.asr.language': { label: 'Language Hint', type: 'text', nullable: true, hint: 'ISO-639-1, e.g. zh; empty = auto-detect' },
  'rag.asr.segment_seconds': { label: 'Segment Length (s)', type: 'number', hint: 'Long audio is uploaded in pieces cut at silences (max 780 s ≈ 25 MB)' },
  'rag.asr.timeout': { label: 'Timeout per Segment (s)', type: 'number' },
  'rag.asr.max_retries': { label: 'Retries', type: 'number', hint: 'Also re-asks on empty or runaway transcripts' },
  'rag.retrieval.default_top_k': { label: 'Top K', type: 'number' },
  'rag.retrieval.default_similarity_cutoff': { label: 'Similarity Cutoff', type: 'number', step: 0.05, range: [0, 1] },
  'rag.retrieval.default_sparse_top_k': { label: 'BM25 Candidates', type: 'number' },
  'rag.retrieval.default_hybrid_alpha': { label: 'Hybrid Alpha', type: 'number', step: 0.05, range: [0, 1], hint: 'Note: ignored by pgvector hybrid — use Hybrid Fusion below instead' },
  'rag.retrieval.hybrid_fusion': { label: 'Hybrid Fusion', type: 'select', options: ['rrf', 'concat'], hint: 'rrf = Reciprocal Rank Fusion (recommended). concat = llama_index native — dense dominates.' },
  'rag.retrieval.expand_context_default': { label: 'Expand Neighbors by Default', type: 'bool' },
  'rag.retrieval.expand_context_neighbors': { label: 'Neighbor Count', type: 'number' },
  'rag.retrieval.auto_merging.enabled': { label: 'Auto-Merging', type: 'bool' },
  'rag.retrieval.auto_merging.merge_threshold': { label: 'Merge Threshold', type: 'number', step: 0.05, range: [0, 1] },
}

export const PRESETS = {
  precision: { label: 'High Precision', sub: 'Fewer, high-confidence results', patch: { 'rag.retrieval.default_top_k': 3, 'rag.retrieval.default_similarity_cutoff': 0.5, 'rag.rerank.enabled': true, 'rag.rerank.score_threshold': 0.3 } },
  balanced: { label: 'Balanced', sub: 'Default tuning', patch: { 'rag.retrieval.default_top_k': 5, 'rag.retrieval.default_similarity_cutoff': 0.35, 'rag.retrieval.default_sparse_top_k': 12, 'rag.rerank.enabled': true, 'rag.rerank.score_threshold': 0.1 } },
  recall: { label: 'High Recall', sub: 'Cast a wide net', patch: { 'rag.retrieval.default_top_k': 12, 'rag.retrieval.default_similarity_cutoff': 0.15, 'rag.retrieval.default_sparse_top_k': 30, 'rag.rerank.enabled': true, 'rag.rerank.score_threshold': 0.0 } },
}

export function sectionOf(path) {
  let best = ''
  for (const sec of Object.keys(META)) if (path.startsWith(sec + '.') && sec.length > best.length) best = sec
  return best || path.split('.').slice(0, -1).join('.')
}

// Empty select / text values of nullable fields mean "unset" (inherit / follow the profile).
export function normalizeValue(f, v) {
  if (f && f.nullable && v === '') return null
  return v
}

// Key indicator for a secret setting: never the key, only whether one is set and where from.
export function keyIndicator(entry) {
  if (!entry || !entry.is_set) return { cls: 'mute', text: 'Not set' }
  if (entry.env) return { cls: 'ok', text: `Set · from $${entry.env}` }
  return { cls: 'warn', text: 'Set · literal value' }
}

// Where a role runs, for the roles overview.
export function runsLabel(role) {
  if (!role || role.runs === 'off') return { cls: 'mute', text: 'Off' }
  if (role.runs === 'in-process') return { cls: 'run', text: 'In-process' }
  return { cls: 'ok', text: 'Endpoint' }
}

// Render a probe result ({status, detail, latency_ms}) as a status line.
export function probeText(r) {
  if (!r) return { cls: '', text: '' }
  const ms = r.latency_ms != null ? ` · ${r.latency_ms} ms` : ''
  if (r.status === 'ok') return { cls: 'ok', text: `✓ ${r.detail}${ms}` }
  if (r.status === 'local') return { cls: 'ok', text: `✓ ${r.detail}` }
  if (r.status === 'disabled' || r.status === 'unset') return { cls: '', text: r.detail }
  return { cls: 'bad', text: `✗ ${r.detail}${ms}` }
}
