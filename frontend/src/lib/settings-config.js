// Settings model (ported from the standalone console). Section metadata + per-path
// field definitions drive the Settings page; values come from GET /api/admin/settings.

export const META = {
  'rag.embedding': { title: 'Embedding', icon: 'layers', probe: true, desc: 'Vectorization for documents and queries', warn: 'Changing any field makes existing folder vectors incompatible with the new model — those folders must be re-indexed. New uploads use the new settings immediately.' },
  'rag.llm': { title: 'LLM', icon: 'spark', probe: true, desc: 'Contextual Retrieval prefix generation' },
  'rag.contextual_retrieval': { title: 'Contextual Retrieval', icon: 'list', desc: 'LLM-generated context prefix per chunk at index time' },
  'rag.rerank': { title: 'Reranker', icon: 'sort', probe: true, desc: 'Cross-encoder re-scoring of retrieval results' },
  'rag.asr': { title: 'Speech-to-Text', icon: 'mic', probe: true, desc: 'Transcription source for audio indexing and /v1/transcriptions' },
  'rag.retrieval': { title: 'Retrieval', icon: 'sliders', desc: 'Hybrid search and auto-merging behavior' },
}

export const FIELDS = {
  'rag.embedding.provider': { label: 'Provider', type: 'select', options: ['vllm', 'openai', 'azure', 'ollama', 'huggingface'] },
  'rag.embedding.model': { label: 'Model', type: 'text', hint: 'e.g. intfloat/multilingual-e5-large / BAAI/bge-m3' },
  'rag.embedding.dimension': { label: 'Dimension', type: 'number', hint: "Must equal the model's actual output dimension (e5-large / bge-m3 = 1024)" },
  'rag.embedding.base_url': { label: 'Endpoint', type: 'text' },
  'rag.embedding.api_key': { label: 'API Key', type: 'secret' },
  'rag.embedding.query_prefix': { label: 'Query Prefix', type: 'text', hint: 'Required "query: " for e5 family; leave empty for bge' },
  'rag.embedding.passage_prefix': { label: 'Passage Prefix', type: 'text', hint: 'Required "passage: " for e5 family; leave empty for bge' },
  'rag.llm.provider': { label: 'Provider', type: 'select', options: ['vllm', 'openai', 'azure', 'ollama'] },
  'rag.llm.model': { label: 'Model', type: 'text' },
  'rag.llm.base_url': { label: 'Endpoint', type: 'text' },
  'rag.llm.api_key': { label: 'API Key', type: 'secret' },
  'rag.contextual_retrieval.enabled': { label: 'Enabled', type: 'bool', hint: 'Disabling speeds up indexing significantly at the cost of retrieval quality' },
  'rag.contextual_retrieval.max_context_length': { label: 'Max Prefix Length (chars)', type: 'number' },
  'rag.contextual_retrieval.max_concurrent': { label: 'Concurrent LLM Requests', type: 'number' },
  'rag.contextual_retrieval.max_doc_chars': { label: 'Max Document Chars', type: 'number' },
  'rag.contextual_retrieval.max_tokens': { label: 'Max Generation Tokens', type: 'number' },
  'rag.contextual_retrieval.reasoning_effort': { label: 'Reasoning Effort', type: 'select', options: ['low', 'medium', 'high'] },
  'rag.rerank.enabled': { label: 'Enabled', type: 'bool' },
  'rag.rerank.model': { label: 'Model', type: 'text', hint: 'Must match the name served by the rerank service, otherwise results silently fall back to original order' },
  'rag.rerank.base_url': { label: 'Endpoint', type: 'text' },
  'rag.rerank.api_key': { label: 'API Key', type: 'secret' },
  'rag.rerank.top_n': { label: 'Top N', type: 'number', nullable: true, hint: 'Empty = determined by query top_k' },
  'rag.rerank.score_threshold': { label: 'Score Threshold', type: 'number', step: 0.05, range: [0, 1] },
  'rag.rerank.query_template': { label: 'Query Template', type: 'textarea' },
  'rag.rerank.document_template': { label: 'Document Template', type: 'textarea' },
  'rag.asr.enabled': { label: 'Enable Audio Transcription', type: 'bool' },
  'rag.asr.provider': { label: 'Provider', type: 'select', options: ['fireredasr', 'docling-whisper', 'openai-compatible'], hint: 'fireredasr = local Traditional Chinese (requires weights); openai-compatible = cloud / self-hosted' },
  'rag.asr.base_url': { label: 'Cloud Endpoint', type: 'text', hint: 'openai-compatible only' },
  'rag.asr.api_key': { label: 'API Key', type: 'secret' },
  'rag.asr.model': { label: 'Cloud Model', type: 'text', hint: 'openai-compatible only, e.g. whisper-1' },
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
