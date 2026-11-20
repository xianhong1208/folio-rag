// Search page: which endpoint a run uses and how the answer is shown. One folder with
// passages keeps the folder playground (answer, trace); "All folders" or the
// Documents view use POST /api/search (src/domain/rag/doc_search.py).

export const ALL_FOLDERS = 'all'

// The folder playground can generate an answer and show the retrieval trace;
// /api/search does neither.
export const usesPlayground = (fid, view) => fid !== ALL_FOLDERS && view === 'passages'

// Body of POST /api/search for the current selection; `filters` is a document filter
// (lib/libraryFilters.js toFilter) or null.
export function searchBody({ fid, view, query, filters }) {
  const body = { query, group_by: view === 'documents' ? 'document' : 'passage' }
  if (fid && fid !== ALL_FOLDERS) body.folders = [Number(fid)]
  if (filters) body.filters = filters
  return body
}

// The Search page's filter: the rail's state as a document filter. With one folder
// chosen, the rail's folder list does not apply (and is hidden).
export function searchFilter(state, fid, toFilter) {
  const own = fid && fid !== ALL_FOLDERS ? { ...state, folders: [] } : state
  return toFilter(own)
}

const REASONS = {
  nothing_indexed: 'nothing indexed',
  embedding_model: 'other embedding model, re-index',
  folder_limit: 'over the folder limit',
  error: 'search failed',
}

// "2 folders not searched: a (nothing indexed), b (other embedding model, re-index)"
export function skippedNote(skipped) {
  if (!Array.isArray(skipped) || skipped.length === 0) return ''
  const n = skipped.length
  const list = skipped.map((s) => `${s.name} (${REASONS[s.reason] || s.reason})`).join(', ')
  return `${n} folder${n === 1 ? '' : 's'} not searched: ${list}`
}

// "1 passage" / "3 passages"
export const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`

// "p.3 · Intro › Scope" from a snippet (either part may be missing).
export function snippetPlace(s) {
  const parts = []
  if (s && s.page != null) parts.push(`p.${s.page}`)
  if (s && Array.isArray(s.headings) && s.headings.length) parts.push(s.headings.join(' › '))
  return parts.join(' · ')
}
