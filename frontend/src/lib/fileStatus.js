// Per-file status flags shown next to the status pill in the folder view.

// Indexed, but the parser reported pages it could not read (see FileIndex.index_warning).
export function isPartial(f) {
  return !!(f && f.status === 'indexed' && f.warning)
}

// Indexed by an older chunking pipeline, or by an embedding model (or dimension) other
// than the configured one: a re-index is recommended. The server decides.
export function needsReindex(f) {
  return !!(f && f.status === 'indexed' && f.reindex_recommended)
}

// Why, in words (the server's reindex_reason: 'embedding' or 'pipeline').
export function reindexNote(f) {
  if (!needsReindex(f)) return ''
  if (f.reindex_reason === 'embedding') {
    const was = f.embedding_model ? ` (${f.embedding_model}${f.embedding_dimension ? `, ${f.embedding_dimension} dims` : ''})` : ''
    return `Indexed with another embedding model${was} than the one configured now: its vectors do not match new queries. Re-index it.`
  }
  return 'Indexed by an older chunking pipeline. Still searchable; re-index for section-aligned chunks.'
}

export function countNeedsReindex(files) {
  return (files || []).filter(needsReindex).length
}

// Folder list: the server's per-folder count of files needing a re-index.
export function folderStaleCount(folder) {
  return (folder && folder.reindex_recommended_files) || 0
}

export function staleFileIds(files) {
  return (files || []).filter(needsReindex).map((f) => f.id)
}

// POST /folders/{id}/retry takes at most this many file_ids per request.
export const RETRY_BATCH = 1000

export function batches(ids, size = RETRY_BATCH) {
  const out = []
  for (let i = 0; i < ids.length; i += size) out.push(ids.slice(i, i + size))
  return out
}
