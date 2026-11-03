// Per-file status flags shown next to the status pill in the folder view.

// Indexed, but the parser reported pages it could not read (see FileIndex.index_warning).
export function isPartial(f) {
  return !!(f && f.status === 'indexed' && f.warning)
}

// Indexed by an older chunking pipeline: still searchable, but a re-index gives it
// the current chunks (section-aligned, with breadcrumbs). The server decides.
export function needsReindex(f) {
  return !!(f && f.status === 'indexed' && f.reindex_recommended)
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
