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
