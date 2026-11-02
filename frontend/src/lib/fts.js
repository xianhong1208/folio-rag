// Keyword-index (FTS) status for a folder, from GET /api/admin/manage/folders/:id/fts.
// Returns the warning to show when the folder's index was built with a different
// Chinese segmenter than the one configured, or null when nothing needs doing.
export function ftsNotice(status) {
  if (!status || !status.needs_rebuild) return null
  const was = status.indexed_with || 'another segmenter'
  const now = status.configured || 'the configured segmenter'
  return `Keyword index built with ${was}, but queries now use ${now}. Run Rebuild FTS to restore keyword recall.`
}
