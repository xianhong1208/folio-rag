// Per-file status flags shown next to the status pill in the folder view.

// Indexed, but the parser reported pages it could not read (see FileIndex.index_warning).
export function isPartial(f) {
  return !!(f && f.status === 'indexed' && f.warning)
}
