// Index-job and file-status helpers shared by the Folders and Jobs pages.

// Pill colour class per file / job status.
const PILL = {
  indexed: 'ok', succeeded: 'ok', completed: 'ok',
  failed: 'bad',
  running: 'run', loading: 'run', contextualizing: 'run', embedding: 'run', writing: 'run',
  pending: 'warn', queued: 'warn', partial_success: 'warn', interrupted: 'warn',
  cancelled: 'mute', unindexed: 'mute',
}

export function pillClass(status) {
  return PILL[status] || 'run'
}

// "partial_success" → "Partial success"
export function statusLabel(status) {
  if (!status) return '—'
  const s = String(status).replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

// File statuses whose last attempt did not produce an index — Retry is the main action.
const NEEDS_RETRY = new Set(['failed', 'interrupted', 'cancelled'])

export function needsRetry(file) {
  return !!file && !file._stage && NEEDS_RETRY.has(file.status)
}

// Retry (forced reindex) is offered for any file with no live work on it.
export function canRetry(file) {
  return !!file && !file._stage
}

// Cancel is offered while the file is queued or being indexed.
export function canCancel(file) {
  return !!file && !!file._stage
}

const KIND_LABEL = { index: 'Index', rebuild_fts: 'Rebuild FTS' }

export function jobKindLabel(kind) {
  return KIND_LABEL[kind || 'index'] || statusLabel(kind)
}

export const ACTIVE_JOB = new Set(['running', 'pending', 'queued'])

export function isJobActive(job) {
  return !!job && ACTIVE_JOB.has(job.status)
}
