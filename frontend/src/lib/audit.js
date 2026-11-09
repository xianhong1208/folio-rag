// How one audit-log entry reads. Besides runtime settings changes (set / reset), the
// log records folders renamed while being claimed for their owner (claim_rename): a
// token's pre-1.5 folder moved to its person, who already had a folder of that name.
export function auditEntry(e) {
  const reset = e.action === 'reset'
  const rename = e.action === 'claim_rename'
  return {
    cls: reset ? 'reset' : 'set',
    act: reset ? 'RESET' : rename ? 'RENAME' : 'SET',
    key: rename && e.old != null ? `${e.key} ${e.old}` : e.key,
    value: String(e.new ?? '—'),
  }
}
