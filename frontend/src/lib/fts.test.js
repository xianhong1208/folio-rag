import { describe, it, expect } from 'vitest'
import { ftsNotice } from './fts'

describe('ftsNotice', () => {
  it('is null when no rebuild is needed or the status is missing', () => {
    expect(ftsNotice(null)).toBeNull()
    expect(ftsNotice({ needs_rebuild: false, indexed_with: 'ckip', configured: 'ckip' })).toBeNull()
  })
  it('names both segmenters when the index is stale', () => {
    const msg = ftsNotice({ needs_rebuild: true, indexed_with: 'ckip', configured: 'bigram' })
    expect(msg).toContain('ckip')
    expect(msg).toContain('bigram')
    expect(msg).toContain('Rebuild FTS')
  })
})
