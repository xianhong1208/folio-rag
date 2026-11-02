import { describe, it, expect } from 'vitest'
import { isPartial, needsReindex, countNeedsReindex } from './fileStatus'

describe('isPartial', () => {
  it('flags an indexed file that carries a parse warning', () => {
    expect(isPartial({ status: 'indexed', warning: 'Partial parse: page 3 …' })).toBe(true)
  })
  it('ignores files without a warning or not indexed', () => {
    expect(isPartial({ status: 'indexed', warning: null })).toBe(false)
    expect(isPartial({ status: 'failed', warning: 'x' })).toBe(false)
    expect(isPartial(null)).toBe(false)
  })
})

describe('needsReindex', () => {
  it('follows the server flag for indexed files only', () => {
    expect(needsReindex({ status: 'indexed', reindex_recommended: true })).toBe(true)
    expect(needsReindex({ status: 'indexed', reindex_recommended: false })).toBe(false)
    expect(needsReindex({ status: 'failed', reindex_recommended: true })).toBe(false)
    expect(needsReindex(undefined)).toBe(false)
  })
  it('counts stale files', () => {
    expect(countNeedsReindex([{ status: 'indexed', reindex_recommended: true }, { status: 'indexed' }])).toBe(1)
    expect(countNeedsReindex(null)).toBe(0)
  })
})
