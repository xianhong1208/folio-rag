import { describe, it, expect } from 'vitest'
import { isPartial } from './fileStatus'

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
