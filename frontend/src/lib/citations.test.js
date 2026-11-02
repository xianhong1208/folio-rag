import { describe, it, expect } from 'vitest'
import { citationParts } from './citations'

describe('citationParts', () => {
  it('links only valid citations', () => {
    expect(citationParts('A [2] B [9].', [2])).toEqual([
      { t: 'text', v: 'A ' }, { t: 'cite', n: 2 }, { t: 'text', v: ' B [9].' },
    ])
  })
  it('falls back to the result count while streaming', () => {
    expect(citationParts('[1][3]', undefined, 2)).toEqual([{ t: 'cite', n: 1 }, { t: 'text', v: '[3]' }])
  })
  it('handles empty text', () => {
    expect(citationParts('', [1])).toEqual([])
    expect(citationParts(null)).toEqual([])
  })
})
