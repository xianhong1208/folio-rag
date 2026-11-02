import { describe, it, expect } from 'vitest'
import { gapRows, gapDesc } from './gaps'

describe('gapRows', () => {
  it('formats gap_queries with their best score', () => {
    const rows = gapRows({ gap_queries: [
      { query: 'weak', count: 3, result_count: 4, top_score: 0.213 },
      { query: 'none', count: 1, result_count: 0, top_score: null },
      { query: 'unscored', count: 2, result_count: 2, top_score: null },
    ] })
    expect(rows).toEqual([
      { query: 'weak', count: 3, score: '0.21' },
      { query: 'none', count: 1, score: 'no results' },
      { query: 'unscored', count: 2, score: '—' },
    ])
  })
  it('falls back to zero_queries from older servers', () => {
    expect(gapRows({ zero_queries: [{ query: 'q', count: 2 }] })).toEqual([{ query: 'q', count: 2, score: 'no results' }])
    expect(gapRows(null)).toEqual([])
  })
})

describe('gapDesc', () => {
  it('names the threshold when known', () => {
    expect(gapDesc({ gap_threshold: 0.5 })).toContain('0.5')
    expect(gapDesc({})).toContain('returned nothing')
  })
})
