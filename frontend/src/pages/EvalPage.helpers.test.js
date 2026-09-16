import { describe, it, expect } from 'vitest'
import { ndcgOf } from './EvalPage'

// ndcgOf backs the baseline delta: it must read one *specific* mode's nDCG@k so
// deltas compare like-for-like across runs (not the current best vs a different
// mode in the previous run).
describe('ndcgOf', () => {
  const summary = {
    rerank: { 'ndcg@10': 0.6 },
    rrf: { 'ndcg@10': 0.7 },
  }

  it('returns the requested mode’s nDCG@k', () => {
    expect(ndcgOf(summary, 'rrf', 10)).toBe(0.7)
    expect(ndcgOf(summary, 'rerank', 10)).toBe(0.6)
  })

  it('returns null when the mode is absent (so no bogus delta is shown)', () => {
    expect(ndcgOf(summary, 'vector', 10)).toBeNull()
  })

  it('returns null when the k is absent', () => {
    expect(ndcgOf(summary, 'rrf', 5)).toBeNull()
  })

  it('returns null for a missing summary', () => {
    expect(ndcgOf(null, 'rrf', 10)).toBeNull()
    expect(ndcgOf(undefined, 'rrf', 10)).toBeNull()
  })

  it('compares the same mode across runs (regression for the delta bug)', () => {
    // prev best was rerank(0.6); this run rrf wins(0.75). The delta must be
    // rrf-vs-rrf (0.75 - 0.70), never rrf-vs-rerank (0.75 - 0.60).
    const prev = { rerank: { 'ndcg@10': 0.6 }, rrf: { 'ndcg@10': 0.7 } }
    const curBestMode = 'rrf'
    expect(ndcgOf(prev, curBestMode, 10)).toBe(0.7)
  })
})
