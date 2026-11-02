import { describe, it, expect } from 'vitest'
import { ndcgOf, headlineNdcg, modeLabel } from './EvalPage'

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

describe('headlineNdcg', () => {
  it('prefers the production mode', () => {
    expect(headlineNdcg({ vector: { 'ndcg@10': 0.9 }, production: { 'ndcg@10': 0.5 } }, 10)).toBe(0.5)
  })
  it('falls back to rerank / rrf for runs from before the production mode', () => {
    expect(headlineNdcg({ rrf: { 'ndcg@10': 0.4 }, hybrid: { 'ndcg@10': 0.9 } }, 10)).toBe(0.4)
    expect(headlineNdcg({ vector: { 'ndcg@10': 0.9 } }, 10)).toBeNull()
  })
})

describe('modeLabel', () => {
  it('uses the report labels, then built-ins, then the raw key', () => {
    expect(modeLabel({ mode_labels: { bm25: 'BM25!' } }, 'bm25')).toBe('BM25!')
    expect(modeLabel(null, 'vector')).toBe('Vector only (dense embeddings)')
    expect(modeLabel({}, 'rrf')).toBe('Hybrid (dense + BM25, RRF fusion, no rerank)')
    expect(modeLabel({}, 'hybrid')).toBe('hybrid')
  })
})
