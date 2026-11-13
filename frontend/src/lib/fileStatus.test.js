import { describe, it, expect } from 'vitest'
import { isPartial, needsReindex, reindexNote, countNeedsReindex, folderStaleCount, staleFileIds, batches, RETRY_BATCH } from './fileStatus'

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

describe('folder list stale count', () => {
  it('reads the server count, defaulting to 0', () => {
    expect(folderStaleCount({ reindex_recommended_files: 12 })).toBe(12)
    expect(folderStaleCount({})).toBe(0)
    expect(folderStaleCount(null)).toBe(0)
  })
  it('picks the ids of stale files only', () => {
    const files = [
      { id: 'a', status: 'indexed', reindex_recommended: true },
      { id: 'b', status: 'indexed', reindex_recommended: false },
      { id: 'c', status: 'failed', reindex_recommended: true },
      { id: 'd', status: 'indexed', reindex_recommended: true },
    ]
    expect(staleFileIds(files)).toEqual(['a', 'd'])
    expect(staleFileIds(undefined)).toEqual([])
  })
  it('splits ids into retry-sized batches', () => {
    expect(batches([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]])
    expect(batches([], 2)).toEqual([])
    const ids = Array.from({ length: RETRY_BATCH + 1 }, (_, i) => i)
    expect(batches(ids).map((b) => b.length)).toEqual([RETRY_BATCH, 1])
  })
})

describe('reindexNote', () => {
  it('names the embedding model the file was indexed with', () => {
    const n = reindexNote({ status: 'indexed', reindex_recommended: true, reindex_reason: 'embedding', embedding_model: 'old-emb', embedding_dimension: 768 })
    expect(n).toContain('another embedding model (old-emb, 768 dims)')
  })
  it('falls back to the pipeline note', () => {
    expect(reindexNote({ status: 'indexed', reindex_recommended: true, reindex_reason: 'pipeline' })).toContain('older chunking pipeline')
  })
  it('is empty when no re-index is needed', () => {
    expect(reindexNote({ status: 'indexed', reindex_recommended: false, reindex_reason: null })).toBe('')
  })
})
