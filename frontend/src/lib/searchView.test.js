import { describe, expect, it } from 'vitest'
import { ALL_FOLDERS, plural, searchBody, skippedNote, snippetPlace, usesPlayground } from './searchView'

describe('searchView', () => {
  it('uses the folder playground only for one folder and passages', () => {
    expect(usesPlayground('3', 'passages')).toBe(true)
    expect(usesPlayground('3', 'documents')).toBe(false)
    expect(usesPlayground(ALL_FOLDERS, 'passages')).toBe(false)
  })

  it('builds the /api/search body', () => {
    expect(searchBody({ fid: ALL_FOLDERS, view: 'documents', query: 'tea' }))
      .toEqual({ query: 'tea', group_by: 'document' })
    expect(searchBody({ fid: '7', view: 'documents', query: 'tea' }))
      .toEqual({ query: 'tea', group_by: 'document', folders: [7] })
    expect(searchBody({ fid: ALL_FOLDERS, view: 'passages', query: 'tea' }))
      .toEqual({ query: 'tea', group_by: 'passage' })
  })

  it('explains skipped folders', () => {
    expect(skippedNote([])).toBe('')
    expect(skippedNote(undefined)).toBe('')
    expect(skippedNote([{ name: 'a', reason: 'nothing_indexed' }]))
      .toBe('1 folder not searched: a (nothing indexed)')
    expect(skippedNote([{ name: 'a', reason: 'embedding_model' }, { name: 'b', reason: 'new_reason' }]))
      .toBe('2 folders not searched: a (other embedding model, re-index), b (new_reason)')
  })

  it('formats counts and snippet places', () => {
    expect(plural(1, 'passage')).toBe('1 passage')
    expect(plural(3, 'passage')).toBe('3 passages')
    expect(snippetPlace({ page: 3, headings: ['Intro', 'Scope'] })).toBe('p.3 · Intro › Scope')
    expect(snippetPlace({ page: null, headings: [] })).toBe('')
    expect(snippetPlace(null)).toBe('')
  })
})
