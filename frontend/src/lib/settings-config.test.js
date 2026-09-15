import { describe, it, expect } from 'vitest'
import { sectionOf, META, FIELDS, PRESETS } from './settings-config'

describe('sectionOf', () => {
  it('maps a field path to its longest matching section', () => {
    expect(sectionOf('rag.embedding.model')).toBe('rag.embedding')
    expect(sectionOf('rag.rerank.score_threshold')).toBe('rag.rerank')
  })
  it('prefers the longest prefix for nested paths', () => {
    // 'rag.retrieval' must win even though the path nests deeper.
    expect(sectionOf('rag.retrieval.auto_merging.enabled')).toBe('rag.retrieval')
  })
  it('falls back to the parent path when no section matches', () => {
    expect(sectionOf('foo.bar.baz')).toBe('foo.bar')
  })
})

describe('settings model integrity', () => {
  it('every field resolves to a defined section', () => {
    for (const path of Object.keys(FIELDS)) {
      expect(Object.keys(META)).toContain(sectionOf(path))
    }
  })
  it('every preset patch only targets known fields', () => {
    for (const preset of Object.values(PRESETS)) {
      for (const path of Object.keys(preset.patch)) {
        expect(FIELDS).toHaveProperty(path)
      }
    }
  })
})
