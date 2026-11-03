import { describe, it, expect } from 'vitest'
import { META, PRESETS } from './settings-config'
import { TABS, tabFromParam, tabOfSection, diffChanges, changesPerTab, stagePreset, needsReindex, showValue } from './settings-changes'

const state = {
  'rag.retrieval.default_top_k': { value: 10 },
  'rag.retrieval.default_similarity_cutoff': { value: 0.25 },
  'rag.embedding.model': { value: 'intfloat/multilingual-e5-large' },
  'rag.rerank.enabled': { value: true },
  'rag.llm.api_key': { value: '•••', is_set: true },
}

describe('tabs', () => {
  it('every settings section is on exactly one tab', () => {
    const placed = TABS.flatMap((t) => t.sections)
    expect([...placed].sort()).toEqual(Object.keys(META).sort())
    expect(new Set(placed).size).toBe(placed.length)
  })
  it('old /settings/<section> links land on the right tab', () => {
    expect(tabFromParam('embedding')).toBe('models')
    expect(tabFromParam('reranker')).toBe('retrieval')
    expect(tabFromParam('asr')).toBe('speech')
    expect(tabFromParam('parsing')).toBe('indexing')
    expect(tabFromParam('retrieval')).toBe('retrieval')
    expect(tabFromParam('settings')).toBe('models')
    expect(tabFromParam(undefined)).toBe('models')
  })
  it('maps sections to tabs', () => {
    expect(tabOfSection('rag.rerank')).toBe('retrieval')
    expect(tabOfSection('rag.answer_llm')).toBe('models')
  })
})

describe('pending changes', () => {
  it('lists real changes with from/to, in field order, across tabs', () => {
    const c = diffChanges({ 'rag.retrieval.default_top_k': 12, 'rag.embedding.model': 'BAAI/bge-m3' }, state)
    expect(c.map((x) => [x.path, x.from, x.to, x.tab])).toEqual([
      ['rag.embedding.model', 'intfloat/multilingual-e5-large', 'BAAI/bge-m3', 'models'],
      ['rag.retrieval.default_top_k', 10, 12, 'retrieval'],
    ])
    expect(changesPerTab(c)).toEqual({ models: 1, retrieval: 1 })
  })
  it('an edit typed back to the saved value is not a change', () => {
    expect(diffChanges({ 'rag.retrieval.default_top_k': 10 }, state)).toEqual([])
  })
  it('ignores paths the server did not report', () => {
    expect(diffChanges({ 'rag.retrieval.default_top_k': 3, 'rag.asr.model': 'x' }, state).map((c) => c.path))
      .toEqual(['rag.retrieval.default_top_k'])
  })
  it('only embedding changes need the re-index warning', () => {
    expect(needsReindex(diffChanges({ 'rag.retrieval.default_top_k': 12 }, state))).toBe(false)
    expect(needsReindex(diffChanges({ 'rag.embedding.model': 'BAAI/bge-m3' }, state))).toBe(true)
  })
})

describe('presets', () => {
  it('stage into the pending set instead of being saved, overriding edits to the same field', () => {
    const d = stagePreset({ 'rag.retrieval.default_top_k': 7, 'rag.embedding.model': 'x' }, PRESETS.recall.patch)
    expect(d['rag.retrieval.default_top_k']).toBe(PRESETS.recall.patch['rag.retrieval.default_top_k'])
    expect(d['rag.embedding.model']).toBe('x')
  })
})

describe('showValue', () => {
  it('never echoes a secret', () => {
    expect(showValue('sk-live-123', true)).toBe('new key')
    expect(showValue('', true)).toBe('unset')
  })
  it('formats plain values', () => {
    expect(showValue(true)).toBe('on')
    expect(showValue(null)).toBe('—')
    expect(showValue('a'.repeat(60))).toHaveLength(48)
  })
})
