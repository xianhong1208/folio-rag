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

import { normalizeValue, keyIndicator, runsLabel, probeText } from './settings-config'

describe('per-role settings', () => {
  it('every probing section names a backend role', () => {
    const roles = ['embedding', 'llm', 'contextual_llm', 'answer_llm', 'eval_llm', 'rerank', 'vision', 'asr']
    for (const meta of Object.values(META)) if (meta.probe) expect(roles).toContain(meta.probe)
  })
  it('every api_key field is a secret input', () => {
    for (const [path, f] of Object.entries(FIELDS)) if (path.endsWith('.api_key')) expect(f.type).toBe('secret')
  })
  it('empty nullable values mean unset (inherit / follow the profile)', () => {
    expect(normalizeValue(FIELDS['rag.answer_llm.model'], '')).toBeNull()
    expect(normalizeValue(FIELDS['rag.parsing.pdf'], '')).toBeNull()
    expect(normalizeValue(FIELDS['rag.llm.model'], '')).toBe('')
    expect(normalizeValue(undefined, 'x')).toBe('x')
  })
})

describe('keyIndicator', () => {
  it('never shows a key, only where it comes from', () => {
    expect(keyIndicator(undefined)).toEqual({ cls: 'mute', text: 'Not set' })
    expect(keyIndicator({ is_set: false })).toEqual({ cls: 'mute', text: 'Not set' })
    expect(keyIndicator({ is_set: true, env: 'OPENAI_API_KEY' })).toEqual({ cls: 'ok', text: 'Set · from $OPENAI_API_KEY' })
    expect(keyIndicator({ is_set: true, env: null, value: 'sk-x' }).text).toBe('Set · literal value')
  })
})

describe('runsLabel / probeText', () => {
  it('labels where a role runs', () => {
    expect(runsLabel({ runs: 'off' }).text).toBe('Off')
    expect(runsLabel({ runs: 'in-process' }).text).toBe('In-process')
    expect(runsLabel({ runs: 'endpoint' }).text).toBe('Endpoint')
    expect(runsLabel(null).text).toBe('Off')
  })
  it('renders probe results', () => {
    expect(probeText(null)).toEqual({ cls: '', text: '' })
    expect(probeText({ status: 'ok', detail: '1024-dim vector', latency_ms: 42 })).toEqual({ cls: 'ok', text: '✓ 1024-dim vector · 42 ms' })
    expect(probeText({ status: 'local', detail: 'Local weights found' }).cls).toBe('ok')
    expect(probeText({ status: 'disabled', detail: 'Disabled in settings' })).toEqual({ cls: '', text: 'Disabled in settings' })
    expect(probeText({ status: 'down', detail: 'HTTP 401', latency_ms: null })).toEqual({ cls: 'bad', text: '✗ HTTP 401' })
  })
})
