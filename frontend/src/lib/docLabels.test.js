import { describe, it, expect } from 'vitest'
import { parseTags, formatTags, parseMetadata, formatMetadata } from './docLabels'

describe('tags', () => {
  it('trims, collapses whitespace and drops case-insensitive duplicates', () => {
    expect(parseTags(' invoice,  Signed   copy ,INVOICE,, ')).toEqual(['invoice', 'Signed copy'])
    expect(parseTags('')).toEqual([])
    expect(parseTags(null)).toEqual([])
  })
  it('round-trips through the form text', () => {
    expect(parseTags(formatTags(['a', 'B c']))).toEqual(['a', 'B c'])
    expect(formatTags(null)).toBe('')
  })
})

describe('metadata', () => {
  it('parses one key: value per line with types', () => {
    expect(parseMetadata('client: Acme\nyear: 2024\nratio: 0.5\npaid: true\ndue: 2024-03-01\n\nnote: a: b')).toEqual({
      value: { client: 'Acme', year: 2024, ratio: 0.5, paid: true, due: '2024-03-01', note: 'a: b' },
    })
  })
  it('keeps quoted values as strings', () => {
    expect(parseMetadata('code: "2024"\nflag: "true"').value).toEqual({ code: '2024', flag: 'true' })
  })
  it('reports lines it cannot read', () => {
    expect(parseMetadata('client Acme').error).toMatch(/Line 1/)
    expect(parseMetadata(': x').error).toMatch(/Line 1/)
    expect(parseMetadata('a: 1\na: 2').error).toMatch(/set twice/)
  })
  it('round-trips, quoting strings that would read as another type', () => {
    const meta = { client: 'Acme', year: 2024, code: '2024', paid: false, due: '2024-03-01' }
    expect(parseMetadata(formatMetadata(meta)).value).toEqual(meta)
    expect(formatMetadata(null)).toBe('')
  })
})
