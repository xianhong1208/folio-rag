import { describe, it, expect } from 'vitest'
import { parseMarkdownTable } from './ChunkText'

describe('parseMarkdownTable', () => {
  it('parses a GFM table with leading/trailing pipes', () => {
    const md = [
      '| Name | Qty |',
      '| --- | --- |',
      '| Apple | 3 |',
      '| Pear | 5 |',
    ].join('\n')
    const t = parseMarkdownTable(md)
    expect(t.header).toEqual(['Name', 'Qty'])
    expect(t.body).toEqual([['Apple', '3'], ['Pear', '5']])
  })

  it('parses a table without outer pipes', () => {
    const md = 'a | b\n--- | ---\n1 | 2'
    const t = parseMarkdownTable(md)
    expect(t.header).toEqual(['a', 'b'])
    expect(t.body).toEqual([['1', '2']])
  })

  it('ignores non-table lines around the table', () => {
    const md = 'Caption text\n| x | y |\n|---|---|\n| 1 | 2 |'
    const t = parseMarkdownTable(md)
    expect(t.header).toEqual(['x', 'y'])
    expect(t.body).toEqual([['1', '2']])
  })

  it('returns null for plain prose (no separator row)', () => {
    expect(parseMarkdownTable('just a sentence with | a pipe')).toBeNull()
    expect(parseMarkdownTable('no pipes at all')).toBeNull()
  })

  it('returns null for a header + separator but no body', () => {
    expect(parseMarkdownTable('| a | b |\n| --- | --- |')).toBeNull()
  })
})
