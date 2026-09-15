import { describe, it, expect } from 'vitest'
import { parseMarkdownTable, parseTripletTable } from './ChunkText'

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

describe('parseTripletTable (Docling chunker serialization)', () => {
  // Verbatim shape of a real indexed xlsx chunk: contextual prefix + triplets.
  const chunk = 'mm_table_test.xlsx表格列出各伺服器主機名稱、角色與硬體配置細節。\n\n'
    + 'rag-api-01, Role = API. rag-api-01, CPU cores = 16. rag-api-01, RAM (GB) = 64. '
    + 'rag-db-01, Role = PostgreSQL+pgvector. rag-db-01, CPU cores = 32. rag-db-01, RAM (GB) = 256.'

  it('rebuilds rows and columns in first-seen order', () => {
    const t = parseTripletTable(chunk)
    expect(t.header).toEqual(['', 'Role', 'CPU cores', 'RAM (GB)'])
    expect(t.body).toEqual([
      ['rag-api-01', 'API', '16', '64'],
      ['rag-db-01', 'PostgreSQL+pgvector', '32', '256'],
    ])
  })

  it('keeps the contextual prefix as a caption', () => {
    expect(parseTripletTable(chunk).caption).toBe('mm_table_test.xlsx表格列出各伺服器主機名稱、角色與硬體配置細節。')
  })

  it('leaves missing cells blank', () => {
    const t = parseTripletTable('a, x = 1. a, y = 2. b, x = 3.')
    expect(t.body).toEqual([['a', '1', '2'], ['b', '3', '']])
  })

  it('returns null for prose', () => {
    expect(parseTripletTable('Just a sentence, with a comma = but no table.')).toBeNull()
    expect(parseTripletTable('')).toBeNull()
  })
})
