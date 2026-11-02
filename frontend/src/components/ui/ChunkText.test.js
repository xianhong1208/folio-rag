import { describe, it, expect } from 'vitest'
import { parseMarkdownTable, parseTripletTable, isTripletLine, splitChunkSegments } from './ChunkText'

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
  // Verbatim shape of a real indexed xlsx chunk: contextual prefix + triplets,
  // joined with ". " and with NO trailing period after the last triplet (Docling's
  // real output — the ". ".join produces no terminal period).
  const chunk = 'mm_table_test.xlsx表格列出各伺服器主機名稱、角色與硬體配置細節。\n\n'
    + 'rag-api-01, Role = API. rag-api-01, CPU cores = 16. rag-api-01, RAM (GB) = 64. '
    + 'rag-db-01, Role = PostgreSQL+pgvector. rag-db-01, CPU cores = 32. rag-db-01, RAM (GB) = 256'

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

  // Regression: real Docling data contains periods in row labels and values, and
  // never terminates with a period. The previous regex parser silently dropped or
  // collided on all three of these.
  it('keeps the final cell when there is no trailing period', () => {
    const t = parseTripletTable('r1, A = x. r1, B = y')
    expect(t.body).toEqual([['r1', 'x', 'y']])
  })

  it('does not collide float-rendered row labels (1.0 vs 2.0)', () => {
    const t = parseTripletTable('1.0, Status = Active. 2.0, Status = Idle')
    expect(t.header).toEqual(['', 'Status'])
    expect(t.body).toEqual([['1.0', 'Active'], ['2.0', 'Idle']])
  })

  it('keeps decimal cell values intact', () => {
    const t = parseTripletTable('Server, RAM = 64.5. Server, CPU = 8')
    expect(t.body).toEqual([['Server', '64.5', '8']])
  })

  it('returns null for prose', () => {
    expect(parseTripletTable('Just a sentence, with a comma = but no table.')).toBeNull()
    expect(parseTripletTable('')).toBeNull()
  })
})

describe('splitChunkSegments (prose merged with a table in one section)', () => {
  it('keeps a lead-in paragraph as text and the triplet line as a table', () => {
    const segs = splitChunkSegments('Power consumption depends on the load:\nIdle, Consumption = 11 W. Peak, Consumption = 18 W')
    expect(segs.map((s) => s.type)).toEqual(['text', 'table'])
    expect(segs[0].text).toBe('Power consumption depends on the load:')
    expect(segs[1].table.body).toEqual([['Idle', '11 W'], ['Peak', '18 W']])
  })

  it('handles text after the table and the old prefix + blank line layout', () => {
    const segs = splitChunkSegments('Prefix sentence.\n\nA, x = 1. B, x = 2\nFootnote.')
    expect(segs.map((s) => s.type)).toEqual(['text', 'table', 'text'])
    expect(segs[2].text).toBe('Footnote.')
  })

  it('does not turn prose with an equals sign into a table', () => {
    expect(isTripletLine('Set the value x = 3, then restart.')).toBe(false)
    expect(isTripletLine('Basic, Fee = 15. Pro, Fee = 35')).toBe(true)
    expect(splitChunkSegments('Set the value x = 3, then restart.').map((s) => s.type)).toEqual(['text'])
  })
})
