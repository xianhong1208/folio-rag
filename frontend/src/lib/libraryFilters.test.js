import { describe, it, expect } from 'vitest'
import {
  emptyState, stateFromParams, paramsFromState, parseMetaParam, metaConditions, metaValue,
  toFilter, listBody, contentBody, isContentMode, chipsFor, removeChip, clearFilters,
  activeFilterCount, railMode, RAIL_BREAKPOINT, bulkPatches, nextDay, rangeLabel, pageCount,
  fileType, docId, toggle, rowComplete,
} from './libraryFilters'

const P = (qs) => new URLSearchParams(qs)
const S = (over) => ({ ...emptyState(), ...over })

describe('filter state <-> URL (LIB-URL)', () => {
  it('LIB-URL-001 reads every field', () => {
    const s = stateFromParams(P('tag=invoice&tag=signed&tagmode=all&folder=3&folder=7&type=PDF&from=2024-01-01'
      + '&to=2024-12-31&name=q3&m=year:gte:2024&m=client:eq:Acme:West&q=payment+terms&sort=name&order=asc&page=2&size=100'))
    expect(s).toEqual({
      tags: ['invoice', 'signed'], tagMode: 'all', folders: [3, 7], types: ['pdf'],
      from: '2024-01-01', to: '2024-12-31', name: 'q3',
      meta: [{ key: 'year', op: 'gte', value: '2024' }, { key: 'client', op: 'eq', value: 'Acme:West' }],
      q: 'payment terms', sort: 'name', order: 'asc', page: 2, size: 100,
    })
  })
  it('LIB-URL-002 round-trips and leaves defaults out', () => {
    const s = S({ tags: ['a b'], folders: [2], meta: [{ key: 'x', op: 'exists', value: '' }], page: 3 })
    const p = paramsFromState(s)
    expect(p.toString()).toBe('tag=a+b&folder=2&m=x%3Aexists&page=3')
    expect(stateFromParams(p)).toEqual(s)
    expect(paramsFromState(emptyState()).toString()).toBe('')
  })
  it('LIB-URL-003 bad values fall back to defaults', () => {
    const s = stateFromParams(P('folder=abc&folder=-1&folder=4&from=yesterday&to=2024-13&m=nocolon&m=k:gt:1'
      + '&sort=evil&order=up&page=0&size=7&tagmode=most'))
    expect(s.folders).toEqual([4])
    expect(s.from).toBe('')
    expect(s.to).toBe('')
    expect(s.meta).toEqual([])
    expect(s.sort).toBe('uploaded')
    expect(s.order).toBe('desc')
    expect(s.page).toBe(1)
    expect(s.size).toBe(50)
    expect(s.tagMode).toBe('any')
  })
  it('LIB-URL-004 the tag mode only reaches the URL with two tags or more', () => {
    expect(paramsFromState(S({ tags: ['a'], tagMode: 'all' })).has('tagmode')).toBe(false)
    expect(paramsFromState(S({ tags: ['a', 'b'], tagMode: 'all' })).get('tagmode')).toBe('all')
  })
  it('LIB-URL-005 metadata params keep colons in the value and refuse unknown operators', () => {
    expect(parseMetaParam('url:eq:https://x.test/a')).toEqual({ key: 'url', op: 'eq', value: 'https://x.test/a' })
    expect(parseMetaParam('k:missing')).toEqual({ key: 'k', op: 'missing', value: '' })
    expect(parseMetaParam(':eq:1')).toBeNull()
    expect(parseMetaParam('k:like:1')).toBeNull()
  })
})

describe('state -> document filter (LIB-FLT)', () => {
  it('LIB-FLT-001 an empty state is no filter', () => {
    expect(toFilter(emptyState())).toBeNull()
    expect(listBody(emptyState())).toEqual({ sort: 'uploaded', order: 'desc', limit: 50, offset: 0 })
  })
  it('LIB-FLT-002 maps each field', () => {
    expect(toFilter(S({
      tags: ['invoice'], folders: [3], types: ['pdf'], from: '2024-01-01', to: '2024-12-31', name: ' q3 ',
      meta: [{ key: 'year', op: 'gte', value: '2024' }],
    }))).toEqual({
      tags_any: ['invoice'], folders: [3], file_type: ['pdf'], uploaded_after: '2024-01-01',
      uploaded_before: '2025-01-01', name_contains: 'q3', metadata: { year: { gte: 2024 } },
    })
    expect(toFilter(S({ tags: ['a', 'b'], tagMode: 'all' }))).toEqual({ tags_all: ['a', 'b'] })
  })
  it('LIB-FLT-003 the inclusive "to" date becomes the exclusive next day', () => {
    expect(nextDay('2024-02-28')).toBe('2024-02-29')
    expect(nextDay('2023-12-31')).toBe('2024-01-01')
    expect(nextDay('bad')).toBeNull()
  })
  it('LIB-FLT-004 page and size become limit and offset', () => {
    expect(listBody(S({ page: 3, size: 25, sort: 'size', order: 'asc' })))
      .toEqual({ sort: 'size', order: 'asc', limit: 25, offset: 50 })
  })
  it('LIB-FLT-005 content search uses /api/search by document over the same filters', () => {
    const s = S({ q: '  payment terms ', tags: ['signed'] })
    expect(isContentMode(s)).toBe(true)
    expect(isContentMode(S({ q: ' a ' }))).toBe(false)
    expect(contentBody(s)).toEqual({ query: 'payment terms', group_by: 'document', top_k: 50, snippets: 1,
      filters: { tags_any: ['signed'] } })
  })
})

describe('metadata condition rows (LIB-META)', () => {
  it('LIB-META-001 typed values', () => {
    expect(metaValue('2024')).toBe(2024)
    expect(metaValue('"2024"')).toBe('2024')
    expect(metaValue('true')).toBe(true)
    expect(metaValue(' Acme ')).toBe('Acme')
    expect(metaValue('2024-03-01')).toBe('2024-03-01')
  })
  it('LIB-META-002 each operator', () => {
    expect(metaConditions([
      { key: 'client', op: 'eq', value: 'Acme' },
      { key: 'lang', op: 'in', value: 'zh, en,' },
      { key: 'due', op: 'lte', value: '2024-12-31' },
      { key: 'reviewed', op: 'exists', value: '' },
      { key: 'draft', op: 'missing', value: 'ignored' },
    ])).toEqual({
      client: 'Acme', lang: { in: ['zh', 'en'] }, due: { lte: '2024-12-31' },
      reviewed: { exists: true }, draft: { exists: false },
    })
  })
  it('LIB-META-003 incomplete rows are skipped', () => {
    expect(rowComplete({ key: '', op: 'eq', value: '1' })).toBe(false)
    expect(rowComplete({ key: 'k', op: 'gte', value: ' ' })).toBe(false)
    expect(rowComplete({ key: 'k', op: 'exists', value: '' })).toBe(true)
    expect(metaConditions([{ key: 'k', op: 'eq', value: '' }, { key: '', op: 'eq', value: 'x' },
      { key: 'k', op: 'in', value: ' , ' }])).toEqual({})
  })
  it('LIB-META-004 rows on one key merge; an "=" joins them as "in"', () => {
    expect(metaConditions([{ key: 'year', op: 'gte', value: '2020' }, { key: 'year', op: 'lte', value: '2024' }]))
      .toEqual({ year: { gte: 2020, lte: 2024 } })
    expect(metaConditions([{ key: 'year', op: 'eq', value: '2020' }, { key: 'year', op: 'in', value: '2021' }]))
      .toEqual({ year: { in: [2021, 2020] } })
    expect(metaConditions([{ key: 'year', op: 'eq', value: '2020' }, { key: 'year', op: 'exists', value: '' }]))
      .toEqual({ year: { exists: true, in: [2020] } })
  })
})

describe('chips (LIB-CHIP)', () => {
  const s = S({
    tags: ['invoice', 'signed'], tagMode: 'all', folders: [3], types: ['pdf'], from: '2024-01-01', to: '2024-12-31',
    name: 'q3', meta: [{ key: 'year', op: 'gte', value: '2024' }, { key: '', op: 'eq', value: '' }], page: 4,
  })
  it('LIB-CHIP-001 one chip per active condition, folders named', () => {
    expect(chipsFor(s, (id) => (id === 3 ? 'contracts' : '?')).map((c) => c.label)).toEqual([
      'tag (all) invoice', 'tag (all) signed', 'folder contracts', 'type pdf', 'from 2024-01-01',
      'to 2024-12-31', 'name ~ q3', 'year ≥ 2024',
    ])
    expect(activeFilterCount(s)).toBe(8)
    expect(activeFilterCount(emptyState())).toBe(0)
  })
  it('LIB-CHIP-002 removing a chip drops only its condition and goes back to page 1', () => {
    for (const c of chipsFor(s)) {
      const n = removeChip(s, c.key)
      expect(n.page).toBe(1)
      expect(chipsFor(n).map((x) => x.key)).toEqual(chipsFor(s).map((x) => x.key).filter((k) => k !== c.key))
    }
    expect(removeChip(s, 'm:0').meta).toEqual([{ key: '', op: 'eq', value: '' }])
    expect(removeChip(S({ tags: ['a:b', 'c'] }), 'tag:a:b').tags).toEqual(['c'])
  })
  it('LIB-CHIP-003 clear keeps the query, sort and page size', () => {
    const c = clearFilters({ ...s, q: 'x y', sort: 'name', order: 'asc', size: 100 })
    expect(toFilter(c)).toBeNull()
    expect([c.q, c.sort, c.order, c.size, c.page]).toEqual(['x y', 'name', 'asc', 100, 1])
  })
  it('LIB-CHIP-004 toggle', () => {
    expect(toggle(['a'], 'b')).toEqual(['a', 'b'])
    expect(toggle(['a', 'b'], 'a')).toEqual(['b'])
  })
})

describe('rail collapse (LIB-RAIL)', () => {
  it('LIB-RAIL-001 a sheet below the breakpoint, the rail from it up', () => {
    expect(railMode(390)).toBe('sheet')
    expect(railMode(RAIL_BREAKPOINT - 1)).toBe('sheet')
    expect(railMode(RAIL_BREAKPOINT)).toBe('rail')
    expect(railMode(1440)).toBe('rail')
  })
})

describe('bulk edit requests (LIB-BULK)', () => {
  const a = { id: 'a1', name: 'a.pdf', folder: { id: 3 }, tags: ['Invoice'], metadata: { year: 2024 } }
  const b = { file_id: 'b2', name: 'b.md', folder: { id: 7 }, tags: [], metadata: {} }
  it('LIB-BULK-001 add tags: only documents missing one, case-insensitive, existing spelling kept', () => {
    const p = bulkPatches([a, b], { kind: 'add_tags', tags: 'invoice, signed' })
    expect(p.map((x) => [x.path, x.body])).toEqual([
      ['/api/admin/manage/folders/3/files/a1', { tags: ['Invoice', 'signed'] }],
      ['/api/admin/manage/folders/7/files/b2', { tags: ['invoice', 'signed'] }],
    ])
    expect(bulkPatches([a], { kind: 'add_tags', tags: 'INVOICE' })).toEqual([])
  })
  it('LIB-BULK-002 remove tags: only documents that carry one', () => {
    const p = bulkPatches([a, b], { kind: 'remove_tags', tags: 'INVOICE' })
    expect(p.map((x) => [x.doc.name, x.body])).toEqual([['a.pdf', { tags: [] }]])
  })
  it('LIB-BULK-003 set a metadata key keeps the other keys, typed; unchanged documents are skipped', () => {
    const p = bulkPatches([a, b], { kind: 'set_meta', key: 'status', value: 'final' })
    expect(p.map((x) => x.body)).toEqual([{ metadata: { year: 2024, status: 'final' } }, { metadata: { status: 'final' } }])
    expect(bulkPatches([a], { kind: 'set_meta', key: 'year', value: '2024' })).toEqual([])
    expect(bulkPatches([a], { kind: 'set_meta', key: 'year', value: '"2024"' })[0].body).toEqual({ metadata: { year: '2024' } })
    expect(bulkPatches([a], { kind: 'set_meta', key: 'k', value: '  ' })).toEqual([])
    expect(bulkPatches([a], { kind: 'set_meta', key: '', value: 'x' })).toEqual([])
  })
  it('LIB-BULK-004 clear a metadata key: only documents that have it', () => {
    const p = bulkPatches([a, b], { kind: 'clear_meta', key: 'year' })
    expect(p.map((x) => [x.doc.name, x.body])).toEqual([['a.pdf', { metadata: {} }]])
  })
  it('LIB-BULK-005 ids from the listing and from a search result', () => {
    expect(docId(a)).toBe('a1')
    expect(docId(b)).toBe('b2')
  })
})

describe('table helpers (LIB-TBL)', () => {
  it('LIB-TBL-001 range and pages', () => {
    expect(rangeLabel(S({ page: 2, size: 50 }), 50, 213)).toBe('51–100 of 213')
    expect(rangeLabel(S(), 0, 0)).toBe('0 documents')
    expect(pageCount(S({ size: 50 }), 213)).toBe(5)
    expect(pageCount(S(), 0)).toBe(1)
  })
  it('LIB-TBL-002 file type', () => {
    expect(fileType({ name: 'A.PDF' })).toBe('pdf')
    expect(fileType({ name: 'README', mime: 'text/plain' })).toBe('plain')
    expect(fileType({ name: 'x' })).toBe('—')
  })
})
