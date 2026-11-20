// Library page: the filter state, its URL form, the document filter the API takes
// (docs/api.md#document-filters), the chips that show it, and the bulk edits.
//
// The URL is the single source of truth, so a filtered view can be bookmarked:
//   /library?tag=invoice&tag=signed&tagmode=all&folder=3&type=pdf&from=2024-01-01
//     &to=2024-12-31&name=q3&m=year:gte:2024&q=payment+terms&sort=name&order=asc&page=2&size=100
// Defaults are left out of the URL.

import { parseTags } from './docLabels'

export const PAGE_SIZES = [25, 50, 100, 200]
export const SORTS = ['uploaded', 'name', 'size']
// Metadata operators offered in a condition row. "missing" is exists: false.
export const META_OPS = [
  { op: 'eq', label: '=' },
  { op: 'in', label: 'one of' },
  { op: 'gte', label: '≥' },
  { op: 'lte', label: '≤' },
  { op: 'exists', label: 'is set' },
  { op: 'missing', label: 'is not set' },
]
const OPS = new Set(META_OPS.map((o) => o.op))
export const opTakesValue = (op) => op !== 'exists' && op !== 'missing'
// File types offered in the rail (extensions; the API also takes MIME types).
export const FILE_TYPES = ['pdf', 'docx', 'doc', 'md', 'txt', 'html', 'xlsx', 'pptx', 'csv', 'rtf']
// Below this width the rail becomes a sheet behind a "Filters" button.
export const RAIL_BREAKPOINT = 900

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/

export function emptyState() {
  return {
    tags: [], tagMode: 'any', folders: [], types: [], from: '', to: '', name: '',
    meta: [], q: '', sort: 'uploaded', order: 'desc', page: 1, size: 50,
  }
}

const uniq = (xs) => [...new Set(xs)]

// "year:gte:2024" -> {key, op, value}; the value keeps any further colons.
export function parseMetaParam(s) {
  const a = String(s).indexOf(':')
  if (a <= 0) return null
  const b = String(s).indexOf(':', a + 1)
  const key = s.slice(0, a)
  const op = b < 0 ? s.slice(a + 1) : s.slice(a + 1, b)
  const value = b < 0 ? '' : s.slice(b + 1)
  if (!OPS.has(op)) return null
  return { key, op, value }
}

export const metaParam = (r) => `${r.key}:${r.op}${opTakesValue(r.op) ? ':' + r.value : ''}`

const posInt = (v, dflt) => {
  const n = Number(v)
  return Number.isInteger(n) && n > 0 ? n : dflt
}

// URLSearchParams -> state (unknown or malformed values fall back to the defaults).
export function stateFromParams(params) {
  const s = emptyState()
  s.tags = uniq(params.getAll('tag').map((t) => t.trim()).filter(Boolean))
  s.tagMode = params.get('tagmode') === 'all' ? 'all' : 'any'
  s.folders = uniq(params.getAll('folder').map(Number).filter((n) => Number.isInteger(n) && n > 0))
  s.types = uniq(params.getAll('type').map((t) => t.trim().toLowerCase()).filter(Boolean))
  const from = params.get('from') || ''
  const to = params.get('to') || ''
  s.from = DATE_RE.test(from) ? from : ''
  s.to = DATE_RE.test(to) ? to : ''
  s.name = params.get('name') || ''
  s.meta = params.getAll('m').map(parseMetaParam).filter(Boolean)
  s.q = params.get('q') || ''
  s.sort = SORTS.includes(params.get('sort')) ? params.get('sort') : 'uploaded'
  s.order = params.get('order') === 'asc' ? 'asc' : 'desc'
  s.page = posInt(params.get('page'), 1)
  s.size = PAGE_SIZES.includes(Number(params.get('size'))) ? Number(params.get('size')) : 50
  return s
}

// state -> URLSearchParams (defaults left out).
export function paramsFromState(s) {
  const p = new URLSearchParams()
  for (const t of s.tags) p.append('tag', t)
  if (s.tags.length > 1 && s.tagMode === 'all') p.set('tagmode', 'all')
  for (const f of s.folders) p.append('folder', String(f))
  for (const t of s.types) p.append('type', t)
  if (s.from) p.set('from', s.from)
  if (s.to) p.set('to', s.to)
  if (s.name) p.set('name', s.name)
  for (const r of s.meta) if (r.key) p.append('m', metaParam(r))
  if (s.q) p.set('q', s.q)
  if (s.sort !== 'uploaded') p.set('sort', s.sort)
  if (s.order !== 'desc') p.set('order', s.order)
  if (s.page > 1) p.set('page', String(s.page))
  if (s.size !== 50) p.set('size', String(s.size))
  return p
}

// A typed metadata value: true/false, plain numbers, a "quoted" string kept as text.
export function metaValue(raw) {
  const v = String(raw ?? '').trim()
  if (v.length >= 2 && v.startsWith('"') && v.endsWith('"')) return v.slice(1, -1)
  if (v === 'true') return true
  if (v === 'false') return false
  if (/^-?\d+(\.\d+)?$/.test(v)) return Number(v)
  return v
}

// A complete row only: a key, and a value when the operator takes one.
export const rowComplete = (r) => !!(r && r.key && OPS.has(r.op) && (!opTakesValue(r.op) || String(r.value ?? '').trim()))

// Condition rows -> the filter's metadata object. Several rows on one key are merged
// into one condition (an "=" then becomes "in": [value]); the last row of an op wins.
export function metaConditions(rows) {
  const out = {}
  for (const r of rows || []) {
    if (!rowComplete(r)) continue
    let cond
    if (r.op === 'eq') cond = { eq: metaValue(r.value) }
    else if (r.op === 'in') {
      const vals = String(r.value).split(',').map((x) => x.trim()).filter(Boolean).map(metaValue)
      if (!vals.length) continue
      cond = { in: vals }
    } else if (r.op === 'exists') cond = { exists: true }
    else if (r.op === 'missing') cond = { exists: false }
    else cond = { [r.op]: metaValue(r.value) }
    out[r.key] = { ...(out[r.key] || {}), ...cond }
  }
  const meta = {}
  for (const [k, c] of Object.entries(out)) {
    const ops = Object.keys(c)
    if (ops.length === 1 && ops[0] === 'eq') meta[k] = c.eq
    else if ('eq' in c) {
      const { eq, ...rest } = c
      meta[k] = { ...rest, in: [...(rest.in || []), eq] }
    } else meta[k] = c
  }
  return meta
}

// "2024-12-31" -> "2025-01-01" (the API's uploaded_before is exclusive; the rail's
// "to" date is inclusive).
export function nextDay(d) {
  if (!DATE_RE.test(d || '')) return null
  const t = new Date(d + 'T00:00:00Z')
  t.setUTCDate(t.getUTCDate() + 1)
  return t.toISOString().slice(0, 10)
}

// state -> the API's document filter, or null when nothing is set.
export function toFilter(s) {
  const f = {}
  if (s.tags?.length) f[s.tagMode === 'all' ? 'tags_all' : 'tags_any'] = [...s.tags]
  if (s.folders?.length) f.folders = [...s.folders]
  if (s.types?.length) f.file_type = [...s.types]
  if (s.from) f.uploaded_after = s.from
  if (s.to && nextDay(s.to)) f.uploaded_before = nextDay(s.to)
  if (s.name?.trim()) f.name_contains = s.name.trim()
  const meta = metaConditions(s.meta)
  if (Object.keys(meta).length) f.metadata = meta
  return Object.keys(f).length ? f : null
}

// Body of POST /api/documents/search (the plain listing).
export function listBody(s) {
  const body = { sort: s.sort, order: s.order, limit: s.size, offset: (s.page - 1) * s.size }
  const filters = toFilter(s)
  if (filters) body.filters = filters
  return body
}

// Body of POST /api/search when the content search box is used: documents ranked by
// their best passage, over the same filters.
export const CONTENT_TOP_K = 50
export function contentBody(s) {
  const body = { query: s.q.trim(), group_by: 'document', top_k: CONTENT_TOP_K, snippets: 1 }
  const filters = toFilter(s)
  if (filters) body.filters = filters
  return body
}

// Content mode needs at least 2 characters (the API's minimum).
export const isContentMode = (s) => (s.q || '').trim().length >= 2

const opLabel = (op) => (META_OPS.find((o) => o.op === op) || {}).label || op

// The active filters as removable chips: [{key, label}]. `folderName(id)` names folders.
export function chipsFor(s, folderName = (id) => `#${id}`) {
  const chips = []
  for (const t of s.tags) chips.push({ key: 'tag:' + t, label: (s.tags.length > 1 && s.tagMode === 'all' ? 'tag (all) ' : 'tag ') + t })
  for (const f of s.folders) chips.push({ key: 'folder:' + f, label: 'folder ' + folderName(f) })
  for (const t of s.types) chips.push({ key: 'type:' + t, label: 'type ' + t })
  if (s.from) chips.push({ key: 'from', label: 'from ' + s.from })
  if (s.to) chips.push({ key: 'to', label: 'to ' + s.to })
  if (s.name) chips.push({ key: 'name', label: 'name ~ ' + s.name })
  s.meta.forEach((r, i) => {
    if (!rowComplete(r)) return
    chips.push({ key: 'm:' + i, label: `${r.key} ${opLabel(r.op)}${opTakesValue(r.op) ? ' ' + r.value : ''}` })
  })
  return chips
}

// The state without one chip's condition (back to page 1).
export function removeChip(s, key) {
  const n = { ...s, page: 1 }
  const [kind, ...rest] = key.split(':')
  const val = rest.join(':')
  if (kind === 'tag') n.tags = s.tags.filter((t) => t !== val)
  else if (kind === 'folder') n.folders = s.folders.filter((f) => String(f) !== val)
  else if (kind === 'type') n.types = s.types.filter((t) => t !== val)
  else if (kind === 'from') n.from = ''
  else if (kind === 'to') n.to = ''
  else if (kind === 'name') n.name = ''
  else if (kind === 'm') n.meta = s.meta.filter((_, i) => String(i) !== val)
  return n
}

// Everything the rail sets, cleared (the content query, sort and page size stay).
export const clearFilters = (s) => ({ ...emptyState(), q: s.q, sort: s.sort, order: s.order, size: s.size })

// How many filter conditions are active (the badge on the phone's Filters button).
export const activeFilterCount = (s) => chipsFor(s).length

// 'rail' beside the table, or 'sheet' behind a button on narrow screens.
export const railMode = (width) => (width < RAIL_BREAKPOINT ? 'sheet' : 'rail')

// Toggle a value in a list (tags, folders, types).
export const toggle = (list, v) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v])

// ---- bulk edit ----

const sameTag = (a, b) => a.toLowerCase() === b.toLowerCase()

// One PATCH per selected document whose tags or metadata actually change:
// [{doc, path, body}]. action: {kind: 'add_tags'|'remove_tags', tags: 'a, b'} or
// {kind: 'set_meta', key, value} or {kind: 'clear_meta', key}. PATCH replaces the
// stored list / object, so each body carries the document's whole new value.
export function bulkPatches(docs, action) {
  const out = []
  for (const d of docs) {
    const id = docId(d)
    const path = `/api/admin/manage/folders/${d.folder.id}/files/${id}`
    const tags = Array.isArray(d.tags) ? d.tags : []
    const meta = d.metadata && typeof d.metadata === 'object' ? d.metadata : {}
    if (action.kind === 'add_tags') {
      const add = parseTags(action.tags).filter((t) => !tags.some((x) => sameTag(x, t)))
      if (add.length) out.push({ doc: d, path, body: { tags: [...tags, ...add] } })
    } else if (action.kind === 'remove_tags') {
      const rm = parseTags(action.tags)
      const kept = tags.filter((t) => !rm.some((x) => sameTag(x, t)))
      if (kept.length !== tags.length) out.push({ doc: d, path, body: { tags: kept } })
    } else if (action.kind === 'set_meta') {
      const v = metaValue(action.value)
      if (!action.key || v === '') continue
      if (meta[action.key] !== v) out.push({ doc: d, path, body: { metadata: { ...meta, [action.key]: v } } })
    } else if (action.kind === 'clear_meta') {
      if (action.key && action.key in meta) {
        const { [action.key]: _drop, ...rest } = meta // eslint-disable-line no-unused-vars
        out.push({ doc: d, path, body: { metadata: rest } })
      }
    }
  }
  return out
}

// A document's id: the listing calls it `id`, a search result `file_id`.
export const docId = (d) => String(d.id ?? d.file_id)

// "1–50 of 213"
export function rangeLabel(s, count, total) {
  if (!total) return '0 documents'
  const first = (s.page - 1) * s.size + 1
  return `${first}–${first + count - 1} of ${total}`
}

export const pageCount = (s, total) => Math.max(1, Math.ceil((total || 0) / s.size))

// The file type shown in the table: the extension, else the MIME subtype.
export function fileType(d) {
  const m = /\.([a-z0-9]{1,8})$/i.exec(d.name || '')
  if (m) return m[1].toLowerCase()
  return (d.mime || '').split('/').pop() || '—'
}
