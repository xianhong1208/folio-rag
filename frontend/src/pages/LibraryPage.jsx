import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ChevronLeft, ChevronRight, SlidersHorizontal, X } from 'lucide-react'
import { get, post } from '../services/api'
import { fmtBytes, fmtDate } from '../lib/format'
import { pillClass, statusLabel } from '../lib/jobs'
import { formatMetadata } from '../lib/docLabels'
import { skippedNote, snippetPlace } from '../lib/searchView'
import {
  PAGE_SIZES, activeFilterCount, chipsFor, clearFilters, contentBody, docId, fileType, isContentMode,
  listBody, pageCount, paramsFromState, railMode, rangeLabel, removeChip, stateFromParams, CONTENT_TOP_K,
} from '../lib/libraryFilters'
import FilterRail from '../components/FilterRail'
import DocDrawer from '../components/DocDrawer'
import BulkEditDialog from '../components/BulkEditDialog'
import EmptyState from '../components/ui/EmptyState'

// Library: every document of the person's folders in one server-paginated table
// (POST /api/documents/search), a filter rail, a content search (POST /api/search by
// document over the same filters), bulk tag / metadata edits and a document panel.
// The filter state lives in the URL (lib/libraryFilters.js).

const SORT_OPTIONS = [
  { v: 'uploaded:desc', label: 'Newest first' },
  { v: 'uploaded:asc', label: 'Oldest first' },
  { v: 'name:asc', label: 'Name A–Z' },
  { v: 'name:desc', label: 'Name Z–A' },
  { v: 'size:desc', label: 'Largest first' },
  { v: 'size:asc', label: 'Smallest first' },
]

function SortTh({ col, label, s, setState, className }) {
  const on = s.sort === col
  const next = on ? (s.order === 'asc' ? 'desc' : 'asc') : (col === 'name' ? 'asc' : 'desc')
  return (
    <th className={'sortable ' + (on ? s.order : '') + (className ? ' ' + className : '')}
      aria-sort={on ? (s.order === 'asc' ? 'ascending' : 'descending') : 'none'}>
      <button type="button" className="th-btn" onClick={() => setState({ ...s, sort: col, order: next, page: 1 })}>
        {label}<span className="arrow">{on && s.order === 'asc' ? '▲' : '▼'}</span>
      </button>
    </th>
  )
}

function TagCell({ d }) {
  const n = Object.keys(d.metadata || {}).length
  if (!(d.tags?.length) && !n) return null
  return (
    <div className="tag-row" title={formatMetadata(d.metadata)}>
      {(d.tags || []).map((t) => <span className="tag-chip" key={t}>{t}</span>)}
      {n > 0 && <span className="meta-count">{n} metadata</span>}
    </div>
  )
}

export default function LibraryPage() {
  const [params, setParams] = useSearchParams()
  const key = params.toString()
  const s = useMemo(() => stateFromParams(new URLSearchParams(key)), [key])
  const setState = useCallback((next) => setParams(paramsFromState(next)), [setParams])

  const [folders, setFolders] = useState(null)
  const [tags, setTags] = useState(null)
  const [keys, setKeys] = useState(null)
  const [data, setData] = useState(null) // { docs, total, content, skipped } | { err }
  const [loading, setLoading] = useState(false)
  const [nonce, setNonce] = useState(0)
  const [sel, setSel] = useState(new Map()) // id -> doc (kept across pages)
  const [open, setOpen] = useState(null) // the document in the panel
  const [bulk, setBulk] = useState(null) // bulk edit kind
  const [sheet, setSheet] = useState(false)
  const [qText, setQText] = useState(s.q)
  useEffect(() => { setQText(s.q) }, [s.q])

  const loadFacets = useCallback(() => {
    get('/api/tags').then((r) => setTags(r.data.tags)).catch(() => setTags([]))
    get('/api/metadata/keys').then((r) => setKeys(r.data.keys)).catch(() => setKeys([]))
  }, [])
  useEffect(() => {
    get('/api/admin/folders').then((r) => setFolders(r.data.folders)).catch(() => setFolders([]))
    loadFacets()
  }, [loadFacets])

  // The phone's filter sheet closes by itself when the window grows past the breakpoint.
  useEffect(() => {
    const onResize = () => { if (railMode(window.innerWidth) === 'rail') setSheet(false) }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const content = isContentMode(s)
  const seq = useRef(0)
  useEffect(() => {
    const my = ++seq.current
    setLoading(true)
    const req = content
      ? post('/api/search', contentBody(s)).then((r) => ({
        docs: r.data.documents || [], total: (r.data.documents || []).length, content: true, skipped: r.data.skipped_folders,
      }))
      : post('/api/documents/search', listBody(s)).then((r) => ({ docs: r.data.documents, total: r.data.total }))
    req.then((d) => { if (seq.current === my) setData(d) })
      .catch((e) => { if (seq.current === my) setData({ err: e.message }) })
      .finally(() => { if (seq.current === my) setLoading(false) })
  }, [key, nonce]) // eslint-disable-line react-hooks/exhaustive-deps

  // A page past the end (documents removed meanwhile): go to the last page.
  useEffect(() => {
    if (data && !data.err && !data.content && data.docs.length === 0 && s.page > 1 && data.total > 0) {
      setState({ ...s, page: pageCount(s, data.total) })
    }
  }, [data]) // eslint-disable-line react-hooks/exhaustive-deps

  const folderName = (id) => (folders || []).find((f) => f.id === id)?.name || `#${id}`
  const chips = chipsFor(s, folderName)
  const docs = data?.docs || []

  const runSearch = () => setState({ ...s, q: qText.trim(), page: 1 })
  const clearSearch = () => { setQText(''); setState({ ...s, q: '', page: 1 }) }

  const toggleSel = (d) => setSel((m) => { const n = new Map(m); const id = docId(d); n.has(id) ? n.delete(id) : n.set(id, d); return n })
  const allOnPage = docs.length > 0 && docs.every((d) => sel.has(docId(d)))
  const togglePage = () => setSel((m) => {
    const n = new Map(m)
    if (allOnPage) docs.forEach((d) => n.delete(docId(d)))
    else docs.forEach((d) => n.set(docId(d), d))
    return n
  })

  const refresh = () => { setNonce((n) => n + 1); loadFacets() }
  const bulkDone = () => { setBulk(null); setSel(new Map()); refresh() }
  const closeDrawer = useCallback(() => setOpen(null), [])
  const saved = (doc) => {
    setData((dt) => (dt && dt.docs ? { ...dt, docs: dt.docs.map((x) => (docId(x) === docId(doc) ? { ...x, ...doc } : x)) } : dt))
    loadFacets()
  }

  const rail = <FilterRail value={s} onChange={setState} tags={tags} keys={keys} folders={folders} />
  const nFilters = activeFilterCount(s)
  const pages = pageCount(s, data?.total)
  const cols = content ? 5 : 7

  return (
    <div id="view-library" className="lib">
      <div className="lib-top">
        <div className="sp-bar lib-search">
          <span className="sp-prompt">&#10095;</span>
          <input className="sp-query" placeholder="search inside documents…" aria-label="Search inside documents" value={qText}
            onChange={(e) => setQText(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') runSearch() }} />
          {s.q && <button className="act-btn" type="button" aria-label="Clear the search" title="Back to the list" onClick={clearSearch}><X size={14} /></button>}
          <button className="btn-cyber sm" type="button" onClick={runSearch} disabled={qText.trim().length > 0 && qText.trim().length < 2}>Search</button>
        </div>
        <button className="btn-ghost lib-filter-btn" type="button" onClick={() => setSheet(true)} aria-expanded={sheet}>
          <SlidersHorizontal size={13} /> Filters{nFilters ? <span className="lib-badge">{nFilters}</span> : null}
        </button>
      </div>

      {chips.length > 0 && (
        <div className="lib-chips" aria-label="Active filters">
          {chips.map((c) => (
            <button key={c.key} type="button" className="lib-chip" title="Remove this filter" onClick={() => setState(removeChip(s, c.key))}>
              <span>{c.label}</span><X size={11} />
            </button>
          ))}
          <button type="button" className="lib-clear" onClick={() => setState(clearFilters(s))}>Clear all</button>
        </div>
      )}

      <div className="lib-body">
        <div className="lib-rail" role="region" aria-label="Filters">{rail}</div>
        <section className="lib-main">
          <div className="lib-bar">
            <div className="lib-count">
              {data?.err ? 'error' : !data ? 'loading…'
                : content ? <>{data.total} document{data.total === 1 ? '' : 's'} ranked by content{data.total >= CONTENT_TOP_K ? ` · top ${CONTENT_TOP_K}` : ''}</>
                  : rangeLabel(s, docs.length, data.total)}
              {loading && data && <span className="lib-busy"> · updating…</span>}
            </div>
            <div className="grow" />
            {!content && (
              <>
                <select className="lib-select" aria-label="Sort" value={`${s.sort}:${s.order}`}
                  onChange={(e) => { const [sort, order] = e.target.value.split(':'); setState({ ...s, sort, order, page: 1 }) }}>
                  {SORT_OPTIONS.map((o) => <option key={o.v} value={o.v}>{o.label}</option>)}
                </select>
                <select className="lib-select" aria-label="Page size" value={s.size}
                  onChange={(e) => setState({ ...s, size: Number(e.target.value), page: 1 })}>
                  {PAGE_SIZES.map((n) => <option key={n} value={n}>{n} / page</option>)}
                </select>
              </>
            )}
          </div>
          {data?.skipped?.length ? <div className="sp-skipped">{skippedNote(data.skipped)}</div> : null}

          {sel.size > 0 && (
            <div className="bulkbar lib-bulk">
              <span>{sel.size} selected</span>
              <div className="grow" />
              <button className="btn-ghost sm" onClick={() => setBulk('add_tags')}>Add tags</button>
              <button className="btn-ghost sm" onClick={() => setBulk('remove_tags')}>Remove tags</button>
              <button className="btn-ghost sm" onClick={() => setBulk('set_meta')}>Set metadata</button>
              <button className="btn-ghost sm" onClick={() => setBulk('clear_meta')}>Clear metadata</button>
              <button className="act-btn" title="Clear selection" aria-label="Clear selection" onClick={() => setSel(new Map())}><X size={14} /></button>
            </div>
          )}

          <div className={'tablewrap lib-grid' + (content ? ' content' : '')}>
            <table>
              <thead><tr>
                <th className="selcol"><input type="checkbox" className="selcheck" aria-label="Select this page"
                  checked={allOnPage} onChange={togglePage} disabled={!docs.length} /></th>
                {content ? <th>Document</th> : <SortTh col="name" label="Document" s={s} setState={setState} />}
                <th className="c-folder">Folder</th>
                {content ? <><th className="num c-hits">Hits</th><th className="num c-score">Score</th></> : <>
                  <th className="c-type">Type</th>
                  <SortTh col="size" label="Size" s={s} setState={setState} className="c-size" />
                  <SortTh col="uploaded" label="Uploaded" s={s} setState={setState} className="c-up" />
                  <th className="statuscol c-status">Status</th>
                </>}
              </tr></thead>
              <tbody>
                {data == null ? <tr><td className="empty" colSpan={cols}>loading…</td></tr>
                  : data.err ? <tr><td colSpan={cols}><EmptyState t={content ? 'Search failed' : 'Could not load the documents'} d={data.err} /></td></tr>
                    : docs.length === 0 ? <tr><td colSpan={cols}>
                      <EmptyState t={content ? 'No document matches' : (nFilters ? 'No document matches these filters' : 'No documents yet')}
                        d={content ? 'Try other words, or loosen the filters.' : (nFilters ? 'Remove a filter, or clear them all.' : 'Upload files into a folder (Folders) and they show up here.')} />
                    </td></tr>
                      : docs.map((d) => {
                        const id = docId(d)
                        const sn = d.snippets?.[0]
                        return (
                          <tr key={id} className={'rowlink' + (sel.has(id) ? ' sel' : '')} tabIndex={0}
                            onClick={() => setOpen(d)} onKeyDown={(e) => { if (e.key === 'Enter') setOpen(d) }}>
                            <td className="selcol" onClick={(e) => e.stopPropagation()}>
                              <input type="checkbox" className="selcheck" aria-label={'Select ' + d.name} checked={sel.has(id)} onChange={() => toggleSel(d)} />
                            </td>
                            <td className="cell-main lib-name">
                              <span className="lib-doc">{d.name}</span>
                              <div className="lib-phone sub">{d.folder?.name}{content ? ` · ${d.hits} hit${d.hits === 1 ? '' : 's'} · ${d.score != null ? d.score.toFixed(3) : '—'}` : ` · ${fmtBytes(d.size_bytes)} · ${fmtDate(d.uploaded_at)}`}</div>
                              <TagCell d={d} />
                              {sn && (
                                <div className="lib-snippet">
                                  {snippetPlace(sn) && <span className="c-meta">{snippetPlace(sn)} · </span>}{sn.text}
                                </div>
                              )}
                            </td>
                            <td className="c-folder">{d.folder?.name}</td>
                            {content ? <>
                              <td className="num c-hits">{d.hits}</td>
                              <td className="num c-score">{d.score != null ? d.score.toFixed(3) : '—'}</td>
                            </> : <>
                              <td className="c-type"><span className="lib-type">{fileType(d)}</span></td>
                              <td className="num c-size">{fmtBytes(d.size_bytes)}</td>
                              <td className="num c-up">{fmtDate(d.uploaded_at)}</td>
                              <td className="statuscol c-status"><span className={'pill ' + pillClass(d.status)}>{statusLabel(d.status)}</span></td>
                            </>}
                          </tr>
                        )
                      })}
              </tbody>
            </table>
          </div>

          {!content && data && !data.err && data.total > s.size && (
            <div className="lib-pager">
              <button className="act-btn" aria-label="Previous page" disabled={s.page <= 1} onClick={() => setState({ ...s, page: s.page - 1 })}><ChevronLeft size={14} /></button>
              <span>page {s.page} of {pages}</span>
              <button className="act-btn" aria-label="Next page" disabled={s.page >= pages} onClick={() => setState({ ...s, page: s.page + 1 })}><ChevronRight size={14} /></button>
            </div>
          )}
        </section>
      </div>

      {sheet && (
        <div className="sheet-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) setSheet(false) }}>
          <div className="sheet" role="dialog" aria-label="Filters">
            <div className="sheet-head">
              <b>Filters</b>{nFilters ? <span className="lib-badge">{nFilters}</span> : null}
              <span className="grow" />
              {nFilters > 0 && <button type="button" className="lib-clear" onClick={() => setState(clearFilters(s))}>Clear all</button>}
              <button type="button" className="btn-cyber sm" onClick={() => setSheet(false)}>Done</button>
            </div>
            <div className="sheet-body">{rail}</div>
          </div>
        </div>
      )}
      {open && <DocDrawer doc={open} onClose={closeDrawer} onSaved={saved} />}
      {bulk && <BulkEditDialog kind={bulk} docs={[...sel.values()]} keys={keys} onClose={() => setBulk(null)} onDone={bulkDone} />}
    </div>
  )
}
