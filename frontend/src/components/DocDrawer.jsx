import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Download, Search, X } from 'lucide-react'
import { api, post } from '../services/api'
import { fmtBytes, fmtDate } from '../lib/format'
import { pillClass, statusLabel } from '../lib/jobs'
import { parseTags, formatTags, parseMetadata, formatMetadata } from '../lib/docLabels'
import { docId, fileType } from '../lib/libraryFilters'
import ChunkText from './ui/ChunkText'
import { snippetPlace } from '../lib/searchView'

// One document of the Library: its details, its tags / metadata / description
// (editable), a download, and a search inside it (a folder search limited by
// filters.file_ids). `doc` is a row of the listing or of a content search; the full
// record is read again by id, so a search result (which has no size or status) shows
// everything too.
export default function DocDrawer({ doc, onClose, onSaved }) {
  const id = docId(doc)
  const [full, setFull] = useState(null)
  const [loadErr, setLoadErr] = useState(null)
  const [form, setForm] = useState(null) // { tags, meta, desc }
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState(null) // { kind: 'ok'|'bad', text }
  const [q, setQ] = useState('')
  const [hits, setHits] = useState(null) // { results, err, running }
  const closeRef = useRef(null)

  useEffect(() => {
    let alive = true
    setFull(null); setLoadErr(null); setForm(null); setMsg(null); setHits(null); setQ('')
    post('/api/documents/search', { filters: { file_ids: [id] }, limit: 1 })
      .then((r) => {
        if (!alive) return
        const d = r.data.documents[0]
        if (!d) { setLoadErr('This document no longer exists.'); return }
        setFull(d)
        setForm({ tags: formatTags(d.tags), meta: formatMetadata(d.metadata), desc: d.description || '' })
      })
      .catch((e) => { if (alive) setLoadErr(e.message) })
    return () => { alive = false }
  }, [id])

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const d = full || doc
  const folder = d.folder || {}
  const base = `/api/admin/manage/folders/${folder.id}/files/${id}`

  const save = async () => {
    const meta = parseMetadata(form.meta)
    if (meta.error) { setMsg({ kind: 'bad', text: meta.error }); return }
    setSaving(true); setMsg(null)
    try {
      await api('PATCH', base, { tags: parseTags(form.tags), metadata: meta.value, description: form.desc.trim() })
      const next = { ...d, tags: parseTags(form.tags), metadata: meta.value, description: form.desc.trim() }
      setFull(next)
      setMsg({ kind: 'ok', text: 'Saved.' })
      onSaved?.(next)
    } catch (e) { setMsg({ kind: 'bad', text: e.message }) }
    setSaving(false)
  }

  const searchInside = async () => {
    if (q.trim().length < 2) return
    setHits({ running: true })
    try {
      const r = await post(`/api/admin/manage/folders/${folder.id}/query`, { query: q.trim(), filters: { file_ids: [id] } })
      setHits({ results: r.data.results || [] })
    } catch (e) { setHits({ err: e.message }) }
  }

  return (
    <div className="drawer-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="drawer" role="dialog" aria-label={'Document ' + d.name}>
        <div className="dr-head">
          <div className="dr-title">
            <div className="dr-kicker">document</div>
            <h3>{d.name}</h3>
          </div>
          <button ref={closeRef} className="act-btn" aria-label="Close" title="Close (Esc)" onClick={onClose}><X size={14} /></button>
        </div>
        <div className="dr-body">
          {loadErr && <div className="dr-msg bad">{loadErr}</div>}
          <dl className="dr-facts">
            <dt>Folder</dt><dd><Link to={`/folders?folder=${folder.id}`}>{folder.name}</Link></dd>
            <dt>Type</dt><dd>{fileType(d)}{d.mime ? <span className="dr-sub"> · {d.mime}</span> : null}</dd>
            <dt>Size</dt><dd>{full ? fmtBytes(full.size_bytes) : '…'}</dd>
            <dt>Status</dt><dd>{full ? <span className={'pill ' + pillClass(full.status)}>{statusLabel(full.status)}</span> : '…'}</dd>
            <dt>Uploaded</dt><dd>{full ? fmtDate(full.uploaded_at) : '…'}</dd>
            <dt>Updated</dt><dd>{full ? fmtDate(full.updated_at) : '…'}</dd>
          </dl>
          <div className="dr-actions">
            <a className="btn-ghost sm" href={base + '/download'} target="_blank" rel="noreferrer"><Download size={13} /> Download</a>
          </div>

          <div className="dr-sec">
            <div className="dr-sec-h">Labels</div>
            {form ? (
              <div className="dr-form">
                <label htmlFor="dr-tags">Tags (comma-separated)</label>
                <input id="dr-tags" placeholder="invoice, signed" value={form.tags} onChange={(e) => setForm({ ...form, tags: e.target.value })} />
                <label htmlFor="dr-meta">Metadata (one key: value per line)</label>
                <textarea id="dr-meta" rows={5} placeholder={'client: Acme\nyear: 2024\nsigned: 2024-03-01'} value={form.meta}
                  onChange={(e) => setForm({ ...form, meta: e.target.value })} />
                <label htmlFor="dr-desc">Description</label>
                <input id="dr-desc" placeholder="optional" value={form.desc} onChange={(e) => setForm({ ...form, desc: e.target.value })} />
                <div className="dr-row">
                  {msg && <span className={'dr-msg ' + msg.kind}>{msg.text}</span>}
                  <span className="grow" />
                  <button className="btn-cyber sm" onClick={save} disabled={saving}>{saving ? 'Saving…' : 'Save'}</button>
                </div>
                <div className="dr-note">Keys: lowercase letters, digits and _. Numbers and true/false are typed; write YYYY-MM-DD for a date, or "quote" a value to keep it as text.</div>
              </div>
            ) : !loadErr && <div className="skel" style={{ height: 120 }} />}
          </div>

          <div className="dr-sec">
            <div className="dr-sec-h">Search inside this document</div>
            <div className="sp-bar dr-search">
              <span className="sp-prompt">&#10095;</span>
              <input className="sp-query" placeholder="what to look for…" value={q} aria-label="Search inside this document"
                onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && searchInside()} />
              <button className="act-btn" aria-label="Search" title="Search" onClick={searchInside} disabled={hits?.running}><Search size={14} /></button>
            </div>
            <div className="dr-hits">
              {hits == null ? null
                : hits.running ? <div className="dr-note">searching…</div>
                  : hits.err ? <div className="dr-msg bad">{hits.err}</div>
                    : hits.results.length === 0 ? <div className="dr-note">No passage of this document matches.</div>
                      : hits.results.map((h, i) => (
                        <div className="chunk-row" key={i}>
                          <div className="c-head">
                            <span className="c-idx">#{i + 1}</span>
                            {snippetPlace({ page: h.metadata?.page, headings: h.metadata?.headings }) &&
                              <span className="c-meta">{snippetPlace({ page: h.metadata?.page, headings: h.metadata?.headings })}</span>}
                            <span className="c-meta" style={{ marginLeft: 'auto' }}>score {h.score != null ? h.score.toFixed(3) : '—'}</span>
                          </div>
                          <ChunkText text={h.text} contentType={h.metadata?.content_type} className="c-text" />
                        </div>
                      ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
