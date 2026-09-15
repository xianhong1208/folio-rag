import { useEffect, useState, useCallback, useRef } from 'react'
import { Folder, Play, Pencil, Trash2, Download, Eye, X, RefreshCw } from 'lucide-react'
import { get, post, api } from '../services/api'
import { fmtNum, fmtBytes, fmtTime, shortModel } from '../lib/format'
import EmptyState from '../components/ui/EmptyState'
import ChunkText, { ContentBadge } from '../components/ui/ChunkText'
import { useModal } from '../contexts/ModalContext'
import { useToast } from '../contexts/ToastContext'
import { useI18n } from '../contexts/I18nContext'

function statusPill(s) {
  const map = { indexed: 'ok', failed: 'bad', running: 'run', pending: 'warn', queued: 'warn', unindexed: 'mute' }
  const cls = map[s] || 'run'
  return <span className={'pill ' + cls}>{s.charAt(0).toUpperCase() + s.slice(1)}</span>
}

export default function FoldersPage() {
  const { confirm, form } = useModal()
  const toast = useToast()
  const { t } = useI18n()
  const [folders, setFolders] = useState(null)
  const [q, setQ] = useState('')
  const [cur, setCur] = useState(null) // {id,name} when viewing files
  const [files, setFiles] = useState(null)
  const [fq, setFq] = useState('')
  const [sel, setSel] = useState(new Set())
  const [chunks, setChunks] = useState(null) // { name, items } | null
  const fileInput = useRef(null)

  const loadFolders = useCallback(async () => {
    const r = await get('/api/admin/folders'); setFolders(r.data.folders)
  }, [])
  useEffect(() => { loadFolders().catch(() => {}) }, [loadFolders])

  const loadFiles = useCallback(async (fid) => {
    const [r, sg] = await Promise.all([
      get(`/api/admin/folders/${fid}/files`),
      get('/api/admin/active-stages').catch(() => ({ data: { stages: {} } })),
    ])
    const stages = (sg.data && sg.data.stages) || {}
    const fs = r.data.files.map((f) => {
      const st = stages[f.id]
      if (st && f.status !== 'indexed') {
        f._stage = st.stage || 'queued'
        if (st.done != null && st.total) f._stagePct = Math.round(st.done / st.total * 100)
      }
      return f
    })
    setFiles(fs)
    return fs
  }, [])

  // poll while files parse — React reconciles, so no flicker
  useEffect(() => {
    if (!cur) return
    let alive = true
    let timer = null
    const tick = async () => {
      try {
        const fs = await loadFiles(cur.id)
        const pending = fs.some((f) => f._stage || ['unindexed', 'running', 'pending'].includes(f.status))
        if (alive && pending) timer = setTimeout(tick, 1500)
      } catch { /* ignore */ }
    }
    tick()
    return () => { alive = false; if (timer) clearTimeout(timer) }
  }, [cur, loadFiles])

  const openFolder = (f) => { setCur({ id: f.id, name: f.name }); setFiles(null); setSel(new Set()) }
  const back = () => { setCur(null); setFiles(null); setSel(new Set()); loadFolders() }

  const toggleSel = (id) => setSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n })
  const bulkReindex = async () => {
    const ids = [...sel]
    try { await Promise.all(ids.map((id) => post(`/api/admin/manage/folders/${cur.id}/files/${id}/retry`, {}))); toast(`Reindexing ${ids.length} file(s)`, 'ok'); setSel(new Set()); loadFiles(cur.id) }
    catch (e) { toast('Reindex failed: ' + e.message, 'bad') }
  }
  const bulkDelete = async () => {
    const ids = [...sel]
    const ok = await confirm({ title: `Delete ${ids.length} file(s)?`, action: 'Delete', danger: true, note: 'Removes the selected files and their index.' })
    if (!ok) return
    try { await Promise.all(ids.map((id) => api('DELETE', `/api/admin/manage/folders/${cur.id}/files/${id}`))); toast(`${ids.length} file(s) deleted`, 'ok'); setSel(new Set()); loadFiles(cur.id) }
    catch (e) { toast('Delete failed: ' + e.message, 'bad') }
  }
  const viewChunks = async (f) => {
    setChunks({ name: f.name, items: null })
    try { const r = await get(`/api/admin/folders/${cur.id}/files/${f.id}/chunks?limit=500`); setChunks({ name: f.name, items: r.data.chunks || [] }) }
    catch (e) { setChunks({ name: f.name, items: [], err: e.message }) }
  }

  const createFolder = async () => {
    const r = await form({
      title: 'New Folder', action: 'Create',
      fields: [
        { id: 'name', label: 'Name', placeholder: 'e.g. product-docs', required: true },
        { id: 'owner_token', label: 'Owner Token', placeholder: 'user token that owns this folder', required: true },
      ],
      note: 'Folders are token-scoped — users access this folder with this token via REST / MCP.',
    })
    if (!r) return
    try { await post('/api/admin/manage/folders', { name: r.name.trim(), owner_token: r.owner_token.trim() }); toast('Folder created', 'ok'); loadFolders() }
    catch (e) { toast('Create failed: ' + e.message, 'bad') }
  }
  const reindexFolder = async (fid, skip) => {
    try { const r = await post(`/api/admin/manage/folders/${fid}/index`, { skip_existing: skip }); toast(r.message || 'Indexing started', 'ok') }
    catch (e) { toast('Reindex failed: ' + e.message, 'bad') }
  }
  const [rebuilding, setRebuilding] = useState(false)
  const rebuildFts = async () => {
    const ok = await confirm({
      title: 'Rebuild full-text search?', action: 'Rebuild',
      note: 'Re-segments every indexed chunk with CKIP and rewrites the search vector — no re-embedding, so it is safe and usually quick. Improves Chinese keyword recall for folders indexed before the CKIP upgrade.',
    })
    if (!ok) return
    setRebuilding(true)
    try { const r = await post(`/api/admin/manage/folders/${cur.id}/rebuild-fts`, {}); toast(r.message || 'Full-text search rebuilt', 'ok') }
    catch (e) { toast('Rebuild failed: ' + e.message, 'bad') }
    setRebuilding(false)
  }
  const editFolder = async (f) => {
    const r = await form({ title: 'Rename Folder', action: 'Save', fields: [{ id: 'name', label: 'Name', defaultValue: f.name, required: true }] })
    if (!r) return
    try { await api('PATCH', `/api/admin/manage/folders/${f.id}`, { name: r.name.trim() || null }); toast('Folder updated', 'ok'); loadFolders() }
    catch (e) { toast('Update failed: ' + e.message, 'bad') }
  }
  const deleteFolder = async (f) => {
    const r = await form({
      title: 'Delete Folder', action: 'Delete', danger: true, match: f.name,
      fields: [{ id: 'confirm', label: `Type “${f.name}” to confirm`, placeholder: f.name }],
      note: 'Deletes the folder, all its files, vectors and index records. This cannot be undone.',
    })
    if (!r) return
    try { await api('DELETE', `/api/admin/manage/folders/${f.id}`); toast(`Folder "${f.name}" deleted`, 'ok'); loadFolders() }
    catch (e) { toast('Delete failed: ' + e.message, 'bad') }
  }
  const deleteFile = async (fileId) => {
    const ok = await confirm({ title: 'Delete this file?', action: 'Delete', danger: true, note: 'Removes the file and its index from this folder.' })
    if (!ok) return
    try { await api('DELETE', `/api/admin/manage/folders/${cur.id}/files/${fileId}`); toast('File deleted', 'ok'); loadFiles(cur.id) }
    catch (e) { toast('Delete failed: ' + e.message, 'bad') }
  }
  const upload = async (e) => {
    const list = e.target.files; if (!list?.length) return
    const fd = new FormData(); for (const f of list) fd.append('files', f)
    try { await fetch(`/api/admin/manage/folders/${cur.id}/files`, { method: 'POST', body: fd }); toast(`Uploading ${list.length} file(s)`, 'ok'); loadFiles(cur.id) }
    catch (err) { toast('Upload failed: ' + err.message, 'bad') }
    e.target.value = ''
  }

  // ---- files detail view ----
  if (cur) {
    const list = files || []
    const shown = fq ? list.filter((f) => f.name.toLowerCase().includes(fq.toLowerCase())) : list
    const totalChunks = list.reduce((a, f) => a + (f.chunks || 0), 0)
    const indexed = list.filter((f) => f.status === 'indexed').length
    const failed = list.filter((f) => f.status === 'failed').length
    const models = [...new Set(list.map((f) => f.embedding_model).filter(Boolean))]
    const pct = list.length ? Math.round(indexed / list.length * 100) : 0
    const mdl = models.length === 1 ? shortModel(models[0]) : (models.length > 1 ? models.length + ' models' : '—')
    return (
      <div id="folder-files">
        <div className="crumbs"><a onClick={back} style={{ cursor: 'pointer' }}>{t('folders.crumb')}</a><span className="sep">/</span><span>{cur.name}</span></div>
        <div className="ff-head">
          <div className="ff-title-wrap"><h2>{cur.name}</h2>
            <div className="desc">{t('folders.stat.files')}: {list.length} · {failed ? t('folders.failedN', { n: failed }) : t('folders.allHealthy')}{models.length === 1 ? ' · ' + models[0] : ''}</div>
          </div>
          <div className="ff-actions">
            <div className="search-box"><input placeholder={t('folders.searchFiles')} value={fq} onChange={(e) => setFq(e.target.value)} /></div>
            <button className="btn-ghost" onClick={rebuildFts} disabled={rebuilding} title={t('folders.rebuildFts.title')}>
              <RefreshCw size={13} style={{ marginRight: 6, verticalAlign: '-2px', animation: rebuilding ? 'spin 1s linear infinite' : 'none' }} />{rebuilding ? t('common.rebuilding') : t('folders.rebuildFts')}
            </button>
            <button className="btn-ghost" onClick={() => reindexFolder(cur.id, false)}>{t('folders.reindex')}</button>
            <button className="btn-cyber" onClick={() => fileInput.current?.click()}>{t('folders.upload')}</button>
            <input type="file" ref={fileInput} multiple hidden onChange={upload} />
          </div>
        </div>
        <div className="ff-stats">
          <div className="s"><div className="k">{t('folders.stat.files')}</div><div className="v">{fmtNum(list.length)}</div><div className="d">{failed ? t('folders.failedN', { n: failed }) : t('folders.allHealthy')}</div></div>
          <div className="s"><div className="k">{t('folders.stat.chunks')}</div><div className="v" style={{ color: 'var(--signal)' }}>{fmtNum(totalChunks)}</div><div className="d">{t('folders.vectorsIndexed')}</div></div>
          <div className="s"><div className="k">{t('folders.stat.coverage')}</div><div className="v" style={{ color: 'var(--matrix)' }}>{pct}<small>%</small></div><div className="d">{t('folders.nIndexed', { a: indexed, b: list.length })}</div></div>
          <div className="s"><div className="k">{t('folders.stat.embedding')}</div><div className="v mdl" style={{ color: 'var(--cyber)' }}>{mdl}</div><div className="d">{models.length > 1 ? t('folders.rebuildRecommended') : 'e5 · 1024d'}</div></div>
        </div>
        {sel.size > 0 && (
          <div className="bulkbar">
            <span>{t('folders.selected', { n: sel.size })}</span>
            <div className="grow" />
            <button className="btn-ghost" onClick={bulkReindex}>{t('folders.reindexSelected')}</button>
            <button className="btn-danger" style={{ padding: '7px 14px' }} onClick={bulkDelete}>{t('folders.deleteSelected')}</button>
            <button className="act-btn" title={t('folders.clearSel')} onClick={() => setSel(new Set())}><X size={14} /></button>
          </div>
        )}
        <div className="tablewrap ff-grid">
          <table>
            <thead><tr>
              <th className="selcol"><input type="checkbox" className="selcheck"
                checked={shown.length > 0 && shown.every((f) => sel.has(f.id))}
                onChange={(e) => setSel(e.target.checked ? new Set(shown.map((f) => f.id)) : new Set())} /></th>
              <th>{t('folders.col.file')}</th><th>{t('folders.col.status')}</th><th>{t('folders.col.chunks')}</th><th>{t('folders.col.size')}</th><th>{t('folders.col.indexedAt')}</th><th>{t('folders.col.actions')}</th>
            </tr></thead>
            <tbody>
              {files == null ? <tr><td className="empty" colSpan={7}>{t('common.loading')}</td></tr>
                : shown.length === 0 ? <tr><td className="empty" colSpan={7}>{t('folders.noFiles')}</td></tr>
                  : shown.map((f) => (
                    <tr key={f.id} title={f.error || f.embedding_model || ''}>
                      <td className="selcol"><input type="checkbox" className="selcheck" checked={sel.has(f.id)} onChange={() => toggleSel(f.id)} /></td>
                      <td className="cell-main file-cell">{f.name}{f.error && <div className="sub" style={{ color: 'var(--alert-hi)', whiteSpace: 'normal' }}>{f.error}</div>}</td>
                      <td>{f._stage ? <>{statusPill(f._stage)}{f._stagePct != null && <span className="stage-bar"><div style={{ width: f._stagePct + '%' }} /></span>}</> : statusPill(f.status)}</td>
                      <td className="num">{fmtNum(f.chunks)}</td>
                      <td className="num">{fmtBytes(f.size_bytes)}</td>
                      <td className="num">{fmtTime(f.indexed_at)}</td>
                      <td><div className="actions">
                        {f.status === 'indexed' && <button className="act-btn" title={t('folders.action.viewChunks')} onClick={() => viewChunks(f)}><Eye size={14} /></button>}
                        <button className="act-btn" title={t('folders.action.download')} onClick={() => window.open(`/api/admin/manage/folders/${cur.id}/files/${f.id}/download`, '_blank')}><Download size={14} /></button>
                        <button className="act-btn danger" title={t('folders.action.deleteFile')} onClick={() => deleteFile(f.id)}><Trash2 size={14} /></button>
                      </div></td>
                    </tr>
                  ))}
            </tbody>
          </table>
        </div>

        {chunks && (
          <div className="modal-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) setChunks(null) }}>
            <div className="modal wide" role="dialog">
              <h3 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>{t('folders.chunksTitle', { name: chunks.name })}
                <button className="act-btn" style={{ marginLeft: 'auto' }} onClick={() => setChunks(null)}><X size={14} /></button>
              </h3>
              <div className="m-body">
                {chunks.items == null ? <div className="sec">{t('common.loading')}</div>
                  : chunks.err ? <EmptyState t={t('folders.chunks.failTitle')} d={chunks.err} />
                    : chunks.items.length === 0 ? <EmptyState t={t('folders.chunks.noneTitle')} d={t('folders.chunks.noneDesc')} />
                      : chunks.items.map((c, i) => (
                        <div className="chunk-row" key={i}>
                          <div className="c-head">
                            <span className="c-idx">#{i + 1}</span>
                            {c.node_role && <span className="c-meta">{c.node_role}</span>}
                            {c.headings?.length ? <span className="c-meta">{c.headings.join(' › ')}</span> : null}
                            <ContentBadge type={c.content_type} />
                          </div>
                          <ChunkText text={c.text} contentType={c.content_type} className="c-text" />
                        </div>
                      ))}
              </div>
              <div className="m-foot"><button className="btn-ghost" onClick={() => setChunks(null)}>{t('common.close')}</button></div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // ---- folder list view ----
  const shown = folders ? (q ? folders.filter((f) => (f.name + ' ' + (f.description || '')).toLowerCase().includes(q.toLowerCase())) : folders) : []
  return (
    <div id="view-folders">
      <div className="card">
        <div className="card-head">
          <div className="icon"><Folder size={16} /></div>
          <div><h2>{t('folders.title')}</h2><div className="desc">{t('folders.desc')}</div></div>
          <div className="grow" />
          <div className="search-box"><input placeholder={t('folders.searchFolders')} value={q} onChange={(e) => setQ(e.target.value)} /></div>
          <button className="btn-cyber" onClick={createFolder}>{t('folders.new')}</button>
        </div>
        <div className="tablewrap">
          <table>
            <thead><tr><th>{t('folders.col.name')}</th><th>{t('folders.col.files')}</th><th>{t('folders.col.size')}</th><th>{t('folders.col.indexed')}</th><th>{t('folders.col.chunks')}</th><th>{t('folders.col.lastIndexed')}</th><th>{t('folders.col.actions')}</th></tr></thead>
            <tbody>
              {folders == null ? <tr><td className="empty" colSpan={7}>{t('common.loading')}</td></tr>
                : shown.length === 0 ? <tr><td colSpan={7}><EmptyState t={t('folders.empty.title')} d={t('folders.empty.desc')} /></td></tr>
                  : shown.map((f) => (
                    <tr key={f.id} className="rowlink" onClick={() => openFolder(f)}>
                      <td className="cell-main name-cell">{f.name}<div className="sub">{f.description || ''}</div></td>
                      <td className="num">{fmtNum(f.file_count)}</td>
                      <td className="num">{fmtBytes(f.total_size_bytes)}</td>
                      <td className="num"><span style={{ color: 'var(--matrix-hi)' }}>{fmtNum(f.indexed_files)}</span>{f.failed_files ? <span style={{ color: 'var(--alert-hi)' }}> · {fmtNum(f.failed_files)} failed</span> : null}</td>
                      <td className="num">{fmtNum(f.chunks)}</td>
                      <td className="num">{fmtTime(f.last_indexed_at)}</td>
                      <td onClick={(e) => e.stopPropagation()}><div className="actions">
                        <button className="act-btn" title={t('folders.action.index')} onClick={() => reindexFolder(f.id, true)}><Play size={14} /></button>
                        <button className="act-btn" title={t('folders.action.rename')} onClick={() => editFolder(f)}><Pencil size={14} /></button>
                        <button className="act-btn danger" title={t('folders.action.deleteFolder')} onClick={() => deleteFolder(f)}><Trash2 size={14} /></button>
                      </div></td>
                    </tr>
                  ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
