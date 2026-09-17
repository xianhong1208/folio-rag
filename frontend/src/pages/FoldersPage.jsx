import { useEffect, useState, useCallback, useRef } from 'react'
import { Folder, Play, Pencil, Trash2, Download, Eye, X, RefreshCw } from 'lucide-react'
import { get, post, api } from '../services/api'
import { fmtNum, fmtBytes, fmtTime, shortModel } from '../lib/format'
import EmptyState from '../components/ui/EmptyState'
import ChunkText, { ContentBadge } from '../components/ui/ChunkText'
import { useModal } from '../contexts/ModalContext'
import { useToast } from '../contexts/ToastContext'

function statusPill(s) {
  const map = { indexed: 'ok', failed: 'bad', running: 'run', pending: 'warn', queued: 'warn', unindexed: 'mute' }
  const cls = map[s] || 'run'
  return <span className={'pill ' + cls}>{s.charAt(0).toUpperCase() + s.slice(1)}</span>
}

export default function FoldersPage() {
  const { confirm, form } = useModal()
  const toast = useToast()
  const [folders, setFolders] = useState(null)
  const [foldersErr, setFoldersErr] = useState(null)
  const [q, setQ] = useState('')
  const [cur, setCur] = useState(null) // {id,name} when viewing files
  const [files, setFiles] = useState(null)
  const [fq, setFq] = useState('')
  const [sel, setSel] = useState(new Set())
  const [chunks, setChunks] = useState(null) // { name, items } | null
  const fileInput = useRef(null)
  // The folder whose files are currently shown — used to drop a stale files
  // response that resolves after the user has already switched folders.
  const curIdRef = useRef(null)
  // Bumped to (re)start the file poll after an action that creates work (upload,
  // reindex). The poll stops itself once nothing is pending, so a new upload into
  // an already-indexed folder would otherwise never refresh without this.
  const [pollNonce, setPollNonce] = useState(0)
  const kickPoll = useCallback(() => setPollNonce((n) => n + 1), [])

  const loadFolders = useCallback(async () => {
    try { const r = await get('/api/admin/folders'); setFolders(r.data.folders); setFoldersErr(null) }
    catch (e) { setFoldersErr(e.message || 'Failed to load folders') }
  }, [])
  useEffect(() => { loadFolders() }, [loadFolders])

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
    // Ignore a response for a folder the user has since navigated away from.
    if (curIdRef.current === fid) setFiles(fs)
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
      } catch {
        // Transient failure — retry instead of freezing the view on "loading…".
        if (alive) timer = setTimeout(tick, 3000)
      }
    }
    tick()
    return () => { alive = false; if (timer) clearTimeout(timer) }
  }, [cur, loadFiles, pollNonce])

  // Esc closes the chunk viewer, matching the app's other modals.
  useEffect(() => {
    if (!chunks) return
    const onKey = (e) => { if (e.key === 'Escape') setChunks(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [chunks])

  const openFolder = (f) => { curIdRef.current = f.id; setCur({ id: f.id, name: f.name }); setFiles(null); setSel(new Set()) }
  const back = () => { curIdRef.current = null; setCur(null); setFiles(null); setSel(new Set()); loadFolders() }

  const toggleSel = (id) => setSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n })
  const bulkReindex = async () => {
    const ids = [...sel]
    try { await Promise.all(ids.map((id) => post(`/api/admin/manage/folders/${cur.id}/files/${id}/retry`, {}))); toast(`Reindexing ${ids.length} file(s)`, 'ok'); setSel(new Set()); kickPoll() }
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
    try { const r = await post(`/api/admin/manage/folders/${fid}/index`, { skip_existing: skip }); toast(r.message || 'Indexing started', 'ok'); kickPoll() }
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
    const list = [...(e.target.files || [])]
    e.target.value = ''
    if (!list.length) return
    // The endpoint takes ONE file per request in field `file` (it used to be posted
    // as `files`, which the API rejected with 422 — and the result was never checked,
    // so uploads silently did nothing). Post sequentially so each failure is named.
    let ok = 0
    for (const f of list) {
      const fd = new FormData()
      fd.append('file', f)
      fd.append('auto_index', 'true')
      try {
        const res = await fetch(`/api/admin/manage/folders/${cur.id}/files`, { method: 'POST', body: fd })
        if (!res.ok) {
          let msg = `HTTP ${res.status}`
          try {
            const d = await res.json()
            msg = d.detail?.error || d.detail?.[0]?.msg || (typeof d.detail === 'string' ? d.detail : null) || d.message || msg
          } catch { /* non-JSON */ }
          throw new Error(msg)
        }
        ok++
      } catch (err) { toast(`Upload failed: ${f.name} — ${err.message}`, 'bad') }
    }
    if (ok) toast(`Uploaded ${ok} file(s) — indexing`, 'ok')
    kickPoll() // restart polling so the new file's status refreshes to indexed automatically
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
        <div className="crumbs"><a role="button" tabIndex={0} onClick={back} style={{ cursor: 'pointer' }}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); back() } }}>Folders</a><span className="sep">/</span><span>{cur.name}</span></div>
        <div className="ff-head">
          <div className="ff-title-wrap"><h2>{cur.name}</h2>
            <div className="desc">{list.length} file(s) · {failed ? failed + ' failed' : 'all healthy'}{models.length === 1 ? ' · ' + models[0] : ''}</div>
          </div>
          <div className="ff-actions">
            <div className="search-box"><input placeholder="Search files…" value={fq} onChange={(e) => setFq(e.target.value)} /></div>
            <button className="btn-ghost tip" onClick={rebuildFts} disabled={rebuilding}
              data-tip="Re-segments existing chunks with CKIP and rebuilds only the keyword-search (BM25) vector — no re-embedding and no re-parsing, unlike Reindex Folder. Use it for folders indexed before the CKIP upgrade to fix Chinese keyword recall.">
              <RefreshCw size={13} style={{ animation: rebuilding ? 'spin 1s linear infinite' : 'none' }} />{rebuilding ? 'Rebuilding…' : 'Rebuild FTS'}
            </button>
            <button className="btn-ghost" onClick={() => reindexFolder(cur.id, false)}>Reindex Folder</button>
            <button className="btn-cyber" onClick={() => fileInput.current?.click()}>Upload Files</button>
            <input type="file" ref={fileInput} multiple hidden onChange={upload} />
          </div>
        </div>
        <div className="ff-stats">
          <div className="s"><div className="k">Files</div><div className="v">{fmtNum(list.length)}</div><div className="d">{failed ? failed + ' failed' : 'all healthy'}</div></div>
          <div className="s"><div className="k">Chunks</div><div className="v" style={{ color: 'var(--signal)' }}>{fmtNum(totalChunks)}</div><div className="d">vectors indexed</div></div>
          <div className="s"><div className="k">Coverage</div><div className="v" style={{ color: 'var(--matrix)' }}>{pct}<small>%</small></div><div className="d">{indexed}/{list.length} indexed</div></div>
          <div className="s"><div className="k">Embedding</div><div className="v mdl" style={{ color: 'var(--cyber)' }}>{mdl}</div><div className="d">{models.length > 1 ? 'rebuild recommended' : 'e5 · 1024d'}</div></div>
        </div>
        {sel.size > 0 && (
          <div className="bulkbar">
            <span>{sel.size} selected</span>
            <div className="grow" />
            <button className="btn-ghost" onClick={bulkReindex}>Reindex selected</button>
            <button className="btn-danger" style={{ padding: '7px 14px' }} onClick={bulkDelete}>Delete selected</button>
            <button className="act-btn" title="Clear selection" onClick={() => setSel(new Set())}><X size={14} /></button>
          </div>
        )}
        <div className="tablewrap ff-grid">
          <table>
            <thead><tr>
              <th className="selcol"><input type="checkbox" className="selcheck"
                checked={shown.length > 0 && shown.every((f) => sel.has(f.id))}
                onChange={(e) => setSel(e.target.checked ? new Set(shown.map((f) => f.id)) : new Set())} /></th>
              <th>File</th><th className="statuscol">Status</th><th>Chunks</th><th>Size</th><th>Indexed At</th><th>Actions</th>
            </tr></thead>
            <tbody>
              {files == null ? <tr><td className="empty" colSpan={7}>loading…</td></tr>
                : shown.length === 0 ? <tr><td className="empty" colSpan={7}>No files</td></tr>
                  : shown.map((f) => (
                    <tr key={f.id} title={f.error || f.embedding_model || ''}>
                      <td className="selcol"><input type="checkbox" className="selcheck" checked={sel.has(f.id)} onChange={() => toggleSel(f.id)} /></td>
                      <td className="cell-main file-cell">{f.name}{f.error && <div className="sub" style={{ color: 'var(--alert-hi)', whiteSpace: 'normal' }}>{f.error}</div>}</td>
                      <td className="statuscol">{f._stage ? <>{statusPill(f._stage)}{f._stagePct != null && <span className="stage-bar"><div style={{ width: f._stagePct + '%' }} /></span>}</> : statusPill(f.status)}</td>
                      <td className="num">{fmtNum(f.chunks)}</td>
                      <td className="num">{fmtBytes(f.size_bytes)}</td>
                      <td className="num">{fmtTime(f.indexed_at)}</td>
                      <td><div className="actions">
                        {f.status === 'indexed' && <button className="act-btn" title="View chunks" onClick={() => viewChunks(f)}><Eye size={14} /></button>}
                        <button className="act-btn" title="Download" onClick={() => window.open(`/api/admin/manage/folders/${cur.id}/files/${f.id}/download`, '_blank')}><Download size={14} /></button>
                        <button className="act-btn danger" title="Delete file" onClick={() => deleteFile(f.id)}><Trash2 size={14} /></button>
                      </div></td>
                    </tr>
                  ))}
            </tbody>
          </table>
        </div>

        {chunks && (
          <div className="modal-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) setChunks(null) }}>
            <div className="modal wide" role="dialog">
              <h3 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>Chunks · {chunks.name}
                <button className="act-btn" style={{ marginLeft: 'auto' }} onClick={() => setChunks(null)}><X size={14} /></button>
              </h3>
              <div className="m-body">
                {chunks.items == null ? <div className="sec">loading…</div>
                  : chunks.err ? <EmptyState t="Failed to load" d={chunks.err} />
                    : chunks.items.length === 0 ? <EmptyState t="No chunks" d="This file has no indexed chunks" />
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
              <div className="m-foot"><button className="btn-ghost" onClick={() => setChunks(null)}>Close</button></div>
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
          <div><h2>Folders</h2><div className="desc">Files and index status per folder — click a row for details</div></div>
          <div className="grow" />
          <div className="search-box"><input placeholder="Search folders…" value={q} onChange={(e) => setQ(e.target.value)} /></div>
          <button className="btn-cyber" onClick={createFolder}>+ New Folder</button>
        </div>
        <div className="tablewrap">
          <table>
            <thead><tr><th>Name</th><th>Files</th><th>Size</th><th>Indexed</th><th>Chunks</th><th>Last Indexed</th><th>Actions</th></tr></thead>
            <tbody>
              {folders == null && foldersErr
                ? <tr><td colSpan={7}><EmptyState t="Couldn’t load folders" d={foldersErr} />
                    <div style={{ textAlign: 'center', marginTop: 8 }}><button className="btn-ghost" onClick={loadFolders}>Retry</button></div></td></tr>
                : folders == null ? <tr><td className="empty" colSpan={7}>loading…</td></tr>
                : shown.length === 0 ? <tr><td colSpan={7}><EmptyState t="No folders yet" d="Create one with “+ New Folder”" /></td></tr>
                  : shown.map((f) => (
                    <tr key={f.id} className="rowlink" tabIndex={0} role="button"
                      onClick={() => openFolder(f)}
                      onKeyDown={(e) => { if (e.key === 'Enter') openFolder(f) }}>
                      <td className="cell-main name-cell">{f.name}<div className="sub">{f.description || ''}</div></td>
                      <td className="num">{fmtNum(f.file_count)}</td>
                      <td className="num">{fmtBytes(f.total_size_bytes)}</td>
                      <td className="num"><span style={{ color: 'var(--matrix-hi)' }}>{fmtNum(f.indexed_files)}</span>{f.failed_files ? <span style={{ color: 'var(--alert-hi)' }}> · {fmtNum(f.failed_files)} failed</span> : null}</td>
                      <td className="num">{fmtNum(f.chunks)}</td>
                      <td className="num">{fmtTime(f.last_indexed_at)}</td>
                      <td onClick={(e) => e.stopPropagation()}><div className="actions">
                        <button className="act-btn" title="Start indexing" onClick={() => reindexFolder(f.id, true)}><Play size={14} /></button>
                        <button className="act-btn" title="Rename" onClick={() => editFolder(f)}><Pencil size={14} /></button>
                        <button className="act-btn danger" title="Delete folder" onClick={() => deleteFolder(f)}><Trash2 size={14} /></button>
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
