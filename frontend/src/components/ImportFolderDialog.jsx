import { useEffect, useRef, useState } from 'react'
import { FolderInput, X } from 'lucide-react'
import { uploadRaw } from '../services/api'
import { fmtBytes } from '../lib/format'
import { importPath, importProblem, looksLikeArchive } from '../lib/folderArchive'

// Import a folder archive (from Export here, or `main.py export-folder`) as a new
// folder of the signed-in person. Resolves onDone(result) on success.
export default function ImportFolderDialog({ onClose, onDone }) {
  const [file, setFile] = useState(null)
  const [name, setName] = useState('')
  const [filesOnly, setFilesOnly] = useState(false)
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState(null)
  const input = useRef(null)

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose])

  const pick = (e) => {
    const f = (e.target.files || [])[0] || null
    e.target.value = ''
    setFile(f)
    setProblem(f && !looksLikeArchive(f)
      ? { title: 'Not a folder archive', text: 'Choose the .tar file an export produced.', offerFilesOnly: false, blocking: true }
      : null)
  }

  const run = async (onlyFiles) => {
    if (!file || busy) return
    setBusy(true)
    setProblem(null)
    try {
      const r = await uploadRaw(importPath({ name, filesOnly: onlyFiles }), file)
      onDone(r.data)
    } catch (err) {
      setProblem(importProblem(err))
      setBusy(false)
    }
  }

  return (
    <div className="modal-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose() }}>
      <div className="modal import-dialog" role="dialog" aria-labelledby="import-title">
        <h3 id="import-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>Import a folder
          <button className="act-btn" style={{ marginLeft: 'auto' }} aria-label="Close" onClick={onClose} disabled={busy}><X size={14} /></button>
        </h3>
        <div className="m-body">
          <div>
          <label>Archive</label>
          <div className="import-pick">
            <button className="btn-ghost" onClick={() => input.current?.click()} disabled={busy}>
              <FolderInput size={13} />Choose file…
            </button>
            <span className="import-file" title={file ? file.name : ''}>
              {file ? <>{file.name} <span className="sub">· {fmtBytes(file.size)}</span></> : <span className="sub">A .tar from Export, or from main.py export-folder</span>}
            </span>
            <input type="file" ref={input} accept=".tar,application/x-tar" hidden onChange={pick} />
          </div>
          </div>
          <div>
          <label htmlFor="import-name">Name</label>
          <input id="import-name" placeholder="Same as the exported folder" value={name} disabled={busy}
            onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run(filesOnly)} />
          </div>
          <label className="import-check">
            <input type="checkbox" checked={filesOnly} disabled={busy} onChange={(e) => setFilesOnly(e.target.checked)} />
            <span>Files only — leave the archive’s index out and index the files here</span>
          </label>
          {problem && (
            <div className="import-problem" role="alert">
              <strong>{problem.title}</strong>
              <div>{problem.text}</div>
              {problem.offerFilesOnly && (
                <button className="btn-ghost" onClick={() => { setFilesOnly(true); run(true) }} disabled={busy}>
                  Import files only, then index
                </button>
              )}
            </div>
          )}
          <div className="m-note">
            The folder becomes a new folder of yours; an existing name gets -2, -3, …
            Its search index comes along when this Folio uses the same embedding model.
          </div>
        </div>
        <div className="m-foot">
          <button className="btn-ghost" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn-cyber" onClick={() => run(filesOnly)}
            disabled={busy || !file || !!(problem && problem.blocking)}>
            {busy ? 'Importing…' : 'Import'}
          </button>
        </div>
      </div>
    </div>
  )
}
