import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { bulkPatches } from '../lib/libraryFilters'

// Bulk edit of the selected documents: add / remove tags, set / clear one metadata
// key. One PATCH per document that changes (PATCH replaces a document's whole tag
// list or metadata object), two at a time, with progress and per-document errors.

const TITLES = {
  add_tags: 'Add tags', remove_tags: 'Remove tags', set_meta: 'Set a metadata key', clear_meta: 'Clear a metadata key',
}

const CONCURRENCY = 2

export default function BulkEditDialog({ kind, docs, keys, onClose, onDone }) {
  const [vals, setVals] = useState({ tags: '', key: '', value: '' })
  const [run, setRun] = useState(null) // { done, total, errors: [{name, message}], finished, skipped }

  const busy = run && !run.finished
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })
  const close = () => { if (run?.finished) onDone(); else onClose() }

  const isTags = kind === 'add_tags' || kind === 'remove_tags'
  const ready = isTags ? vals.tags.trim() : (vals.key.trim() && (kind === 'clear_meta' || vals.value.trim()))

  const apply = async () => {
    const patches = bulkPatches(docs, { kind, tags: vals.tags, key: vals.key.trim(), value: vals.value })
    const state = { done: 0, total: patches.length, errors: [], finished: false, skipped: docs.length - patches.length }
    setRun({ ...state })
    let next = 0
    const worker = async () => {
      while (next < patches.length) {
        const p = patches[next++]
        try { await api('PATCH', p.path, p.body) }
        catch (e) { state.errors.push({ name: p.doc.name, message: e.message }) }
        state.done++
        setRun({ ...state, errors: [...state.errors] })
      }
    }
    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, patches.length) }, worker))
    setRun({ ...state, errors: [...state.errors], finished: true })
  }

  const pct = run && run.total ? Math.round(run.done / run.total * 100) : 100
  return (
    <div className="modal-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) close() }}>
      <div className="modal bulk-dialog" role="dialog" aria-label={TITLES[kind]}>
        <h3>{TITLES[kind]} · <span className="mono">{docs.length} document{docs.length === 1 ? '' : 's'}</span></h3>
        <div className="m-body">
          {!run && (isTags ? (
            <div>
              <label htmlFor="bulk-tags">Tags (comma-separated)</label>
              <input id="bulk-tags" autoFocus placeholder="invoice, signed" value={vals.tags}
                onChange={(e) => setVals({ ...vals, tags: e.target.value })} onKeyDown={(e) => e.key === 'Enter' && ready && apply()} />
            </div>
          ) : (
            <>
              <div>
                <label htmlFor="bulk-key">Key</label>
                <input id="bulk-key" autoFocus list="bulk-keys" placeholder="client" value={vals.key}
                  onChange={(e) => setVals({ ...vals, key: e.target.value })} />
                <datalist id="bulk-keys">{(keys || []).map((k) => <option key={k.key} value={k.key} />)}</datalist>
              </div>
              {kind === 'set_meta' && (
                <div>
                  <label htmlFor="bulk-value">Value</label>
                  <input id="bulk-value" placeholder="Acme · 2024 · 2024-03-01 · true" value={vals.value}
                    onChange={(e) => setVals({ ...vals, value: e.target.value })} onKeyDown={(e) => e.key === 'Enter' && ready && apply()} />
                </div>
              )}
            </>
          ))}
          {!run && <div className="m-note">{kind === 'set_meta'
            ? 'Numbers and true/false are typed; write YYYY-MM-DD for a date, or "quote" a value to keep it as text. Documents that already have this value are left alone.'
            : 'Only documents that change are written; the others are left alone.'}</div>}
          {run && (
            <div className="bulk-run" aria-live="polite">
              <div className="bulk-prog"><div style={{ width: pct + '%' }} className={run.errors.length ? 'warn' : ''} /></div>
              <div className="bulk-count">
                {run.finished ? 'Done' : 'Writing'} · {run.done} / {run.total} updated
                {run.skipped ? ` · ${run.skipped} unchanged` : ''}
                {run.errors.length ? ` · ${run.errors.length} failed` : ''}
              </div>
              {run.errors.length > 0 && (
                <ul className="bulk-errors">
                  {run.errors.map((e, i) => <li key={i}><b>{e.name}</b> — {e.message}</li>)}
                </ul>
              )}
            </div>
          )}
        </div>
        <div className="m-foot">
          {!run && <button className="btn-ghost" onClick={onClose}>Cancel</button>}
          {!run && <button className="btn-cyber" onClick={apply} disabled={!ready}>Apply</button>}
          {run && <button className="btn-cyber" onClick={close} disabled={busy}>{busy ? 'Working…' : 'Close'}</button>}
        </div>
      </div>
    </div>
  )
}
