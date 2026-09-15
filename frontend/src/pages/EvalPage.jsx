import { useEffect, useState, useRef, useCallback } from 'react'
import { FlaskConical } from 'lucide-react'
import { get, post } from '../services/api'
import EmptyState from '../components/ui/EmptyState'

const pct = (x) => (x == null ? '—' : (x * 100).toFixed(1) + '%')

export default function EvalPage() {
  const [folders, setFolders] = useState([])
  const [fid, setFid] = useState('')
  const [n, setN] = useState('15')
  const [regen, setRegen] = useState(false)
  const [report, setReport] = useState(null)
  const [status, setStatus] = useState(null)
  const [running, setRunning] = useState(false)
  const poll = useRef(null)

  useEffect(() => {
    get('/api/admin/folders').then((r) => {
      const fs = r.data.folders.filter((f) => f.indexed_files > 0)
      setFolders(fs); if (fs.length) setFid(String(fs[0].id))
    }).catch(() => {})
    return () => { if (poll.current) clearInterval(poll.current) }
  }, [])

  const loadReport = useCallback(async (id) => {
    if (!id) { setReport(null); return }
    try { const r = await get('/api/admin/eval/report?folder_id=' + id); setReport(r.data || null) }
    catch { setReport(null) }
  }, [])
  useEffect(() => { loadReport(fid) }, [fid, loadReport])

  const run = async () => {
    if (!fid) return
    setRunning(true); setStatus({ message: 'Starting…' })
    try {
      await post('/api/admin/eval/run', { folder_id: Number(fid), n: Number(n), regenerate: regen })
      if (poll.current) clearInterval(poll.current)
      poll.current = setInterval(async () => {
        try {
          const s = (await get('/api/admin/eval/status?folder_id=' + fid)).data
          setStatus(s)
          if (!s || !s.running) {
            clearInterval(poll.current); poll.current = null
            setRunning(false); setStatus(null); loadReport(fid)
          }
        } catch { clearInterval(poll.current); poll.current = null; setRunning(false) }
      }, 1500)
    } catch (e) { setRunning(false); setStatus({ error: e.message }) }
  }

  let rows = []
  let bestMode = null
  if (report) {
    const k = report.k
    const modes = report.modes || Object.keys(report.summary || {})
    rows = modes.map((m) => { const s = report.summary[m]; return { m, rec: s['recall@' + k], mrr: s.mrr, ndcg: s['ndcg@' + k] } })
    if (rows.length) bestMode = rows.reduce((a, b) => (b.ndcg > a.ndcg ? b : a), rows[0]).m
  }

  return (
    <div id="view-eval">
      <div className="card">
        <div className="card-head">
          <div className="icon"><FlaskConical size={16} /></div>
          <div><h2>Retrieval Evaluation</h2><div className="desc">Auto-generates questions from your chunks, then scores vector / hybrid / rerank — Recall@k · MRR · nDCG@k</div></div>
          <div className="grow" />
          <select value={fid} onChange={(e) => setFid(e.target.value)} style={selStyle}>
            {folders.length ? folders.map((f) => <option key={f.id} value={f.id}>{f.name} ({f.indexed_files} files)</option>) : <option value="">No indexed folders</option>}
          </select>
          <select value={n} onChange={(e) => setN(e.target.value)} style={selStyle}>
            <option value="10">10 Q</option><option value="15">15 Q</option><option value="25">25 Q</option>
          </select>
          <button className="btn-cyber" onClick={run} disabled={running || !fid}>{running ? 'Running…' : 'Run evaluation'}</button>
        </div>
        <div style={{ padding: '8px 22px 20px' }}>
          <label className="sp-check" style={{ marginBottom: 12 }}>
            <input type="checkbox" checked={regen} onChange={(e) => setRegen(e.target.checked)} /> Regenerate question set (use after re-indexing — chunk IDs change)
          </label>
          {status && <div className="sec" style={{ marginBottom: 12 }}>{status.error ? 'Error: ' + status.error : (status.message || 'Evaluating…')}</div>}

          {report && rows.length ? (
            <div className="tablewrap">
              <table>
                <thead><tr><th>Mode</th><th>Recall@{report.k}</th><th>MRR</th><th>nDCG@{report.k}</th></tr></thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.m} style={r.m === bestMode ? { background: 'var(--matrix-soft)' } : undefined}>
                      <td className="cell-main">{r.m}{r.m === bestMode ? ' · best' : ''}</td>
                      <td className="num">{pct(r.rec)}</td>
                      <td className="num">{r.mrr != null ? r.mrr.toFixed(3) : '—'}</td>
                      <td className="num">{pct(r.ndcg)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : !running && <EmptyState t="No evaluation yet" d="Run one to score this folder’s retrieval quality" />}
        </div>
      </div>
    </div>
  )
}

const selStyle = { padding: '7px 10px', borderRadius: 9, background: 'var(--input-bg)', border: '1px solid var(--hairline)', color: 'var(--ink)', fontSize: 12 }
