import { useEffect, useState, useRef, useCallback } from 'react'
import { FlaskConical } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { get, post } from '../services/api'
import EmptyState from '../components/ui/EmptyState'

const pct = (x) => (x == null ? '—' : (x * 100).toFixed(1) + '%')

// Best available mode for a summary (rerank > rrf > hybrid > vector) and its nDCG@k.
// Used only for the multi-run trend line ("best available" per run).
const bestNdcg = (summary, k) => {
  if (!summary) return null
  for (const m of ['rerank', 'rrf', 'hybrid', 'vector']) {
    if (summary[m] && summary[m]['ndcg@' + k] != null) return summary[m]['ndcg@' + k]
  }
  return null
}

// nDCG@k for one *specific* mode (null if absent). Baseline deltas must compare the
// same mode across runs — comparing the current best mode against a different mode
// in the previous run produces bogus "improvement" numbers.
const ndcgOf = (summary, mode, k) =>
  (summary && summary[mode] && summary[mode]['ndcg@' + k] != null) ? summary[mode]['ndcg@' + k] : null

function Delta({ cur, prev }) {
  if (cur == null || prev == null) return null
  const d = cur - prev
  if (Math.abs(d) < 0.0005) return <span className="c-meta">± 0</span>
  const up = d > 0
  return (
    <span style={{ color: up ? 'var(--matrix-hi)' : 'var(--alert-hi)', fontSize: 11, marginLeft: 6 }}>
      {up ? '▲' : '▼'} {(Math.abs(d) * 100).toFixed(1)} pts
    </span>
  )
}

export default function EvalPage() {
  const [folders, setFolders] = useState([])
  const [fid, setFid] = useState('')
  const [n, setN] = useState('15')
  const [regen, setRegen] = useState(false)
  const [judge, setJudge] = useState(false)
  const [report, setReport] = useState(null)
  const [history, setHistory] = useState([])
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
    if (!id) { setReport(null); setHistory([]); return }
    try { const r = await get('/api/admin/eval/report?folder_id=' + id); setReport(r.data || null) }
    catch { setReport(null) }
    try { const h = await get('/api/admin/eval/history?folder_id=' + id); setHistory((h.data && h.data.entries) || []) }
    catch { setHistory([]) }
  }, [])
  useEffect(() => { loadReport(fid) }, [fid, loadReport])

  const run = async () => {
    if (!fid) return
    setRunning(true); setStatus({ message: 'Starting…' })
    try {
      await post('/api/admin/eval/run', { folder_id: Number(fid), n: Number(n), regenerate: regen, judge })
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
  const aq = report && report.answer_quality
  const prev = report && report.previous
  const trend = history.map((h, i) => ({
    i: i + 1,
    ndcg: bestNdcg(h.summary, h.k),
    faith: h.answer_quality ? h.answer_quality.faithfulness : null,
  }))

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
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 12 }}>
            <label className="sp-check">
              <input type="checkbox" checked={regen} onChange={(e) => setRegen(e.target.checked)} /> Regenerate question set (use after re-indexing — chunk IDs change)
            </label>
            <label className="sp-check">
              <input type="checkbox" checked={judge} onChange={(e) => setJudge(e.target.checked)} /> Grade answer quality (LLM-judged faithfulness + relevance — slower)
            </label>
          </div>
          {status && <div className="sec" style={{ marginBottom: 12 }}>{status.error ? 'Error: ' + status.error : (status.message || `Evaluating… ${status.done || 0}/${status.total || '?'}`)}</div>}
          {report?.regenerate_reason && (
            <div className="sec" style={{ marginBottom: 12, color: 'var(--signal-hi)' }}>⚠ {report.regenerate_reason}</div>
          )}
          {report?.stale_dropped > 0 && (
            <div className="sec" style={{ marginBottom: 12, color: 'var(--signal-hi)' }}>⚠ {report.stale_dropped} question(s) skipped — their source chunk no longer exists (file reindexed). Regenerate the set for a full run.</div>
          )}

          {report && rows.length ? (
            <>
              <div className="tablewrap">
                <table>
                  <thead><tr><th>Mode</th><th>Recall@{report.k}</th><th>MRR</th><th>nDCG@{report.k}</th></tr></thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.m} style={r.m === bestMode ? { background: 'var(--matrix-soft)' } : undefined}>
                        <td className="cell-main">{r.m}{r.m === bestMode ? ' · best' : ''}</td>
                        <td className="num">{pct(r.rec)}</td>
                        <td className="num">{r.mrr != null ? r.mrr.toFixed(3) : '—'}</td>
                        <td className="num">{pct(r.ndcg)}
                          {r.m === bestMode && prev && <Delta cur={r.ndcg} prev={ndcgOf(prev.summary, r.m, prev.k)} />}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {aq && (
                <div className="ff-stats" style={{ marginTop: 16 }}>
                  <div className="s"><div className="k">Faithfulness</div><div className="v" style={{ color: 'var(--matrix)' }}>{pct(aq.faithfulness)}</div><div className="d">grounded in context</div></div>
                  <div className="s"><div className="k">Relevance</div><div className="v" style={{ color: 'var(--cyber)' }}>{pct(aq.relevance)}</div><div className="d">answers the question</div></div>
                  <div className="s"><div className="k">Judged</div><div className="v">{aq.n_judged}</div><div className="d">answers over {aq.mode}</div></div>
                </div>
              )}

              {trend.length > 1 && (
                <div style={{ marginTop: 20 }}>
                  <div className="nav-group" style={{ paddingLeft: 0 }}>Baseline trend · nDCG (best mode){trend.some((t) => t.faith != null) ? ' + faithfulness' : ''} across {trend.length} runs</div>
                  <div style={{ height: 180, marginTop: 8 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trend} margin={{ top: 6, right: 12, bottom: 0, left: -18 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--hairline-soft)" />
                        <XAxis dataKey="i" tick={{ fontSize: 10, fill: 'var(--ink-subtle)' }} />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--ink-subtle)' }} />
                        <Tooltip contentStyle={{ background: 'var(--surface-solid)', border: '1px solid var(--hairline)', fontSize: 12 }} />
                        <Line type="monotone" dataKey="ndcg" stroke="var(--matrix-hi)" strokeWidth={2} dot={{ r: 2 }} name="nDCG" isAnimationActive={false} />
                        <Line type="monotone" dataKey="faith" stroke="var(--cyber-hi)" strokeWidth={2} dot={{ r: 2 }} name="faithfulness" connectNulls isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          ) : !running && <EmptyState t="No evaluation yet" d="Run one to score this folder’s retrieval quality" />}
        </div>
      </div>
    </div>
  )
}

const selStyle = { padding: '7px 10px', borderRadius: 9, background: 'var(--input-bg)', border: '1px solid var(--hairline)', color: 'var(--ink)', fontSize: 12 }
