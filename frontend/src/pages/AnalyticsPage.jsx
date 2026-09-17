import { useEffect, useState, useCallback, useRef } from 'react'
import { BarChart3, Search } from 'lucide-react'
import { get } from '../services/api'
import { fmtNum } from '../lib/format'
import EmptyState from '../components/ui/EmptyState'

export default function AnalyticsPage() {
  const [folders, setFolders] = useState([])
  const [fid, setFid] = useState('')
  const [days, setDays] = useState('30')
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  const reqRef = useRef(0) // guards against out-of-order responses when filters change fast

  useEffect(() => {
    get('/api/admin/folders').then((r) => setFolders(r.data.folders)).catch(() => {})
  }, [])

  const load = useCallback(async () => {
    setErr(null)
    const my = ++reqRef.current
    try {
      const qs = `?days=${days}${fid ? `&folder_id=${fid}` : ''}`
      const r = await get('/api/admin/analytics' + qs)
      if (my === reqRef.current) setD(r.data)
    } catch (e) { if (my === reqRef.current) setErr(e.message) }
  }, [days, fid])
  useEffect(() => { load() }, [load])

  const by = d?.by_day || []
  const max = Math.max(1, ...by.map((x) => x.count))

  return (
    <div id="view-analytics">
      <div className="card">
        <div className="card-head">
          <div className="icon"><BarChart3 size={16} /></div>
          <div><h2>Query Analytics</h2><div className="desc">What users actually search — volume, latency and gaps in your knowledge base</div></div>
          <div className="grow" />
          <select value={fid} onChange={(e) => setFid(e.target.value)} style={{ padding: '7px 10px', borderRadius: 9, background: 'var(--input-bg)', border: '1px solid var(--hairline)', color: 'var(--ink)', fontSize: 12 }}>
            <option value="">All folders</option>
            {folders.map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
          </select>
          <select value={days} onChange={(e) => setDays(e.target.value)} style={{ padding: '7px 10px', borderRadius: 9, background: 'var(--input-bg)', border: '1px solid var(--hairline)', color: 'var(--ink)', fontSize: 12 }}>
            <option value="7">Last 7 days</option><option value="30">Last 30 days</option><option value="90">Last 90 days</option>
          </select>
        </div>

        <div className="kpi-grid" id="an-kpis">
          {err ? <div style={{ gridColumn: '1/-1' }}><EmptyState t="Failed to load" d={err} /></div>
            : d == null ? <div className="sec">loading…</div>
              : [
                { label: 'Total Queries', value: fmtNum(d.total), foot: `over ${d.days} days` },
                { label: 'Avg Latency', value: d.avg_latency_ms != null ? d.avg_latency_ms + ' ms' : '—', foot: 'retrieval time' },
                { label: 'Zero-Result Rate', value: (d.zero_rate * 100).toFixed(1) + '%', foot: `${fmtNum(d.zero_result)} empty searches` },
              ].map((k) => (
                <div className="kpi" key={k.label}>
                  <div className="k-label">{k.label}</div><div className="k-value">{k.value}</div><div className="k-foot">{k.foot}</div>
                </div>
              ))}
        </div>

        <div style={{ padding: '4px 0 18px' }}>
          <div className="sec" style={{ marginBottom: 10 }}>Query volume by day</div>
          {by.length ? (
            <>
              <div className="spark">
                {by.map((x, i) => <div key={i} className={'bar' + (x.count === max ? ' hot' : '')} style={{ height: Math.max(Math.round(x.count / max * 100), 4) + '%' }} title={`${x.date}: ${x.count}`} />)}
              </div>
              <div className="spark-labels"><span>{by[0].date.slice(5)}</span><span>{by[by.length - 1].date.slice(5)}</span></div>
            </>
          ) : <div style={{ color: 'var(--ink-subtle)', fontSize: 12, padding: '8px 0', fontFamily: 'JetBrains Mono' }}>No queries in this window</div>}
        </div>
      </div>

      <div className="an-tables">
        <div className="card">
          <div className="card-head"><div className="icon good"><BarChart3 size={16} /></div>
            <div><h2>Top Queries</h2><div className="desc">Most frequent searches · avg results returned</div></div></div>
          <div className="tablewrap">
            {d?.top_queries?.length ? (
              <table><thead><tr><th>Query</th><th>Count</th><th>Avg Hits</th></tr></thead>
                <tbody>{d.top_queries.map((q, i) => <tr key={i}><td className="cell-main">{q.query}</td><td className="num">{fmtNum(q.count)}</td><td className="num">{q.avg_hits}</td></tr>)}</tbody></table>
            ) : <EmptyState t="No queries yet" d="Run some searches to see analytics" />}
          </div>
        </div>
        <div className="card">
          <div className="card-head"><div className="icon warn"><Search size={16} /></div>
            <div><h2>Knowledge Gaps</h2><div className="desc">Searches that returned nothing — candidates for new content</div></div></div>
          <div className="tablewrap">
            {d?.zero_queries?.length ? (
              <table><thead><tr><th>Search returned nothing</th><th>Count</th></tr></thead>
                <tbody>{d.zero_queries.map((q, i) => <tr key={i}><td className="cell-main">{q.query}</td><td className="num">{fmtNum(q.count)}</td></tr>)}</tbody></table>
            ) : <EmptyState t="No knowledge gaps" d="Every search returned results — nice" />}
          </div>
        </div>
      </div>
    </div>
  )
}
