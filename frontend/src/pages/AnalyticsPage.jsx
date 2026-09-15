import { useEffect, useState, useCallback } from 'react'
import { BarChart3, Search } from 'lucide-react'
import { get } from '../services/api'
import { fmtNum } from '../lib/format'
import EmptyState from '../components/ui/EmptyState'
import { useI18n } from '../contexts/I18nContext'

export default function AnalyticsPage() {
  const { t } = useI18n()
  const [folders, setFolders] = useState([])
  const [fid, setFid] = useState('')
  const [days, setDays] = useState('30')
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    get('/api/admin/folders').then((r) => setFolders(r.data.folders)).catch(() => {})
  }, [])

  const load = useCallback(async () => {
    setErr(null)
    try {
      const qs = `?days=${days}${fid ? `&folder_id=${fid}` : ''}`
      const r = await get('/api/admin/analytics' + qs)
      setD(r.data)
    } catch (e) { setErr(e.message) }
  }, [days, fid])
  useEffect(() => { load() }, [load])

  const by = d?.by_day || []
  const max = Math.max(1, ...by.map((x) => x.count))

  return (
    <div id="view-analytics">
      <div className="card">
        <div className="card-head">
          <div className="icon"><BarChart3 size={16} /></div>
          <div><h2>{t('an.title')}</h2><div className="desc">{t('an.desc')}</div></div>
          <div className="grow" />
          <select value={fid} onChange={(e) => setFid(e.target.value)} style={{ padding: '7px 10px', borderRadius: 9, background: 'var(--input-bg)', border: '1px solid var(--hairline)', color: 'var(--ink)', fontSize: 12 }}>
            <option value="">{t('an.allFolders')}</option>
            {folders.map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
          </select>
          <select value={days} onChange={(e) => setDays(e.target.value)} style={{ padding: '7px 10px', borderRadius: 9, background: 'var(--input-bg)', border: '1px solid var(--hairline)', color: 'var(--ink)', fontSize: 12 }}>
            <option value="7">{t('an.last7')}</option><option value="30">{t('an.last30')}</option><option value="90">{t('an.last90')}</option>
          </select>
        </div>

        <div className="kpi-grid" id="an-kpis">
          {err ? <div style={{ gridColumn: '1/-1' }}><EmptyState t={t('an.failTitle')} d={err} /></div>
            : d == null ? <div className="sec">{t('common.loading')}</div>
              : [
                { label: t('an.kpi.total'), value: fmtNum(d.total), foot: t('an.kpi.totalFoot', { n: d.days }) },
                { label: t('an.kpi.latency'), value: d.avg_latency_ms != null ? d.avg_latency_ms + ' ms' : '—', foot: t('an.kpi.latencyFoot') },
                { label: t('an.kpi.zeroRate'), value: (d.zero_rate * 100).toFixed(1) + '%', foot: t('an.kpi.zeroFoot', { n: fmtNum(d.zero_result) }) },
              ].map((k) => (
                <div className="kpi" key={k.label}>
                  <div className="k-label">{k.label}</div><div className="k-value">{k.value}</div><div className="k-foot">{k.foot}</div>
                </div>
              ))}
        </div>

        <div style={{ padding: '4px 0 18px' }}>
          <div className="sec" style={{ marginBottom: 10 }}>{t('an.volByDay')}</div>
          {by.length ? (
            <>
              <div className="spark">
                {by.map((x, i) => <div key={i} className={'bar' + (x.count === max ? ' hot' : '')} style={{ height: Math.max(Math.round(x.count / max * 100), 4) + '%' }} title={`${x.date}: ${x.count}`} />)}
              </div>
              <div className="spark-labels"><span>{by[0].date.slice(5)}</span><span>{by[by.length - 1].date.slice(5)}</span></div>
            </>
          ) : <div style={{ color: 'var(--ink-subtle)', fontSize: 12, padding: '8px 0', fontFamily: 'JetBrains Mono' }}>{t('an.noQueriesWindow')}</div>}
        </div>
      </div>

      <div className="an-tables">
        <div className="card">
          <div className="card-head"><div className="icon good"><BarChart3 size={16} /></div>
            <div><h2>{t('an.topQueries')}</h2><div className="desc">{t('an.topDesc')}</div></div></div>
          <div className="tablewrap">
            {d?.top_queries?.length ? (
              <table><thead><tr><th>{t('an.col.query')}</th><th>{t('an.col.count')}</th><th>{t('an.col.avgHits')}</th></tr></thead>
                <tbody>{d.top_queries.map((q, i) => <tr key={i}><td className="cell-main">{q.query}</td><td className="num">{fmtNum(q.count)}</td><td className="num">{q.avg_hits}</td></tr>)}</tbody></table>
            ) : <EmptyState t={t('an.topEmpty.title')} d={t('an.topEmpty.desc')} />}
          </div>
        </div>
        <div className="card">
          <div className="card-head"><div className="icon warn"><Search size={16} /></div>
            <div><h2>{t('an.gaps')}</h2><div className="desc">{t('an.gapsDesc')}</div></div></div>
          <div className="tablewrap">
            {d?.zero_queries?.length ? (
              <table><thead><tr><th>{t('an.col.gap')}</th><th>{t('an.col.count')}</th></tr></thead>
                <tbody>{d.zero_queries.map((q, i) => <tr key={i}><td className="cell-main">{q.query}</td><td className="num">{fmtNum(q.count)}</td></tr>)}</tbody></table>
            ) : <EmptyState t={t('an.gapsEmpty.title')} d={t('an.gapsEmpty.desc')} />}
          </div>
        </div>
      </div>
    </div>
  )
}
