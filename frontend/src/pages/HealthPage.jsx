import { useEffect, useState, useCallback } from 'react'
import { Activity, Database, Layers, Sparkles, ArrowDownUp, Mic, HardDrive } from 'lucide-react'
import { get } from '../services/api'
import EmptyState from '../components/ui/EmptyState'
import { useI18n } from '../contexts/I18nContext'

// [css class, i18n key] — labels resolved with t() at render time
const LABEL = { ok: ['ok', 'status.online'], down: ['bad', 'status.offline'], disabled: ['mute', 'status.disabled'], unset: ['mute', 'status.notset'] }
const OVERALL = { ok: ['ok', 'health.overall.ok'], down: ['bad', 'health.overall.down'] }

function kindIcon(kind) {
  const k = (kind || '').toLowerCase()
  if (k.includes('db') || k.includes('database')) return Database
  if (k.includes('embed') || k.includes('layer')) return Layers
  if (k.includes('llm') || k.includes('spark')) return Sparkles
  if (k.includes('rank') || k.includes('sort')) return ArrowDownUp
  if (k.includes('asr') || k.includes('mic') || k.includes('speech')) return Mic
  if (k.includes('disk') || k.includes('store')) return HardDrive
  return Activity
}

export default function HealthPage() {
  const { t } = useI18n()
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)

  const load = useCallback(async () => {
    try { const r = await get('/api/admin/health'); setD(r.data) } catch (e) { setErr(e.message) }
  }, [])

  useEffect(() => {
    let alive = true
    const run = () => load()
    run()
    const t = setInterval(() => { if (alive) run() }, 15000)
    return () => { alive = false; clearInterval(t) }
  }, [load])

  const overall = d && (OVERALL[d.overall] || ['mute', 'health.overall.idle'])

  return (
    <div id="view-health">
      <div className="card">
        <div className="card-head">
          <div className="icon"><Activity size={16} /></div>
          <div><h2>{t('health.title')}</h2><div className="desc">{t('health.desc')}</div></div>
          <div className="grow" />
          {overall && <span className={'pill ' + overall[0]}>{t(overall[1])}</span>}
        </div>
        <div className="health-grid">
          {err ? <div style={{ gridColumn: '1/-1' }}><EmptyState t={t('health.failTitle')} d={err} /></div>
            : d == null ? <div className="sec" style={{ marginTop: 4 }}>{t('common.loading')}</div>
              : d.checks.map((c, i) => {
                const Icon = kindIcon(c.kind)
                const [cls, label] = LABEL[c.status] || LABEL.unset
                return (
                  <div key={i} className={'htile ' + c.status}>
                    <div className="h-icon"><Icon size={17} /></div>
                    <div className="h-body">
                      <div className="h-name">{c.name}<span className={'pill ' + cls}>{t(label)}</span></div>
                      <div className="h-detail">{c.detail || ''}</div>
                      {c.endpoint && <div className="h-ep" title={c.endpoint}>{c.endpoint}</div>}
                    </div>
                    {c.latency_ms != null && <span className="h-lat">{c.latency_ms} ms</span>}
                  </div>
                )
              })}
        </div>
      </div>
    </div>
  )
}
