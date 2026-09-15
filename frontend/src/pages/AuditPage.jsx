import { useEffect, useState } from 'react'
import { ScrollText } from 'lucide-react'
import { get } from '../services/api'
import { fmtTime } from '../lib/format'
import EmptyState from '../components/ui/EmptyState'
import { useI18n } from '../contexts/I18nContext'

export default function AuditPage() {
  const { t } = useI18n()
  const [rows, setRows] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    get('/api/admin/audit?limit=200').then((r) => setRows(r.data.entries)).catch((e) => setErr(e.message))
  }, [])

  return (
    <div id="view-audit">
      <div className="card">
        <div className="card-head">
          <div className="icon"><ScrollText size={16} /></div>
          <div><h2>{t('audit.title')}</h2><div className="desc">{t('audit.desc')}</div></div>
        </div>
        <div className="timeline">
          {err ? <EmptyState t={t('audit.failTitle')} d={err} />
            : rows == null ? <div className="sec" style={{ marginTop: 8 }}>{t('common.loading')}</div>
              : rows.length === 0 ? <EmptyState t={t('audit.empty.title')} d={t('audit.empty.desc')} />
                : rows.map((e, i) => {
                  const reset = e.action === 'reset'
                  return (
                    <div key={i} className={'tl-item ' + (reset ? 'reset' : 'set')}>
                      <span className="tl-dot" />
                      <div className="tl-time">{fmtTime(e.at)}</div>
                      <div className="tl-line">
                        <span className={'tl-act ' + (reset ? 'reset' : 'set')}>{reset ? t('audit.reset') : t('audit.set')}</span>
                        <span className="tl-key">{e.key}</span>
                        <span className="tl-arrow">→</span>
                        <span className="tl-val">{String(e.new ?? '—')}</span>
                      </div>
                      <div className="tl-by">{t('audit.by', { who: e.by || 'console' })}</div>
                    </div>
                  )
                })}
        </div>
      </div>
    </div>
  )
}
