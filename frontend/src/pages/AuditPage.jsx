import { useEffect, useState } from 'react'
import { ScrollText } from 'lucide-react'
import { get } from '../services/api'
import { fmtTime } from '../lib/format'
import EmptyState from '../components/ui/EmptyState'

export default function AuditPage() {
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
          <div><h2>Audit Log</h2><div className="desc">Every runtime settings change — who, when, what</div></div>
        </div>
        <div className="timeline">
          {err ? <EmptyState t="Failed to load" d={err} />
            : rows == null ? <div className="sec" style={{ marginTop: 8 }}>loading…</div>
              : rows.length === 0 ? <EmptyState t="No changes yet" d="Runtime settings changes will appear here" />
                : rows.map((e, i) => {
                  const reset = e.action === 'reset'
                  return (
                    <div key={i} className={'tl-item ' + (reset ? 'reset' : 'set')}>
                      <span className="tl-dot" />
                      <div className="tl-time">{fmtTime(e.at)}</div>
                      <div className="tl-line">
                        <span className={'tl-act ' + (reset ? 'reset' : 'set')}>{reset ? 'RESET' : 'SET'}</span>
                        <span className="tl-key">{e.key}</span>
                        <span className="tl-arrow">→</span>
                        <span className="tl-val">{String(e.new ?? '—')}</span>
                      </div>
                      <div className="tl-by">by {e.by || 'console'}</div>
                    </div>
                  )
                })}
        </div>
      </div>
    </div>
  )
}
