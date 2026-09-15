import { useEffect, useState, useCallback } from 'react'
import { Zap, XCircle } from 'lucide-react'
import { get, post } from '../services/api'
import { fmtTime } from '../lib/format'
import EmptyState from '../components/ui/EmptyState'
import { useI18n } from '../contexts/I18nContext'

const PILL = { running: 'run', pending: 'warn', queued: 'warn', completed: 'ok', failed: 'bad', cancelled: 'mute' }

export default function JobsPage() {
  const { t } = useI18n()
  const [jobs, setJobs] = useState(null)

  const load = useCallback(async () => {
    const r = await get('/api/admin/jobs?limit=30')
    setJobs(r.data.jobs)
  }, [])

  useEffect(() => {
    let alive = true
    const run = () => load().catch(() => {})
    run()
    const t = setInterval(() => { if (alive) run() }, 10000)
    return () => { alive = false; clearInterval(t) }
  }, [load])

  const cancel = async (j) => {
    if (!window.confirm(t('jobs.cancelConfirm'))) return
    try { await post(`/api/admin/manage/folders/${j.folder_id}/jobs/${j.job_id}/cancel`); load() } catch (e) { alert(t('jobs.cancelFail') + e.message) }
  }

  return (
    <div id="view-jobs">
      <div className="card">
        <div className="card-head">
          <div className="icon"><Zap size={16} /></div>
          <div><h2>{t('jobs.title')}</h2><div className="desc">{t('jobs.desc')}</div></div>
        </div>
        <div className="jobstream">
          {jobs == null ? <div className="sec" style={{ marginTop: 8 }}>{t('common.loading')}</div>
            : jobs.length === 0 ? <EmptyState t={t('jobs.empty.title')} d={t('jobs.empty.desc')} />
              : jobs.map((j) => {
                const running = ['running', 'pending', 'queued'].includes(j.status)
                const pct = j.total_files ? Math.round(j.processed_files / j.total_files * 100) : 0
                return (
                  <div key={j.job_id} className="js-row">
                    <div className="js-status"><span className={'pill ' + (PILL[j.status] || 'mute')}>{t('status.' + j.status)}</span></div>
                    <div className="js-main">
                      <div className="js-top">
                        <span className="js-folder">{j.folder_name || ('#' + j.folder_id)}</span>
                        <span className="js-count">{j.processed_files}/{j.total_files}</span>
                        {running && (
                          <button className="act-btn danger" title={t('jobs.cancelTitle')} onClick={() => cancel(j)}><XCircle size={14} /></button>
                        )}
                      </div>
                      <div className="js-bar"><div className={'js-fill' + (running ? ' live' : '')} style={{ width: (running && pct === 0 ? 6 : pct) + '%' }} /></div>
                      <div className="js-msg">{j.error || j.message || j.current_file_name || '—'} · {fmtTime(j.started_at)}{j.completed_at ? ` → ${fmtTime(j.completed_at)}` : ''}</div>
                    </div>
                  </div>
                )
              })}
        </div>
      </div>
    </div>
  )
}
