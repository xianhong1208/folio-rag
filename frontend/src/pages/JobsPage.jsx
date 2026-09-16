import { useEffect, useState, useCallback } from 'react'
import { Zap, XCircle } from 'lucide-react'
import { get, post } from '../services/api'
import { fmtTime } from '../lib/format'
import EmptyState from '../components/ui/EmptyState'
import { useModal } from '../contexts/ModalContext'
import { useToast } from '../contexts/ToastContext'

const PILL = { running: 'run', pending: 'warn', queued: 'warn', completed: 'ok', failed: 'bad', cancelled: 'mute' }

export default function JobsPage() {
  const { confirm } = useModal()
  const toast = useToast()
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
    const ok = await confirm({
      title: 'Cancel this job?', action: 'Cancel job', danger: true,
      note: 'It stops after the current file; already-indexed files are kept.',
    })
    if (!ok) return
    try { await post(`/api/admin/manage/folders/${j.folder_id}/jobs/${j.job_id}/cancel`); toast('Job cancelled', 'ok'); load() }
    catch (e) { toast('Cancel failed: ' + e.message, 'bad') }
  }

  return (
    <div id="view-jobs">
      <div className="card">
        <div className="card-head">
          <div className="icon"><Zap size={16} /></div>
          <div><h2>Index Jobs</h2><div className="desc">Recent jobs · refreshes every 10s</div></div>
        </div>
        <div className="jobstream">
          {jobs == null ? <div className="sec" style={{ marginTop: 8 }}>loading…</div>
            : jobs.length === 0 ? <EmptyState t="No indexing jobs yet" d="Jobs appear here when folders are indexed" />
              : jobs.map((j) => {
                const running = ['running', 'pending', 'queued'].includes(j.status)
                const pct = j.total_files ? Math.round(j.processed_files / j.total_files * 100) : 0
                return (
                  <div key={j.job_id} className="js-row">
                    <div className="js-status"><span className={'pill ' + (PILL[j.status] || 'mute')}>{j.status}</span></div>
                    <div className="js-main">
                      <div className="js-top">
                        <span className="js-folder">{j.folder_name || ('#' + j.folder_id)}</span>
                        <span className="js-count">{j.processed_files}/{j.total_files}</span>
                        {running && (
                          <button className="act-btn danger" title="Cancel job" onClick={() => cancel(j)}><XCircle size={14} /></button>
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
