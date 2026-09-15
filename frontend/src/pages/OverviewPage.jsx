import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { get, post } from '../services/api'
import { fmtNum, fmtBytes, shortModel } from '../lib/format'
import { useI18n } from '../contexts/I18nContext'

const Svg = ({ children }) => <svg viewBox="0 0 24 24">{children}</svg>

const STAGES = [
  { key: 'ingest', cls: 'on', name: 'INGEST', icon: <Svg><path d="M6 2h9l5 5v15H6z" /><path d="M15 2v5h5" /></Svg> },
  { key: 'chunk', cls: 'on', name: 'CHUNK', icon: <Svg><path d="M4 5h16M4 12h10M4 19h16" /></Svg> },
  { key: 'embed', cls: 'on', name: 'EMBED', icon: <Svg><circle cx="12" cy="12" r="3" /><path d="M12 2v4M12 18v4M2 12h4M18 12h4" /></Svg> },
  { key: 'vector', cls: 'on', name: 'VECTOR', icon: <Svg><ellipse cx="12" cy="6" rx="8" ry="3" /><path d="M4 6v12c0 1.6 3.6 3 8 3s8-1.4 8-3V6" /></Svg> },
  { key: 'retrieve', cls: 'sig', name: 'RETRIEVE', icon: <Svg><circle cx="11" cy="11" r="7" /><path d="M21 21l-4-4" /></Svg> },
  { key: 'rerank', cls: 'sig', name: 'RERANK', icon: <Svg><path d="M7 4v16M7 4l-3 3M7 4l3 3M17 20V4M17 20l-3-3M17 20l3-3" /></Svg> },
  { key: 'synth', cls: 'out', name: 'SYNTHESIZE', icon: <Svg><path d="M12 3a9 9 0 100 18 9 9 0 000-18z" /><path d="M8 12l3 3 5-6" /></Svg> },
]

function Pill({ state }) {
  const { t } = useI18n()
  const map = {
    online: ['ok', 'status.online'], offline: ['bad', 'status.offline'],
    unset: ['mute', 'status.notset'], checking: ['mute', 'status.checking'],
    ready: ['ok', 'status.ready'], disabled: ['mute', 'status.disabled'],
  }
  const [cls, label] = map[state] || map.checking
  return <span className={`pill ${cls}`}>{t(label)}</span>
}

export default function OverviewPage() {
  const nav = useNavigate()
  const { t } = useI18n()
  const [d, setD] = useState(null)
  const [res, setRes] = useState(null)
  const [health, setHealth] = useState({})

  const load = useCallback(async () => {
    const r = await get('/api/admin/overview')
    setD(r.data)
    const m = r.data.models
    const probe = async (id, url) => {
      if (!url) { setHealth((h) => ({ ...h, [id]: 'unset' })); return }
      try {
        const p = await post('/api/admin/probe', { base_url: url })
        setHealth((h) => ({ ...h, [id]: p.data && p.data.reachable ? 'online' : 'offline' }))
      } catch { setHealth((h) => ({ ...h, [id]: 'offline' })) }
    }
    probe('embed', m.embedding.base_url)
    probe('llm', m.llm.base_url)
    if (m.rerank.enabled) probe('rerank', m.rerank.base_url)
    try { const rr = await get('/api/admin/resources'); setRes(rr.data) } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    let alive = true
    const run = () => load().catch(() => {})
    run()
    const t = setInterval(() => { if (alive) run() }, 10000)
    return () => { alive = false; clearInterval(t) }
  }, [load])

  if (!d) {
    return (
      <div className="term-hero">
        <div className="term-cap"><h1>{t('ov.title')}</h1></div>
        <div className="sec" style={{ marginTop: 20 }}>{t('common.loading')}</div>
      </div>
    )
  }

  const ix = d.index, m = d.models
  const total = Math.max(ix.indexed_files + ix.failed_files + ix.unindexed_files, 1)
  const jobsFoot = Object.entries(d.jobs.by_status).map(([k, v]) => `${k} ${v}`).join(' · ') || t('ov.rb.queueIdle')

  const stageStat = {
    ingest: <><b>{fmtNum(d.files)}</b> file · {fmtBytes(d.total_size_bytes)}</>,
    chunk: <><b>{fmtNum(ix.total_chunks)}</b> chunks</>,
    embed: <>{shortModel(m.embedding.model)} · <b>{m.embedding.dimension || '?'}</b>d</>,
    vector: <><b>{fmtNum(ix.total_chunks)}</b> vec · pgvector</>,
    retrieve: <>hybrid · <b>.75</b>/<b>.25</b></>,
    rerank: m.rerank.enabled ? <>{shortModel(m.rerank.model)} · top-6</> : 'off',
    synth: shortModel(m.llm.model),
  }

  const ribbon = [
    { k: t('ov.rb.folders'), v: fmtNum(d.folders), cls: 'cy', d: t('ov.rb.tokenScoped'), nav: '/folders' },
    { k: t('ov.rb.files'), v: fmtNum(d.files), cls: '', d: fmtBytes(d.total_size_bytes) + ' ' + t('ov.rb.indexed') },
    { k: t('ov.rb.chunks'), v: fmtNum(ix.total_chunks), cls: 'am', d: t('ov.rb.vectorsInStore') },
    { k: t('ov.rb.activeJobs'), v: fmtNum(d.jobs.active.length), cls: 'gr', d: jobsFoot, nav: '/jobs' },
  ]

  const asrState = !m.asr.enabled ? 'disabled' : (m.asr.available ? 'ready' : 'offline')

  return (
    <div id="view-overview">
      <div className="term-hero">
        <div className="term-cap">
          <h1>{t('ov.title')}</h1>
          <div className="term-q">&#10095; hierarchical · auto-merge · hybrid retrieval<span className="cursor blink">&#9613;</span></div>
        </div>
        <div className="pipe">
          <div className="rail" /><div className="flow"><i /><i /><i /></div>
          {STAGES.map((s) => (
            <div key={s.key} className={`stage ${s.cls}`}>
              <div className="orb">{s.icon}</div>
              <div className="nm">{s.name}</div>
              <div className="st">{stageStat[s.key]}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="ribbon">
        {ribbon.map((x) => (
          <div key={x.k} className={'ro' + (x.nav ? ' clickable' : '')} onClick={x.nav ? () => nav(x.nav) : undefined}>
            <div className="k">{x.k}</div>
            <div className={'v ' + x.cls}>{x.v}</div>
            <div className="d">{x.d}</div>
          </div>
        ))}
      </div>

      <div className="term-cols">
        <div className="term-col">
          <div className="sec">{t('ov.indexHealth')} <span className="r">{t('ov.coverage')}</span></div>
          <div className="distbar">
            <div style={{ width: `${ix.indexed_files / total * 100}%`, background: 'var(--matrix)' }} />
            <div style={{ width: `${ix.failed_files / total * 100}%`, background: 'var(--alert)' }} />
            <div style={{ width: `${ix.unindexed_files / total * 100}%`, background: 'var(--ink-subtle)' }} />
          </div>
          <div className="dist-legend">
            <span><i style={{ background: 'var(--matrix)' }} />{t('ov.legend.indexed')} {fmtNum(ix.indexed_files)}</span>
            <span><i style={{ background: 'var(--alert)' }} />{t('ov.legend.failed')} {fmtNum(ix.failed_files)}</span>
            <span><i style={{ background: 'var(--ink-subtle)' }} />{t('ov.legend.notIndexed')} {fmtNum(ix.unindexed_files)}</span>
          </div>

          <div className="sec" style={{ marginTop: 24 }}>{t('ov.retrievalShape')} <span className="r">auto-merge · hybrid .75/.25</span></div>
          <div className="term-tree">
            <div className="n"><span className="p hit">{t('ov.tree.parent')}</span><small>{t('ov.tree.autoMerged')}</small></div>
            <div className="n l"><span className="p hit">{t('ov.tree.leaf')}</span><small>{t('ov.tree.matched')}</small></div>
            <div className="n l"><span className="p hit">{t('ov.tree.leaf')}</span><small>{t('ov.tree.matched')}</small></div>
            <div className="n l"><span className="p">{t('ov.tree.leaf')}</span><small>{t('ov.tree.neighbor')}</small></div>
          </div>
        </div>

        <div className="term-col">
          <div className="sec">{t('ov.services')} <span className="r">{t('ov.liveHealth')}</span></div>
          <div className="svc-grid">
            <div className="svc"><div className="grow"><div className="name">{t('ov.svc.embedding')}</div><div className="val">{m.embedding.model || '—'} · {m.embedding.dimension || '?'}d</div></div><Pill state={health.embed || 'checking'} /></div>
            <div className="svc"><div className="grow"><div className="name">{t('ov.svc.llm')}</div><div className="val">{m.llm.model || '—'}</div></div><Pill state={health.llm || 'checking'} /></div>
            <div className="svc"><div className="grow"><div className="name">{t('ov.svc.reranker')}</div><div className="val">{m.rerank.enabled ? (m.rerank.model || '—') : t('ov.disabled')}</div></div><Pill state={m.rerank.enabled ? (health.rerank || 'checking') : 'disabled'} /></div>
            <div className="svc"><div className="grow"><div className="name">{t('ov.svc.asr')}</div><div className="val">{m.asr.provider || '—'}</div></div><Pill state={asrState} /></div>
          </div>

          <div className="sec" style={{ marginTop: 24 }}>{t('ov.store')} <span className="r">agentic_rag</span></div>
          <div className="svc-grid">
            {res ? (
              <>
                <div className="svc"><div className="grow"><div className="name">{t('ov.svc.database')}</div><div className="val">{fmtBytes(res.db_size_bytes)}</div></div></div>
                <div className="svc"><div className="grow"><div className="name">{t('ov.svc.vectorStorage')} · {res.vector_tables} table(s)</div><div className="val">{fmtBytes(res.vector_bytes)}</div></div></div>
                <div className="svc"><div className="grow"><div className="name">{t('ov.svc.diskUsage')}</div><div className="val">{fmtBytes(res.disk_used_bytes)} / {fmtBytes(res.disk_total_bytes)}</div></div></div>
              </>
            ) : <div className="svc"><div className="grow"><div className="val">…</div></div></div>}
          </div>
        </div>
      </div>
    </div>
  )
}
