import { useEffect, useState, useRef } from 'react'
import { get, post, stream } from '../services/api'
import EmptyState from '../components/ui/EmptyState'
import ChunkText, { ContentBadge } from '../components/ui/ChunkText'

const COLORS = { vector: 'var(--cyber)', bm25: 'var(--signal)', hybrid: 'var(--matrix)', rerank: 'var(--cyber-deep)' }

function Meta({ md }) {
  return (
    <>
      <span className="chunk-head-crumb">{md.file_name || ''}</span>
      {md.headings?.length ? <span className="c-meta">{md.headings.join(' › ')}</span> : null}
      {md.page != null ? <span className="c-meta">p.{md.page}</span> : null}
      <ContentBadge type={md.content_type} />
    </>
  )
}

function Bar({ score, max, rank, color }) {
  const pct = score != null ? Math.round(Math.abs(score) / max * 100) : 0
  return (
    <>
      <div className="tb-track"><div style={{ width: pct + '%', background: color }} /></div>
      <div className="tb-num">{score != null ? score.toFixed(3) : '—'}
        <span className={'rank-chip' + (rank == null ? ' miss' : '')}>{rank != null ? '#' + rank : '—'}</span>
      </div>
    </>
  )
}

function TraceResults({ trace }) {
  const rs = trace.results || []
  const maxOf = (k) => Math.max(1e-9, ...rs.map((r) => Math.abs(r[k] || 0)))
  const vmax = maxOf('vector_score'), bmax = maxOf('bm25_score'), hmax = maxOf('hybrid_score'), rmax = maxOf('rerank_score')
  const fusionLabel = trace.fusion === 'rrf' ? 'RRF' : 'Hybrid'
  return (
    <>
      <div style={{ fontSize: 12, color: 'var(--ink-subtle)', marginBottom: 10, fontFamily: 'JetBrains Mono' }}>
        {rs.length} result(s) from {trace.candidates} {fusionLabel} candidate(s){trace.reranked ? ' · reranked' : ' · rerank off'}.
      </div>
      {rs.map((r, i) => (
        <div className="chunk-row" key={i}>
          <div className="c-head">
            <span className="c-idx">#{i + 1}</span><Meta md={r.metadata || {}} />
            <span className="c-meta" style={{ marginLeft: 'auto' }}>final {r.final_score != null ? r.final_score.toFixed(3) : '—'}</span>
          </div>
          <div className="trace-bars">
            <span className="tb-label">Vector</span><Bar score={r.vector_score} max={vmax} rank={r.vector_rank} color={COLORS.vector} />
            <span className="tb-label">BM25</span><Bar score={r.bm25_score} max={bmax} rank={r.bm25_rank} color={COLORS.bm25} />
            <span className="tb-label">{fusionLabel}</span><Bar score={r.hybrid_score} max={hmax} rank={r.hybrid_rank} color={COLORS.hybrid} />
            {r.reranked && <><span className="tb-label">Rerank</span><Bar score={r.rerank_score} max={rmax} rank={i + 1} color={COLORS.rerank} /></>}
          </div>
          <ChunkText text={r.text} contentType={(r.metadata || {}).content_type} className="c-text" />
        </div>
      ))}
    </>
  )
}

export default function SearchPage() {
  const [folders, setFolders] = useState([])
  const [fid, setFid] = useState('')
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState(false)
  const [trace, setTrace] = useState(false)
  const [running, setRunning] = useState(false)
  const [res, setRes] = useState(null) // {hits, trace, answer, err}
  const abortRef = useRef(null)

  useEffect(() => {
    get('/api/admin/folders').then((r) => {
      const fs = r.data.folders.filter((f) => f.indexed_files > 0)
      setFolders(fs)
      if (fs.length) setFid(String(fs[0].id))
    }).catch(() => {})
    // Abort any in-flight stream when the page unmounts, so the SSE read loop
    // stops and the backend stops generating for a client that has navigated away.
    return () => abortRef.current?.abort()
  }, [])

  const run = async () => {
    if (!fid) { setRes({ err: 'No indexed folder — index some files first' }); return }
    if (query.trim().length < 2) return
    abortRef.current?.abort() // cancel a previous run still streaming
    const ac = new AbortController()
    abortRef.current = ac
    setRunning(true); setRes(null)

    // Streaming path: only when generating an answer without the trace view
    // (the trace needs the full non-streaming trace object).
    if (answer && !trace) {
      setRes({ hits: [], answer: '', streaming: true })
      try {
        await stream(`/api/admin/manage/folders/${fid}/query/stream`, { query }, (ev) => {
          if (ev.type === 'sources') setRes((r) => ({ ...r, hits: ev.results || [], total: ev.total }))
          else if (ev.type === 'meta') setRes((r) => ({ ...r, meta: ev }))
          else if (ev.type === 'token') setRes((r) => ({ ...r, answer: (r.answer || '') + ev.text }))
          else if (ev.type === 'done') setRes((r) => ({ ...r, answer: ev.answer ?? r.answer, streaming: false }))
          else if (ev.type === 'error') setRes((r) => ({ ...r, err: ev.error, streaming: false }))
        }, ac.signal)
      } catch (e) { if (!ac.signal.aborted) setRes({ err: e.message }) }
      if (!ac.signal.aborted) setRunning(false)
      return
    }

    try {
      const r = await post(`/api/admin/manage/folders/${fid}/query`, { query, answer, trace })
      if (!ac.signal.aborted) setRes({ hits: r.data.results || [], trace: r.data.trace, answer: r.data.answer, total: r.data.total_results })
    } catch (e) { if (!ac.signal.aborted) setRes({ err: e.message }) }
    if (!ac.signal.aborted) setRunning(false)
  }

  return (
    <div id="view-search">
      <div className="sp-head">
        <h2>Search Playground</h2>
        <div className="desc">Live retrieval against a folder — which chunks match, and how each signal ranks them.</div>
      </div>
      <div className="sp-bar">
        <span className="sp-prompt">&#10095;</span>
        <select className="sp-folder" value={fid} onChange={(e) => setFid(e.target.value)}>
          {folders.length ? folders.map((f) => <option key={f.id} value={f.id}>{f.name} ({f.indexed_files} files)</option>)
            : <option value="">No indexed folders</option>}
        </select>
        <input className="sp-query" placeholder="query the corpus…" value={query}
          onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run()} />
        <button className="btn-cyber" onClick={run} disabled={running}>{running ? 'Running…' : 'Run'}</button>
      </div>
      <div className="sp-opts">
        <label className="sp-check"><input type="checkbox" checked={answer} onChange={(e) => setAnswer(e.target.checked)} /> generate answer (RAG)</label>
        <label className="sp-check"><input type="checkbox" checked={trace} onChange={(e) => setTrace(e.target.checked)} /> retrieval trace</label>
        <span className="sp-sig">vector · BM25/CKIP · rerank · auto-merge</span>
      </div>

      {(res?.answer || res?.streaming) && (
        <div className="answer-card">
          <div className="a-label">
            ANSWER
            {res.streaming && <span style={{ marginLeft: 8, color: 'var(--cyber)' }}>streaming…</span>}
            {res.meta && !res.streaming && (res.meta.kept != null) && (
              <span style={{ marginLeft: 8, color: 'var(--ink-subtle)', fontWeight: 400 }}>
                {res.meta.kept} kept · {res.meta.dropped} dropped · {res.meta.confidence}
              </span>
            )}
          </div>
          <div className="a-text">
            {res.answer || (res.streaming ? '' : null)}
            {res.streaming && <span style={{ animation: 'blink 1s step-end infinite', color: 'var(--cyber)' }}>▋</span>}
          </div>
        </div>
      )}
      <div id="sp-results">
        {res == null ? null
          : res.err ? <EmptyState t="Search failed" d={res.err} />
            : (res.hits?.length === 0 && !res.streaming) ? <EmptyState t="No matches" d="Try lowering similarity or rephrasing" />
              : res.trace ? <TraceResults trace={res.trace} />
                : res.hits.map((h, i) => (
                  <div className="chunk-row" key={i}>
                    <div className="c-head">
                      <span className="c-idx">#{i + 1}</span><Meta md={h.metadata || {}} />
                      <span className="c-meta" style={{ marginLeft: 'auto' }}>score {h.score != null ? h.score.toFixed(3) : '—'}</span>
                    </div>
                    <ChunkText text={h.text} contentType={(h.metadata || {}).content_type} className="c-text" />
                  </div>
                ))}
      </div>
    </div>
  )
}
