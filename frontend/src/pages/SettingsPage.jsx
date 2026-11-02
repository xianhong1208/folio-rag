import { useEffect, useState, useCallback, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Layers, Sparkles, List, ArrowDownUp, Mic, SlidersHorizontal, MessageSquare, FlaskConical, ScanText } from 'lucide-react'
import { get, post, api } from '../services/api'
import { fmtNum } from '../lib/format'
import { META, FIELDS, PRESETS, sectionOf, normalizeValue, keyIndicator, runsLabel, probeText } from '../lib/settings-config'
import { useModal } from '../contexts/ModalContext'
import { useToast } from '../contexts/ToastContext'
import { useSession } from '../contexts/SessionContext'

const ICON = { layers: Layers, spark: Sparkles, list: List, sort: ArrowDownUp, mic: Mic, sliders: SlidersHorizontal, message: MessageSquare, flask: FlaskConical, scan: ScanText }
const PARAM_TO_SEC = { embedding: 'rag.embedding', llm: 'rag.llm', 'contextual-llm': 'rag.contextual_llm', 'answer-llm': 'rag.answer_llm', 'eval-llm': 'rag.eval_llm', contextual: 'rag.contextual_retrieval', reranker: 'rag.rerank', parsing: 'rag.parsing', asr: 'rag.asr', retrieval: 'rag.retrieval' }
const SEC_TO_PARAM = Object.fromEntries(Object.entries(PARAM_TO_SEC).map(([p, s]) => [s, p]))

function Field({ path, f, value, entry, onChange, disabled }) {
  const wide = f.type === 'textarea'
  const isBool = f.type === 'bool'
  const overridden = entry?.overridden
  const setV = (v) => onChange(path, normalizeValue(f, v))
  const key = f.type === 'secret' ? keyIndicator(entry) : null

  return (
    <div className={'field' + (wide ? ' wide' : '')}>
      <div className="f-head">
        <label htmlFor={isBool ? undefined : path}>{f.label}</label>
        <div className="grow" />
        {key && <span className={'pill ' + key.cls} title="API keys are never shown; only whether one is set and which environment variable it comes from">{key.text}</span>}
        {overridden && <span className="ovr" title="Differs from config.yaml">Overridden</span>}
        {isBool && (
          <label className="toggle"><input type="checkbox" aria-label={f.label} checked={!!value} disabled={disabled} onChange={(e) => setV(e.target.checked)} /><span /></label>
        )}
      </div>
      {f.hint && <div className="hint">{f.hint}</div>}
      {isBool ? null
        : f.type === 'select' ? (
          <select id={path} value={value ?? ''} disabled={disabled} onChange={(e) => setV(e.target.value)}>
            {f.options.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        ) : f.type === 'textarea' ? (
          <textarea id={path} value={value ?? ''} disabled={disabled} onChange={(e) => setV(e.target.value)} />
        ) : f.type === 'secret' ? (
          <input id={path} type="password" autoComplete="off" placeholder={'Type ${ENV_VAR} (recommended) or a new key'} value={value === '•••' ? '' : (value ?? '')} disabled={disabled} onChange={(e) => setV(e.target.value)} />
        ) : f.range ? (
          <div className="slider-wrap">
            <input id={path} type="range" min={f.range[0]} max={f.range[1]} step={f.step || 0.05} disabled={disabled}
              value={value ?? f.range[0]}
              style={{ '--fill': (((value ?? f.range[0]) - f.range[0]) / (f.range[1] - f.range[0]) * 100) + '%' }}
              onChange={(e) => setV(parseFloat(e.target.value))} />
            <input type="number" aria-label={f.label + ' (number)'} step={f.step || 0.05} value={value ?? ''} disabled={disabled} onChange={(e) => setV(e.target.value === '' ? null : parseFloat(e.target.value))} />
          </div>
        ) : (
          <input id={path} type={f.type === 'number' ? 'number' : 'text'} value={value ?? ''} disabled={disabled}
            onChange={(e) => setV(f.type === 'number' ? (e.target.value === '' ? null : Number(e.target.value)) : e.target.value)} />
        )}
    </div>
  )
}

function RolesPanel({ view, readOnly, probes, onProbe }) {
  if (!view || !view.roles || !view.roles.length) return null
  const usage = view.usage || {}
  return (
    <section className="card" id="model-roles">
      <div className="card-head">
        <div className="icon"><Layers size={16} /></div>
        <div>
          <h2>Model Roles · profile <code>{view.profile}</code></h2>
          <div className="desc">Where each job runs and which model serves it. Keys are never shown — only whether one is set.</div>
        </div>
        <div className="grow" />
      </div>
      <div className="role-grid">
        {view.roles.map((r) => {
          const runs = runsLabel(r)
          const key = r.runs === 'endpoint' ? keyIndicator({ is_set: r.key_set, env: r.key_env }) : null
          const pr = probeText(probes[r.role])
          return (
            <div className="role" key={r.role}>
              <div className="role-head">
                <span className="name">{r.label}</span>
                <span className={'pill ' + runs.cls}>{runs.text}</span>
              </div>
              <div className="val" title={r.base_url || ''}>{r.runs === 'off' ? '—' : `${r.provider || '—'} · ${r.model || '—'}`}</div>
              {r.runs === 'endpoint' && <div className="sub" title={r.base_url || ''}>{r.base_url || 'provider default endpoint'}</div>}
              {r.inherits && r.runs !== 'off' && <div className="sub">inherits the LLM section</div>}
              <div className="role-foot">
                {key && <span className={'pill ' + key.cls}>{key.text}</span>}
                <div className="grow" />
                {!readOnly && r.runs !== 'off' && <button className="btn-ghost sm" onClick={() => onProbe(r.role)}>Test</button>}
              </div>
              {pr.text && <div className={'probe-result ' + pr.cls}>{pr.text}</div>}
            </div>
          )
        })}
      </div>
      {(usage.vision_pages || usage.prefix_tokens) ? (
        <div className="card-foot"><span className="probe-result">Today (UTC): {fmtNum(usage.vision_pages || 0)} cloud-parsed page(s) · {fmtNum(usage.prefix_tokens || 0)} estimated prefix token(s) against daily caps</span></div>
      ) : null}
    </section>
  )
}

export default function SettingsPage() {
  const { confirm } = useModal()
  const toast = useToast()
  const { isAdmin } = useSession()
  const readOnly = !isAdmin
  const [state, setState] = useState(null) // path -> {value, overridden, is_set, env}
  const [view, setView] = useState(null) // {profile, roles, usage}
  const [roleProbe, setRoleProbe] = useState({}) // role -> probe result
  const [dirty, setDirty] = useState({})
  const [saving, setSaving] = useState('')
  const [probe, setProbe] = useState({}) // sec -> {cls,text}

  const loc = useLocation()
  const navigate = useNavigate()
  const sectionParam = loc.pathname.split('/').filter(Boolean).pop()
  // The section currently scrolled into view (kept in sync by the scrollspy below).
  // Lets the click-to-scroll effect tell "user clicked a different section" apart
  // from "the spy just synced the URL to where we already are".
  const inViewRef = useRef(null)

  const loadSettings = useCallback(async () => {
    const r = await get('/api/admin/settings')
    const m = {}
    for (const s of r.data.settings) m[s.path] = s
    setState(m)
    setView({ profile: r.data.profile, roles: r.data.roles || [], usage: r.data.usage || {} })
  }, [])
  useEffect(() => { loadSettings().catch(() => {}) }, [loadSettings])

  // Scroll to the section chosen in the sidebar (/settings/<section>).
  // Skip when the URL already matches the section in view — that means the
  // scrollspy synced the URL to the user's scroll, and scrolling again would
  // fight them. Only a genuine nav to a *different* section scrolls.
  useEffect(() => {
    if (state == null) return
    if (sectionParam === inViewRef.current) return
    const sec = PARAM_TO_SEC[sectionParam]
    const t = setTimeout(() => {
      if (!sec) { window.scrollTo({ top: 0 }); return }
      const el = document.getElementById('sec-' + sec.replaceAll('.', '-'))
      if (el) el.scrollIntoView({ block: 'start' })
    }, 120)
    return () => clearTimeout(t)
  }, [sectionParam, state])

  // Scrollspy: highlight the sidebar item for whichever section is near the top
  // of the viewport as the user scrolls (updates the URL with replace, so the
  // sidebar NavLink active state follows without adding history entries).
  useEffect(() => {
    if (state == null) return
    const secs = Array.from(document.querySelectorAll('section[id^="sec-"]'))
    if (!secs.length) return
    const visible = new Set()
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) visible.add(e.target)
        else visible.delete(e.target)
      }
      if (!visible.size) return
      // Topmost currently-visible section wins (viewport-relative, container-agnostic).
      const top = [...visible].sort(
        (a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top)[0]
      const param = SEC_TO_PARAM[top.id.slice(4).replaceAll('-', '.')]
      if (param && param !== inViewRef.current) {
        inViewRef.current = param
        navigate('/settings/' + param, { replace: true })
      }
    }, { rootMargin: '-72px 0px -60% 0px' })
    secs.forEach((s) => io.observe(s))
    return () => io.disconnect()
  }, [state, navigate])

  const onChange = (path, v) => setDirty((d) => ({ ...d, [path]: v }))
  const valOf = (p) => (p in dirty ? dirty[p] : state?.[p]?.value)

  const saveSection = async (sec) => {
    const patch = {}
    for (const p of Object.keys(dirty)) if (sectionOf(p) === sec) patch[p] = dirty[p]
    if (!Object.keys(patch).length) return
    if (META[sec].warn) {
      let impact = 'Existing vectors become incompatible with the new model and must be rebuilt.'
      try {
        const o = (await get('/api/admin/overview')).data
        impact = `${fmtNum(o.index.indexed_files)} indexed file(s) across ${fmtNum(o.folders)} folder(s) will need re-indexing to use the new settings.`
      } catch { /* ignore */ }
      const ok = await confirm({ title: `Apply ${META[sec].title} change?`, danger: true, action: 'Apply & Save', note: impact + ' New uploads use the new settings immediately.' })
      if (!ok) return
    }
    setSaving(sec)
    try {
      const r = await api('PUT', '/api/admin/settings', { settings: patch })
      setDirty((d) => { const n = { ...d }; for (const p of Object.keys(patch)) delete n[p]; return n })
      await loadSettings()
      const warns = (r.data && r.data.warnings) || []
      toast(warns.length ? 'Applied — note: ' + warns[0] : 'Settings applied', warns.length ? 'warn' : 'ok')
    } catch (e) { toast('Save failed: ' + e.message, 'bad') }
    setSaving('')
  }

  const applyPreset = async (key) => {
    const p = PRESETS[key]
    const ok = await confirm({ title: `Apply "${p.label}" preset?`, action: 'Apply', note: `Updates ${Object.keys(p.patch).length} retrieval & reranker settings. Takes effect immediately; fine-tune afterwards.` })
    if (!ok) return
    try { await api('PUT', '/api/admin/settings', { settings: p.patch }); await loadSettings(); toast(`Applied "${p.label}" preset`, 'ok') }
    catch (e) { toast('Preset failed: ' + e.message, 'bad') }
  }

  // Per-role probe: one minimal real request with the *saved* settings (the key stays on the server).
  const runProbe = async (role) => {
    setRoleProbe((p) => ({ ...p, [role]: { status: 'unset', detail: 'Testing…' } }))
    try {
      const r = await post('/api/admin/probe/' + role, {})
      setRoleProbe((p) => ({ ...p, [role]: r.data }))
      return r.data
    } catch (e) {
      const res = { status: 'down', detail: e.message }
      setRoleProbe((p) => ({ ...p, [role]: res }))
      return res
    }
  }

  const testConn = async (sec) => {
    const dirtyHere = Object.keys(dirty).some((p) => sectionOf(p) === sec)
    setProbe((p) => ({ ...p, [sec]: { cls: '', text: 'Testing…' } }))
    const res = probeText(await runProbe(META[sec].probe))
    setProbe((p) => ({ ...p, [sec]: dirtyHere ? { ...res, text: res.text + ' (saved settings — save to test your edits)' } : res }))
  }

  return (
    <div id="view-settings">
      {readOnly && (
        <div className="ro-note" role="note">
          <span>🔒</span>
          <span>Read-only. Your token can view these settings but not change them — changing settings (and testing connections) needs a token with the <code>rag:admin</code> scope.</span>
        </div>
      )}
      <p className="lede">Changes take effect <strong>immediately</strong> and persist across restarts. An <strong>Overridden</strong> badge means the value differs from config.yaml. Changes in the <span style={{ color: 'var(--signal-hi)' }}>amber section (Embedding)</span> require re-indexing existing folders. API keys are best given as <code>{'${ENV_VAR}'}</code> references — the server stores the reference, never the key (required under the cloud profile).</p>
      <RolesPanel view={view} readOnly={readOnly} probes={roleProbe} onProbe={runProbe} />
      <div className="preset-bar">
        <span className="preset-label">Retrieval presets</span>
        {Object.entries(PRESETS).map(([k, p]) => (
          <button key={k} className="preset-btn" disabled={readOnly} title={readOnly ? 'Needs rag:admin' : undefined} onClick={() => applyPreset(k)}><b>{p.label}</b><span>{p.sub}</span></button>
        ))}
      </div>

      {state == null ? <div className="sec">loading…</div> : Object.entries(META).map(([sec, meta]) => {
        const paths = Object.keys(FIELDS).filter((p) => sectionOf(p) === sec && p in state)
        if (!paths.length) return null
        const Icon = ICON[meta.icon] || SlidersHorizontal
        const secDirty = paths.some((p) => p in dirty)
        const pr = probe[sec]
        return (
          <section key={sec} id={'sec-' + sec.replaceAll('.', '-')} className={'card' + (meta.warn ? ' warnband' : '')}>
            <div className="card-head">
              <div className="icon"><Icon size={16} /></div>
              <div><h2>{meta.title}</h2><div className="desc">{meta.desc}</div></div>
              <div className="grow" />
            </div>
            {meta.warn && <div className="warnnote"><span>⚠</span><span>{meta.warn}</span></div>}
            <div className="fields">
              {paths.map((p) => (
                <Field key={p} path={p} f={FIELDS[p]} value={valOf(p)} entry={state[p]} onChange={onChange} disabled={readOnly} />
              ))}
            </div>
            <div className="card-foot">
              <button className="btn-cyber" disabled={readOnly || !secDirty || saving === sec} title={readOnly ? 'Needs rag:admin' : undefined} onClick={() => saveSection(sec)}>{saving === sec ? 'Saving…' : 'Save Changes'}</button>
              {meta.probe && !readOnly && <button className="btn-ghost" onClick={() => testConn(sec)}>Test Connection</button>}
              {pr && <span className={'probe-result ' + pr.cls}>{pr.text}</span>}
              <div className="grow" />
            </div>
          </section>
        )
      })}
    </div>
  )
}
