import { useEffect, useState, useCallback, useMemo } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Layers, Sparkles, List, ArrowDownUp, Mic, SlidersHorizontal, MessageSquare, FlaskConical, ScanText } from 'lucide-react'
import { get, post, api } from '../services/api'
import { fmtNum } from '../lib/format'
import { META, FIELDS, PRESETS, sectionOf, normalizeValue, keyIndicator, runsLabel, probeText } from '../lib/settings-config'
import { useModal } from '../contexts/ModalContext'
import { useToast } from '../contexts/ToastContext'
import { useSession } from '../contexts/SessionContext'
import PageCap from '../components/PageCap'
import { TABS, OVERRIDE_SECTIONS, tabFromParam, diffChanges, changesPerTab, stagePreset, needsReindex, showValue } from '../lib/settings-changes'

const ICON = { layers: Layers, spark: Sparkles, list: List, sort: ArrowDownUp, mic: Mic, sliders: SlidersHorizontal, message: MessageSquare, flask: FlaskConical, scan: ScanText }

function Field({ path, f, value, entry, onChange, disabled, pending }) {
  const wide = f.type === 'textarea'
  const isBool = f.type === 'bool'
  const overridden = entry?.overridden
  const setV = (v) => onChange(path, normalizeValue(f, v))
  const key = f.type === 'secret' ? keyIndicator(entry) : null

  return (
    <div className={'field' + (wide ? ' wide' : '') + (pending ? ' pending' : '')}>
      <div className="f-head">
        <label htmlFor={isBool ? undefined : path}>{f.label}</label>
        <div className="grow" />
        {key && <span className={'pill ' + key.cls} title="API keys are never shown; only whether one is set and which environment variable it comes from">{key.text}</span>}
        {pending && <span className="pend" title="Changed here, not applied yet">pending</span>}
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
    <section className="set-block" id="model-roles">
      <div className="sec">Model roles <span className="r">profile {view.profile} · keys are never shown, only whether one is set</span></div>
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
        <div className="set-foot"><span className="probe-result">Today (UTC): {fmtNum(usage.vision_pages || 0)} cloud-parsed page(s) · {fmtNum(usage.prefix_tokens || 0)} estimated prefix token(s) against daily caps</span></div>
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
  const [dirty, setDirty] = useState({}) // path -> pending value, across every tab
  const [applying, setApplying] = useState(false)
  const [review, setReview] = useState(false)
  const [probe, setProbe] = useState({}) // sec -> {cls,text}

  const loc = useLocation()
  const navigate = useNavigate()
  const param = loc.pathname.split('/').filter(Boolean)[1]
  const tabKey = tabFromParam(param)
  const tab = TABS.find((t) => t.key === tabKey)
  // Old /settings/<section> links (and bare /settings) settle on their tab's URL.
  useEffect(() => { if (param !== tabKey) navigate('/settings/' + tabKey, { replace: true }) }, [param, tabKey, navigate])

  const loadSettings = useCallback(async () => {
    const r = await get('/api/admin/settings')
    const m = {}
    for (const s of r.data.settings) m[s.path] = s
    setState(m)
    setView({ profile: r.data.profile, roles: r.data.roles || [], usage: r.data.usage || {} })
  }, [])
  useEffect(() => { loadSettings().catch(() => {}) }, [loadSettings])

  const changes = useMemo(() => diffChanges(dirty, state), [dirty, state])
  const perTab = changesPerTab(changes)
  const pendingPaths = new Set(changes.map((c) => c.path))

  // Leaving the page with unapplied changes asks first (tab switches keep them).
  useEffect(() => {
    if (!changes.length) return
    const onBefore = (e) => { e.preventDefault(); e.returnValue = '' }
    window.addEventListener('beforeunload', onBefore)
    return () => window.removeEventListener('beforeunload', onBefore)
  }, [changes.length])

  const onChange = (path, v) => setDirty((d) => ({ ...d, [path]: v }))
  const valOf = (p) => (p in dirty ? dirty[p] : state?.[p]?.value)

  const discard = () => { setDirty({}); setReview(false) }

  const apply = async () => {
    if (!changes.length) return
    if (needsReindex(changes)) {
      let impact = 'Existing vectors become incompatible with the new model and must be rebuilt.'
      try {
        const o = (await get('/api/admin/overview')).data
        impact = `${fmtNum(o.index.indexed_files)} indexed file(s) across ${fmtNum(o.folders)} folder(s) will need re-indexing to use the new embedding settings.`
      } catch { /* ignore */ }
      const ok = await confirm({ title: `Apply ${changes.length} change(s)?`, danger: true, action: 'Apply', note: impact + ' New uploads use the new settings immediately.' })
      if (!ok) return
    }
    const patch = Object.fromEntries(changes.map((c) => [c.path, c.to]))
    setApplying(true)
    try {
      const r = await api('PUT', '/api/admin/settings', { settings: patch })
      setDirty({})
      setReview(false)
      await loadSettings()
      const warns = (r.data && r.data.warnings) || []
      toast(warns.length ? 'Applied — note: ' + warns[0] : `Applied ${changes.length} change(s)`, warns.length ? 'warn' : 'ok')
    } catch (e) { toast('Apply failed: ' + e.message, 'bad') }
    setApplying(false)
  }

  // A preset only fills the form; it is reviewed and applied like any other change.
  const usePreset = (key) => {
    setDirty((d) => stagePreset(d, PRESETS[key].patch))
    toast(`"${PRESETS[key].label}" staged — review and apply below`, 'ok')
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
    const pendingHere = changes.some((c) => c.section === sec)
    setProbe((p) => ({ ...p, [sec]: { cls: '', text: 'Testing…' } }))
    const res = probeText(await runProbe(META[sec].probe))
    setProbe((p) => ({ ...p, [sec]: pendingHere ? { ...res, text: res.text + ' (tests the applied settings — apply to test your edits)' } : res }))
  }

  const block = (sec) => {
    const meta = META[sec]
    const paths = Object.keys(FIELDS).filter((p) => sectionOf(p) === sec && p in state)
    if (!paths.length) return null
    const Icon = ICON[meta.icon] || SlidersHorizontal
    const pr = probe[sec]
    return (
      <section key={sec} id={'sec-' + sec.replaceAll('.', '-')} className={'set-block' + (meta.warn ? ' warnband' : '')}>
        <div className="sec"><Icon size={12} className="sec-ico" />{meta.title} <span className="r">{meta.desc}</span></div>
        {meta.warn && <div className="warnnote"><span>⚠</span><span>{meta.warn}</span></div>}
        <div className="fields">
          {paths.map((p) => (
            <Field key={p} path={p} f={FIELDS[p]} value={valOf(p)} entry={state[p]} onChange={onChange} disabled={readOnly} pending={pendingPaths.has(p)} />
          ))}
        </div>
        {meta.probe && !readOnly && (
          <div className="set-foot">
            <button className="btn-ghost sm" onClick={() => testConn(sec)}>Test connection</button>
            {pr && <span className={'probe-result ' + pr.cls}>{pr.text}</span>}
          </div>
        )}
      </section>
    )
  }

  const main = tab.sections.filter((s) => !OVERRIDE_SECTIONS.includes(s))
  const overrides = tab.sections.filter((s) => OVERRIDE_SECTIONS.includes(s))
  const overridesPending = changes.some((c) => overrides.includes(c.section))

  return (
    <div id="view-settings" className={changes.length ? 'has-savebar' : undefined}>
      <PageCap title="Settings" query={tab.query} />
      <nav className="subtabs" role="tablist" aria-label="Settings">
        {TABS.map((t) => (
          <button key={t.key} role="tab" aria-selected={t.key === tabKey} className={'subtab' + (t.key === tabKey ? ' active' : '')}
            onClick={() => navigate('/settings/' + t.key)}>
            {t.label}{perTab[t.key] ? <span className="tab-dot" title={`${perTab[t.key]} pending change(s)`}>{perTab[t.key]}</span> : null}
          </button>
        ))}
      </nav>
      {readOnly && (
        <div className="ro-note" role="note">
          <span>🔒</span>
          <span>Read-only. Your token can view these settings but not change them — changing settings (and testing connections) needs a token with the <code>rag:admin</code> scope.</span>
        </div>
      )}
      {tabKey === 'models' && (
        <dl className="set-notes" aria-label="How settings work">
          <dt>apply</dt>
          <dd>Edits collect in the bar at the bottom and take effect when you <b>Apply</b> them. They survive restarts.</dd>
          <dt>override</dt>
          <dd><span className="ovr">Overridden</span> marks a value that differs from <code>config.yaml</code>.</dd>
          <dt>keys</dt>
          <dd>Give API keys as <code>{'${ENV_VAR}'}</code>: the server stores the reference, never the key. Required under the cloud profile.</dd>
        </dl>
      )}
      {tabKey === 'models' && <RolesPanel view={view} readOnly={readOnly} probes={roleProbe} onProbe={runProbe} />}
      {tabKey === 'retrieval' && (
        <div className="preset-bar">
          <span className="preset-label">start from</span>
          {Object.entries(PRESETS).map(([k, p]) => (
            <button key={k} className="preset-btn" disabled={readOnly} title={readOnly ? 'Needs rag:admin' : 'Fills the form; nothing is saved until you apply'} onClick={() => usePreset(k)}><b>{p.label}</b><span>{p.sub}</span></button>
          ))}
        </div>
      )}

      {state == null ? <div className="sec">loading…</div> : (
        <>
          {main.map(block)}
          {overrides.length > 0 && (
            <details className="overrides" open={overridesPending || undefined}>
              <summary><span className="sec">Per-role overrides <span className="r">optional · empty fields inherit the LLM above</span></span></summary>
              {overrides.map(block)}
            </details>
          )}
        </>
      )}

      {changes.length > 0 && (
        <div className="savebar" role="region" aria-label="Pending changes">
          {review && (
            <div className="savebar-review">
              {changes.map((c) => (
                <div key={c.path} className="chg">
                  <button className="chg-tab" onClick={() => navigate('/settings/' + c.tab)}>{c.tab}</button>
                  <span className="chg-name" title={c.path}>{c.label}</span>
                  <span className="chg-from">{showValue(c.from, c.secret)}</span>
                  <span className="chg-arrow">→</span>
                  <span className="chg-to">{showValue(c.to, c.secret)}</span>
                  <button className="chg-undo" title="Undo this change" aria-label={'Undo ' + c.label}
                    onClick={() => setDirty((d) => { const n = { ...d }; delete n[c.path]; return n })}>undo</button>
                </div>
              ))}
            </div>
          )}
          <div className="savebar-row">
            <span className="savebar-count"><span className="cursor blink">&#9613;</span> {changes.length} pending change{changes.length > 1 ? 's' : ''}</span>
            <span className="savebar-peek">{changes.slice(0, 2).map((c) => `${c.label.split(' · ').pop()} ${showValue(c.from, c.secret)}→${showValue(c.to, c.secret)}`).join(' · ')}{changes.length > 2 ? ' …' : ''}</span>
            <div className="grow" />
            <button className="btn-ghost sm" onClick={() => setReview((r) => !r)} aria-expanded={review}>{review ? 'Hide' : 'Review'}</button>
            <button className="btn-ghost sm" onClick={discard}>Discard</button>
            <button className="btn-cyber sm" disabled={applying} onClick={apply}>{applying ? 'Applying…' : 'Apply'}</button>
          </div>
        </div>
      )}
    </div>
  )
}
