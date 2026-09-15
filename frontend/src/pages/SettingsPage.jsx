import { useEffect, useState, useCallback } from 'react'
import { useLocation } from 'react-router-dom'
import { Layers, Sparkles, List, ArrowDownUp, Mic, SlidersHorizontal } from 'lucide-react'
import { get, post, api } from '../services/api'
import { fmtNum } from '../lib/format'
import { META, FIELDS, PRESETS, sectionOf } from '../lib/settings-config'
import { useModal } from '../contexts/ModalContext'
import { useToast } from '../contexts/ToastContext'
import { useI18n } from '../contexts/I18nContext'

const ICON = { layers: Layers, spark: Sparkles, list: List, sort: ArrowDownUp, mic: Mic, sliders: SlidersHorizontal }
const PARAM_TO_SEC = { embedding: 'rag.embedding', llm: 'rag.llm', contextual: 'rag.contextual_retrieval', reranker: 'rag.rerank', asr: 'rag.asr', retrieval: 'rag.retrieval' }

function Field({ path, f, value, overridden, onChange }) {
  const { t } = useI18n()
  const wide = f.type === 'textarea'
  const isBool = f.type === 'bool'
  const setV = (v) => onChange(path, v)

  return (
    <div className={'field' + (wide ? ' wide' : '')}>
      <div className="f-head">
        <label>{f.label}</label>
        <div className="grow" />
        {overridden && <span className="ovr" title={t('set.overriddenTip')}>{t('set.overridden')}</span>}
        {isBool && (
          <label className="toggle"><input type="checkbox" checked={!!value} onChange={(e) => setV(e.target.checked)} /><span /></label>
        )}
      </div>
      {f.hint && <div className="hint">{f.hint}</div>}
      {isBool ? null
        : f.type === 'select' ? (
          <select value={value ?? ''} onChange={(e) => setV(e.target.value)}>
            {f.options.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        ) : f.type === 'textarea' ? (
          <textarea value={value ?? ''} onChange={(e) => setV(e.target.value)} />
        ) : f.type === 'secret' ? (
          <input type="password" placeholder="Set — type to replace" value={value ?? ''} onChange={(e) => setV(e.target.value)} />
        ) : f.range ? (
          <div className="slider-wrap">
            <input type="range" min={f.range[0]} max={f.range[1]} step={f.step || 0.05}
              value={value ?? f.range[0]}
              style={{ '--fill': (((value ?? f.range[0]) - f.range[0]) / (f.range[1] - f.range[0]) * 100) + '%' }}
              onChange={(e) => setV(parseFloat(e.target.value))} />
            <input type="number" step={f.step || 0.05} value={value ?? ''} onChange={(e) => setV(e.target.value === '' ? null : parseFloat(e.target.value))} />
          </div>
        ) : (
          <input type={f.type === 'number' ? 'number' : 'text'} value={value ?? ''}
            onChange={(e) => setV(f.type === 'number' ? (e.target.value === '' ? null : Number(e.target.value)) : e.target.value)} />
        )}
    </div>
  )
}

export default function SettingsPage() {
  const { confirm } = useModal()
  const toast = useToast()
  const { t } = useI18n()
  const [state, setState] = useState(null) // path -> {value, overridden}
  const [dirty, setDirty] = useState({})
  const [saving, setSaving] = useState('')
  const [probe, setProbe] = useState({}) // sec -> {cls,text}

  const loc = useLocation()
  const sectionParam = loc.pathname.split('/').filter(Boolean).pop()

  const loadSettings = useCallback(async () => {
    const r = await get('/api/admin/settings')
    const m = {}
    for (const s of r.data.settings) m[s.path] = s
    setState(m)
  }, [])
  useEffect(() => { loadSettings().catch(() => {}) }, [loadSettings])

  // Scroll to the section chosen in the sidebar (/settings/<section>)
  useEffect(() => {
    if (state == null) return
    const sec = PARAM_TO_SEC[sectionParam]
    const t = setTimeout(() => {
      if (!sec) { window.scrollTo({ top: 0 }); return }
      const el = document.getElementById('sec-' + sec.replaceAll('.', '-'))
      if (el) el.scrollIntoView({ block: 'start' })
    }, 120)
    return () => clearTimeout(t)
  }, [sectionParam, state])

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

  const testConn = async (sec) => {
    setProbe((p) => ({ ...p, [sec]: { cls: '', text: 'Testing…' } }))
    try {
      if (sec === 'rag.asr' && valOf('rag.asr.provider') !== 'openai-compatible') {
        const a = (await get('/api/admin/overview')).data.models.asr
        setProbe((p) => ({ ...p, [sec]: a.available ? { cls: 'ok', text: '✓ Ready — local weights found' } : { cls: 'bad', text: '✗ Not ready — weights missing under assets/' } }))
        return
      }
      const base = valOf(sec + '.base_url')
      if (!base) { setProbe((p) => ({ ...p, [sec]: { cls: 'bad', text: 'No endpoint set' } })); return }
      const r = await post('/api/admin/probe', { base_url: base })
      const ok = r.data && r.data.reachable
      setProbe((p) => ({ ...p, [sec]: ok ? { cls: 'ok', text: '✓ Reachable' } : { cls: 'bad', text: '✗ Unreachable' } }))
    } catch (e) { setProbe((p) => ({ ...p, [sec]: { cls: 'bad', text: '✗ ' + e.message } })) }
  }

  return (
    <div id="view-settings">
      <p className="lede">Changes take effect <strong>immediately</strong> and persist across restarts. An <strong>Overridden</strong> badge means the value differs from config.yaml. Changes in the <span style={{ color: 'var(--signal-hi)' }}>amber section (Embedding)</span> require re-indexing existing folders.</p>
      <div className="preset-bar">
        <span className="preset-label">{t('set.presets')}</span>
        {Object.entries(PRESETS).map(([k, p]) => (
          <button key={k} className="preset-btn" onClick={() => applyPreset(k)}><b>{p.label}</b><span>{p.sub}</span></button>
        ))}
      </div>

      {state == null ? <div className="sec">{t('common.loading')}</div> : Object.entries(META).map(([sec, meta]) => {
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
                <Field key={p} path={p} f={FIELDS[p]} value={valOf(p)} overridden={state[p]?.overridden} onChange={onChange} />
              ))}
            </div>
            <div className="card-foot">
              <button className="btn-cyber" disabled={!secDirty || saving === sec} onClick={() => saveSection(sec)}>{saving === sec ? t('common.saving') : t('common.save')}</button>
              {meta.probe && <button className="btn-ghost" onClick={() => testConn(sec)}>{t('set.testConn')}</button>}
              {pr && <span className={'probe-result ' + pr.cls}>{pr.text}</span>}
              <div className="grow" />
            </div>
          </section>
        )
      })}
    </div>
  )
}
