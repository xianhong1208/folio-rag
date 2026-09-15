import { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react'

const Ctx = createContext(null)
export const useModal = () => useContext(Ctx)

export function ModalProvider({ children }) {
  const [m, setM] = useState(null) // { kind, opts, resolve }

  const confirm = useCallback((opts) => new Promise((resolve) => setM({ kind: 'confirm', opts, resolve })), [])
  const form = useCallback((opts) => new Promise((resolve) => setM({ kind: 'form', opts, resolve })), [])

  const settle = (val) => { if (m) m.resolve(val); setM(null) }

  return (
    <Ctx.Provider value={{ confirm, form }}>
      {children}
      {m && <ModalUI m={m} settle={settle} />}
    </Ctx.Provider>
  )
}

function ModalUI({ m, settle }) {
  const { kind, opts } = m
  const [vals, setVals] = useState(() => {
    const o = {}
    for (const f of opts.fields || []) o[f.id] = f.defaultValue || ''
    return o
  })
  const firstRef = useRef(null)
  useEffect(() => { firstRef.current?.focus() }, [])
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') settle(kind === 'confirm' ? false : null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, []) // eslint-disable-line

  const matchOk = !opts.match || (vals[(opts.fields?.[0]?.id)] || '') === opts.match
  const required = (opts.fields || []).filter((f) => f.required)
  const canSubmit = matchOk && required.every((f) => (vals[f.id] || '').trim())

  const submit = () => {
    if (kind === 'confirm') return settle(true)
    if (!canSubmit) return
    settle(vals)
  }

  return (
    <div className="modal-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) settle(kind === 'confirm' ? false : null) }}>
      <div className="modal" role="dialog">
        <h3>{opts.title}</h3>
        <div className="m-body">
          {(opts.fields || []).map((f, i) => (
            <div key={f.id}>
              <label>{f.label}</label>
              <input
                ref={i === 0 ? firstRef : null}
                placeholder={f.placeholder || ''}
                value={vals[f.id]}
                onChange={(e) => setVals((v) => ({ ...v, [f.id]: e.target.value }))}
                onKeyDown={(e) => e.key === 'Enter' && submit()}
              />
            </div>
          ))}
          {opts.note && <div className={'m-note' + (opts.danger ? ' warn' : '')}>{opts.note}</div>}
        </div>
        <div className="m-foot">
          <button className="btn-ghost" onClick={() => settle(kind === 'confirm' ? false : null)}>Cancel</button>
          <button
            className={opts.danger ? 'btn-danger' : 'btn-cyber'}
            disabled={kind !== 'confirm' && !canSubmit}
            onClick={submit}
          >{opts.action || (kind === 'confirm' ? 'Confirm' : 'OK')}</button>
        </div>
      </div>
    </div>
  )
}
