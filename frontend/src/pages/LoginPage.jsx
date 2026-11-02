import { useState, useRef, useEffect } from 'react'
import { KeyRound, LogIn } from 'lucide-react'
import { useSession } from '../contexts/SessionContext'
import { cleanToken, loginErrorMessage } from '../lib/session'

export default function LoginPage() {
  const { login, notice } = useSession()
  const [token, setToken] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const ref = useRef(null)
  useEffect(() => { ref.current?.focus() }, [])

  const submit = async (e) => {
    e.preventDefault()
    const t = cleanToken(token)
    if (!t) { setErr(loginErrorMessage('login_required')); return }
    setBusy(true); setErr(null)
    try {
      await login(t)
    } catch (ex) {
      setErr(loginErrorMessage(ex.code, ex.message))
      setBusy(false)
    }
    // Never keep the token around in the page once it has been sent.
    setToken('')
  }

  return (
    <div className="login-wrap">
      <form className="login-card card" onSubmit={submit}>
        <div className="login-brand">
          <div className="logo">Agentic RAG</div>
          <div className="sub">Retrieval Terminal</div>
        </div>
        <div className="card-head">
          <div className="icon"><KeyRound size={16} /></div>
          <div>
            <h2>Sign in</h2>
            <div className="desc">Paste an MCP Gate access token issued for this service</div>
          </div>
        </div>
        <div className="login-body">
          {notice && !err && <div className="login-msg warn" role="status">{notice}</div>}
          {err && <div className="login-msg bad" role="alert">{err}</div>}
          <label htmlFor="token">Access token</label>
          <textarea
            id="token" ref={ref} value={token} spellCheck={false} autoComplete="off"
            placeholder="eyJhbGciOiJSUzI1NiIs…"
            onChange={(e) => setToken(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit(e) }}
          />
          <ul className="login-help">
            <li>You will see and manage the folders owned by this token — the same ones it reaches over REST and MCP.</li>
            <li>Changing settings needs a token with the <code>rag:admin</code> scope.</li>
            <li>The token is checked once and is not stored; the session ends when the token expires or after the configured session length (8 hours by default).</li>
          </ul>
        </div>
        <div className="card-foot">
          <div className="grow" />
          <button type="submit" className="btn-cyber" disabled={busy || !token.trim()}>
            <LogIn size={14} />{busy ? 'Signing in…' : 'Sign in'}
          </button>
        </div>
      </form>
    </div>
  )
}
