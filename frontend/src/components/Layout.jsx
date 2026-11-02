import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useEffect, useState } from 'react'
import {
  Home, Folder, Zap, Search, BarChart3, FlaskConical, Layers, Sparkles,
  ListTree, ArrowDownUp, Mic, SlidersHorizontal, Activity, ScrollText, Sun, Moon, LogOut, Menu, ScanText,
} from 'lucide-react'
import { useSession } from '../contexts/SessionContext'
import { sessionLabel, expiresIn } from '../lib/session'
import { saveTheme } from '../lib/theme'

const NAV = [
  { group: 'Console', items: [
    { to: '/', end: true, icon: Home, label: 'Overview' },
    { to: '/folders', icon: Folder, label: 'Folders' },
    { to: '/jobs', icon: Zap, label: 'Index Jobs' },
  ] },
  { group: 'Search & Insights', items: [
    { to: '/search', icon: Search, label: 'Search Playground' },
    { to: '/analytics', icon: BarChart3, label: 'Query Analytics' },
    { to: '/eval', icon: FlaskConical, label: 'Evaluation' },
  ] },
  { group: 'Settings', items: [
    { to: '/settings/embedding', icon: Layers, label: 'Embedding' },
    { to: '/settings/llm', icon: Sparkles, label: 'LLM' },
    { to: '/settings/contextual', icon: ListTree, label: 'Contextual Retrieval' },
    { to: '/settings/reranker', icon: ArrowDownUp, label: 'Reranker' },
    { to: '/settings/parsing', icon: ScanText, label: 'Document Parsing' },
    { to: '/settings/asr', icon: Mic, label: 'Speech-to-Text' },
    { to: '/settings/retrieval', icon: SlidersHorizontal, label: 'Retrieval' },
  ] },
  { group: 'System', items: [
    { to: '/health', icon: Activity, label: 'System Health' },
    { to: '/audit', icon: ScrollText, label: 'Audit Log' },
  ] },
]

const ALL = NAV.flatMap((g) => g.items)

function UserBadge() {
  const { session, isAdmin, logout } = useSession()
  if (!session) return null
  if (session.mode === 'open') {
    return (
      <div className="user-badge" title="auth.enabled is false — anyone who can reach this port has full access">
        <span className="pill warn">Open mode</span>
        <span className="ub-meta">no sign-in</span>
      </div>
    )
  }
  const scopes = (session.scopes || []).join(' ') || 'no scopes'
  const tip = `Token ${session.jti_prefix || '?'} · ${scopes} · expires ${expiresIn(session.expires_at)}`
  return (
    <div className="user-badge" title={tip}>
      <span className="dot ok" />
      <span className="ub-name">{sessionLabel(session)}</span>
      {session.sub && session.jti_prefix && <span className="ub-meta">{session.jti_prefix}</span>}
      {isAdmin && <span className="pill run">admin</span>}
      <button className="theme-btn" onClick={logout} title="Sign out" aria-label="Sign out"><LogOut size={13} /><span className="btn-label">Sign out</span></button>
    </div>
  )
}

export default function Layout() {
  const loc = useLocation()
  const [theme, setTheme] = useState(document.documentElement.className || 'dark')
  // Below 900px the sidebar is off-canvas; the menu button in the top bar opens it.
  const [navOpen, setNavOpen] = useState(false)
  useEffect(() => { setNavOpen(false) }, [loc.pathname]) // close after navigating
  useEffect(() => {
    if (!navOpen) return
    const onKey = (e) => { if (e.key === 'Escape') setNavOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navOpen])

  const active = [...ALL].sort((a, b) => b.to.length - a.to.length)
    .find((i) => (i.end ? loc.pathname === '/' : loc.pathname.startsWith(i.to)))
  const title = active ? active.label : 'Overview'

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    document.documentElement.className = next
    try { saveTheme(window.localStorage, next) } catch { /* ignore */ }
    setTheme(next)
  }

  return (
    <>
      <aside id="sidebar" className={navOpen ? 'open' : undefined}>
        <div className="brand">
          <div className="logo">Folio</div>
          <div className="sub">Retrieval Terminal</div>
        </div>
        <nav>
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="nav-group">{g.group}</div>
              {g.items.map((it) => {
                const Icon = it.icon
                return (
                  <NavLink
                    key={it.to}
                    to={it.to}
                    end={it.end}
                    className={({ isActive }) => 'nav-item' + (isActive ? ' active' : '')}
                  >
                    <Icon strokeWidth={1.7} />
                    {it.label}
                  </NavLink>
                )
              })}
            </div>
          ))}
        </nav>
        <div className="side-foot">
          <div className="side-ver">Folio v1.1.0</div>
          <button className="theme-btn" onClick={toggleTheme}>
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
      </aside>

      <div className={'scrim' + (navOpen ? ' show' : '')} onClick={() => setNavOpen(false)} aria-hidden="true" />

      <main>
        <header className="top">
          <button className="menu-btn" type="button" onClick={() => setNavOpen((o) => !o)}
            aria-label="Open navigation" aria-controls="sidebar" aria-expanded={navOpen}>
            <Menu />
          </button>
          <div className="title">{title}</div>
          <div className="grow" />
          <UserBadge />
        </header>
        <div className="content">
          <Outlet />
        </div>
      </main>
    </>
  )
}
