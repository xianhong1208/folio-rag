import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { Home, Library, Search, BarChart3, SlidersHorizontal, Activity, Sun, Moon, LogOut, Menu } from 'lucide-react'
import { useSession } from '../contexts/SessionContext'
import { sessionLabel, expiresIn } from '../lib/session'
import { saveTheme } from '../lib/theme'
import { NAV, navEntryFor, activeTab, crumbs } from '../lib/nav'
import PageCap from './PageCap'

const ICONS = { overview: Home, library: Library, search: Search, insights: BarChart3, settings: SlidersHorizontal, system: Activity }

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

function SubTabs({ entry }) {
  if (!entry.tabs) return null
  return (
    <nav className="subtabs" aria-label={entry.label}>
      {entry.tabs.map((t) => (
        <NavLink key={t.to} to={t.to} className={({ isActive }) => 'subtab' + (isActive ? ' active' : '')}>{t.label}</NavLink>
      ))}
    </nav>
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

  const entry = navEntryFor(loc.pathname)
  const tab = activeTab(entry, loc.pathname)
  const path = crumbs(loc.pathname)
  // Settings draws its own caption + tab strip (its tabs are part of the form, not routes of their own).
  const ownCaption = entry.key === 'overview' || entry.key === 'settings'

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
          <div className="logo">Folio<span className="cursor blink">&#9613;</span></div>
          <div className="sub">retrieval terminal</div>
        </div>
        <nav className="side-nav">
          {NAV.map((it, i) => {
            const Icon = ICONS[it.key]
            const active = it.key === entry.key
            return (
              <NavLink key={it.key} to={it.to} end={it.end} aria-current={active ? 'page' : undefined}
                className={'nav-item' + (active ? ' active' : '')}>
                <span className="nav-no">{String(i + 1).padStart(2, '0')}</span>
                <Icon strokeWidth={1.7} />
                <span className="nav-label">{it.label}</span>
                {active && <span className="nav-caret blink" aria-hidden="true">&#9613;</span>}
              </NavLink>
            )
          })}
        </nav>
        <div className="side-foot">
          <div className="side-ver">folio v1.3.0</div>
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
          <div className="title crumbs" aria-label={tab ? tab.label : entry.label}>
            {path.map((p, i) => <span key={i} className={i === path.length - 1 ? 'here' : undefined}>{p}</span>)}
          </div>
          <div className="grow" />
          <UserBadge />
        </header>
        <div className="content">
          {!ownCaption && <PageCap title={entry.label} query={entry.query} />}
          {!ownCaption && <SubTabs entry={entry} />}
          <Outlet />
        </div>
      </main>
    </>
  )
}
