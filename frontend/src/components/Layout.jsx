import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useState } from 'react'
import {
  Home, Folder, Zap, Search, BarChart3, FlaskConical, Layers, Sparkles,
  ListTree, ArrowDownUp, Mic, SlidersHorizontal, Activity, ScrollText, Sun, Moon,
  Languages,
} from 'lucide-react'
import { useI18n } from '../contexts/I18nContext'

// `key` is an i18n key (nav.*); labels are resolved through t() at render time.
const NAV = [
  { group: 'nav.group.console', items: [
    { to: '/', end: true, icon: Home, key: 'nav.overview' },
    { to: '/folders', icon: Folder, key: 'nav.folders' },
    { to: '/jobs', icon: Zap, key: 'nav.jobs' },
  ] },
  { group: 'nav.group.search', items: [
    { to: '/search', icon: Search, key: 'nav.search' },
    { to: '/analytics', icon: BarChart3, key: 'nav.analytics' },
    { to: '/eval', icon: FlaskConical, key: 'nav.eval' },
  ] },
  { group: 'nav.group.settings', items: [
    { to: '/settings/embedding', icon: Layers, key: 'nav.embedding' },
    { to: '/settings/llm', icon: Sparkles, key: 'nav.llm' },
    { to: '/settings/contextual', icon: ListTree, key: 'nav.contextual' },
    { to: '/settings/reranker', icon: ArrowDownUp, key: 'nav.reranker' },
    { to: '/settings/asr', icon: Mic, key: 'nav.asr' },
    { to: '/settings/retrieval', icon: SlidersHorizontal, key: 'nav.retrieval' },
  ] },
  { group: 'nav.group.system', items: [
    { to: '/health', icon: Activity, key: 'nav.health' },
    { to: '/audit', icon: ScrollText, key: 'nav.audit' },
  ] },
]

const ALL = NAV.flatMap((g) => g.items)

export default function Layout() {
  const loc = useLocation()
  const { t, lang, setLang } = useI18n()
  const [theme, setTheme] = useState(document.documentElement.className || 'dark')

  const active = [...ALL].sort((a, b) => b.to.length - a.to.length)
    .find((i) => (i.end ? loc.pathname === '/' : loc.pathname.startsWith(i.to)))
  const title = t(active ? active.key : 'nav.overview')

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    document.documentElement.className = next
    try { localStorage.setItem('agentic_theme', next) } catch { /* ignore */ }
    setTheme(next)
  }
  const toggleLang = () => setLang(lang === 'en' ? 'zh-TW' : 'en')

  return (
    <>
      <aside>
        <div className="brand">
          <div className="logo">Agentic RAG</div>
          <div className="sub">{t('brand.sub')}</div>
        </div>
        <nav>
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="nav-group">{t(g.group)}</div>
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
                    {t(it.key)}
                  </NavLink>
                )
              })}
            </div>
          ))}
        </nav>
        <div className="side-foot">
          <div className="side-ver">{t('side.version')}</div>
          <div className="side-foot-btns">
            <button className="theme-btn" onClick={toggleLang} title={t('lang.label')}>
              <Languages size={14} />
              {lang === 'en' ? '中文' : 'EN'}
            </button>
            <button className="theme-btn" onClick={toggleTheme}>
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
              {theme === 'dark' ? t('theme.light') : t('theme.dark')}
            </button>
          </div>
        </div>
      </aside>

      <main>
        <header className="top">
          <div className="title">{title}</div>
          <div className="grow" />
          <div className="statusline"><span className="dot ok" /> {t('top.connected')}</div>
        </header>
        <div className="content">
          <Outlet />
        </div>
      </main>
    </>
  )
}
