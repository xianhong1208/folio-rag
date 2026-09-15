import { createContext, useContext, useState, useCallback, useMemo } from 'react'
import { DICT, LANGS } from '../lib/i18n'

const I18nContext = createContext(null)

// Pick the initial language: a stored choice wins; otherwise infer from the
// browser (zh-* -> zh-TW) and default to English. Wrapped in try/catch because
// localStorage can throw in private windows / previews.
function initialLang() {
  try {
    const saved = localStorage.getItem('agentic_lang')
    if (saved && DICT[saved]) return saved
  } catch { /* ignore */ }
  try {
    if ((navigator.language || '').toLowerCase().startsWith('zh')) return 'zh-TW'
  } catch { /* ignore */ }
  return 'en'
}

export function I18nProvider({ children }) {
  const [lang, setLangState] = useState(initialLang)

  const setLang = useCallback((next) => {
    if (!DICT[next]) return
    setLangState(next)
    try { localStorage.setItem('agentic_lang', next) } catch { /* ignore */ }
    try { document.documentElement.lang = next } catch { /* ignore */ }
  }, [])

  // t(key, vars): current language -> English fallback -> the key itself.
  // {name} placeholders are filled from vars.
  const t = useCallback((key, vars) => {
    let s = (DICT[lang] && DICT[lang][key]) ?? DICT.en[key] ?? key
    if (vars) for (const k of Object.keys(vars)) s = s.replaceAll(`{${k}}`, String(vars[k]))
    return s
  }, [lang])

  const value = useMemo(() => ({ lang, setLang, t, langs: LANGS }), [lang, setLang, t])
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useI18n must be used within I18nProvider')
  return ctx
}
