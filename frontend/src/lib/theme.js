// Console theme, remembered per viewer in localStorage.
//
// The key was 'agentic_theme' before the project became Folio. migrateStorageKeys()
// moves a value stored under an old key to its new key once (read old, write
// new, delete old), so the rename does not reset anyone's theme.

export const THEME_KEY = 'folio_theme'

// new key -> old key
export const RENAMED_KEYS = { [THEME_KEY]: 'agentic_theme' }

export function migrateStorageKeys(storage) {
  if (!storage) return
  for (const [key, old] of Object.entries(RENAMED_KEYS)) {
    try {
      const legacy = storage.getItem(old)
      if (legacy === null) continue
      if (storage.getItem(key) === null) storage.setItem(key, legacy)
      storage.removeItem(old)
    } catch { /* storage unavailable (private mode, blocked): nothing to migrate */ }
  }
}

export function readTheme(storage) {
  try {
    const t = storage && storage.getItem(THEME_KEY)
    return t === 'light' || t === 'dark' ? t : 'dark'
  } catch { return 'dark' }
}

export function saveTheme(storage, theme) {
  try { storage && storage.setItem(THEME_KEY, theme) } catch { /* ignore */ }
}
