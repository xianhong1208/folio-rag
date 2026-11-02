import { describe, expect, it } from 'vitest'
import { THEME_KEY, migrateStorageKeys, readTheme, saveTheme } from './theme.js'

function memoryStorage(init = {}) {
  const m = new Map(Object.entries(init))
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)) },
    removeItem: (k) => { m.delete(k) },
    dump: () => Object.fromEntries(m),
  }
}

describe('theme storage', () => {
  it('uses the folio_theme key', () => {
    expect(THEME_KEY).toBe('folio_theme')
  })

  it('moves the pre-rename key to the new one and deletes the old', () => {
    const s = memoryStorage({ agentic_theme: 'light' })
    migrateStorageKeys(s)
    expect(s.dump()).toEqual({ folio_theme: 'light' })
    expect(readTheme(s)).toBe('light')
  })

  it('keeps an existing new value and still drops the old key', () => {
    const s = memoryStorage({ agentic_theme: 'light', folio_theme: 'dark' })
    migrateStorageKeys(s)
    expect(s.dump()).toEqual({ folio_theme: 'dark' })
  })

  it('is a no-op when there is nothing to migrate', () => {
    const s = memoryStorage({ other: 'x' })
    migrateStorageKeys(s)
    expect(s.dump()).toEqual({ other: 'x' })
  })

  it('defaults to dark and ignores junk values', () => {
    expect(readTheme(memoryStorage())).toBe('dark')
    expect(readTheme(memoryStorage({ folio_theme: 'neon' }))).toBe('dark')
    expect(readTheme(null)).toBe('dark')
  })

  it('survives a storage that throws', () => {
    const broken = { getItem() { throw new Error('blocked') }, setItem() { throw new Error('blocked') }, removeItem() {} }
    expect(() => migrateStorageKeys(broken)).not.toThrow()
    expect(readTheme(broken)).toBe('dark')
    expect(() => saveTheme(broken, 'light')).not.toThrow()
  })

  it('saves under the new key', () => {
    const s = memoryStorage()
    saveTheme(s, 'light')
    expect(s.dump()).toEqual({ folio_theme: 'light' })
  })
})
