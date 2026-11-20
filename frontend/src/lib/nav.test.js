import { describe, it, expect } from 'vitest'
import { NAV, navEntryFor, activeTab, crumbs } from './nav'

describe('navEntryFor', () => {
  it('maps every sub-page route to its sidebar entry', () => {
    expect(navEntryFor('/folders').key).toBe('folders')
    expect(navEntryFor('/jobs').key).toBe('folders')
    expect(navEntryFor('/library').key).toBe('library')
    expect(navEntryFor('/analytics').key).toBe('insights')
    expect(navEntryFor('/eval').key).toBe('insights')
    expect(navEntryFor('/health').key).toBe('system')
    expect(navEntryFor('/audit').key).toBe('system')
    expect(navEntryFor('/search').key).toBe('search')
  })
  it('keeps the settings tabs under Settings, old section links included', () => {
    expect(navEntryFor('/settings').key).toBe('settings')
    expect(navEntryFor('/settings/retrieval').key).toBe('settings')
    expect(navEntryFor('/settings/embedding/').key).toBe('settings')
  })
  it('falls back to the Overview for / and unknown paths', () => {
    expect(navEntryFor('/').key).toBe('overview')
    expect(navEntryFor('/nope').key).toBe('overview')
  })
  it('does not match on a bare prefix (/searching is not /search)', () => {
    expect(navEntryFor('/searching').key).toBe('overview')
  })
})

describe('activeTab and crumbs', () => {
  it('finds the active tab', () => {
    const fol = NAV.find((e) => e.key === 'folders')
    expect(activeTab(fol, '/jobs').label).toBe('Index Jobs')
    expect(activeTab(NAV.find((e) => e.key === 'library'), '/library')).toBeNull()
    expect(activeTab(NAV[0], '/')).toBeNull()
  })
  it('builds the top-bar path', () => {
    expect(crumbs('/jobs')).toEqual(['folio', 'folders', 'index jobs'])
    expect(crumbs('/folders')).toEqual(['folio', 'folders'])
    expect(crumbs('/library')).toEqual(['folio', 'library'])
    expect(crumbs('/search')).toEqual(['folio', 'search'])
    expect(crumbs('/')).toEqual(['folio', 'overview'])
  })
  it('the sidebar stays short and every entry has a caption', () => {
    expect(NAV.length).toBeLessThanOrEqual(7)
    for (const e of NAV) expect(e.query).toBeTruthy()
  })
})
