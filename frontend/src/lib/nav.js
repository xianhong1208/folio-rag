// Console navigation model: six sidebar entries, some with sub-pages shown as tabs under the
// page caption. Routes are unchanged (/folders, /jobs, …) so bookmarks and links keep working;
// only the grouping moved out of the sidebar.

export const NAV = [
  { key: 'overview', label: 'Overview', to: '/', end: true, query: 'hierarchical · auto-merge · hybrid retrieval' },
  { key: 'library', label: 'Library', to: '/folders', query: 'folders · files · index jobs',
    tabs: [{ to: '/folders', label: 'Folders' }, { to: '/jobs', label: 'Index Jobs' }] },
  { key: 'search', label: 'Search', to: '/search', query: 'ask a folder · inspect every hit' },
  { key: 'insights', label: 'Insights', to: '/analytics', query: 'what people ask · how well it answers',
    tabs: [{ to: '/analytics', label: 'Query Analytics' }, { to: '/eval', label: 'Evaluation' }] },
  { key: 'settings', label: 'Settings', to: '/settings', query: 'models · indexing · retrieval · speech' },
  { key: 'system', label: 'System', to: '/health', query: 'health · audit trail',
    tabs: [{ to: '/health', label: 'Health' }, { to: '/audit', label: 'Audit Log' }] },
]

const under = (path, prefix) => path === prefix || path.startsWith(prefix + '/')

// The sidebar entry a pathname belongs to (the Overview for "/" and anything unknown).
export function navEntryFor(pathname) {
  const path = pathname.replace(/\/+$/, '') || '/'
  for (const e of NAV) {
    if (e.end) continue
    if (under(path, e.to) || (e.tabs || []).some((t) => under(path, t.to))) return e
  }
  return NAV[0]
}

// The active sub-page tab of an entry, if it has tabs.
export function activeTab(entry, pathname) {
  return (entry.tabs || []).find((t) => under(pathname.replace(/\/+$/, ''), t.to)) || null
}

// "folio / library / index jobs" — the path shown in the top bar.
export function crumbs(pathname) {
  const e = navEntryFor(pathname)
  const t = activeTab(e, pathname)
  return ['folio', e.label.toLowerCase(), ...(t && t.label.toLowerCase() !== e.label.toLowerCase() ? [t.label.toLowerCase()] : [])]
}
