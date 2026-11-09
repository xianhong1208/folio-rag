// Caption under the Overview's Folders KPI. Admins also get the number of folders,
// server-wide, that still belong to a pre-1.5 per-token key: each moves to its person
// the next time its token is used.
export function foldersFoot(d) {
  const base = d.scope === 'all' ? 'all owners' : 'owned by you'
  const n = d.legacy_owned_folders
  return typeof n === 'number' && n > 0 ? `${base} · ${n} legacy unclaimed` : base
}
