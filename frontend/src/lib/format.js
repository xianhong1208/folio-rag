// The console is English-only, so format numbers and dates with a fixed en-US
// locale rather than the viewer's browser locale (which would render e.g. a
// zh-TW month as "9月" in the Indexed At column instead of "Sep").
export const fmtNum = (n) => (n == null ? '—' : Number(n).toLocaleString('en-US'))

export function fmtBytes(b) {
  if (b == null) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0, v = Number(b)
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`
}

export function fmtTime(t) {
  if (!t) return '—'
  const d = new Date(t)
  if (isNaN(d)) return '—'
  return d.toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

export const shortModel = (s) => (s || '—').split('/').pop()
