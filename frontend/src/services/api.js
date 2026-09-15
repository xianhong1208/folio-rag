// Agentic RAG admin API client. The console is unauthenticated (admin module),
// same-origin under /admin, hitting /api/admin/* on the backend.
const API_BASE = ''

export async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } }
  if (body != null) opts.body = JSON.stringify(body)
  const res = await fetch(API_BASE + path, opts)
  let data = null
  try { data = await res.json() } catch { /* non-JSON */ }
  if (!res.ok) {
    const msg =
      (data && (data.detail?.error || (typeof data.detail === 'string' ? data.detail : null) || data.message)) ||
      `HTTP ${res.status}`
    throw new Error(msg)
  }
  return data
}

export const get = (path) => api('GET', path)
export const post = (path, body) => api('POST', path, body)
