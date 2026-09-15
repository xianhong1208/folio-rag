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

// POST a request and consume a Server-Sent Events stream, invoking onEvent(obj)
// once per `data:` frame. Resolves when the stream ends; rejects on HTTP error
// or an aborted signal. Frames are JSON; malformed frames are skipped.
export async function stream(path, body, onEvent, signal) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    let msg = `HTTP ${res.status}`
    try {
      const d = await res.json()
      msg = (d && (d.detail?.error || (typeof d.detail === 'string' ? d.detail : null) || d.message)) || msg
    } catch { /* non-JSON */ }
    throw new Error(msg)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      const line = frame.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      try { onEvent(JSON.parse(line.slice(5).trim())) } catch { /* skip malformed frame */ }
    }
  }
}
