// Folio admin API client. Same-origin under /admin, hitting /api/admin/* and
// /api/console/session on the backend. Authentication is the HttpOnly session cookie
// set at sign-in (the browser sends it; the token itself never reaches JS storage).
// Every non-GET request carries the X-RAG-Console CSRF header the backend requires.
import { requestHeaders, parseApiError } from '../lib/session'

const API_BASE = ''

// Listeners told when the server says the session is gone (401), so the app can
// drop back to the login screen. Receives the error code (e.g. session_expired).
const unauthorizedListeners = new Set()
export function onUnauthorized(fn) {
  unauthorizedListeners.add(fn)
  return () => unauthorizedListeners.delete(fn)
}

export class ApiError extends Error {
  constructor(message, status, code) {
    super(message)
    this.status = status
    this.code = code
  }
}

async function toError(res, { notify = true } = {}) {
  let data = null
  try { data = await res.json() } catch { /* non-JSON */ }
  const { code, message } = parseApiError(data, res.status)
  if (res.status === 401 && notify) unauthorizedListeners.forEach((fn) => fn(code))
  return new ApiError(message, res.status, code)
}

export async function api(method, path, body, { notify = true } = {}) {
  const opts = { method, headers: requestHeaders(method), credentials: 'same-origin' }
  if (body != null) opts.body = JSON.stringify(body)
  const res = await fetch(API_BASE + path, opts)
  if (!res.ok) throw await toError(res, { notify })
  try { return await res.json() } catch { return null }
}

export const get = (path) => api('GET', path)
export const post = (path, body) => api('POST', path, body)

// Multipart upload (one FormData per request); the browser sets the content type.
export async function upload(path, formData) {
  const res = await fetch(API_BASE + path, {
    method: 'POST', body: formData, credentials: 'same-origin',
    headers: requestHeaders('POST', { json: false }),
  })
  if (!res.ok) throw await toError(res)
  try { return await res.json() } catch { return null }
}

// POST a request and consume a Server-Sent Events stream, invoking onEvent(obj)
// once per `data:` frame. Resolves when the stream ends; rejects on HTTP error
// or an aborted signal. Frames are JSON; malformed frames are skipped.
export async function stream(path, body, onEvent, signal) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: requestHeaders('POST'),
    credentials: 'same-origin',
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) throw await toError(res)
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

// ---- console session ----
export const getSession = () => api('GET', '/api/console/session', null, { notify: false })
export const signIn = (token) => api('POST', '/api/console/session', { token }, { notify: false })
export const signOut = () => api('DELETE', '/api/console/session', null, { notify: false })
