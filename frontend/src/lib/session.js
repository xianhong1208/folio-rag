// Pure helpers for the console session (no DOM, no fetch) — unit-tested in node.

// Header the backend requires on every cookie-authenticated non-GET request (CSRF).
export const CSRF_HEADER = 'X-RAG-Console'

const SAFE = new Set(['GET', 'HEAD', 'OPTIONS'])

export function needsCsrf(method) {
  return !SAFE.has(String(method || 'GET').toUpperCase())
}

// Headers for a request: JSON content type when there is a JSON body, plus the
// CSRF header on unsafe methods.
export function requestHeaders(method, { json = true } = {}) {
  const h = {}
  if (json) h['Content-Type'] = 'application/json'
  if (needsCsrf(method)) h[CSRF_HEADER] = '1'
  return h
}

// The backend reports errors as {detail: {error, error_description}}, {detail: "..."},
// a FastAPI validation list, or {message}. Return {code, message}.
export function parseApiError(data, status) {
  const d = data && data.detail
  if (d && typeof d === 'object' && !Array.isArray(d)) {
    return { code: d.error || null, message: d.error_description || d.error || `HTTP ${status}` }
  }
  if (typeof d === 'string') return { code: null, message: d }
  if (Array.isArray(d) && d[0] && d[0].msg) return { code: null, message: d[0].msg }
  if (data && data.message) return { code: data.error || null, message: data.message }
  return { code: null, message: `HTTP ${status}` }
}

const LOGIN_MESSAGES = {
  token_expired: 'This token has expired. Issue a new one in MCP Gate.',
  wrong_audience: 'This token was issued for a different service. Issue a token for Agentic RAG in MCP Gate.',
  wrong_issuer: 'This token was not issued by the MCP Gate this server trusts.',
  bad_signature: 'The token signature is invalid.',
  unknown_signing_key: 'The token was not signed by the trusted MCP Gate.',
  malformed_token: 'That does not look like an access token.',
  missing_jti: 'This token has no token ID (jti). Use an MCP Gate personal access token.',
  insufficient_scope: 'This token is missing a scope the server requires.',
  jwks_unavailable: 'Cannot reach MCP Gate to check the token. Try again shortly.',
  login_required: 'Paste an access token to sign in.',
  csrf_failed: 'The sign-in request was refused (cross-origin). Reload the page and try again.',
}

// A login-screen message for an error code, falling back to the server's text.
export function loginErrorMessage(code, fallback) {
  return LOGIN_MESSAGES[code] || fallback || 'Sign-in failed.'
}

// Why the user is looking at the login screen after having been signed in.
export function expiredNotice(code) {
  if (code === 'session_expired') return 'Your session has expired. Sign in again.'
  if (code === 'invalid_session') return 'Your session is no longer valid. Sign in again.'
  return null
}

// Display name for the user badge: the token subject, else its id prefix.
export function sessionLabel(s) {
  if (!s) return ''
  if (s.mode === 'open') return 'Open mode'
  return s.sub || s.jti_prefix || 'Signed in'
}

// "in 3h 20m" style countdown to the session expiry (seconds since epoch).
export function expiresIn(expiresAt, nowMs = Date.now()) {
  if (!expiresAt) return ''
  const s = Math.floor(expiresAt - nowMs / 1000)
  if (s <= 0) return 'expired'
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (h >= 24) return `in ${Math.floor(h / 24)}d ${h % 24}h`
  if (h > 0) return `in ${h}h ${m}m`
  return `in ${Math.max(m, 1)}m`
}

// Strip whitespace and a pasted "Bearer " prefix from a token.
export function cleanToken(raw) {
  const t = String(raw || '').trim()
  return t.replace(/^bearer\s+/i, '').trim()
}
