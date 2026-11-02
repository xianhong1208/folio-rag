// Map GET /api/admin/health checks to the Overview service pills.
// The health endpoint probes the *configured* endpoints server-side, so any signed-in
// user can see service status without the admin-only arbitrary-URL probe.
const BY_NAME = { Embedding: 'embed', LLM: 'llm', Reranker: 'rerank' }
const STATE = { ok: 'online', down: 'offline', unset: 'unset', disabled: 'disabled' }

export function serviceStates(checks) {
  const out = {}
  for (const c of checks || []) {
    const id = BY_NAME[c.name]
    if (id) out[id] = STATE[c.status] || 'offline'
  }
  return out
}
