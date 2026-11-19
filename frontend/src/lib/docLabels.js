// Tags and metadata of one document, as the console edits them. The server applies
// the real rules (src/domain/rag/doc_meta.py) and answers 422 with a message; these
// helpers only turn the form's text into the request body and back.

// "invoice, Signed ,invoice" -> ["invoice", "Signed"]: trimmed, inner whitespace
// collapsed, case-insensitive duplicates dropped (first spelling kept).
export function parseTags(text) {
  const out = []
  const seen = new Set()
  for (const raw of String(text || '').split(',')) {
    const t = raw.trim().replace(/\s+/g, ' ')
    if (!t || seen.has(t.toLowerCase())) continue
    seen.add(t.toLowerCase())
    out.push(t)
  }
  return out
}

export const formatTags = (tags) => (Array.isArray(tags) ? tags.join(', ') : '')

// One "key: value" per line. true / false become booleans, plain numbers numbers,
// anything else (dates included, written YYYY-MM-DD) a string. Wrap a value in
// double quotes to keep it a string ("2024", "true").
export function parseMetadata(text) {
  const value = {}
  const lines = String(text || '').split('\n')
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) continue
    const at = line.indexOf(':')
    if (at <= 0) return { error: `Line ${i + 1}: write key: value` }
    const key = line.slice(0, at).trim()
    const raw = line.slice(at + 1).trim()
    if (key in value) return { error: `Line ${i + 1}: "${key}" is set twice` }
    value[key] = toValue(raw)
  }
  return { value }
}

function toValue(raw) {
  if (raw.length >= 2 && raw.startsWith('"') && raw.endsWith('"')) return raw.slice(1, -1)
  if (raw === 'true') return true
  if (raw === 'false') return false
  if (/^-?\d+(\.\d+)?$/.test(raw)) return Number(raw)
  return raw
}

export function formatMetadata(meta) {
  if (!meta || typeof meta !== 'object') return ''
  return Object.entries(meta).map(([k, v]) => {
    const keepString = typeof v === 'string' && toValue(v) !== v
    return `${k}: ${keepString ? `"${v}"` : String(v)}`
  }).join('\n')
}
