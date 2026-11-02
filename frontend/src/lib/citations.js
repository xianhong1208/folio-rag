// Split an answer into text and citation parts so [n] can link to result card #n.
// `valid` (from the server's `citations`) limits which numbers become links; when
// it is absent, any number that matches a result card (1..maxN) does.
export function citationParts(text, valid, maxN) {
  const out = []
  if (!text) return out
  const ok = (n) => (Array.isArray(valid) ? valid.includes(n) : n >= 1 && n <= (maxN || 0))
  const re = /\[(\d+)\]/g
  let last = 0
  let m
  while ((m = re.exec(text)) !== null) {
    const n = Number(m[1])
    if (!ok(n)) continue
    if (m.index > last) out.push({ t: 'text', v: text.slice(last, m.index) })
    out.push({ t: 'cite', n })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ t: 'text', v: text.slice(last) })
  return out
}
