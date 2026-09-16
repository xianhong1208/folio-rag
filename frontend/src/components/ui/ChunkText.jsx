// Renders a retrieval/chunk body. When the chunk is a Docling table
// (content_type === 'table'), its text is GitHub-flavored Markdown; parse it into
// a real HTML table. Anything that fails to parse falls back to plain text, so a
// mis-flagged chunk never renders blank.

export function parseMarkdownTable(text) {
  const lines = text.trim().split('\n').map((l) => l.trim()).filter(Boolean)
  const rows = lines.filter((l) => l.includes('|'))
  if (rows.length < 2) return null
  const cells = (l) => l.replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim())
  const isSep = (l) => l.includes('-') && /^[\s:|-]+$/.test(l.replace(/\|/g, ''))
  const sepIdx = rows.findIndex((l, i) => i > 0 && isSep(l))
  if (sepIdx < 1) return null
  const header = cells(rows[sepIdx - 1])
  const body = rows.slice(sepIdx + 1).filter((l) => !isSep(l)).map(cells)
  if (!body.length) return null
  return { header, body }
}

// Docling's chunker serializes a table as row/column triplets:
//   "<row label>, <column> = <value>. <row label>, <column> = <value>. …"
// often preceded by a contextual-retrieval prefix paragraph. Rebuild the grid:
// rows in first-seen order, columns in first-seen order, blank for missing cells.
export function parseTripletTable(text) {
  const src = (text || '').trim()
  if (!src) return null
  // A prefix paragraph (contextual retrieval) ends with a blank line.
  const cut = src.lastIndexOf('\n\n')
  const body = cut >= 0 && src.slice(cut).includes(' = ') ? src.slice(cut + 2) : src
  const caption = cut >= 0 && body !== src ? src.slice(0, cut).trim() : ''
  const re = /([^.\n]+?),\s([^=\n]+?)\s=\s([^.\n]*?)\.(?=\s|$)/g
  const rows = new Map()
  const cols = []
  let m
  while ((m = re.exec(body)) !== null) {
    const [, row, col, val] = m.map((s) => (s == null ? s : s.trim()))
    if (!rows.has(row)) rows.set(row, {})
    if (!cols.includes(col)) cols.push(col)
    rows.get(row)[col] = val
  }
  if (rows.size < 1 || cols.length < 1 || rows.size * cols.length < 2) return null
  return {
    caption,
    header: ['', ...cols],
    body: [...rows.entries()].map(([row, cells]) => [row, ...cols.map((c) => cells[c] ?? '')]),
  }
}

export function ContentBadge({ type }) {
  if (type !== 'table' && type !== 'picture') return null
  return <span className={'type-badge ' + type}>{type === 'table' ? 'TABLE' : 'FIGURE'}</span>
}

export default function ChunkText({ text, contentType, className }) {
  if (contentType === 'table') {
    const t = parseMarkdownTable(text || '') || parseTripletTable(text || '')
    if (t) {
      return (
        // Don't reuse the caller's `c-text` class here: under `.chunk-row .c-text`
        // it carries a 160px max-height clamp meant for long plain-text chunks,
        // which would squeeze the table into a tiny inner scrollbox. The wrapper
        // has its own overflow-x + spacing in `.chunk-table-wrap`.
        <div className="chunk-table-wrap">
          {t.caption && <div className="c-meta" style={{ padding: '6px 9px 0' }}>{t.caption}</div>}
          <table className="chunk-table">
            <thead><tr>{t.header.map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
            <tbody>{t.body.map((r, i) => (
              <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>
            ))}</tbody>
          </table>
        </div>
      )
    }
  }
  return <div className={className}>{text}</div>
}
