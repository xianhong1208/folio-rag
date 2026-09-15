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

export function ContentBadge({ type }) {
  if (type !== 'table' && type !== 'picture') return null
  return <span className={'type-badge ' + type}>{type === 'table' ? 'TABLE' : 'FIGURE'}</span>
}

export default function ChunkText({ text, contentType, className }) {
  if (contentType === 'table') {
    const t = parseMarkdownTable(text || '')
    if (t) {
      return (
        <div className={'chunk-table-wrap ' + (className || '')}>
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
