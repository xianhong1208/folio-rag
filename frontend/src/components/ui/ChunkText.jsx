// Renders a retrieval/chunk body. When the chunk is a Docling table
// (content_type === 'table'), its text is the chunker's *triplet* serialization
// ("<row>, <col> = <val>. …") — see parseTripletTable. (Markdown pipe tables are
// also accepted for safety, but Docling's default TripletTableSerializer never
// emits them.) Anything that fails to parse falls back to plain text, so a
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
//   "<row label>, <column> = <value>. <row label>, <column> = <value> …"
// joined with ". " (period + space) and with NO trailing period after the last
// triplet, often preceded by a contextual-retrieval prefix paragraph. Rebuild the
// grid: rows in first-seen order, columns in first-seen order, blank for missing
// cells.
//
// We split on the ". " triplet delimiter rather than regex-matching the period,
// because real cell data routinely contains periods — float-rendered row labels
// ("1.0"), decimal values ("64.5"), etc. Splitting on ". " keeps those intact
// (a decimal has no space after its point) and never drops the final cell.
// (Known limit: a cell whose text itself contains ". " can over-split; the robust
// fix is a structured backend payload — tracked in the issue.)
export function parseTripletTable(text) {
  const src = (text || '').trim()
  if (!src) return null
  // A prefix paragraph (contextual retrieval) ends with a blank line.
  const cut = src.lastIndexOf('\n\n')
  const body = cut >= 0 && src.slice(cut).includes(' = ') ? src.slice(cut + 2) : src
  const caption = cut >= 0 && body !== src ? src.slice(0, cut).trim() : ''
  const rows = new Map()
  const cols = []
  for (let part of body.replace(/\.\s*$/, '').split('. ')) {
    part = part.trim()
    if (!part) continue
    const eq = part.indexOf(' = ')
    if (eq < 0) continue
    const left = part.slice(0, eq)
    const val = part.slice(eq + 3).trim()
    const comma = left.indexOf(', ') // first ", " splits row label from column
    if (comma < 0) continue
    const row = left.slice(0, comma).trim()
    const col = left.slice(comma + 2).trim()
    if (!row || !col) continue
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

// True when a single line is entirely Docling triplets ("<row>, <col> = <val>" parts
// joined by ". "). Stricter than parseTripletTable, which skips non-triplet parts: a
// prose line that happens to contain " = " must not turn into a table.
export function isTripletLine(line) {
  const parts = (line || '').trim().replace(/\.\s*$/, '').split('. ').map((p) => p.trim()).filter(Boolean)
  if (!parts.length) return false
  return parts.every((p) => {
    const eq = p.indexOf(' = ')
    const comma = p.indexOf(', ')
    return eq > 0 && comma > 0 && comma < eq
  })
}

// A table chunk can mix prose and tables: a short paragraph merged with the table
// that follows it in the same section, or (older indexes) a contextual prefix above
// the table. Docling writes each table on one line, so split by line and turn every
// triplet line into a table, keeping the prose lines as text around it.
export function splitChunkSegments(text) {
  const out = []
  for (const line of (text || '').split('\n')) {
    if (isTripletLine(line)) {
      const t = parseTripletTable(line)
      if (t) { out.push({ type: 'table', table: t }); continue }
    }
    const last = out[out.length - 1]
    if (last && last.type === 'text') last.text += '\n' + line
    else out.push({ type: 'text', text: line })
  }
  return out
    .map((s) => (s.type === 'text' ? { ...s, text: s.text.trim() } : s))
    .filter((s) => s.type === 'table' || s.text)
}

function TableView({ t }) {
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

export function ContentBadge({ type }) {
  if (type !== 'table' && type !== 'picture') return null
  return <span className={'type-badge ' + type}>{type === 'table' ? 'TABLE' : 'FIGURE'}</span>
}

export default function ChunkText({ text, contentType, className }) {
  if (contentType === 'table') {
    const md = parseMarkdownTable(text || '')
    if (md) return <TableView t={md} />
    const segments = splitChunkSegments(text)
    if (segments.some((s) => s.type === 'table')) {
      return (
        <div>
          {segments.map((s, i) => (s.type === 'table'
            ? <TableView key={i} t={s.table} />
            : <div key={i} className={className}>{s.text}</div>))}
        </div>
      )
    }
  }
  return <div className={className}>{text}</div>
}
