import { useEffect, useRef, useState } from 'react'
import { Plus, X } from 'lucide-react'
import { FILE_TYPES, META_OPS, opTakesValue, rowComplete, toggle } from '../lib/libraryFilters'

// The document filter controls shared by the Library page (full rail) and the Search
// page (compact: tags, metadata and folders). `value` is a filter state
// (lib/libraryFilters.js); every change calls onChange with the next state (page 1).

const TAGS_SHOWN = 14

function Section({ title, extra, children }) {
  return (
    <div className="fr-sec">
      <div className="fr-head"><span>{title}</span>{extra}</div>
      {children}
    </div>
  )
}

function TagList({ value, tags, set }) {
  const [all, setAll] = useState(false)
  if (!tags) return <div className="fr-empty">loading…</div>
  if (!tags.length) return <div className="fr-empty">No tags yet. Add them to documents with bulk edit or the document panel.</div>
  // Selected tags stay visible even when they are past the cut.
  const shown = all ? tags : tags.filter((t, i) => i < TAGS_SHOWN || value.tags.some((x) => x.toLowerCase() === t.tag.toLowerCase()))
  return (
    <>
      <div className="fr-tags">
        {shown.map((t) => {
          const on = value.tags.some((x) => x.toLowerCase() === t.tag.toLowerCase())
          return (
            <button key={t.tag} type="button" className={'fr-tag' + (on ? ' on' : '')} aria-pressed={on}
              onClick={() => set({ tags: on ? value.tags.filter((x) => x.toLowerCase() !== t.tag.toLowerCase()) : [...value.tags, t.tag] })}>
              <span className="fr-tag-name">{t.tag}</span><span className="fr-count">{t.count}</span>
            </button>
          )
        })}
      </div>
      {tags.length > TAGS_SHOWN && (
        <button type="button" className="fr-more" onClick={() => setAll((a) => !a)}>
          {all ? 'Show fewer' : `Show all ${tags.length}`}
        </button>
      )}
    </>
  )
}

function CheckList({ items, selected, onToggle, empty }) {
  if (!items) return <div className="fr-empty">loading…</div>
  if (!items.length) return <div className="fr-empty">{empty}</div>
  return (
    <div className="fr-checks">
      {items.map((it) => (
        <label key={it.value} className="fr-check">
          <input type="checkbox" checked={selected.includes(it.value)} onChange={() => onToggle(it.value)} />
          <span className="fr-check-label">{it.label}</span>
          {it.count != null && <span className="fr-count">{it.count}</span>}
        </label>
      ))}
    </div>
  )
}

const sameRows = (a, b) => JSON.stringify(a) === JSON.stringify(b)

// Metadata condition rows. Rows being typed live here; only complete rows reach the
// filter (and the URL), a value 400 ms after the last keystroke.
function MetaRows({ value, keys, set }) {
  const [rows, setRows] = useState(value.meta)
  const committed = useRef(value.meta)
  const timer = useRef(null)
  // A change from outside (a chip removed, Clear all, back/forward) resets the rows.
  useEffect(() => {
    if (!sameRows(value.meta, committed.current)) { committed.current = value.meta; setRows(value.meta) }
  }, [value.meta])
  useEffect(() => () => clearTimeout(timer.current), [])

  const commit = (next, delay = 0) => {
    clearTimeout(timer.current)
    const run = () => {
      const done = next.filter(rowComplete)
      if (sameRows(done, committed.current)) return
      committed.current = done
      set({ meta: done })
    }
    if (delay) timer.current = setTimeout(run, delay)
    else run()
  }
  const update = (i, patch, delay = 0) => {
    const next = rows.map((r, j) => (j === i ? { ...r, ...patch } : r))
    setRows(next)
    commit(next, delay)
  }
  const add = () => setRows([...rows, { key: keys?.[0]?.key || '', op: 'eq', value: '' }])
  const remove = (i) => { const next = rows.filter((_, j) => j !== i); setRows(next); commit(next) }

  const keyNames = (keys || []).map((k) => k.key)
  return (
    <div className="fr-meta">
      {rows.map((r, i) => (
        <div className="fr-mrow" key={i}>
          <select aria-label="Metadata key" value={r.key} onChange={(e) => update(i, { key: e.target.value })}>
            {!keyNames.includes(r.key) && <option value={r.key}>{r.key || 'key…'}</option>}
            {keyNames.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
          <select aria-label="Operator" value={r.op} onChange={(e) => update(i, { op: e.target.value })}>
            {META_OPS.map((o) => <option key={o.op} value={o.op}>{o.label}</option>)}
          </select>
          {opTakesValue(r.op)
            ? <input aria-label="Value" placeholder={r.op === 'in' ? 'a, b' : 'value'} value={r.value}
              onChange={(e) => update(i, { value: e.target.value }, 400)}
              onKeyDown={(e) => { if (e.key === 'Enter') commit(rows) }} />
            : <span className="fr-mfill" />}
          <button type="button" className="fr-x" aria-label="Remove condition" title="Remove condition" onClick={() => remove(i)}><X size={12} /></button>
        </div>
      ))}
      {keys && keys.length === 0 && rows.length === 0
        ? <div className="fr-empty">No metadata yet.</div>
        : <button type="button" className="fr-more" onClick={add}><Plus size={12} /> Add condition</button>}
    </div>
  )
}

// A text box that reaches the filter 400 ms after the last keystroke (or on Enter).
function NameBox({ value, set }) {
  const [text, setText] = useState(value.name)
  const timer = useRef(null)
  useEffect(() => { setText(value.name) }, [value.name])
  useEffect(() => () => clearTimeout(timer.current), [])
  const commit = (t) => { clearTimeout(timer.current); if (t !== value.name) set({ name: t }) }
  return (
    <input className="fr-input" placeholder="part of the file name" aria-label="Name contains" value={text}
      onChange={(e) => { const t = e.target.value; setText(t); clearTimeout(timer.current); timer.current = setTimeout(() => commit(t), 400) }}
      onKeyDown={(e) => { if (e.key === 'Enter') commit(text) }} />
  )
}

export default function FilterRail({ value, onChange, tags, keys, folders, compact = false, showFolders = true }) {
  // Debounced inputs call set() after later renders: always build on the latest value.
  const latest = useRef(value)
  latest.current = value
  const set = (patch) => onChange({ ...latest.current, ...patch, page: 1 })
  const folderItems = folders && folders.map((f) => ({ value: f.id, label: f.name, count: f.file_count ?? f.total_files }))
  return (
    <div className={'frail' + (compact ? ' compact' : '')}>
      <Section title="Tags" extra={
        <div className="fr-mode" role="group" aria-label="Match tags">
          {['any', 'all'].map((m) => (
            <button key={m} type="button" className={value.tagMode === m ? 'on' : ''} aria-pressed={value.tagMode === m}
              onClick={() => set({ tagMode: m })} title={m === 'any' ? 'Documents with at least one of the tags' : 'Documents with all of the tags'}>{m}</button>
          ))}
        </div>}>
        <TagList value={value} tags={tags} set={set} />
      </Section>
      {showFolders && (
        <Section title="Folders">
          <CheckList items={folderItems} selected={value.folders} empty="No folders."
            onToggle={(id) => set({ folders: toggle(value.folders, id) })} />
        </Section>
      )}
      {!compact && (
        <Section title="File type">
          <div className="fr-types">
            {FILE_TYPES.map((t) => (
              <button key={t} type="button" className={'fr-type' + (value.types.includes(t) ? ' on' : '')}
                aria-pressed={value.types.includes(t)} onClick={() => set({ types: toggle(value.types, t) })}>{t}</button>
            ))}
          </div>
        </Section>
      )}
      {!compact && (
        <Section title="Uploaded">
          <div className="fr-dates">
            <label><span>from</span><input type="date" value={value.from} max={value.to || undefined} onChange={(e) => set({ from: e.target.value })} /></label>
            <label><span>to</span><input type="date" value={value.to} min={value.from || undefined} onChange={(e) => set({ to: e.target.value })} /></label>
          </div>
        </Section>
      )}
      {!compact && (
        <Section title="Name contains">
          <NameBox value={value} set={set} />
        </Section>
      )}
      <Section title="Metadata">
        <MetaRows value={value} keys={keys} set={set} />
      </Section>
    </div>
  )
}
