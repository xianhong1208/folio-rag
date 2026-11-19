// Settings page model: which tab shows which section, and the pending-change set behind the
// single "Apply" bar. Edits from every tab accumulate in one `dirty` map (path -> new value);
// nothing reaches the server until Apply, which sends them in one PUT.
import { FIELDS, META, sectionOf } from './settings-config'

export const TABS = [
  { key: 'models', label: 'Models', query: 'who embeds, who answers',
    sections: ['rag.embedding', 'rag.llm', 'rag.contextual_llm'] },
  { key: 'indexing', label: 'Indexing', query: 'how documents become chunks',
    sections: ['rag.parsing', 'rag.contextual_retrieval'] },
  { key: 'retrieval', label: 'Retrieval', query: 'how a query finds them again',
    sections: ['rag.retrieval', 'rag.rerank'] },
  { key: 'speech', label: 'Speech', query: 'audio in, text out',
    sections: ['rag.asr'] },
]

// The contextual-prefix override is optional; the Models tab folds it away under the main LLM.
export const OVERRIDE_SECTIONS = ['rag.contextual_llm']

// The old console linked /settings/<section> (e.g. /settings/embedding); keep those links working.
// answer-llm / eval-llm were removed in 1.5 (answers and evaluation use the LLM); their links
// still open the Models tab.
const LEGACY = { embedding: 'models', llm: 'models', 'contextual-llm': 'models', 'answer-llm': 'models', 'eval-llm': 'models',
  contextual: 'indexing', parsing: 'indexing', reranker: 'retrieval', retrieval: 'retrieval', asr: 'speech' }

export function tabFromParam(param) {
  if (TABS.some((t) => t.key === param)) return param
  return LEGACY[param] || 'models'
}

export function tabOfSection(sec) {
  return (TABS.find((t) => t.sections.includes(sec)) || TABS[0]).key
}

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null)

// Pending changes, in field order: [{path, label, section, tab, from, to}]. An edit that was
// typed back to the saved value is not a change.
export function diffChanges(dirty, state) {
  const out = []
  for (const path of Object.keys(FIELDS)) {
    if (!(path in dirty) || !state || !(path in state)) continue
    const from = state[path].value
    const to = dirty[path]
    if (same(from, to)) continue
    const section = sectionOf(path)
    out.push({ path, label: `${META[section]?.title || section} · ${FIELDS[path].label}`, section, tab: tabOfSection(section), from, to, secret: FIELDS[path].type === 'secret' })
  }
  return out
}

// Number of pending changes per tab, for the dots on the tab strip.
export function changesPerTab(changes) {
  const n = {}
  for (const c of changes) n[c.tab] = (n[c.tab] || 0) + 1
  return n
}

// A preset only fills the form: its values join the pending set (overwriting edits to the same
// fields) and are reviewed and applied like any other change.
export function stagePreset(dirty, patch) {
  return { ...dirty, ...patch }
}

// Changes that invalidate existing vectors (embedding section) need a re-index warning on Apply.
export function needsReindex(changes) {
  return changes.some((c) => META[c.section]?.warn)
}

// How a value is shown in the review list. Secrets are never echoed back.
export function showValue(v, secret) {
  if (secret) return v ? 'new key' : 'unset'
  if (v === null || v === undefined || v === '') return '—'
  if (typeof v === 'boolean') return v ? 'on' : 'off'
  const s = String(v)
  return s.length > 48 ? s.slice(0, 47) + '…' : s
}
