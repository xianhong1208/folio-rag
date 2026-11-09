import { describe, it, expect } from 'vitest'
import { auditEntry } from './audit'

describe('auditEntry', () => {
  it('shows settings changes as before', () => {
    expect(auditEntry({ action: 'set', key: 'rag.top_k', new: 8 }))
      .toEqual({ cls: 'set', act: 'SET', key: 'rag.top_k', value: '8' })
    expect(auditEntry({ action: 'reset', key: 'rag.top_k', new: null }))
      .toEqual({ cls: 'reset', act: 'RESET', key: 'rag.top_k', value: '—' })
  })
  it('shows a folder renamed while being claimed as old → new', () => {
    expect(auditEntry({ action: 'claim_rename', key: 'folder:12', old: 'docs', new: 'docs-2' }))
      .toEqual({ cls: 'set', act: 'RENAME', key: 'folder:12 docs', value: 'docs-2' })
  })
})
