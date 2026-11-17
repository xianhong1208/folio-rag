import { describe, it, expect } from 'vitest'
import { importPath, importProblem, needsIndexAfterImport, importWarnings, importSummary, exportProblem, looksLikeArchive } from './folderArchive'

describe('importPath', () => {
  it('sends only the name and files_only, never an owner', () => {
    expect(importPath()).toBe('/api/admin/manage/folders/import')
    expect(importPath({ name: '  docs copy ', filesOnly: true }))
      .toBe('/api/admin/manage/folders/import?name=docs+copy&files_only=true')
    expect(importPath({ name: '   ', filesOnly: false })).toBe('/api/admin/manage/folders/import')
    expect(importPath({ name: 'x', owner: 'sub:someone' })).not.toContain('owner')
  })
})

describe('importProblem', () => {
  it('offers files-only only for an embedding mismatch', () => {
    const p = importProblem({ code: 'embedding_mismatch', message: 'its vectors were made with bge' })
    expect(p.offerFilesOnly).toBe(true)
    expect(p.text).toContain('bge')
    expect(p.text.startsWith('Its vectors')).toBe(true)
    expect(p.text).toContain('import the files only')
    for (const code of ['newer_version', 'damaged_archive', 'too_large', null]) {
      expect(importProblem({ code, message: 'x' }).offerFilesOnly).toBe(false)
    }
  })
  it('explains a newer export and falls back to the message', () => {
    expect(importProblem({ code: 'newer_version', message: 'made by Folio 9' }).text).toContain('Upgrade')
    expect(importProblem({ code: 'damaged_archive', message: 'checksum mismatch' }).text).toBe('Checksum mismatch')
    expect(importProblem(null).text).toBe('Import failed')
  })
})

describe('after an import', () => {
  it('indexes only a files-only import that has files', () => {
    expect(needsIndexAfterImport({ indexed: false, files: 3 })).toBe(true)
    expect(needsIndexAfterImport({ indexed: false, files: 0 })).toBe(false)
    expect(needsIndexAfterImport({ indexed: true, files: 3 })).toBe(false)
    expect(needsIndexAfterImport(null)).toBe(false)
  })
  it('summarises the result', () => {
    expect(importSummary({ name: 'docs-2', files: 1, chunks: 4, indexed: true })).toBe('Imported as "docs-2" — 1 file, 4 chunks')
    expect(importSummary({ name: 'd', files: 2, chunks: 0, indexed: false })).toBe('Imported as "d" — 2 files, not indexed yet')
  })
})

describe('exportProblem / looksLikeArchive', () => {
  it('tells the person to wait for a running index job', () => {
    expect(exportProblem({ code: 'index_job_running', message: '1 index job(s) are running' })).toContain('Jobs')
    expect(exportProblem({ message: 'boom' })).toBe('boom')
  })
  it('accepts .tar files only', () => {
    expect(looksLikeArchive({ name: 'folio-folder-docs.TAR' })).toBe(true)
    expect(looksLikeArchive({ name: 'notes.zip' })).toBe(false)
    expect(looksLikeArchive(null)).toBe(false)
  })
})

describe('importWarnings', () => {
  it('drops the index-it-yourself hint only when the console indexes the folder', () => {
    const w = ['imported without the index (files only): index the folder in the console', 'renamed to docs-2']
    expect(importWarnings({ indexed: false, files: 2, warnings: w })).toEqual(['renamed to docs-2'])
    expect(importWarnings({ indexed: false, files: 0, warnings: w })).toEqual(w)
    expect(importWarnings({ indexed: true, files: 2, warnings: ['x'] })).toEqual(['x'])
    expect(importWarnings(null)).toEqual([])
  })
})
