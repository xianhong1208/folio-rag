// Pure helpers for exporting and importing a folder archive (no DOM, no fetch) —
// unit-tested in node. The endpoints are in src/api/router/admin_folder_archive.py.

// POST path for an import: the archive is the raw body, options are query parameters.
// The new folder's owner is never sent: the server always uses the session's.
export function importPath({ name, filesOnly } = {}) {
  const q = new URLSearchParams()
  const n = (name || '').trim()
  if (n) q.set('name', n)
  if (filesOnly) q.set('files_only', 'true')
  const s = q.toString()
  return '/api/admin/manage/folders/import' + (s ? '?' + s : '')
}

// How the import dialog explains a refusal. `offerFilesOnly` = show the
// "import the files only, then index them here" way out.
export function importProblem(err) {
  const code = err && err.code
  const raw = (err && err.message) || 'Import failed'
  const msg = raw.charAt(0).toUpperCase() + raw.slice(1)
  switch (code) {
    case 'embedding_mismatch':
      return {
        title: 'Made with another embedding model',
        text: `${msg}. The stored vectors cannot be searched here. You can import the files only and index them with this Folio's model.`,
        offerFilesOnly: true,
      }
    case 'newer_version':
      return { title: 'Made by a newer Folio', text: `${msg}. Upgrade this Folio first.`, offerFilesOnly: false }
    case 'damaged_archive':
      return { title: 'Not a usable folder archive', text: msg, offerFilesOnly: false }
    case 'too_large':
      return { title: 'Archive too large', text: msg, offerFilesOnly: false }
    default:
      return { title: 'Import failed', text: msg, offerFilesOnly: false }
  }
}

// Whether to start indexing right after an import: only a files-only import of a
// folder that has files (an indexed import already carries its index).
export function needsIndexAfterImport(result) {
  return !!result && !result.indexed && (result.files || 0) > 0
}

// One-line summary for the toast after an import.
export function importSummary(result) {
  if (!result) return 'Imported'
  const files = `${result.files} file${result.files === 1 ? '' : 's'}`
  const how = result.indexed ? `${result.chunks} chunks` : 'not indexed yet'
  return `Imported as "${result.name}" — ${files}, ${how}`
}

// What an export error means for the person who asked for it.
export function exportProblem(err) {
  if (err && err.code === 'index_job_running') {
    return `${err.message}. Wait for indexing to finish (see Jobs), then export again.`
  }
  return (err && err.message) || 'Export failed'
}

// A file the browser offers is plausibly a folder archive (the server checks for real).
export function looksLikeArchive(file) {
  return !!file && /\.tar$/i.test(file.name || '')
}

// Warnings worth showing after an import. The "index the folder" one is left out when
// the console starts that indexing itself.
export function importWarnings(result) {
  const ws = (result && result.warnings) || []
  return needsIndexAfterImport(result) ? ws.filter((w) => !/without the index/i.test(w)) : ws
}
