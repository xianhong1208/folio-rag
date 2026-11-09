import { describe, it, expect } from 'vitest'
import { foldersFoot } from './overview'

describe('foldersFoot', () => {
  it('says whose folders are counted', () => {
    expect(foldersFoot({ scope: 'own' })).toBe('owned by you')
    expect(foldersFoot({ scope: 'all' })).toBe('all owners')
  })
  it('adds the legacy count when there is one', () => {
    expect(foldersFoot({ scope: 'own', legacy_owned_folders: 3 })).toBe('owned by you · 3 legacy unclaimed')
    expect(foldersFoot({ scope: 'own', legacy_owned_folders: 0 })).toBe('owned by you')
    expect(foldersFoot({ scope: 'own', legacy_owned_folders: null })).toBe('owned by you')
  })
})
