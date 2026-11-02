import { describe, it, expect } from 'vitest'
import { pillClass, statusLabel, needsRetry, canRetry, canCancel, jobKindLabel, isJobActive } from './jobs'

describe('pillClass', () => {
  it('maps the new statuses', () => {
    expect(pillClass('interrupted')).toBe('warn')
    expect(pillClass('cancelled')).toBe('mute')
    expect(pillClass('succeeded')).toBe('ok')
    expect(pillClass('partial_success')).toBe('warn')
    expect(pillClass('queued')).toBe('warn')
  })
  it('falls back to the running style for live stages', () => {
    expect(pillClass('embedding')).toBe('run')
    expect(pillClass('something-new')).toBe('run')
  })
})

describe('statusLabel', () => {
  it('humanises snake case', () => {
    expect(statusLabel('partial_success')).toBe('Partial success')
    expect(statusLabel('interrupted')).toBe('Interrupted')
    expect(statusLabel(null)).toBe('—')
  })
})

describe('file actions', () => {
  it('retry is the main action for failed / interrupted / cancelled files', () => {
    for (const s of ['failed', 'interrupted', 'cancelled']) expect(needsRetry({ status: s })).toBe(true)
    expect(needsRetry({ status: 'indexed' })).toBe(false)
    expect(needsRetry({ status: 'failed', _stage: 'queued' })).toBe(false)
  })
  it('retry for any idle file, cancel only while work is live', () => {
    expect(canRetry({ status: 'indexed' })).toBe(true)
    expect(canRetry({ status: 'unindexed', _stage: 'embedding' })).toBe(false)
    expect(canCancel({ status: 'unindexed', _stage: 'queued' })).toBe(true)
    expect(canCancel({ status: 'indexed' })).toBe(false)
  })
})

describe('jobs', () => {
  it('labels job kinds', () => {
    expect(jobKindLabel('rebuild_fts')).toBe('Rebuild FTS')
    expect(jobKindLabel(undefined)).toBe('Index')
  })
  it('knows active statuses', () => {
    expect(isJobActive({ status: 'pending' })).toBe(true)
    expect(isJobActive({ status: 'interrupted' })).toBe(false)
  })
})
