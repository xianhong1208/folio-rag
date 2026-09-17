import { describe, it, expect } from 'vitest'
import { fmtNum, fmtBytes, shortModel, fmtTime } from './format'

describe('fmtNum', () => {
  it('renders an em dash for null/undefined', () => {
    expect(fmtNum(null)).toBe('—')
    expect(fmtNum(undefined)).toBe('—')
  })
  it('keeps zero as a number, not a dash', () => {
    expect(fmtNum(0)).toBe('0')
  })
  it('groups thousands', () => {
    expect(fmtNum(1234567)).toBe('1,234,567')
  })
})

describe('fmtBytes', () => {
  it('renders an em dash for null', () => {
    expect(fmtBytes(null)).toBe('—')
  })
  it('shows bytes with no decimal', () => {
    expect(fmtBytes(512)).toBe('512 B')
  })
  it('shows one decimal below 10 units', () => {
    expect(fmtBytes(1536)).toBe('1.5 KB')
  })
  it('drops the decimal at/above 10 units', () => {
    expect(fmtBytes(1024 * 1024 * 20)).toBe('20 MB')
  })
  it('climbs the ladder to GB/TB', () => {
    expect(fmtBytes(1024 ** 3 * 3)).toBe('3.0 GB')
    expect(fmtBytes(1024 ** 4 * 2)).toBe('2.0 TB')
  })
})

describe('shortModel', () => {
  it('keeps only the last path segment', () => {
    expect(shortModel('intfloat/multilingual-e5-large')).toBe('multilingual-e5-large')
    expect(shortModel('bge-m3')).toBe('bge-m3')
  })
  it('renders an em dash for falsy input', () => {
    expect(shortModel('')).toBe('—')
    expect(shortModel(null)).toBe('—')
  })
})

describe('fmtTime', () => {
  it('renders an em dash for empty/invalid input', () => {
    expect(fmtTime(null)).toBe('—')
    expect(fmtTime('not-a-date')).toBe('—')
  })
  it('formats a valid ISO timestamp to a non-dash string', () => {
    const out = fmtTime('2026-09-15T08:30:00Z')
    expect(out).not.toBe('—')
    expect(typeof out).toBe('string')
  })
})
