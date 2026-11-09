import { describe, it, expect } from 'vitest'
import {
  CSRF_HEADER, needsCsrf, requestHeaders, parseApiError, loginErrorMessage,
  expiredNotice, sessionLabel, expiresIn, cleanToken,
} from './session'

describe('needsCsrf / requestHeaders', () => {
  it('only unsafe methods carry the CSRF header', () => {
    for (const m of ['GET', 'get', 'HEAD', 'OPTIONS', undefined]) expect(needsCsrf(m)).toBe(false)
    for (const m of ['POST', 'put', 'PATCH', 'DELETE']) expect(needsCsrf(m)).toBe(true)
  })
  it('builds headers', () => {
    expect(requestHeaders('GET')).toEqual({ 'Content-Type': 'application/json' })
    expect(requestHeaders('POST')).toEqual({ 'Content-Type': 'application/json', [CSRF_HEADER]: '1' })
    // multipart upload: the browser sets the content type
    expect(requestHeaders('POST', { json: false })).toEqual({ [CSRF_HEADER]: '1' })
  })
})

describe('parseApiError', () => {
  it('reads structured auth errors', () => {
    expect(parseApiError({ detail: { error: 'admin_required', error_description: 'Needs rag:admin' } }, 403))
      .toEqual({ code: 'admin_required', message: 'Needs rag:admin' })
  })
  it('reads string and validation details', () => {
    expect(parseApiError({ detail: 'Folder 3 not found' }, 404)).toEqual({ code: null, message: 'Folder 3 not found' })
    expect(parseApiError({ detail: [{ msg: 'field required' }] }, 422)).toEqual({ code: null, message: 'field required' })
  })
  it('falls back to message or status', () => {
    expect(parseApiError({ message: 'boom', error: 'X' }, 500)).toEqual({ code: 'X', message: 'boom' })
    expect(parseApiError(null, 502)).toEqual({ code: null, message: 'HTTP 502' })
    expect(parseApiError({ detail: { error: 'x' } }, 401).message).toBe('x')
  })
})

describe('login messages', () => {
  it('explains the common rejections', () => {
    expect(loginErrorMessage('wrong_audience')).toMatch(/different service/)
    expect(loginErrorMessage('token_expired')).toMatch(/expired/)
    expect(loginErrorMessage('missing_jti')).toMatch(/jti/)
    expect(loginErrorMessage('missing_sub')).toMatch(/no subject/)
  })
  it('falls back to the server text', () => {
    expect(loginErrorMessage('weird', 'server said')).toBe('server said')
    expect(loginErrorMessage(null)).toBe('Sign-in failed.')
  })
  it('explains why a signed-in user was sent back to login', () => {
    expect(expiredNotice('session_expired')).toMatch(/expired/)
    expect(expiredNotice('invalid_session')).toMatch(/no longer valid/)
    expect(expiredNotice('login_required')).toBeNull()
  })
})

describe('sessionLabel', () => {
  it('prefers sub, then jti prefix', () => {
    expect(sessionLabel({ mode: 'token', sub: 'alice', jti_prefix: 'abc…' })).toBe('alice')
    expect(sessionLabel({ mode: 'token', sub: null, jti_prefix: 'abc…' })).toBe('abc…')
    expect(sessionLabel({ mode: 'open' })).toBe('Open mode')
    expect(sessionLabel(null)).toBe('')
  })
})

describe('expiresIn', () => {
  const now = 1_000_000 * 1000
  it('formats the remaining time', () => {
    expect(expiresIn(1_000_000 + 3 * 3600 + 20 * 60, now)).toBe('in 3h 20m')
    expect(expiresIn(1_000_000 + 30, now)).toBe('in 1m')
    expect(expiresIn(1_000_000 + 49 * 3600, now)).toBe('in 2d 1h')
    expect(expiresIn(1_000_000 - 1, now)).toBe('expired')
    expect(expiresIn(null, now)).toBe('')
  })
})

describe('cleanToken', () => {
  it('trims and drops a Bearer prefix', () => {
    expect(cleanToken('  eyJ.abc.def \n')).toBe('eyJ.abc.def')
    expect(cleanToken('Bearer eyJ.abc')).toBe('eyJ.abc')
    expect(cleanToken(null)).toBe('')
  })
})
