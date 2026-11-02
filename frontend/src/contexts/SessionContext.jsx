import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { getSession, signIn, signOut, onUnauthorized } from '../services/api'
import { expiredNotice } from '../lib/session'

const Ctx = createContext(null)
export const useSession = () => useContext(Ctx)

// status: 'loading' (checking the cookie) | 'anon' (show login) | 'ready'
export function SessionProvider({ children }) {
  const [status, setStatus] = useState('loading')
  const [session, setSession] = useState(null)
  const [notice, setNotice] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const r = await getSession()
      setSession(r.data); setStatus('ready')
    } catch {
      setSession(null); setStatus('anon')
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  // Any API call that gets a 401 sends the user back to the login screen.
  useEffect(() => onUnauthorized((code) => {
    setNotice(expiredNotice(code) || 'Please sign in again.')
    setSession(null); setStatus('anon')
  }), [])

  const login = useCallback(async (token) => {
    const r = await signIn(token)
    setNotice(null); setSession(r.data); setStatus('ready')
  }, [])

  const logout = useCallback(async () => {
    try { await signOut() } catch { /* the cookie is cleared server-side; ignore */ }
    setNotice(null); setSession(null); setStatus('anon')
  }, [])

  const isAdmin = !!(session && session.is_admin)
  return (
    <Ctx.Provider value={{ status, session, isAdmin, notice, login, logout, refresh }}>
      {children}
    </Ctx.Provider>
  )
}
