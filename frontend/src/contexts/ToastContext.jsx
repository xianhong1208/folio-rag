import { createContext, useContext, useState, useCallback } from 'react'

const Ctx = createContext(() => {})
export const useToast = () => useContext(Ctx)

let seq = 0

export function ToastProvider({ children }) {
  const [items, setItems] = useState([])

  const toast = useCallback((msg, type = '') => {
    const id = ++seq
    setItems((l) => [...l, { id, msg, type }])
    setTimeout(() => setItems((l) => l.filter((x) => x.id !== id)), type === 'bad' ? 5000 : 3200)
  }, [])

  return (
    <Ctx.Provider value={toast}>
      {children}
      <div className="toast-wrap" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className={'toastx ' + t.type}>{t.msg}</div>
        ))}
      </div>
    </Ctx.Provider>
  )
}
