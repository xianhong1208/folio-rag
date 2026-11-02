import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ModalProvider } from './contexts/ModalContext'
import { ToastProvider } from './contexts/ToastContext'
import { SessionProvider } from './contexts/SessionContext'
import { migrateStorageKeys, readTheme } from './lib/theme'
import './index.css'

// Theme: dark-first, remembered per-viewer (pre-rename keys are migrated first).
let storage = null
try { storage = window.localStorage } catch { /* blocked */ }
migrateStorageKeys(storage)
document.documentElement.className = readTheme(storage)

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter basename="/admin">
      <SessionProvider>
        <ToastProvider>
          <ModalProvider>
            <App />
          </ModalProvider>
        </ToastProvider>
      </SessionProvider>
    </BrowserRouter>
  </React.StrictMode>,
)
