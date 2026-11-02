import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ModalProvider } from './contexts/ModalContext'
import { ToastProvider } from './contexts/ToastContext'
import { SessionProvider } from './contexts/SessionContext'
import './index.css'

// Theme: dark-first, remembered per-viewer.
try {
  const t = localStorage.getItem('agentic_theme') || 'dark'
  document.documentElement.className = t
} catch { document.documentElement.className = 'dark' }

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
