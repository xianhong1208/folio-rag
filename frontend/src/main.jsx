import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { I18nProvider } from './contexts/I18nContext'
import { ModalProvider } from './contexts/ModalContext'
import { ToastProvider } from './contexts/ToastContext'
import './index.css'

// Theme: dark-first, remembered per-viewer.
try {
  const t = localStorage.getItem('agentic_theme') || 'dark'
  document.documentElement.className = t
} catch { document.documentElement.className = 'dark' }

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter basename="/admin">
      <I18nProvider>
        <ToastProvider>
          <ModalProvider>
            <App />
          </ModalProvider>
        </ToastProvider>
      </I18nProvider>
    </BrowserRouter>
  </React.StrictMode>,
)
