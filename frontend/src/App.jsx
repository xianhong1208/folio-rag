import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import OverviewPage from './pages/OverviewPage'
import FoldersPage from './pages/FoldersPage'
import JobsPage from './pages/JobsPage'
import SearchPage from './pages/SearchPage'
import AnalyticsPage from './pages/AnalyticsPage'
import EvalPage from './pages/EvalPage'
import SettingsPage from './pages/SettingsPage'
import HealthPage from './pages/HealthPage'
import AuditPage from './pages/AuditPage'
import LoginPage from './pages/LoginPage'
import { useSession } from './contexts/SessionContext'

export default function App() {
  const { status } = useSession()
  // Route guard: nothing but the login screen until the session is known to be valid.
  // Any 401 later on flips status back to 'anon' (see SessionContext).
  if (status === 'loading') return <div className="login-wrap"><div className="sec">loading…</div></div>
  if (status !== 'ready') return <LoginPage />
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<OverviewPage />} />
        <Route path="folders" element={<FoldersPage />} />
        <Route path="jobs" element={<JobsPage />} />
        <Route path="search" element={<SearchPage />} />
        <Route path="analytics" element={<AnalyticsPage />} />
        <Route path="eval" element={<EvalPage />} />
        <Route path="settings/*" element={<SettingsPage />} />
        <Route path="health" element={<HealthPage />} />
        <Route path="audit" element={<AuditPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
