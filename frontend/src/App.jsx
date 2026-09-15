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

export default function App() {
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
