import { lazy, Suspense } from 'react';
import { Link, Navigate, Route, Routes } from 'react-router-dom';
import Dashboard from './pages/Dashboard.jsx';
import GuidedClaimSetup from './pages/GuidedClaimSetup.jsx';
import TacticalReview from './pages/TacticalReview.jsx';
import ToolPlaceholder from './pages/ToolPlaceholder.jsx';
import { ROUTES } from './routes.js';

const DenialAnalysis = lazy(() => import('./pages/DenialAnalysis.jsx'));
const RecordsReadiness = lazy(() => import('./pages/RecordsReadiness.jsx'));

export default function App() {
  return (
    <div className="app-shell">
      <nav className="app-nav" aria-label="Main">
        <Link to={ROUTES.dashboard}>Dashboard</Link>
        <Link to={ROUTES.denialAnalysis}>Denial analysis</Link>
        <Link to={ROUTES.recordsReadiness}>Records readiness</Link>
        <Link to={ROUTES.guidedSetup}>Guided claim setup</Link>
        <Link to={ROUTES.tacticalReview}>Tactical review</Link>
        <Link to={ROUTES.initialClaim}>Initial claim</Link>
        <Link to={ROUTES.increase}>Increase</Link>
        <Link to={ROUTES.evidenceReview}>Evidence review</Link>
        <Link to={ROUTES.tacticalConditions}>Tactical conditions</Link>
      </nav>

      <Suspense fallback={<div className="route-fallback">Loading...</div>}>
        <Routes>
          <Route path={ROUTES.dashboard} element={<Dashboard />} />
          <Route path="/denial-analysis" element={<DenialAnalysis />} />
          <Route path="/records-readiness" element={<RecordsReadiness />} />
          <Route path={ROUTES.guidedSetup} element={<GuidedClaimSetup />} />
          <Route path={ROUTES.tacticalReview} element={<TacticalReview />} />
          <Route path="/tools/:toolId" element={<ToolPlaceholder />} />
          <Route path="*" element={<Navigate to={ROUTES.dashboard} replace />} />
        </Routes>
      </Suspense>
    </div>
  );
}
