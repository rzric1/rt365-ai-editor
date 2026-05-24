import { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import ClaimComplexityScorer from '../components/ClaimComplexityScorer.jsx';
import { ROUTES } from '../routes.js';

const OGC_ACCREDITATION = 'https://www.va.gov/ogc/apps/accreditation/index.asp';

export default function GuidedClaimSetup() {
  const [tier, setTier] = useState(null);
  const [highAcknowledged, setHighAcknowledged] = useState(false);

  const onScorerComplete = useCallback(({ tier: t }) => {
    setTier(t);
    if (t !== 'high') setHighAcknowledged(true);
    else setHighAcknowledged(false);
  }, []);

  const showMainFlow = tier !== null && (tier !== 'high' || highAcknowledged);

  return (
    <div>
      <h1 className="page-title">Guided claim setup</h1>
      <p className="page-lede">
        Start by estimating claim complexity. If your situation is high complexity, review the escalation
        notes before continuing the guided steps.
      </p>

      <ClaimComplexityScorer onComplete={onScorerComplete} />

      {tier === 'high' && !highAcknowledged && (
        <div className="escalation-panel" role="region" aria-label="High complexity escalation">
          <h3>High complexity — review before you continue</h3>
          <p>
            Your answers suggest layered legal or medical issues. A free VSO is still a good first stop,
            but you may eventually need an accredited claims agent or attorney. Review the{' '}
            <Link to={{ pathname: ROUTES.dashboard, hash: 'vso-escalation' }}>VSO &amp; escalation guide</Link>{' '}
            on the dashboard and consider
            searching accredited representatives via the{' '}
            <a href={OGC_ACCREDITATION} target="_blank" rel="noopener noreferrer">
              VA OGC accreditation search
            </a>
            .
          </p>
          <p style={{ marginBottom: 0 }}>
            Tactical Claims AI can still help you organize evidence and understand options—it does not replace
            formal representation when your claim is this complex.
          </p>
          <div className="btn-group">
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setHighAcknowledged(true)}
            >
              I have read this and want to continue
            </button>
          </div>
        </div>
      )}

      {showMainFlow && (
        <div className="card">
          <h2 className="page-title" style={{ fontSize: '1.2rem' }}>
            Guided setup steps
          </h2>
          <p style={{ marginTop: 0, color: 'var(--color-text-muted)' }}>
            Placeholder for the rest of the guided claim flow (no backend changes in this phase).
          </p>
        </div>
      )}
    </div>
  );
}
