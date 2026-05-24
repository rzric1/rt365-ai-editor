import { Link } from 'react-router-dom';
import ClaimComplexityScorer from '../components/ClaimComplexityScorer.jsx';
import DisclaimerBlock from '../components/DisclaimerBlock.jsx';
import VSOEscalationGuide from '../components/VSOEscalationGuide.jsx';
import { ROUTES } from '../routes.js';

export default function Dashboard() {
  return (
    <div>
      <h1 className="page-title">Dashboard</h1>
      <p className="page-lede">
        Use the scorer to gauge how much support you may need, then review VSO preparation and escalation
        signals.
      </p>

      <section aria-labelledby="next-steps-tools-heading">
        <h2 className="section-label" id="next-steps-tools-heading">
          Next steps — denial &amp; records
        </h2>
        <div className="card dashboard-next-grid">
          <p style={{ marginTop: 0 }}>
            <strong>Help me understand my denial</strong> — upload or paste your decision letter for an
            educational summary of favorable findings, gaps, and options.
          </p>
          <p>
            <Link to={ROUTES.denialAnalysis} className="btn btn-primary" style={{ display: 'inline-flex' }}>
              Open denial analysis
            </Link>
          </p>
          <p>
            <strong>Records readiness</strong> — checklist what you have, what to request, and a simple
            readiness score before you file.
          </p>
          <p style={{ marginBottom: 0 }}>
            <Link to={ROUTES.recordsReadiness} className="btn btn-secondary" style={{ display: 'inline-flex' }}>
              Open records readiness
            </Link>
          </p>
        </div>
      </section>

      <section aria-labelledby="claim-guidance-heading">
        <h2 className="section-label" id="claim-guidance-heading">
          Claim guidance &amp; next steps
        </h2>
        <ClaimComplexityScorer />
        <DisclaimerBlock variant="standard" />
        <VSOEscalationGuide />
      </section>
    </div>
  );
}
