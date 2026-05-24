import DisclaimerBlock from '../components/DisclaimerBlock.jsx';
import VSOEscalationGuide from '../components/VSOEscalationGuide.jsx';

export default function TacticalReview() {
  return (
    <div>
      <h1 className="page-title">Tactical review</h1>
      <p className="page-lede">
        Example review results area. After you read your summary, use the VSO guide for next steps.
      </p>

      <div className="card">
        <h2 className="page-title" style={{ fontSize: '1.2rem' }}>
          Review results
        </h2>
        <p style={{ marginTop: 0, color: 'var(--color-text-muted)' }}>
          Placeholder for tactical review output (no API integration in this task).
        </p>
      </div>

      <VSOEscalationGuide />
      <DisclaimerBlock variant="legal" />
    </div>
  );
}
