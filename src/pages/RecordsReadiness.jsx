import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import DisclaimerBlock from '../components/DisclaimerBlock.jsx';
import { RECORD_ITEMS, readinessTier } from '../data/recordsCatalog.js';
import { ROUTES } from '../routes.js';

function emptyHave() {
  return Object.fromEntries(RECORD_ITEMS.map((r) => [r.id, false]));
}

const TIER_COPY = {
  ready: {
    title: 'Records readiness',
    body: 'You appear records-ready. Proceed with your claim.',
    className: 'readiness-badge readiness-badge--ok',
  },
  partial: {
    title: 'Records readiness',
    body: 'Request missing records before filing when possible.',
    className: 'readiness-badge readiness-badge--warn',
  },
  critical: {
    title: 'Records readiness',
    body: 'Focus on records gathering first. Filing without records increases denial risk.',
    className: 'readiness-badge readiness-badge--alert',
  },
};

export default function RecordsReadiness() {
  const [step, setStep] = useState(1);
  const [have, setHave] = useState(emptyHave);

  const toggle = (id) => {
    setHave((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const missingItems = useMemo(
    () => RECORD_ITEMS.filter((r) => !have[r.id]),
    [have]
  );

  const tier = useMemo(() => readinessTier(have), [have]);
  const tierInfo = TIER_COPY[tier];

  return (
    <div>
      <h1 className="page-title">Records readiness</h1>
      <p className="page-lede">
        Check what you already have, see how to fill gaps, and get a simple readiness read before you
        file.
      </p>

      {step === 1 && (
        <div className="card">
          <h2 className="page-title" style={{ fontSize: '1.15rem' }}>
            Step 1 — What records do you have?
          </h2>
          <p className="form-hint">Check everything you already have in hand or can access online.</p>
          <ul className="records-checklist">
            {RECORD_ITEMS.map((r) => (
              <li key={r.id}>
                <label className="check-row">
                  <input
                    type="checkbox"
                    checked={!!have[r.id]}
                    onChange={() => toggle(r.id)}
                  />
                  <span>{r.label}</span>
                </label>
              </li>
            ))}
          </ul>
          <div className="btn-group">
            <button type="button" className="btn btn-primary" onClick={() => setStep(2)}>
              Continue
            </button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="card">
          <h2 className="page-title" style={{ fontSize: '1.15rem' }}>
            Step 2 — What you still need
          </h2>
          {missingItems.length === 0 ? (
            <p>You checked every category—nice work. You can still review the score in step 3.</p>
          ) : (
            <p className="form-hint">
              For anything you did not check, here is why it matters and how to request it.
            </p>
          )}
          <div className="records-gap-list">
            {missingItems.map((r) => (
              <article key={r.id} className="records-gap-card">
                <h3 className="records-gap-title">{r.label}</h3>
                <p>
                  <strong>Why it matters:</strong> {r.why}
                </p>
                <p>
                  <strong>How to request:</strong> {r.how}
                </p>
                <p>
                  <strong>Where:</strong> {r.where}
                </p>
                <p style={{ marginBottom: 0 }}>
                  <strong>Typical wait:</strong> {r.wait}
                </p>
              </article>
            ))}
          </div>
          <div className="btn-group">
            <button type="button" className="btn btn-secondary" onClick={() => setStep(1)}>
              Back
            </button>
            <button type="button" className="btn btn-primary" onClick={() => setStep(3)}>
              See readiness score
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="card">
          <h2 className="page-title" style={{ fontSize: '1.15rem' }}>
            Step 3 — Records readiness score
          </h2>
          <div className={tierInfo.className}>
            <strong>{tierInfo.title}</strong>
            <p style={{ margin: '0.5rem 0 0' }}>{tierInfo.body}</p>
          </div>
          <p className="form-hint">
            We weight three pillars: your most recent rating decision, service records (STRs or C-file),
            and medical linkage (private records and/or a nexus letter). Other items still strengthen
            your file.
          </p>
          <p>
            Pair this with{' '}
            <Link to={ROUTES.denialAnalysis}>Help me understand my denial</Link> when you are unpacking a
            decision letter.
          </p>
          <div className="btn-group">
            <button type="button" className="btn btn-secondary" onClick={() => setStep(2)}>
              Back
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => {
                setHave(emptyHave());
                setStep(1);
              }}
            >
              Start over
            </button>
          </div>
        </div>
      )}

      <DisclaimerBlock variant="standard" />
    </div>
  );
}
