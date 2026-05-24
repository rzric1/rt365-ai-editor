import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';
import { Link } from 'react-router-dom';
import DisclaimerBlock from './DisclaimerBlock.jsx';
import { ROUTES } from '../routes.js';

const STEPS = [
  'firstClaim',
  'deniedBefore',
  'deniedCount',
  'ratingLetter',
  'cfile',
  'claimType',
  'understandDenial',
  'nexus',
  'secondary',
];

function shouldShowStep(step, answers) {
  if (step === 'deniedBefore' && answers.firstClaim === true) return false;
  if (
    step === 'deniedCount' &&
    (answers.firstClaim === true || answers.deniedBefore === false)
  ) {
    return false;
  }
  return true;
}

function nextVisibleStepIndex(fromIndex, answers) {
  let i = fromIndex + 1;
  while (i < STEPS.length) {
    if (shouldShowStep(STEPS[i], answers)) return i;
    i += 1;
  }
  return STEPS.length;
}

function prevVisibleStepIndex(fromIndex, answers) {
  let i = fromIndex - 1;
  while (i >= 0) {
    if (shouldShowStep(STEPS[i], answers)) return i;
    i -= 1;
  }
  return -1;
}

export function computeClaimComplexityScore(answers) {
  let s = 0;
  const first = answers.firstClaim === true;
  if (!first && answers.deniedBefore === true) s += 1;
  if (
    !first &&
    answers.deniedBefore === true &&
    (answers.deniedCount === '2' || answers.deniedCount === '3plus')
  ) {
    s += 1;
  }
  if (answers.ratingLetter === 'no' || answers.ratingLetter === 'unsure') s += 1;
  if (answers.cfile === 'no' || answers.cfile === 'waiting') s += 1;
  const complexTypes = ['hlr', 'cue', 'board', 'effective_date'];
  if (complexTypes.includes(answers.claimType)) s += 2;
  if (answers.understandDenial === 'no' || answers.understandDenial === 'partial') {
    s += 1;
  }
  if (answers.nexus === 'no' || answers.nexus === 'in_progress') s += 1;
  if (answers.secondary === 'yes' || answers.secondary === 'unsure') s += 1;
  return s;
}

export function complexityTierFromScore(score) {
  if (score <= 3) return 'low';
  if (score <= 6) return 'moderate';
  return 'high';
}

function createInitialWizard() {
  return { phase: 'wizard', stepIndex: 0, answers: {} };
}

function wizardReducer(state, action) {
  switch (action.type) {
    case 'answer': {
      const nextAnswers = { ...state.answers, ...action.patch };
      const nextIdx = nextVisibleStepIndex(state.stepIndex, nextAnswers);
      if (nextIdx >= STEPS.length) {
        return { ...state, answers: nextAnswers, phase: 'result' };
      }
      return { ...state, answers: nextAnswers, stepIndex: nextIdx };
    }
    case 'back': {
      const prev = prevVisibleStepIndex(state.stepIndex, state.answers);
      if (prev < 0) return state;
      return { ...state, stepIndex: prev };
    }
    case 'restart':
      return createInitialWizard();
    default:
      return state;
  }
}

function ToolLinksList({ extraLinks = [] }) {
  return (
    <ul className="tool-links">
      <li>
        <Link to={ROUTES.initialClaim}>Initial Claim (DIY)</Link>
      </li>
      <li>
        <Link to={ROUTES.increase}>Increase</Link>
      </li>
      <li>
        <Link to={ROUTES.evidenceReview}>Evidence Review</Link>
      </li>
      <li>
        <Link to={ROUTES.tacticalConditions}>Tactical Conditions</Link>
      </li>
      {extraLinks.map((item, idx) => (
        <li key={`${item.label}-${idx}`}>
          {typeof item.to === 'string' && item.to.startsWith('http') ? (
            <a href={item.to} target="_blank" rel="noopener noreferrer">
              {item.label}
            </a>
          ) : (
            <Link to={item.to}>{item.label}</Link>
          )}
        </li>
      ))}
    </ul>
  );
}

export default function ClaimComplexityScorer({ onComplete }) {
  const [wizard, dispatch] = useReducer(wizardReducer, undefined, createInitialWizard);
  const { phase, stepIndex, answers } = wizard;
  const completedRef = useRef(false);

  const score = useMemo(
    () => (phase === 'result' ? computeClaimComplexityScore(answers) : 0),
    [phase, answers]
  );
  const tier = useMemo(
    () => (phase === 'result' ? complexityTierFromScore(score) : null),
    [phase, score]
  );

  useEffect(() => {
    if (phase !== 'result') {
      completedRef.current = false;
      return;
    }
    if (!onComplete || completedRef.current) return;
    completedRef.current = true;
    onComplete({ score, tier });
  }, [phase, score, tier, onComplete]);

  const advance = useCallback((patch) => {
    dispatch({ type: 'answer', patch });
  }, []);

  const goBack = useCallback(() => {
    dispatch({ type: 'back' });
  }, []);

  const restart = useCallback(() => {
    dispatch({ type: 'restart' });
    completedRef.current = false;
  }, []);

  const visibleCount = useMemo(() => {
    return STEPS.filter((s) => shouldShowStep(s, answers)).length;
  }, [answers]);

  const visibleIndex = useMemo(() => {
    const vis = STEPS.filter((s) => shouldShowStep(s, answers));
    const cur = STEPS[stepIndex];
    return Math.max(0, vis.indexOf(cur) + 1);
  }, [answers, stepIndex]);

  const currentStep = STEPS[stepIndex];
  const canGoBack = prevVisibleStepIndex(stepIndex, answers) >= 0;

  if (phase === 'result') {
    const ogcUrl = 'https://www.va.gov/ogc/apps/accreditation/index.asp';
    let tierClass = 'result-tier--low';
    let tierLabel = 'Low complexity';
    let message = '';
    let extraLinks = [];

    if (tier === 'low') {
      message =
        'Your claim looks manageable. Use our DIY tools — Initial Claim, Increase, Evidence Review, and Tactical Conditions. Make sure you have your records before filing.';
    } else if (tier === 'moderate') {
      tierClass = 'result-tier--moderate';
      tierLabel = 'Moderate complexity';
      message =
        'Your claim has some complexity. Consider working with a free VSO and use Tactical Claims AI to prep your evidence and understand your options.';
      extraLinks = [
        {
          to: { pathname: ROUTES.dashboard, hash: 'vso-escalation' },
          label: 'VSO meeting & escalation guide',
        },
      ];
    } else {
      tierClass = 'result-tier--high';
      tierLabel = 'High complexity';
      message =
        'Your claim is complex. You may benefit from consulting an accredited claims agent or attorney in addition to using our tools.';
      extraLinks = [
        {
          to: { pathname: ROUTES.dashboard, hash: 'vso-escalation' },
          label: 'VSO & escalation guidance',
        },
        { to: ogcUrl, label: 'Search accredited agents & attorneys (VA OGC)' },
        { to: ROUTES.guidedSetup, label: 'Guided claim setup' },
        { to: ROUTES.tacticalReview, label: 'Tactical review' },
      ];
    }

    return (
      <div className="card claim-complexity-scorer">
        <h2 className="page-title" style={{ fontSize: '1.35rem' }}>
          Your claim complexity score: {score}
        </h2>
        <span className={`result-tier ${tierClass}`}>{tierLabel}</span>
        <p>{message}</p>
        <p className="section-label">Recommended tools</p>
        <ToolLinksList extraLinks={extraLinks} />
        {tier === 'high' && (
          <p style={{ marginTop: '1rem', fontSize: '0.95rem' }}>
            Review accredited representation options on the{' '}
            <a href={ogcUrl} target="_blank" rel="noopener noreferrer">
              VA Office of General Counsel accreditation search
            </a>
            .
          </p>
        )}
        <DisclaimerBlock variant="minimal" />
        <div className="btn-group">
          <button type="button" className="btn btn-secondary" onClick={restart}>
            Start over
          </button>
        </div>
      </div>
    );
  }

  const questionBody = (() => {
    switch (currentStep) {
      case 'firstClaim':
        return (
          <>
            <p>Is this your first VA claim?</p>
            <div className="choice-grid cols-2">
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ firstClaim: true })}
              >
                Yes
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ firstClaim: false })}
              >
                No
              </button>
            </div>
          </>
        );
      case 'deniedBefore':
        return (
          <>
            <p>Have you been denied before for this condition?</p>
            <div className="choice-grid cols-2">
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ deniedBefore: true })}
              >
                Yes
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ deniedBefore: false })}
              >
                No
              </button>
            </div>
          </>
        );
      case 'deniedCount':
        return (
          <>
            <p>If denied — how many times?</p>
            <div className="choice-grid cols-3">
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ deniedCount: '1' })}
              >
                1
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ deniedCount: '2' })}
              >
                2
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ deniedCount: '3plus' })}
              >
                3+
              </button>
            </div>
          </>
        );
      case 'ratingLetter':
        return (
          <>
            <p>Do you have your rating decision letter?</p>
            <div className="choice-grid cols-3">
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ ratingLetter: 'yes' })}
              >
                Yes
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ ratingLetter: 'no' })}
              >
                No
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ ratingLetter: 'unsure' })}
              >
                Not sure
              </button>
            </div>
          </>
        );
      case 'cfile':
        return (
          <>
            <p>Do you have your C-file or STRs?</p>
            <div className="choice-grid cols-3">
              <button type="button" className="choice-btn" onClick={() => advance({ cfile: 'yes' })}>
                Yes
              </button>
              <button type="button" className="choice-btn" onClick={() => advance({ cfile: 'no' })}>
                No
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ cfile: 'waiting' })}
              >
                Requested, waiting
              </button>
            </div>
          </>
        );
      case 'claimType':
        return (
          <>
            <p>What type of claim is this?</p>
            <div className="choice-grid">
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ claimType: 'new' })}
              >
                New claim
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ claimType: 'increase' })}
              >
                Increase
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ claimType: 'supplemental' })}
              >
                Supplemental
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ claimType: 'hlr' })}
              >
                HLR
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ claimType: 'cue' })}
              >
                CUE
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ claimType: 'board' })}
              >
                Board Appeal
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ claimType: 'effective_date' })}
              >
                Effective date dispute
              </button>
            </div>
          </>
        );
      case 'understandDenial':
        return (
          <>
            <p>Do you understand exactly why VA denied or rated you?</p>
            <div className="choice-grid cols-3">
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ understandDenial: 'yes' })}
              >
                Yes
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ understandDenial: 'no' })}
              >
                No
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ understandDenial: 'partial' })}
              >
                Partially
              </button>
            </div>
          </>
        );
      case 'nexus':
        return (
          <>
            <p>Do you have a medical nexus letter or supporting evidence?</p>
            <div className="choice-grid cols-3">
              <button type="button" className="choice-btn" onClick={() => advance({ nexus: 'yes' })}>
                Yes
              </button>
              <button type="button" className="choice-btn" onClick={() => advance({ nexus: 'no' })}>
                No
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ nexus: 'in_progress' })}
              >
                In progress
              </button>
            </div>
          </>
        );
      case 'secondary':
        return (
          <>
            <p>Is this a secondary condition or complex medical theory?</p>
            <div className="choice-grid cols-3">
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ secondary: 'yes' })}
              >
                Yes
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ secondary: 'no' })}
              >
                No
              </button>
              <button
                type="button"
                className="choice-btn"
                onClick={() => advance({ secondary: 'unsure' })}
              >
                Not sure
              </button>
            </div>
          </>
        );
      default:
        return null;
    }
  })();

  return (
    <div className="card claim-complexity-scorer">
      <h2 className="page-title" style={{ fontSize: '1.35rem' }}>
        Claim complexity scorer
      </h2>
      <p className="wizard-progress">
        Step {visibleIndex} of {visibleCount}
      </p>
      {questionBody}
      <div className="btn-group">
        {canGoBack && (
          <button type="button" className="btn btn-secondary" onClick={goBack}>
            Back
          </button>
        )}
      </div>
    </div>
  );
}
