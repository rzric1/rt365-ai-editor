import { useId, useState } from 'react';

const OGC_ACCREDITATION = 'https://www.va.gov/ogc/apps/accreditation/index.asp';

const PREP_ITEMS = [
  'Copy of your rating decision letter',
  'C-file request confirmation or records',
  'List of all conditions (service-connected and claimed)',
  'List of denied conditions',
  'Summary of evidence you have',
  'Summary of evidence you still need',
  'List of questions for your VSO',
];

const ESCALATION_TRIGGERS = [
  {
    title: 'Multiple denials for the same condition',
    body: 'If VA has turned down the same issue more than once, the file may need a closer legal review, new theory, or stronger medical support than a quick intake can cover.',
  },
  {
    title: 'Possible CUE in a prior decision',
    body: 'A clear and unmistakable error claim argues an old decision was wrong based on the record at that time. These are technical and often need experienced help to frame correctly.',
  },
  {
    title: 'Effective date dispute',
    body: 'When the fight is about when benefits should have started—not just how much you are rated—rules about records, filings, and prior decisions get complicated quickly.',
  },
  {
    title: 'Complex secondary condition theory',
    body: 'Linking a new condition to an already service-connected condition can involve long causal chains. VA and examiners may disagree on whether the link is supported.',
  },
  {
    title: 'Conflicting medical opinions',
    body: 'If one doctor supports your claim and another does not, someone usually has to explain why the favorable evidence is more persuasive or why an exam missed key facts.',
  },
  {
    title: 'Board of Veterans Appeals involvement',
    body: 'Once you are at the Board, procedures, deadlines, and evidence rules are different from the regional office. Many veterans add accredited help at this stage.',
  },
];

export default function VSOEscalationGuide() {
  const baseId = useId();
  const [tab, setTab] = useState('prepare');
  const prepId = `${baseId}-prepare`;
  const escalateId = `${baseId}-escalate`;

  return (
    <section className="card" id="vso-escalation" aria-labelledby={`${baseId}-heading`}>
      <h2 className="page-title" style={{ fontSize: '1.35rem' }} id={`${baseId}-heading`}>
        VSO & escalation guide
      </h2>
      <div className="tabs" role="tablist" aria-label="VSO guide sections">
        <button
          type="button"
          role="tab"
          className="tab"
          id={prepId}
          aria-selected={tab === 'prepare'}
          aria-controls={`${prepId}-panel`}
          onClick={() => setTab('prepare')}
        >
          Prepare for Your VSO Meeting
        </button>
        <button
          type="button"
          role="tab"
          className="tab"
          id={escalateId}
          aria-selected={tab === 'escalate'}
          aria-controls={`${escalateId}-panel`}
          onClick={() => setTab('escalate')}
        >
          When to Consider Escalating Beyond a VSO
        </button>
      </div>

      {tab === 'prepare' && (
        <div
          role="tabpanel"
          id={`${prepId}-panel`}
          aria-labelledby={prepId}
        >
          <p>Bring as much of the following as you can so your VSO can give concrete next steps:</p>
          <ul className="checklist">
            {PREP_ITEMS.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {tab === 'escalate' && (
        <div
          role="tabpanel"
          id={`${escalateId}-panel`}
          aria-labelledby={escalateId}
        >
          <ul className="trigger-list">
            {ESCALATION_TRIGGERS.map((row) => (
              <li key={row.title}>
                <span className="trigger-item-title">{row.title}</span>
                {row.body}
              </li>
            ))}
          </ul>
        </div>
      )}

      <footer style={{ marginTop: '1.25rem', fontSize: '0.95rem', color: 'var(--color-text-muted)' }}>
        <p style={{ margin: 0 }}>
          A VSO may refer you to an accredited agent or attorney if your claim requires it. You can also
          search for one yourself.{' '}
          <a href={OGC_ACCREDITATION} target="_blank" rel="noopener noreferrer">
            https://www.va.gov/ogc/apps/accreditation/index.asp
          </a>
        </p>
      </footer>
    </section>
  );
}
