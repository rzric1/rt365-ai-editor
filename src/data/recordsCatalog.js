/** Record types for Records Readiness (step 1 checklist + step 2 guidance). */
export const RECORD_ITEMS = [
  {
    id: 'rating_recent',
    label: 'Rating decision letter (most recent)',
    why: 'The latest decision explains what VA granted, denied, and the reasons given—it is the baseline for any appeal or supplemental filing.',
    how: 'Request a copy in writing from your VA regional office or download available letters from VA.gov if you have online access.',
    where: 'VA.gov (letters / claim status), VA regional office, or printed via a VSO from VBMS when you have authorized representation.',
    wait: 'Digital: often same day; mailed copies commonly a few weeks.',
  },
  {
    id: 'rating_prior',
    label: 'Previous rating decisions',
    why: 'Older decisions show how your ratings evolved and can matter for effective dates or earlier denials.',
    how: 'Submit a written request to the VA regional office or request through your accredited representative.',
    where: 'VA regional office; accredited representative with VBMS access.',
    wait: 'Often several weeks for mailed copies.',
  },
  {
    id: 'strs',
    label: 'Service Treatment Records (STRs)',
    why: 'STRs document in-service symptoms and care; they are central to service connection and many increase claims.',
    how: 'If not already in your C-file, request copies from the National Personnel Records Center or via VA Form 21-4138 / FOIA-style requests as appropriate.',
    where: 'NPRC (military records), or confirm VA already has them in your claims file.',
    wait: 'NPRC requests often take several weeks to months depending on workload.',
  },
  {
    id: 'cfile',
    label: 'C-file (full claims file)',
    why: 'The C-file shows everything VA has on your claim—decisions, exams, evidence received, and internal notes you are allowed to see.',
    how: 'File a Privacy Act / FOIA request or work with a VSO to obtain a copy of your claims file.',
    where: 'VA regional office (Privacy Act office) or through an accredited representative.',
    wait: 'Commonly many weeks to months; expedite rules are limited.',
  },
  {
    id: 'private_medical',
    label: 'Private medical records',
    why: 'Private treatment notes often carry the detail VA needs for diagnosis, severity, and continuity of symptoms.',
    how: 'Sign HIPAA releases (e.g., VA Form 21-4142/4142a) for each provider or request records directly from the provider’s records department.',
    where: 'Private hospitals, specialists, and primary care clinics—not VA unless you also use VA care.',
    wait: 'Typically 2–8 weeks per provider; some release faster electronically.',
  },
  {
    id: 'buddy',
    label: 'Buddy statements',
    why: 'Lay statements can corroborate in-service events or how symptoms affect you over time.',
    how: 'Ask fellow service members, family, or coworkers to complete VA Form 21-10210 (lay/witness statement) or a signed statement with contact information.',
    where: 'Prepared by the witness; you upload or mail to VA or give to your VSO.',
    wait: 'Depends on how quickly witnesses respond—often days to weeks.',
  },
  {
    id: 'lay_statement',
    label: 'Personal statement (lay evidence)',
    why: 'Your own statement ties together timelines, symptoms, and daily impact in your words.',
    how: 'Draft a concise statement with dates and facts; use VA Form 21-4138 or 21-10210 as appropriate.',
    where: 'You prepare it; submit through VA.gov, mail, or your VSO.',
    wait: 'You control the timeline—complete before filing when possible.',
  },
  {
    id: 'nexus',
    label: 'Nexus letter from doctor',
    why: 'A clear medical opinion linking your condition to service (or to another service-connected condition) can be decisive when the record is incomplete.',
    how: 'Ask a treating or examining clinician who has reviewed your records to explain the medical relationship at least as likely as not (or the standard they are using).',
    where: 'Private physician, VA provider (if willing), or independent medical opinion—paid or through treatment relationships.',
    wait: 'Scheduling and drafting often 2–6+ weeks depending on the clinician.',
  },
];

/** Pillars for readiness score (each must be satisfied for “records-ready”). */
export function pillarStatus(have) {
  const hasRating = !!have.rating_recent;
  const hasServiceFile = !!(have.strs || have.cfile);
  const hasMedicalLink = !!(have.nexus || have.private_medical);
  return {
    hasRating,
    hasServiceFile,
    hasMedicalLink,
    satisfied: [hasRating, hasServiceFile, hasMedicalLink].filter(Boolean).length,
  };
}

export function readinessTier(have) {
  const { satisfied } = pillarStatus(have);
  const missing = 3 - satisfied;
  if (missing === 0) return 'ready';
  if (missing <= 2) return 'partial';
  return 'critical';
}
