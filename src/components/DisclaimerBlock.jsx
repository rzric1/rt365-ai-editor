const COPY = {
  minimal:
    'This tool provides general information only and is not legal or medical advice. Always verify details with official VA sources or a qualified representative.',
  standard:
    'Tactical Claims AI offers educational guidance to help veterans organize their claims. It does not replace accredited representation, legal counsel, or decisions from the Department of Veterans Affairs. Outcomes depend on your unique facts and evidence.',
  legal:
    'Nothing on this site establishes an attorney–client or agent–client relationship. We do not guarantee claim outcomes. For legal questions or representation, consult an accredited Veterans Service Organization representative, claims agent, or attorney. Refer to VA and Board of Veterans’ Appeals materials for binding rules and procedures.',
};

export default function DisclaimerBlock({ variant = 'standard' }) {
  const text = COPY[variant] ?? COPY.standard;
  const className =
    variant === 'legal'
      ? 'disclaimer disclaimer--legal'
      : variant === 'minimal'
        ? 'disclaimer disclaimer--minimal'
        : 'disclaimer';

  return (
    <aside className={className} role="note">
      {text}
    </aside>
  );
}
