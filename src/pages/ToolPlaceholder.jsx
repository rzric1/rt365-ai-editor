import { Link, useParams } from 'react-router-dom';
import { ROUTES } from '../routes.js';

const TITLES = {
  'initial-claim': 'Initial Claim (DIY)',
  increase: 'Increase',
  'evidence-review': 'Evidence Review',
  'tactical-conditions': 'Tactical Conditions',
};

export default function ToolPlaceholder() {
  const { toolId } = useParams();
  const title = TITLES[toolId] ?? 'Tool';

  return (
    <div className="placeholder-page">
      <h1 className="page-title">{title}</h1>
      <p className="page-lede">This tool route is reserved for your Tactical Claims AI workflow.</p>
      <p>
        <Link to={ROUTES.dashboard}>Back to dashboard</Link>
      </p>
    </div>
  );
}
