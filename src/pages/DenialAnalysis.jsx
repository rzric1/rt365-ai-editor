import { useCallback, useState } from 'react';
import DisclaimerBlock from '../components/DisclaimerBlock.jsx';
import { getSupabaseAccessToken } from '../lib/supabaseClient.js';

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const r = reader.result;
      if (typeof r !== 'string') {
        reject(new Error('Could not read file'));
        return;
      }
      const comma = r.indexOf(',');
      resolve(comma >= 0 ? r.slice(comma + 1) : r);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function ListSection({ title, items }) {
  if (!items?.length) return null;
  return (
    <div className="denial-section">
      <h3 className="denial-section-title">{title}</h3>
      <ul className="denial-list">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export default function DenialAnalysis() {
  const [pastedText, setPastedText] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);

  const onFileChange = useCallback((e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    setError(null);
  }, []);

  const runAnalysis = useCallback(async () => {
    setError(null);
    setSummary(null);
    if (!pastedText.trim() && !file) {
      setError('Upload a PDF or image, or paste denial text (or both).');
      return;
    }

    setLoading(true);
    try {
      let fileBase64 = null;
      let mimeType = null;
      let filename = null;
      if (file) {
        if (file.size > 4_400_000) {
          setError('File is too large. Use a smaller file or paste text.');
          setLoading(false);
          return;
        }
        fileBase64 = await fileToBase64(file);
        mimeType = file.type || 'application/octet-stream';
        filename = file.name;
      }

      const token = await getSupabaseAccessToken();
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch('/api/claims-intelligence', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          mode: 'denial-analysis',
          text: pastedText.trim() || undefined,
          fileBase64: fileBase64 || undefined,
          mimeType: mimeType || undefined,
          filename: filename || undefined,
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || `Request failed (${res.status})`);
      }
      setSummary(data.summary);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }, [file, pastedText]);

  return (
    <div>
      <h1 className="page-title">Help me understand my denial</h1>
      <p className="page-lede">
        Upload your rating or denial letter (PDF or image), paste the text, or both. We will pull out
        factual points in plain language—no legal advice.
      </p>

      <div className="card">
        <h2 className="page-title" style={{ fontSize: '1.15rem' }}>
          1. Letter or decision file
        </h2>
        <p className="form-hint">PDF or image (PNG, JPG). Optional if you paste the full text below.</p>
        <input
          className="file-input"
          type="file"
          accept=".pdf,application/pdf,image/png,image/jpeg,image/webp"
          onChange={onFileChange}
        />
        {file && (
          <p className="file-meta">
            Selected: <strong>{file.name}</strong> ({Math.round(file.size / 1024)} KB)
          </p>
        )}

        <h2 className="page-title" style={{ fontSize: '1.15rem', marginTop: '1.25rem' }}>
          2. Paste denial or rating text
        </h2>
        <textarea
          className="textarea-input"
          rows={10}
          value={pastedText}
          onChange={(e) => setPastedText(e.target.value)}
          placeholder="Paste the reasons, findings, and evidence sections from your letter…"
          aria-label="Denial or rating letter text"
        />

        {error && <p className="form-error">{error}</p>}

        <div className="btn-group">
          <button type="button" className="btn btn-primary" onClick={runAnalysis} disabled={loading}>
            {loading ? 'Analyzing…' : 'Get educational summary'}
          </button>
        </div>
      </div>

      {summary && (
        <div className="card denial-output">
          <h2 className="page-title" style={{ fontSize: '1.15rem' }}>
            Educational summary
          </h2>
          <p className="form-hint">
            This is not legal advice. It is a plain-language read of what appears in your document.
          </p>

          <ListSection title="Favorable findings VA acknowledged" items={summary.favorableFindings} />
          <ListSection title="Evidence VA said was missing or insufficient" items={summary.missingEvidence} />
          {summary.statedReason ? (
            <div className="denial-section">
              <h3 className="denial-section-title">Stated reason for denial or rating</h3>
              <p className="denial-prose">{summary.statedReason}</p>
            </div>
          ) : null}
          <ListSection title="Possible next steps (educational)" items={summary.nextStepSuggestions} />
          <ListSection title="Tactical Claims AI tools that may help next" items={summary.suggestedTools} />
          {summary.educationalNote ? (
            <p className="denial-note">{summary.educationalNote}</p>
          ) : (
            <p className="denial-note">
              This is an educational summary. Verify with a VSO or accredited representative for
              decisions.
            </p>
          )}
        </div>
      )}

      <DisclaimerBlock variant="legal" />
    </div>
  );
}
