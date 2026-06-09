import { useState } from 'react';

const CHECKOUT_ONETIME =
  import.meta.env.VITE_STRIPE_CHECKOUT_ONETIME || '#pricing';
const CHECKOUT_SUBSCRIPTION =
  import.meta.env.VITE_STRIPE_CHECKOUT_SUBSCRIPTION || '#pricing';
const SUPPORT_EMAIL = import.meta.env.VITE_SUPPORT_EMAIL || 'support@rt365.ai';
const DOWNLOAD_URL = import.meta.env.VITE_APP_DOWNLOAD_URL || '';

const FEATURES = [
  {
    title: 'AI transcription',
    description: 'GPU-accelerated Whisper transcription with speaker-aware timestamps.',
  },
  {
    title: 'Auto-cut',
    description: 'Find highlight moments and export ready-to-edit clips automatically.',
  },
  {
    title: 'GPU-accelerated editing',
    description: 'CUDA-backed pipelines for faster transcoding and analysis on NVIDIA GPUs.',
  },
  {
    title: 'Clip Studio integration',
    description: 'Streamlit dashboard for reviewing, refining, and exporting clips in one place.',
  },
  {
    title: 'DaVinci Resolve EDL export',
    description: 'Send markers and cuts straight into your Resolve timeline.',
  },
  {
    title: 'Streamlit dashboard',
    description: 'Visual session controls, stability tools, and export management built in.',
  },
];

const PRICING = [
  {
    id: 'onetime',
    name: 'One-time purchase',
    price: '$149',
    detail: 'Lifetime license for one machine. Includes all current features and updates.',
    cta: 'Buy now',
    href: CHECKOUT_ONETIME,
    featured: true,
  },
  {
    id: 'subscription',
    name: 'Subscription',
    price: '$19/mo',
    detail: 'Monthly access with priority updates and ongoing AI feature releases.',
    cta: 'Subscribe',
    href: CHECKOUT_SUBSCRIPTION,
    featured: false,
  },
];

export default function App() {
  const [licenseKey, setLicenseKey] = useState('');
  const [licenseStatus, setLicenseStatus] = useState(null);
  const [checking, setChecking] = useState(false);

  async function handleLicenseCheck(e) {
    e.preventDefault();
    setLicenseStatus(null);
    setChecking(true);

    try {
      const res = await fetch('/api/validate-license', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          license_key: licenseKey.trim(),
          verify_only: true,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setLicenseStatus({ ok: false, message: data.error || 'Validation failed.' });
        return;
      }

      if (data.valid) {
        setLicenseStatus({
          ok: true,
          message: data.email
            ? `License key verified for ${data.email}. Download RT365 AI Editor and activate on your machine.`
            : 'License key verified. Download RT365 AI Editor and activate on your machine.',
        });
      } else {
        setLicenseStatus({ ok: false, message: data.error || 'Invalid license key.' });
      }
    } catch {
      setLicenseStatus({ ok: false, message: 'Could not reach the license server. Try again later.' });
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="landing">
      <header className="landing-header">
        <div className="landing-header__inner">
          <span className="landing-logo">RT365 AI Editor</span>
          <nav className="landing-nav" aria-label="Page sections">
            <a href="#features">Features</a>
            <a href="#pricing">Pricing</a>
            <a href="#activate">Activate</a>
            <a href="#support">Support</a>
          </nav>
          <a href="#pricing" className="btn btn-primary landing-header__cta">
            Buy now
          </a>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero__content">
            <p className="hero__eyebrow">Clip Studio for creators</p>
            <h1>RT365 AI Editor — AI-Powered Clip Editing for Creators</h1>
            <p className="hero__lede">
              Transcribe long-form video, find your best moments, and export clips to DaVinci Resolve —
              all from a desktop app built for podcasters, YouTubers, and editors.
            </p>
            <div className="btn-group">
              <a href="#pricing" className="btn btn-primary btn-lg">
                Buy now
              </a>
              {DOWNLOAD_URL ? (
                <a href={DOWNLOAD_URL} className="btn btn-secondary btn-lg">
                  Download
                </a>
              ) : null}
            </div>
          </div>
        </section>

        <section id="features" className="section">
          <div className="section__inner">
            <p className="section-label">Features</p>
            <h2>Everything you need to go from raw footage to publish-ready clips</h2>
            <div className="feature-grid">
              {FEATURES.map((feature) => (
                <article key={feature.title} className="feature-card">
                  <h3>{feature.title}</h3>
                  <p>{feature.description}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section id="pricing" className="section section--alt">
          <div className="section__inner">
            <p className="section-label">Pricing</p>
            <h2>Choose how you want to own RT365</h2>
            <div className="pricing-grid">
              {PRICING.map((tier) => (
                <article
                  key={tier.id}
                  className={`pricing-card${tier.featured ? ' pricing-card--featured' : ''}`}
                >
                  <h3>{tier.name}</h3>
                  <p className="pricing-card__price">{tier.price}</p>
                  <p className="pricing-card__detail">{tier.detail}</p>
                  <a
                    href={tier.href}
                    className={`btn ${tier.featured ? 'btn-primary' : 'btn-secondary'}`}
                    target={tier.href.startsWith('http') ? '_blank' : undefined}
                    rel={tier.href.startsWith('http') ? 'noreferrer' : undefined}
                  >
                    {tier.cta}
                  </a>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section id="activate" className="section">
          <div className="section__inner section__inner--narrow">
            <p className="section-label">License activation</p>
            <h2>Verify your license key</h2>
            <p className="section__lede">
              Enter the key from your purchase confirmation email. Activation on your computer happens
              inside the RT365 AI Editor desktop app.
            </p>
            <form className="license-form card" onSubmit={handleLicenseCheck}>
              <label htmlFor="license-key">License key</label>
              <input
                id="license-key"
                type="text"
                className="text-input"
                placeholder="RT365-XXXX-XXXX-XXXX-XXXX"
                value={licenseKey}
                onChange={(e) => setLicenseKey(e.target.value)}
                autoComplete="off"
                spellCheck={false}
                required
              />
              <button type="submit" className="btn btn-primary" disabled={checking}>
                {checking ? 'Checking…' : 'Verify license'}
              </button>
              {licenseStatus ? (
                <p
                  className={`license-status${licenseStatus.ok ? ' license-status--ok' : ' license-status--error'}`}
                  role="status"
                >
                  {licenseStatus.message}
                </p>
              ) : null}
            </form>
          </div>
        </section>
      </main>

      <footer id="support" className="landing-footer">
        <div className="landing-footer__inner">
          <div>
            <strong>RT365 AI Editor</strong>
            <p>AI-powered clip editing for creators.</p>
          </div>
          <div className="landing-footer__links">
            <a href={`mailto:${SUPPORT_EMAIL}`}>Contact support</a>
            <a href={`mailto:${SUPPORT_EMAIL}?subject=RT365%20license%20help`}>License help</a>
            {DOWNLOAD_URL ? <a href={DOWNLOAD_URL}>Download app</a> : null}
          </div>
        </div>
        <p className="landing-footer__copy">
          © {new Date().getFullYear()} RT365. All rights reserved.
        </p>
      </footer>
    </div>
  );
}
