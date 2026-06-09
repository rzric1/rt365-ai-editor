/**
 * Stripe webhook handler — generates and emails a license key on successful payment.
 * POST /api/stripe-webhook
 *
 * Requires env vars:
 *   STRIPE_SECRET_KEY        — Stripe secret key (sk_live_... or sk_test_...)
 *   STRIPE_WEBHOOK_SECRET    — Webhook signing secret from Stripe dashboard (whsec_...)
 *   SUPABASE_URL             — Supabase project URL
 *   SUPABASE_SERVICE_ROLE_KEY — Supabase service role key (bypasses RLS)
 *   RESEND_API_KEY           — Resend API key
 *   RESEND_FROM_EMAIL        — Verified sender address, e.g. "RT365 <licenses@yourdomain.com>"
 *   SUPPORT_EMAIL            — Support address shown in the email body
 *   APP_DOWNLOAD_URL         — (optional) Download link included in the email
 */

import Stripe from 'stripe';
import { createClient } from '@supabase/supabase-js';
import { Resend } from 'resend';
import { randomBytes } from 'crypto';

// Disable Vercel's JSON body parser — Stripe signs the exact raw bytes.
export const config = {
  api: {
    bodyParser: false,
  },
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function generateLicenseKey() {
  const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  const buf = randomBytes(16);
  const groups = [];
  for (let i = 0; i < 4; i++) {
    let g = '';
    for (let j = 0; j < 4; j++) {
      g += CHARS[buf[i * 4 + j] % CHARS.length];
    }
    groups.push(g);
  }
  return `RT365-${groups.join('-')}`;
}

function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function buildEmailHtml(licenseKey, supportEmail, downloadUrl) {
  const downloadSection = downloadUrl
    ? `<li><a href="${downloadUrl}" style="color:#1e4d7a;">Download RT365 AI Editor</a></li>`
    : '';
  return `<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Your License Key</title></head>
<body style="margin:0;padding:0;background:#f4f6f8;font-family:'Segoe UI',system-ui,sans-serif;">
  <div style="max-width:600px;margin:40px auto;background:#fff;border-radius:10px;overflow:hidden;border:1px solid #d8dee6;">
    <div style="background:#1e4d7a;padding:28px 32px;">
      <h1 style="margin:0;color:#fff;font-size:1.4rem;font-weight:700;">RT365 AI Editor</h1>
      <p style="margin:4px 0 0;color:#b8d0e8;font-size:0.95rem;">License confirmation</p>
    </div>
    <div style="padding:32px;">
      <p style="margin:0 0 20px;color:#1a2332;">Thank you for your purchase! Your license key is ready to activate.</p>

      <div style="background:#f0f4f8;border:2px solid #1e4d7a;border-radius:8px;padding:24px;text-align:center;margin:0 0 28px;">
        <p style="margin:0 0 8px;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#5a6573;">Your License Key</p>
        <code style="font-size:1.6rem;font-weight:700;letter-spacing:0.12em;color:#1e4d7a;word-break:break-all;">${licenseKey}</code>
      </div>

      <h2 style="font-size:1rem;font-weight:700;color:#1a2332;margin:0 0 12px;">How to activate</h2>
      <ol style="margin:0 0 24px;padding-left:1.2rem;color:#1a2332;line-height:1.7;">
        ${downloadSection}
        <li>Launch RT365 AI Editor</li>
        <li>Enter the key above when the activation screen appears</li>
        <li>Click <strong>Activate</strong> — your app is ready</li>
      </ol>

      <p style="margin:0;font-size:0.9rem;color:#5a6573;">
        Questions? Email <a href="mailto:${supportEmail}" style="color:#1e4d7a;">${supportEmail}</a> and we will get back to you promptly.
      </p>
    </div>
    <div style="background:#f4f6f8;padding:16px 32px;border-top:1px solid #d8dee6;">
      <p style="margin:0;font-size:0.75rem;color:#5a6573;">This key is for a single computer. If you need to transfer it to a new machine, contact support.</p>
    </div>
  </div>
</body>
</html>`;
}

// ── Handler ───────────────────────────────────────────────────────────────────

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  // Read raw body as Buffer — Stripe signature verification requires the exact bytes
  const rawBody = await readRawBody(req);
  const sig = req.headers['stripe-signature'];

  if (!sig) {
    console.error('[stripe-webhook] Missing stripe-signature header');
    res.status(400).json({ error: 'Missing stripe-signature header' });
    return;
  }

  if (!Buffer.isBuffer(rawBody) || rawBody.length === 0) {
    console.error('[stripe-webhook] Empty or invalid raw body');
    res.status(400).json({ error: 'Empty request body' });
    return;
  }

  const stripeSecretKey = process.env.STRIPE_SECRET_KEY;
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  console.log('Secret prefix:', process.env.STRIPE_WEBHOOK_SECRET?.slice(0, 10));

  if (!stripeSecretKey || !webhookSecret) {
    console.error('[stripe-webhook] Missing STRIPE_SECRET_KEY or STRIPE_WEBHOOK_SECRET');
    res.status(500).json({ error: 'Server misconfigured' });
    return;
  }

  const stripe = new Stripe(stripeSecretKey, { apiVersion: '2024-12-18.acacia' });

  // Verify signature
  let event;
  try {
    event = stripe.webhooks.constructEvent(rawBody, sig, webhookSecret);
  } catch (err) {
    console.error('[stripe-webhook] Signature verification failed:', err.message);
    res.status(400).json({ error: `Webhook signature invalid: ${err.message}` });
    return;
  }

  // Only process completed checkouts — acknowledge all other events immediately
  if (event.type !== 'checkout.session.completed') {
    res.status(200).json({ received: true });
    return;
  }

  const session = event.data.object;
  const customerEmail = session.customer_details?.email || session.customer_email;
  const stripeSessionId = session.id;

  if (!customerEmail) {
    console.error('[stripe-webhook] No customer email on session', stripeSessionId);
    res.status(400).json({ error: 'No customer email in session' });
    return;
  }

  // Init Supabase with service role key (bypasses RLS)
  const supabaseUrl = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!supabaseUrl || !serviceRoleKey) {
    console.error('[stripe-webhook] Missing Supabase env vars');
    res.status(500).json({ error: 'Server misconfigured' });
    return;
  }
  const supabase = createClient(supabaseUrl, serviceRoleKey);

  // Idempotency: check if this session was already processed
  const { data: existing } = await supabase
    .from('license_keys')
    .select('key, customer_email')
    .eq('stripe_session_id', stripeSessionId)
    .maybeSingle();

  if (existing) {
    console.log('[stripe-webhook] Session already processed, idempotent return', stripeSessionId);
    res.status(200).json({ received: true, duplicate: true });
    return;
  }

  // Generate key and insert
  const licenseKey = generateLicenseKey();
  const { error: insertError } = await supabase.from('license_keys').insert({
    key: licenseKey,
    customer_email: customerEmail,
    stripe_session_id: stripeSessionId,
  });

  if (insertError) {
    console.error('[stripe-webhook] Supabase insert failed:', insertError.message);
    // Return 500 so Stripe retries the webhook
    res.status(500).json({ error: 'Failed to save license key' });
    return;
  }

  console.log('[stripe-webhook] License key created for', customerEmail);

  // Respond 200 immediately — email is fire-and-forget
  res.status(200).json({ received: true });

  // Send email async (do not await — Stripe already got its 200)
  const resendKey = process.env.RESEND_API_KEY;
  const fromEmail = process.env.RESEND_FROM_EMAIL || 'RT365 <licenses@rt365.ai>';
  const supportEmail = process.env.SUPPORT_EMAIL || 'support@rt365.ai';
  const downloadUrl = process.env.APP_DOWNLOAD_URL || '';

  if (!resendKey) {
    console.error('[stripe-webhook] RESEND_API_KEY not set — email not sent for key', licenseKey);
    return;
  }

  const resend = new Resend(resendKey);
  resend.emails
    .send({
      from: fromEmail,
      to: customerEmail,
      subject: 'Your RT365 AI Editor License Key',
      html: buildEmailHtml(licenseKey, supportEmail, downloadUrl),
    })
    .then(() => {
      console.log('[stripe-webhook] Email sent to', customerEmail);
    })
    .catch((err) => {
      console.error('[stripe-webhook] Email send failed for key', licenseKey, '—', err.message);
    });
}
