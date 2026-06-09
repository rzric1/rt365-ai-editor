/**
 * License key validation endpoint.
 * POST /api/validate-license
 * Body: { license_key: string, instance_id: string }
 *
 * Requires env vars:
 *   SUPABASE_URL              — Supabase project URL
 *   SUPABASE_SERVICE_ROLE_KEY — Supabase service role key (bypasses RLS)
 *   SUPPORT_EMAIL             — Shown in the "already activated" error message
 */

import { createClient } from '@supabase/supabase-js';

const KEY_PATTERN = /^RT365-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$/;

async function readJsonBody(req) {
  const buffers = [];
  for await (const chunk of req) {
    buffers.push(chunk);
  }
  const raw = Buffer.concat(buffers).toString('utf8');
  if (!raw) return {};
  return JSON.parse(raw);
}

export default async function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');

  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  let body;
  try {
    body = await readJsonBody(req);
  } catch {
    res.status(400).json({ error: 'Invalid JSON body' });
    return;
  }

  const { license_key, instance_id, verify_only } = body;

  if (!license_key || typeof license_key !== 'string') {
    res.status(400).json({ error: 'license_key is required' });
    return;
  }
  if (!verify_only && (!instance_id || typeof instance_id !== 'string')) {
    res.status(400).json({ error: 'instance_id is required' });
    return;
  }

  // Normalise and basic format-check before hitting the database
  const normKey = license_key.trim().toUpperCase();
  if (!KEY_PATTERN.test(normKey)) {
    res.status(200).json({ valid: false, error: 'Invalid license key format.' });
    return;
  }

  const supabaseUrl = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!supabaseUrl || !serviceRoleKey) {
    console.error('[validate-license] Missing Supabase env vars');
    res.status(500).json({ error: 'Server misconfigured' });
    return;
  }

  const supabase = createClient(supabaseUrl, serviceRoleKey);
  const supportEmail = process.env.SUPPORT_EMAIL || 'support@rt365.ai';

  try {
    const { data: row, error: queryError } = await supabase
      .from('license_keys')
      .select('id, customer_email, activated_at, instance_id')
      .eq('key', normKey)
      .eq('is_active', true)
      .maybeSingle();

    if (queryError) {
      console.error('[validate-license] Query error:', queryError.message);
      res.status(500).json({ error: 'Database error' });
      return;
    }

    // Key not found or deactivated
    if (!row) {
      res.status(200).json({ valid: false, error: 'Invalid license key.' });
      return;
    }

    // Landing-page verification — do not bind instance_id
    if (verify_only) {
      if (row.activated_at && row.instance_id) {
        res.status(200).json({
          valid: true,
          email: row.customer_email,
          activated: true,
        });
        return;
      }
      res.status(200).json({
        valid: true,
        email: row.customer_email,
        activated: false,
      });
      return;
    }

    // Never activated — bind to this instance now
    if (!row.activated_at) {
      const { error: updateError } = await supabase
        .from('license_keys')
        .update({
          activated_at: new Date().toISOString(),
          instance_id: instance_id,
        })
        .eq('id', row.id);

      if (updateError) {
        console.error('[validate-license] Activation update failed:', updateError.message);
        res.status(500).json({ error: 'Activation failed, please try again.' });
        return;
      }

      console.log('[validate-license] Key activated:', normKey, 'instance:', instance_id);
      res.status(200).json({ valid: true, email: row.customer_email });
      return;
    }

    // Already activated — same instance is fine
    if (row.instance_id === instance_id) {
      res.status(200).json({ valid: true, email: row.customer_email });
      return;
    }

    // Activated on a different machine
    res.status(200).json({
      valid: false,
      error: `Key already activated on another machine. Contact ${supportEmail} to transfer your license.`,
    });
  } catch (err) {
    console.error('[validate-license] Unexpected error:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
}
