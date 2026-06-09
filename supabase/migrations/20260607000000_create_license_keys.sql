-- RT365 AI Clip Studio — license key table
-- Run once against your Supabase project (SQL editor or supabase db push)

CREATE TABLE IF NOT EXISTS license_keys (
  id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  key               TEXT        UNIQUE NOT NULL,
  customer_email    TEXT        NOT NULL,
  stripe_session_id TEXT        UNIQUE NOT NULL,
  created_at        TIMESTAMPTZ DEFAULT now(),
  activated_at      TIMESTAMPTZ,
  instance_id       TEXT,
  is_active         BOOLEAN     DEFAULT true
);

-- Server-side operations use the service role key and bypass RLS.
-- Enable RLS so the anon/browser role cannot read this table at all.
ALTER TABLE license_keys ENABLE ROW LEVEL SECURITY;

-- No SELECT / INSERT / UPDATE policies for anon — all access goes through
-- server-side API functions using the service role key.

-- Index for the hot query path: validate-license looks up by key + is_active.
CREATE INDEX IF NOT EXISTS idx_license_keys_key_active
  ON license_keys (key, is_active);

-- Index for Stripe dedup: webhook checks stripe_session_id before inserting.
CREATE INDEX IF NOT EXISTS idx_license_keys_stripe_session
  ON license_keys (stripe_session_id);
