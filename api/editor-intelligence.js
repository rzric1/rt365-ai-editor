/**
 * AI Clip Studio — Seraph Edition — consolidated intelligence API (single Vercel function).
 * POST JSON: { mode: "clip-suggestions", transcript?: string }
 */
const MAX_TEXT_CHARS = 28000;

async function readJsonBody(req) {
  const buffers = [];
  for await (const chunk of req) {
    buffers.push(chunk);
  }
  const raw = Buffer.concat(buffers).toString('utf8');
  if (!raw) return {};
  return JSON.parse(raw);
}

async function verifySupabaseJwt(req) {
  const url = process.env.SUPABASE_URL;
  const anon = process.env.SUPABASE_ANON_KEY;
  if (!url || !anon) return { ok: false, reason: 'server_misconfigured' };

  const auth = req.headers?.authorization || req.headers?.Authorization;
  if (!auth || !auth.startsWith('Bearer ')) {
    return { ok: false, reason: 'missing_bearer' };
  }

  const r = await fetch(`${url.replace(/\/$/, '')}/auth/v1/user`, {
    headers: {
      Authorization: auth,
      apikey: anon,
    },
  });

  if (!r.ok) return { ok: false, reason: 'invalid_session' };
  return { ok: true, user: await r.json() };
}

function truncate(str, max) {
  if (!str || str.length <= max) return str || '';
  return `${str.slice(0, max)}\n\n[…truncated for length]`;
}

const SYSTEM_PROMPT = `You are AI Clip Studio — Seraph Edition's clip intelligence assistant for video creators and podcast editors.

Rules:
- Suggest concrete clip ideas with approximate timestamps when the transcript includes them.
- Focus on hooks, highlights, quotable moments, and chapter boundaries.
- Do not invent quotes — only reference content that appears in the transcript.
- Keep suggestions practical for short-form social clips and long-form chapter cuts.

Respond with ONLY valid JSON (no markdown fences) matching this shape:
{
  "clipSuggestions": [{"title": "string", "startHint": "string", "endHint": "string", "reason": "string"}],
  "chapterMarkers": ["string"],
  "summary": "string"
}`;

async function callOpenAi(transcript) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('Server missing OPENAI_API_KEY.');
  }

  const configured = process.env.OPENAI_MODEL || 'gpt-4o-mini';
  const model =
    typeof configured === 'string' && /^gpt|^o\d/i.test(configured) ? configured : 'gpt-4o-mini';

  const body = {
    model,
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      {
        role: 'user',
        content: `---\nTranscript:\n${truncate(transcript, MAX_TEXT_CHARS)}\n---`,
      },
    ],
  };

  const r = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!r.ok) {
    const errText = await r.text();
    throw new Error(`AI request failed: ${r.status} ${errText.slice(0, 200)}`);
  }

  const data = await r.json();
  const raw = data.choices?.[0]?.message?.content;
  if (!raw) throw new Error('Empty AI response.');
  return JSON.parse(raw);
}

function normalizeSuggestions(parsed) {
  return {
    clipSuggestions: Array.isArray(parsed.clipSuggestions) ? parsed.clipSuggestions : [],
    chapterMarkers: Array.isArray(parsed.chapterMarkers) ? parsed.chapterMarkers : [],
    summary: typeof parsed.summary === 'string' ? parsed.summary : '',
  };
}

export default async function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');

  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }

  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  try {
    const requireAuth =
      process.env.EDITOR_INTEL_REQUIRE_AUTH === 'true' ||
      process.env.EDITOR_INTEL_REQUIRE_AUTH === '1';

    if (requireAuth) {
      const auth = await verifySupabaseJwt(req);
      if (!auth.ok) {
        res.status(401).json({ error: 'Unauthorized', detail: auth.reason });
        return;
      }
    }

    const body = await readJsonBody(req);
    if (body.mode !== 'clip-suggestions') {
      res.status(400).json({ error: 'Unknown or missing mode' });
      return;
    }

    const transcript = typeof body.transcript === 'string' ? body.transcript.trim() : '';
    if (!transcript) {
      res.status(400).json({ error: 'Provide a transcript string.' });
      return;
    }

    const parsed = await callOpenAi(transcript);
    res.status(200).json({ suggestions: normalizeSuggestions(parsed) });
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Server error';
    res.status(500).json({ error: message });
  }
}
