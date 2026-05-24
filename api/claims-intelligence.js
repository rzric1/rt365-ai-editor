/**
 * Consolidated claims intelligence API (single Vercel function).
 * POST JSON: { mode: "denial-analysis", text?, fileBase64?, mimeType?, filename? }
 */
import { createRequire } from 'module';

const require = createRequire(import.meta.url);

let pdfParseFn = null;
try {
  pdfParseFn = require('pdf-parse');
} catch {
  pdfParseFn = null;
}

const MAX_TEXT_CHARS = 28000;
const MAX_FILE_BYTES = 4_500_000;

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

async function extractDocumentParts({ text, fileBase64, mimeType, filename }) {
  let pasted = typeof text === 'string' ? text.trim() : '';
  let image = null;
  let fromFileText = '';

  if (fileBase64) {
    const buf = Buffer.from(fileBase64, 'base64');
    if (buf.length > MAX_FILE_BYTES) {
      throw new Error('File too large (max ~4.5MB). Try pasting text or a smaller export.');
    }
    const mime = (mimeType || '').toLowerCase();
    const name = (filename || '').toLowerCase();

    if (mime === 'application/pdf' || name.endsWith('.pdf')) {
      if (!pdfParseFn) {
        throw new Error('PDF parsing unavailable. Paste the denial text below or upload a PNG/JPG export.');
      }
      const parsed = await pdfParseFn(buf);
      fromFileText = ((parsed && parsed.text) || '').trim();
      if (!fromFileText) {
        throw new Error('Could not read text from this PDF. Paste the denial language manually.');
      }
    } else if (mime.startsWith('image/')) {
      image = { mime, base64: fileBase64 };
    } else {
      throw new Error('Unsupported file type. Use PDF, PNG, JPG, or paste text.');
    }
  }

  const plain = [pasted, fromFileText].filter(Boolean).join('\n\n').trim();
  return { plain, image };
}

const SYSTEM_PROMPT = `You are helping U.S. veterans understand VA rating decisions and denial letters for educational purposes only.

Rules:
- Extract factual content that appears in the document. Do not invent citations or quotes.
- Do not provide legal advice, legal opinions, or predict claim outcomes.
- Use plain, respectful, veteran-friendly language.
- If the input is not a VA decision or is too vague, say so in statedReason and keep other lists short or empty.
- Suggest next steps only as general educational options (e.g., supplemental claim, higher-level review, gathering records)—not as what the veteran "must" do.
- Suggest which Tactical Claims AI tools might help: Initial claim, Increase, Evidence review, Tactical conditions, Records readiness, Denial analysis, Guided claim setup, Tactical review.
- Set educationalNote to exactly: "This is an educational summary. Verify with a VSO or accredited representative for decisions." plus one short sentence if needed.

Respond with ONLY valid JSON (no markdown fences) matching this shape:
{
  "favorableFindings": ["string"],
  "missingEvidence": ["string"],
  "statedReason": "string",
  "nextStepSuggestions": ["string"],
  "suggestedTools": ["string"],
  "educationalNote": "string"
}`;

async function callOpenAi({ documentText, vision, pastedWithImage }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('Server missing OPENAI_API_KEY.');
  }

  const configured = process.env.OPENAI_MODEL || 'gpt-4o-mini';
  const textModel =
    typeof configured === 'string' && /^gpt|^o\d/i.test(configured) ? configured : 'gpt-4o-mini';
  const model = vision ? 'gpt-4o-mini' : textModel;

  let userMessage;
  if (vision) {
    const textPart = truncate(pastedWithImage || documentText, 8000);
    const intro = textPart
      ? `The veteran also pasted this text (may overlap the image):\n${textPart}\n\n`
      : '';
    userMessage = {
      role: 'user',
      content: [
        {
          type: 'text',
          text: `${intro}Using the decision letter or denial image, extract findings and return ONLY the JSON object described in the system message.`,
        },
        {
          type: 'image_url',
          image_url: { url: `data:${vision.mime};base64,${vision.base64}` },
        },
      ],
    };
  } else {
    userMessage = {
      role: 'user',
      content: `---\nDocument text:\n${truncate(documentText, MAX_TEXT_CHARS)}\n---`,
    };
  }

  const body = {
    model,
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      userMessage,
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
      process.env.CLAIMS_INTEL_REQUIRE_AUTH === 'true' ||
      process.env.CLAIMS_INTEL_REQUIRE_AUTH === '1';

    if (requireAuth) {
      const auth = await verifySupabaseJwt(req);
      if (!auth.ok) {
        res.status(401).json({ error: 'Unauthorized', detail: auth.reason });
        return;
      }
    }

    const body = await readJsonBody(req);
    if (body.mode !== 'denial-analysis') {
      res.status(400).json({ error: 'Unknown or missing mode' });
      return;
    }

    const { text, fileBase64, mimeType, filename } = body;
    if (!text?.trim() && !fileBase64) {
      res.status(400).json({ error: 'Provide pasted text and/or an uploaded file.' });
      return;
    }

    const { plain, image } = await extractDocumentParts({ text, fileBase64, mimeType, filename });

    if (image) {
      const parsed = await callOpenAi({
        documentText: plain,
        vision: image,
        pastedWithImage: plain,
      });
      const normalized = normalizeSummary(parsed);
      res.status(200).json({ summary: normalized });
      return;
    }

    if (!plain) {
      res.status(400).json({ error: 'No usable text after processing upload.' });
      return;
    }

    const parsed = await callOpenAi({ documentText: plain, vision: null, pastedWithImage: '' });
    const normalized = normalizeSummary(parsed);
    res.status(200).json({ summary: normalized });
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Server error';
    res.status(500).json({ error: message });
  }
}

function normalizeSummary(parsed) {
  const fallback =
    'This is an educational summary. Verify with a VSO or accredited representative for decisions.';
  return {
    favorableFindings: Array.isArray(parsed.favorableFindings) ? parsed.favorableFindings : [],
    missingEvidence: Array.isArray(parsed.missingEvidence) ? parsed.missingEvidence : [],
    statedReason: typeof parsed.statedReason === 'string' ? parsed.statedReason : '',
    nextStepSuggestions: Array.isArray(parsed.nextStepSuggestions) ? parsed.nextStepSuggestions : [],
    suggestedTools: Array.isArray(parsed.suggestedTools) ? parsed.suggestedTools : [],
    educationalNote: typeof parsed.educationalNote === 'string' ? parsed.educationalNote : fallback,
  };
}
