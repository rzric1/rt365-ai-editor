# RT365 AI Clip Studio — Full App Review Report

**Date:** 2026-05-28  
**Branch:** `feature/opus-quality-upgrade`  
**Python:** `.venv311\Scripts\python.exe`  
**Scope:** Production-hardening and quality-improvement pass (minimal-risk patches)

---

## Executive Summary

### What was reviewed

- `clip_studio_app.py` — Streamlit UI, session state, analysis/export flows, diagnostics
- `clip_engine/` — pipeline, finalizer, scoring, boundaries, cache, GPU prefilter, embeddings, OpenAI resilience

### What was fixed

1. **New Clip Quality Finalizer** (`clip_engine/clip_finalizer.py`) integrated into pipeline, cache reload, UI display, and export guard
2. **Hook title repair** strengthened for dangling fragments, copied transcript lines, and generic titles; `[HOOK REPAIR]` logging added
3. **Merge safety** — adjacent clips merge only when timeline gap **and** transcript similarity (2+ keywords, shared named entities, or embedding ≥ 0.72); removed weak single-keyword hook-only merge
4. **Session/cache** — invalidation logging; session no-op reruns finalize legacy clips; cached clips re-finalized and re-saved
5. **Export safety** — pre-export validation skips invalid clips with warnings instead of crashing the batch
6. **Diagnostics** — duplicate startup diagnostics suppressed with `[DIAGNOSTICS]` log; GPU status probes log failures at debug level

### Stability status

| Area | Status |
|------|--------|
| Compile / import | **Pass** |
| OpenAI wrapper architecture | **Pass** (only `openai_resilience.py` calls API) |
| Streamlit startup (headless) | **Pass** — app served on port 8501, no startup traceback |
| Full podcast E2E | **Not run in this pass** — requires manual test with real media |

### Ready for real podcast test?

**Yes, with caveats.** The app is structurally stable and quality gates are stronger. Run one full 40–90 minute podcast with Discovery Mode + Token Saver using the recommended settings below and review exported SRTs/hooks before batch production use.

---

## Files Modified

### `clip_engine/clip_finalizer.py` (NEW)

- **Reason:** Final quality pass before UI, cache, and export
- **Summary:** Expand boundaries, merge same-story fragments (safe similarity rules), reject unwatchable clips, repair hooks, attach `finalizer_*` metadata, fail-safe fallback to original clips on error

### `clip_engine/clip_pipeline.py`

- **Reason:** Integrate finalizer on normal path and cache hit path
- **Summary:** `_apply_clip_quality_finalizer()`, `PipelineStats` finalizer fields, cache re-save after finalize, `[PIPELINE] completed` log

### `clip_engine/clip_scoring.py`

- **Reason:** Hook quality for UI/export
- **Summary:** Stronger incomplete-title detection usage, transcript-copy detection in `ensure_all_clip_hooks`, `[HOOK REPAIR]` logs, penalties for generic/incomplete titles, improved `_core_message_from_window`

### `clip_engine/clip_boundaries.py`

- **Reason:** Shared incomplete hook detection
- **Summary:** `PARTIAL_HOOK_FRAGMENT_RE`, `INCOMPLETE_ABOUT_TITLE_RE` for user-reported bad title patterns

### `clip_engine/effective_config.py`

- **Reason:** Cache/session observability
- **Summary:** `[CACHE] invalidation reason=...` logs; `[SESSION] no-op widget rerun` log format

### `clip_engine/gpu_pipeline.py`

- **Reason:** GPU stability logging
- **Summary:** Replaced silent `except: pass` with debug logs for CUDA probe failures

### `clip_studio_app.py`

- **Reason:** UI workflow, export safety, finalizer on display/session skip
- **Summary:** Finalizer Report expander (existing), UI finalize for legacy clips, export `validate_clip_for_export`, `[EXPORT] completed` logs, `[DIAGNOSTICS]` skip message

---

## Issues Found and Fixed

### 1. Fragmented / weak final clips

- **Root cause:** Pipeline ended after hook repair without a final boundary/merge/reject pass
- **Fix:** `finalize_clips_for_ui()` after `ensure_all_clip_hooks`, before cache save
- **Before:** Multiple 15–25s fragments from one story, dangling endings, host-question starts
- **After:** Expansion, same-beat merge (with similarity), rejection of unwatchable clips, hook repair pass

### 2. Cached clips bypassing finalizer

- **Root cause:** Cache hit returned clips without running new finalizer
- **Fix:** Cache load path runs `_apply_clip_quality_finalizer()` and updates cache when safe
- **Before:** Old cache could show pre-finalizer clips
- **After:** Cache hits are re-finalized; metadata `finalizer_checked` stored on kept clips

### 3. Weak hook titles reaching export

- **Root cause:** Incomplete fragment titles not always caught or logged
- **Fix:** Expanded `hook_title_is_incomplete`, copy detection, `[HOOK REPAIR]` in `ensure_all_clip_hooks` and finalizer
- **Before:** Titles like `I had a brother that would have been`, `What veterans don't know about`
- **After:** Detected as incomplete; repaired from transcript window or rejected by finalizer

### 4. Unsafe adjacent merge

- **Root cause:** Single shared hook keyword could merge unrelated adjacent clips
- **Fix:** Removed hook-only 1-keyword merge; require 2+ keywords, named entities, or embedding similarity
- **Before:** Risk of merging dense podcast topics by proximity + weak hook overlap
- **After:** Merge requires substantive transcript similarity

### 5. Export batch crash on bad clip

- **Root cause:** No pre-flight validation on edited start/end/title
- **Fix:** `validate_clip_for_export()` — skip with `st.warning` + log, continue batch
- **Before:** One bad clip could disrupt export UX
- **After:** Invalid clips skipped; batch continues

### 6. Legacy session clips without finalizer metadata

- **Root cause:** Session no-op rerun reused old `cs_clips` as-is
- **Fix:** `ensure_clips_finalized()` on session skip and when rendering clip cards
- **Before:** Upgraded code could still show old weak clips until re-analyze
- **After:** Legacy clips finalized on display or session reuse

---

## Clip Quality Improvements

### Finalizer

- Runs after boundary repair, virality, quality gate, series splits, MIN_HOOK filter, and hook repair
- Order: expand → merge → re-expand merged → hook repair → reject
- Metadata: `finalizer_checked`, `finalizer_action`, `finalizer_reason`, `watchability_score`, `merged_from`
- Strong viral moments (virality ≥ 75, watchability ≥ 55) may keep slightly weaker hooks vs. hard reject

### Hook repair

- Patterns for dangling fragments (`that would have been`, `she was`, `so they`, `about`, etc.)
- Transcript-copy detection
- Local declarative rewrite via `_core_message_from_window`
- Logging: `[HOOK REPAIR] clip=X old="..." new="..." score=N`

### Boundary / watchability

- Sentence-span expansion using transcript segments
- Host-question start shift toward guest answer
- Rejection for incomplete start/end, low payoff, short non-standalone clips
- Fail-safe: finalizer exception returns original clips (no app crash)

### Merge / rejection

- Merge gap default 20s + similarity required
- Rejection preserves reasons in logs: `[CLIP FINALIZER] rejected clip=N reason="..."`

---

## Cache and Session State

### Invalidation triggers (only these re-run OpenAI)

| Trigger | Log |
|---------|-----|
| Explicit Re-analyze | `[CACHE] invalidation reason=explicit_reanalyze` |
| Video changed | `video_changed` |
| Transcript changed | `transcript_changed` |
| AI settings fingerprint changed | `ai_settings_changed` |
| Missing transcript/video | respective reasons |

### Must NOT trigger OpenAI

- Hook title edits, export checkboxes, start/end edits, caption/export settings, expanders — confirmed via fingerprint + `log_widget_rerun_noop` → `[SESSION] no-op widget rerun`

### Analysis vs UI state

- Durable: `SESSION_EFFECTIVE_CONFIG`, `SESSION_ANALYSIS_FINGERPRINT`, `cs_analysis_*`, `cs_clips` (analysis output)
- UI widgets: per-clip `hook_*`, `start_*`, `end_*`, `ex_*` — export reads widgets; analysis cache not invalidated by hook edits

---

## GPU / Embedding Safety

| Control | Behavior |
|---------|----------|
| `FORCE_CPU_EMBEDDINGS=1` | Skips GPU embeddings (existing `semantic_ranking.py`) |
| SentenceTransformer load/encode | Broad try/except; safe fallback |
| GPU prefilter fatal error | Logged traceback; returns `([], stats)` — does not kill Streamlit |
| CUDA transcription | **Not modified** in this pass |
| RTX status panel | Probe failures now debug-logged instead of silent pass |

### Remaining GPU risks

- First-run embedding model download can still be slow on cold start
- Very long podcasts + GPU embeddings may use significant VRAM; use `FORCE_CPU_EMBEDDINGS=1` if Streamlit becomes sluggish after analysis

---

## Validation Results

### Compile

```text
.\.venv311\Scripts\python.exe -m compileall clip_engine clip_studio_app.py
→ Exit code 0 (PASS)
```

### OpenAI direct-call search

```text
Select-String -Path .\clip_engine\*.py -Pattern "client.chat.completions.create"
→ Only clip_engine\openai_resilience.py (lines 429, 794, 816) — PASS
```

### Wrapper usage

```text
Select-String call_openai_chat|call_openai_chat_json|call_openai_with_backoff
→ clip_analysis.py, clip_metadata.py, clip_split.py, openai_resilience.py — PASS
```

### Unsafe exits (`clip_engine`, `clip_studio_app.py`)

```text
→ No matches (PASS)
```

### Silent exceptions (`except:` / `except Exception: pass`)

```text
→ No matches in clip_engine or clip_studio_app.py (PASS)
```

### Hook detection smoke

```text
hook_title_is_incomplete([
  "I had a brother that would have been",
  "What veterans dont know about",
  "So ever since I was born, she was"
]) → all True (PASS)
```

### Runtime smoke test

```text
.\.venv311\Scripts\python.exe -m streamlit run clip_studio_app.py --server.headless true
→ Started on http://localhost:8501, no traceback in first 5s (PASS)
```

**Not performed in automation:** CUDA transcribe, full Discovery pipeline, cache hit on real file, export MP4 batch, hook-edit no-op verification in browser.

---

## Remaining Risks

1. **Finalizer may reduce clip count** — quality-first; Discovery Mode may yield fewer than 20 clips if many fragments are rejected
2. **Heuristic host/filler detection** — not true diarization; edge cases may misclassify guest vs host
3. **Named-entity merge** — capitalization heuristic may miss lowercase entities or over-match
4. **No automated E2E** — real podcast test still required for SRT/hook quality confirmation
5. **`clip_finalizer.py` was untracked** — must be committed with this pass

---

## Recommended Next Step

1. Start app: `.\.venv311\Scripts\python.exe -m streamlit run clip_studio_app.py`
2. Run a **40–60 minute** emotional interview with:
   - Discovery Mode ON, Token Saver ON, Target 20, Min 15, Max 90–120, Min gap 35
3. Check **Finalizer Report** expander (expanded/merged/rejected counts)
4. Export 3–5 clips; verify SRT does not start with host questions or end on dangling phrases
5. Edit a hook title and confirm **no new OpenAI call** (status should say session clips still valid)
6. Re-open app and confirm cache hit + finalized clips load correctly

---

## Git Status at Report Time

Modified: `clip_boundaries.py`, `clip_pipeline.py`, `clip_scoring.py`, `effective_config.py`, `gpu_pipeline.py`, `clip_studio_app.py`  
New: `clip_engine/clip_finalizer.py`, `full_app_review_report.md`
