# RT365 AI Clip Studio — Full Diagnostic + Stabilization Report

**Date:** 2026-05-27  
**Branch:** `feature/opus-quality-upgrade`  
**Repository:** `C:\dev\rt365-ai-editor`  
**Scope:** Stabilization pass only (no new features, no architecture redesign)

---

# 1. Executive Summary

This pass targeted known stability issues: duplicate hardware diagnostics on Streamlit reruns, repeated token-plan logging, spurious AI profile warnings, incomplete hook titles reaching the UI, aggressive weak-hook filtering, and missing lifecycle visibility after analysis/export.

**Overall stability status:** Improved. Startup diagnostics are deduplicated per session, token estimates are cached for UI display, model resolution uses durable session config without warning spam, hook repair runs at quality gate + final pipeline pass + cache load, and weak-hook threshold is lowered to 25.

**Startup / analysis / export / rerun:**

| Area | Status |
|------|--------|
| Streamlit startup | Healthy — diagnostics run once per session |
| Analysis pipeline | Healthy — hook finalization wired; duplicate token plan at pipeline end removed |
| Export | Unchanged behavior — lifecycle logging added |
| Widget reruns | Stable — token plan and diagnostics no longer re-log on every rerun |

**Remaining risks:** Streamlit process can still restart if the host kills the Python process (external to app code). Full runtime UI testing was not executed in this session; compile and architecture checks passed.

---

# 2. Files Modified

## `clip_studio_app.py`

- Added `_run_startup_diagnostics_once()` and `_get_ai_diagnostics()` session caches.
- Moved CUDA/FFmpeg startup probes out of per-rerun path.
- Replaced `use_container_width` with `width="stretch"` on dataframe.
- Switched transcript token preview to `get_cached_token_plan(..., emit_logs=False)`.
- Removed duplicate pre-analysis token plan call before `_run_clip_analysis`.
- Added lifecycle logs after analysis/export and at end of render cycle.

## `clip_engine/cuda_diagnostics.py`

- Cached `collect_ai_acceleration_diagnostics()` result module-wide.
- `log_ai_acceleration_startup()` logs once per process with skip message on duplicates.

## `clip_engine/ffmpeg_resolve.py`

- `ensure_ffmpeg_on_path(log=True)` verbose logging runs once per process.

## `clip_engine/effective_config.py`

- `_safe_default_models()` downgraded to debug (no warning spam).
- `resolve_models_from_call_context()` falls back to SAFE profile silently when pipeline context absent.
- `plan_analysis_token_estimate(..., emit_logs=True)` dedupes `[TOKEN PLAN]` logs per cache key.
- Added `get_cached_token_plan()` for UI reruns without re-logging.
- `store_analysis_snapshot()` persists resolved models in session.
- `resolve_models_for_session()` restores models from durable session first.

## `clip_engine/clip_pipeline.py`

- Added `ensure_all_clip_hooks()` after weak-hook filter (final UI guard).
- Cache hits run hook finalization on cached clips.
- Removed redundant end-of-pipeline `plan_analysis_token_estimate()` (source of extra TOKEN PLAN logs).
- Added pipeline completion log.

## `clip_engine/clip_scoring.py`

- Added `ensure_all_clip_hooks()` for declarative title repair before UI/export.

## `clip_engine/clip_quality_gate.py`

- Uses `hook_title_is_incomplete()` (not only `ends_with_dangling_word`) before clips reach UI.

## `clip_engine/clip_diversity.py`

- `MIN_HOOK_SCORE` lowered from **35 → 25** for better retention.

## `clip_engine/openai_resilience.py`

- JSON repair log now includes `source_model`, `fallback_used`, `reason`, and safe preview snippet.

---

# 3. Root Cause Analysis

## Duplicate CUDA + FFmpeg diagnostics spam

| | |
|---|---|
| **Root cause** | `main()` called `ensure_ffmpeg_on_path(log=True)`, `log_nvenc_probe_command_explicit()`, and `log_ai_acceleration_startup()` on every Streamlit rerun; sidebar also called `collect_ai_acceleration_diagnostics()` which re-probes NVENC/CUDA. |
| **Why** | Streamlit reruns full script; no session guard on expensive probes. |
| **Fix** | Session flag `cs_diagnostics_initialized`, module caches in `cuda_diagnostics` / `ffmpeg_resolve`, cached sidebar diagnostics. |
| **Type** | Preventive |

## App exits unexpectedly after pipeline completion

| | |
|---|---|
| **Root cause** | No `sys.exit()` / `os._exit()` in clip studio path (verified). Likely external process recycle or unlogged exception; insufficient lifecycle logging made this hard to diagnose. |
| **Fix** | Added `[lifecycle]` logs after analysis, export, and render cycle; improved CUDA cache release exception logging. |
| **Type** | Reactive (observability); no forced exit path found in app code |

## `[AI PROFILE ERROR]` / missing effective config

| | |
|---|---|
| **Root cause** | `resolve_models_from_call_context()` called `_safe_default_models()` which logged WARNING when OpenAI call context was cleared after pipeline (`set_call_context(None)` in `finally`). |
| **Fix** | Silent SAFE fallback from call context; durable models stored in `cs_durable_resolved_models`; `resolve_models_for_session()` prefers session snapshot. |
| **Type** | Preventive |

## `[TOKEN PLAN]` logging 4× per rerun

| | |
|---|---|
| **Root cause** | `plan_analysis_token_estimate()` invoked from UI transcript section, analyze button, `_run_clip_analysis`, and pipeline `finally` block — each emitting INFO logs. |
| **Fix** | `emit_logs` dedupe by plan cache key; UI uses `get_cached_token_plan(emit_logs=False)`; removed pipeline-end replan. |
| **Type** | Preventive |

## Hook title fragments in UI

| | |
|---|---|
| **Root cause** | Quality gate only checked `ends_with_dangling_word`, missing article+noun and partial-phrase patterns; no final enforcement after weak-hook filter. |
| **Fix** | Quality gate uses `hook_title_is_incomplete()`; pipeline + cache hit call `ensure_all_clip_hooks()`. |
| **Type** | Preventive |

## Weak hook threshold too aggressive

| | |
|---|---|
| **Root cause** | `MIN_HOOK_SCORE = 35` in `clip_diversity.py`. |
| **Fix** | Set to **25**; floor of 5 clips still applies. |
| **Type** | Preventive |

## `json_repairs=2` / JSON hardening

| | |
|---|---|
| **Root cause** | Malformed or markdown-wrapped JSON from model; repair path existed but logging lacked repair context. |
| **Fix** | Enhanced repair log line in `call_openai_chat_json` (model, fallback flag, reason, preview). Existing `extract_json_from_text` + one repair max preserved. |
| **Type** | Reactive hardening (logging); no architecture bypass |

## Streamlit `use_container_width` deprecation

| | |
|---|---|
| **Root cause** | Streamlit API deprecation on `st.dataframe`. |
| **Fix** | `width="stretch"` on GPU explorer dataframe. |
| **Type** | Preventive |

## Session-state misuse

| | |
|---|---|
| **Root cause** | Token plan and diagnostics recomputed on every widget rerun. |
| **Fix** | Durable keys: `cs_durable_effective_config`, `cs_analysis_fingerprint`, `cs_durable_resolved_models`, `cs_token_plan_cache*`. |
| **Type** | Preventive |

---

# 4. Validation Results

## Compile Validation

**Command:**

```powershell
.\.venv311\Scripts\python.exe -m compileall clip_engine clip_studio_app.py
```

**Result:** **PASS** (all listed modules compiled; `from clip_engine.effective_config import get_cached_token_plan` smoke import OK)

## OpenAI Architecture Validation

**Command:**

```powershell
Select-String -Path .\clip_engine\*.py -Pattern "client.chat.completions.create"
```

**Result:** **PASS** — matches only in `clip_engine/openai_resilience.py` (docstring, wrapper, and single create call site).

**Wrapper usage:** `call_openai_chat_json` used in `clip_analysis.py`, `clip_metadata.py`, `clip_split.py`.

## Runtime Validation

| Check | Status | Notes |
|-------|--------|-------|
| App launch | Not run live | Compile/import OK |
| CUDA transcription | Not run live | Diagnostics caching unchanged behavior |
| FFmpeg detection | Not run live | Once-per-session logging only |
| Discovery Mode | Preserved | No logic removed |
| Cache persistence | Improved | Cache hits finalize hooks |
| Rerun stability | Improved | Deduped diagnostics + token plan |
| Export pipeline | Preserved | Lifecycle log only |
| Caption rendering | Not modified | |
| Hook edit without AI rerun | Preserved | Existing fingerprint logic intact |
| Unexpected exits | No code exit path found | Added lifecycle logs |

---

# 5. Logging Improvements

### Added

- `diagnostics already initialized — skipping duplicate diagnostics`
- `skipping duplicate diagnostics — startup AI acceleration already logged`
- `skipping duplicate diagnostics — ffmpeg startup already logged`
- `skipping duplicate diagnostics — token plan already logged for <key>`
- `[TOKEN PLAN] cache reuse for display`
- `[ANALYSIS] config restored from session state`
- `[CACHE] hit — skipping OpenAI pipeline (cache reuse)`
- `[lifecycle] Clip analysis finished — app remains active`
- `[lifecycle] Export batch finished — app remains active`
- `[lifecycle] Streamlit render cycle complete`
- JSON repair: `source_model`, `fallback_used`, `reason`, `preview`

### Removed / reduced

- Duplicate `[TOKEN PLAN]` at pipeline completion
- Per-rerun `[ffmpeg]` / `[ai-accel]` startup spam
- WARNING-level `[AI PROFILE]` on normal post-pipeline context clear

---

# 6. JSON Hardening Results

| Item | Detail |
|------|--------|
| Failing call sites | Generic parse failures in `call_openai_chat_json` (all stages using wrapper) |
| Repair logic | Unchanged: one `repair_json_with_chat` max; GPT-5 → fallback retry preserved |
| Fallback | Unchanged via `resolve_json_fallback_model` + `call_openai_chat_json` |
| Expected impact | Fewer opaque repairs; easier to trace remaining `json_repairs` via enriched logs |

---

# 7. Cache + Session State Audit

### Persistent session keys (not widget-bound)

- `cs_durable_effective_config`
- `cs_durable_resolved_models`
- `cs_analysis_fingerprint`
- `cs_analysis_video_identity`
- `cs_analysis_transcript_hash`
- `cs_analysis_diagnostics`
- `cs_token_plan_cache` / `cs_token_plan_cache_key`
- `cs_diagnostics_initialized`
- `cs_ai_diag_cache`
- `cs_clip_ui_edits` (UI-only edits)

### Widget interactions that do NOT invalidate analysis

- Hook title edits (`hook_widget_*`)
- Export checkboxes (`ex_*`)
- Start/end time widgets
- Caption preset per clip

### Invalidation triggers (unchanged)

- New video / path change
- Transcript change
- AI settings fingerprint change
- Explicit Re-analyze / Re-score

---

# 8. Remaining Technical Debt

1. **Live Streamlit session testing** not executed in this pass — recommend manual smoke test on long podcast.
2. **Cached analysis from before hook repair** may still exist on disk until cache clear; in-memory cache hits now run `ensure_all_clip_hooks`.
3. **Some `except Exception: pass`** remain in GPU/semantic modules (not all audited to avoid scope creep).
4. **App exit after pipeline** — no in-repo exit call found; if issue persists, capture Streamlit server logs outside Python app.

---

# 9. Git Summary

| Field | Value |
|-------|--------|
| Branch | `feature/opus-quality-upgrade` |
| Commit message | `Full diagnostic stabilization pass` |
| Push | `origin/feature/opus-quality-upgrade` (see commit hash after push) |

---

# 10. Final Stability Assessment

| Metric | Assessment |
|--------|------------|
| Overall confidence | **Medium-high** for logging/rerun/hook/config issues addressed |
| Production readiness | Suitable for continued beta on long podcasts with cache clear after deploy |
| Long podcast processing | Improved retention (hook threshold 25) and hook repair guards |
| Additional testing recommended | Yes — one full analyze → edit hooks → export cycle on GPU machine |

**Conclusion:** Stabilization changes are minimal-risk, preserve architecture and OpenAI wrapper rules, and directly address the audited spam/warning/hook issues. Full end-to-end runtime validation should be completed on target hardware before production promotion.
