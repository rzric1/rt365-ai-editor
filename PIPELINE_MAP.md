# RT365 AI Clip Studio — Pipeline Map (Forensic Inventory)

**Audit date:** 2026-06-01  
**Scope:** `c:\dev\rt365-ai-editor` — primary product path is **AI Clip Studio** (`clip_studio_app.py`).  
**Runtime model:** Single-user, local-first, **one Python process** (Streamlit) + **child subprocesses** (FFmpeg, Resolve bridge). No separate backend server for Clip Studio.

---

## 1. Executive summary — what is “running”

When you launch Clip Studio, these components are active:

| Layer | Component | Process |
|-------|-----------|---------|
| Launcher | `launch_ai_clip_studio.ps1` / `run_app.bat` | PowerShell/CMD (parent) |
| UI server | Streamlit + Uvicorn | `python` / `streamlit` on port **8501** |
| App logic | `clip_studio_app.py` + `ui/*` + `clip_engine/*` | Same Python process |
| Media | FFmpeg / ffprobe | **Separate `ffmpeg.exe` per invocation** |
| ML (optional) | faster-whisper, CTranslate2, PyTorch, sentence-transformers, ultralytics | In-process + CUDA |
| Cloud | OpenAI HTTP API | Network from Python |
| Resolve (on demand) | `resolve_bridge.py` | Short-lived Python subprocess |

**Not running unless explicitly started:**

- Vite/React app (`npm run dev` / `dist/`) — separate **Claims** SPA, not Clip Studio.
- `app.py` Streamlit **Resolve companion** — separate entry (`run_app.bat` uses `clip_studio_app.py`).
- `main.py` CLI marker tool — batch/CLI only.
- Webhook server (`rt365-webhook`) — external repo.
- No Celery, Redis, cron, watchdog, or Windows services installed by this repo.

---

## 2. Frontend components

### 2.1 Clip Studio (primary)

| Item | Technology | Entry |
|------|------------|-------|
| UI framework | **Streamlit 1.31+** | `clip_studio_app.py` |
| Layout | Wide layout, sidebar | `st.set_page_config` |
| Modules | `ui/sidebar.py`, `ui/clip_cards.py`, `ui/export_panel.py`, `ui/resolve_panel.py` | Imported each rerun |
| Session | `ui/session_state.py` | Widget keys + clip state in `st.session_state` |
| Styling | Inline HTML in clip map; minimal custom CSS | `clip_cards._render_clip_map` |

**Behavior:** Every widget change triggers a **full script rerun** (Streamlit model). No React/Vue SPA for Clip Studio.

### 2.2 Secondary frontends (same repo, different products)

| Item | Path | Purpose |
|------|------|---------|
| Vite + React 19 | `src/main.jsx`, `src/App.jsx` | Claims / VA tools UI |
| Static `index.html` | Root | Vercel/static landing variant |
| Streamlit companion | `app.py` | Resolve marker companion chat |

---

## 3. Backend services

Clip Studio has **no dedicated API server**. “Backend” is in-process Python:

| Service | Module | Trigger |
|---------|--------|---------|
| Transcription | `clip_engine/transcription.py` | User: Transcribe |
| Clip pipeline | `clip_engine/clip_pipeline.py` | User: Analyze / Re-analyze |
| Export | `clip_engine/export_vertical.py` | User: Export / Preview |
| Resolve bridge | `resolve_bridge.py` | User: Send to Resolve |
| Environment check | `check_environment.py` | Manual CLI |
| Claims API (optional) | `api/claims-intelligence.js` | Vercel serverless — **not Clip Studio** |

---

## 4. Python module inventory (Clip Studio core)

### 4.1 Entry & UI

- `clip_studio_app.py` — main Streamlit entry
- `ui/session_state.py`, `ui/sidebar.py`, `ui/clip_cards.py`, `ui/export_panel.py`, `ui/resolve_panel.py`
- `ui_helpers.py` — folders, Resolve helpers, transcript paths
- `config.py` — paths, env keys, limits

### 4.2 Pipeline / AI

- `clip_engine/clip_pipeline.py` — orchestrator (`run_full_clip_pipeline`)
- `clip_engine/clip_analysis.py` — OpenAI multipass candidates
- `clip_engine/gpu_pipeline.py` — local prefilter + embeddings shortlist
- `clip_engine/local_candidate_discovery.py`, `clip_engine/transcript_candidate_scanner.py`
- `clip_engine/clip_discovery.py` — local fallback candidates
- `clip_engine/clip_diversity.py`, `clip_engine/clip_split.py`, `clip_engine/clip_split_parts.py`
- `clip_engine/clip_boundaries.py`, `clip_engine/clip_scoring.py`, `clip_engine/clip_finalizer.py`
- `clip_engine/clip_duration_governor.py`, `clip_engine/clip_quality_gate.py`
- `clip_engine/clip_expand.py`, `clip_engine/clip_metadata.py`
- `clip_engine/openai_resilience.py` — retries, chunking, JSON repair
- `clip_engine/token_tracking.py`, `clip_engine/effective_config.py`, `clip_engine/ai_profiles.py`

### 4.3 Media / GPU

- `clip_engine/transcription.py`, `clip_engine/transcription_utils.py`
- `clip_engine/audio_extract.py`, `clip_engine/media_probe.py`
- `clip_engine/ffmpeg_resolve.py`, `clip_engine/ffmpeg_gpu.py`
- `clip_engine/export_vertical.py`, `clip_engine/captions.py`, `clip_engine/smart_crop.py`
- `clip_engine/cuda_diagnostics.py`, `clip_engine/semantic_ranking.py`
- `clip_engine/speaker_analysis.py`, `clip_engine/speaker_signals.py`

### 4.4 Storage / upload

- `clip_engine/upload_manifest.py` — browser upload dedupe
- `clip_engine/analysis_cache.py` — JSON analysis cache
- `clip_engine/telemetry.py` — session metrics + rotating logs
- `clip_engine/dependency_status.py` — optional package probes

### 4.5 Resolve

- `resolve_bridge.py`, `resolve_client.py`, `marker_writer.py`
- `clip_engine/resolve_export.py` — EDL generation

---

## 5. FFmpeg processes

| Call site | Purpose | Timeout | Parallelism |
|-----------|---------|---------|-------------|
| `audio_extract.extract_audio_wav` | Video → 16 kHz mono WAV | None (`check=True`) | One at a time (UI blocking) |
| `export_vertical._run_with_fallback` | Clip encode NVENC/x264 | **300 s** | Serialized via `_EXPORT_LOCK` |
| `ffmpeg_gpu._run` / probe | NVENC capability test | 25 s | On ffmpeg resolve |
| `ffmpeg_gpu.run_ffmpeg_checked` | Compress proxy (scripts) | **7200 s** | Scripts only |
| `media_probe.get_media_duration_seconds` | ffprobe duration | 120 s | Per probe |
| `smart_crop` | ffprobe resolution | subprocess | Per export validation |
| `scripts/compress_encode.py` | Batch compress | varies | Manual |

**Process API:** `subprocess.run` only — **no `Popen` registry** in Clip Studio code.

---

## 6. Whisper / faster-whisper

| Path | Model load | Device |
|------|------------|--------|
| `transcription._faster_whisper_run` | **New `WhisperModel` each transcribe** | cuda → cpu fallback |
| `transcription.transcribe_video` | Above or OpenAI `whisper-1` | API reads full WAV bytes |
| `speaker_analysis.diarize_audio_file` | **New `WhisperModel("base")` each run** | cuda/cpu |

**Dependencies:** `faster-whisper` → `ctranslate2`; optional CUDA 12 DLLs (`cublas64_12.dll`). Launcher prepends CUDA 12.9 `bin` to PATH.

---

## 7. OpenAI integrations

| Feature | Module | API surface |
|---------|--------|-------------|
| Clip scoring | `clip_analysis.py` | `chat.completions.create` via `openai_resilience` |
| Metadata / grounding | `clip_metadata.py` | Chat completions |
| Cloud Whisper | `transcription.py` | `audio.transcriptions.create` |
| Clip split (optional) | `clip_split.py` | Chat completions |
| Companion app | `ai_companion.py`, `openai_marker_engine.py` | `responses.create` / chat |

**Rate limiting:** Exponential backoff on 429 (`call_openai_with_backoff`), call delays, token budget checks.

---

## 8. DaVinci Resolve integrations

| Mechanism | File | Notes |
|-----------|------|-------|
| Timeline + markers | `resolve_bridge.py` | JSON stdin, 30 s subprocess timeout |
| EDL export | `clip_engine/resolve_export.py` | In-thread; 5 s executor timeout in UI |
| In-process API | `resolve_client.py` | Used by `app.py` companion |
| Safety | Marker-only in companion; Clip Studio sends clips as markers/timeline |

**No persistent Resolve daemon** started by RT365 — connects to running Resolve instance.

---

## 9. Background workers, queues, schedulers

| Mechanism | Present? | Details |
|-----------|----------|---------|
| `multiprocessing` | **No** | Not used in Clip Studio pipeline |
| `celery` / `rq` | **No** | |
| `asyncio` event loops | **No** (Clip Studio) | |
| `threading` | **Yes** | `export_vertical._EXPORT_LOCK` only |
| `ThreadPoolExecutor` | **Yes** | `ui/resolve_panel.py` EDL (max_workers=1) |
| Streamlit rerun queue | **Yes** | Implicit — user-driven |
| Scheduled jobs | **No** | |
| File watchers | **No** | |

---

## 10. Temporary file systems

| Path | Contents | Cleanup |
|------|----------|---------|
| `outputs/_work/_whisper_input.wav` | Extracted audio | **Not auto-deleted** |
| `outputs/clips/session_*/` | Exported MP4, ASS/SRT sidecars | Persistent |
| `outputs/cache/analysis/<hash>/` | Cached analysis JSON | Manual / cache clear API |
| `outputs/previews/` | Per-clip preview MP4s | **Not auto-deleted** |
| `uploads/` | Browser-saved videos | User-managed |
| `uploads/_duplicates/` | Deduped copies | User button |
| `._tmp.ass` next to export | Burn-in subtitles | **Deleted in `finally`** |
| `*_diarization.json` beside WAV | Speaker cache | Persistent |
| `logs/*.log` | Rotating logs (10 MB × 5) | Rotating handler |

---

## 11. Logging systems

Configured in `clip_engine/telemetry.configure_rotating_logs`:

| File | Logger namespaces |
|------|-------------------|
| `logs/app.log` | clip_studio, clip_engine, telemetry |
| `logs/openai.log` | openai_resilience, telemetry.openai |
| `logs/gpu.log` | gpu_pipeline, export (GPU lines) |
| `logs/exports.log` | export_vertical, telemetry.exports |

Also: `crash_log.txt` (launcher capture), Streamlit console, optional `run_reports/`.

---

## 12. Cache systems

| Cache | Location | Key |
|-------|----------|-----|
| Analysis results | `outputs/cache/analysis/` | Transcript hash + settings digest |
| Analysis progress | Same tree | Resume after rate limit |
| Upload manifest | `uploads/upload_manifest.json` | File fingerprint |
| NVENC probe | In-memory module globals | `ffmpeg_gpu._nvenc_runtime_cached` |
| CUDA probe | `cuda_diagnostics._RUNTIME_PROBE_CACHE` | |
| Embeddings model | `semantic_ranking._MODEL` | Global singleton |
| FFmpeg path | `ffmpeg_resolve._CACHED_EXE` | |

---

## 13. Database usage

**None** for Clip Studio. `.gitignore` mentions `*.sqlite3` but no ORM/SQLite code in pipeline. All persistence is **JSON files + filesystem paths**.

---

## 14. GPU / CUDA / Torch dependency map

```
launch_ai_clip_studio.ps1
  CUDA_VISIBLE_DEVICES=0, CUDA 12.9 bin on PATH
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Streamlit (single Python process)                        │
├──────────────────────────────────────────────────────────┤
│ faster-whisper / CTranslate2  → transcription, diarize   │
│ sentence-transformers + torch → gpu_pipeline embeddings  │
│ ultralytics YOLO + OpenCV     → smart_crop (optional)    │
│ torch.cuda.empty_cache()      → after Analyze only       │
├──────────────────────────────────────────────────────────┤
│ FFmpeg h264_nvenc             → export (separate process) │
│ nvidia-smi                    → VRAM headroom check      │
└──────────────────────────────────────────────────────────┘
```

**Optional packages** (`requirements-ai-upgrades.txt`): ultralytics, opencv, pysubs2, librosa, sentence-transformers, transformers. **librosa is listed but not imported in clip_engine pipeline code** (dependency_status only).

---

## 15. End-to-end pipeline (plain English)

```
[Lauch] PS1/BAT → venv → streamlit run clip_studio_app.py:8501
    │
[1 Upload] Local path (reference only) OR browser → uploads/ (full buffer fingerprint)
    │
[2 Transcribe] FFmpeg extract WAV → faster-whisper GPU/CPU OR OpenAI whisper-1
    │            → cs_segments + cs_formatted in session RAM
    │
[2b Optional] Diarization → second WhisperModel pass on WAV
    │
[3 Analyze] run_full_clip_pipeline:
    │   ├─ Load/skip analysis cache (disk JSON)
    │   ├─ GPU prefilter: local windows + optional embeddings (CUDA)
    │   ├─ OpenAI multipass scoring (network, sequential)
    │   ├─ Diversity, split, boundaries, finalizer, quality gate
    │   └─ torch.cuda.empty_cache() if torch installed
    │
[4 UI] Clip cards — edit hooks/times; optional per-clip preview export
    │
[5 Export] For each selected clip (sequential):
    │   ├─ ASS temp file
    │   ├─ Optional YOLO/OpenCV trajectory (smart_crop)
    │   └─ FFmpeg NVENC/x264 → outputs/clips/session_*/
    │
[6 Resolve] Optional subprocess → resolve_bridge.py (markers/timeline)
```

---

## 16. Related repos / deploy artifacts

| Artifact | Relation |
|----------|----------|
| `rt365-landing` | Static marketing site (Vercel) |
| `rt365-webhook` | Stripe delivery email (Railway) |
| `vercel.json` + `dist/` | Claims app deployment |

---

*This document is inventory only. See RESOURCE_AUDIT.md, CRASH_RISK_REPORT.md, STABILITY_REPORT.md, FIX_PLAN.md, WINDOWS_DIAGNOSTICS.md for risk and remediation.*
