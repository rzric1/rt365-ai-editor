# RT365 AI Clip Studio — Crash Risk Report

Forensic code search results and risk classification.

---

## 1. Concurrency primitives found

| Primitive | Count (approx) | Files |
|-----------|------------------|-------|
| `multiprocessing` | **0** | — |
| `subprocess.run` | **20+** | ffmpeg, probe, resolve, tests |
| `subprocess.Popen` | **0** | — |
| `threading.Lock` | **1** | `export_vertical._EXPORT_LOCK` |
| `ThreadPoolExecutor` | **1** | `ui/resolve_panel.py` (EDL, max_workers=1) |
| `asyncio` | **0** in clip_engine/ui | — |
| `watchdog` | **0** | — |

---

## 2. Loop inventory

### 2.1 Bounded loops (safe)

| Location | Loop | Bound |
|----------|------|-------|
| `clip_pipeline._run_expansion_passes` | `while len(selected) < target` | `max_rounds` ≤ 3 |
| `gpu_pipeline` | `while len(local) > shortlist_min` | Token budget shrink |
| `openai_resilience` | `while compat_attempts` | `max_compat_attempts` |
| `openai_resilience.call_openai_with_backoff` | `for attempt in range(max_attempts)` | Config-limited |
| `clip_discovery` | `while t < media_duration` | Window scan steps |
| `transcript_candidate_scanner` | `while t < media_duration` | Duration list |
| `upload_manifest.write_upload_to_path` | `while offset < total` | File size |
| `transcript_loader` | `while i < len(lines)` | Line count |
| `clip_finalizer` | `while first_i > 0` | Span indices |

### 2.2 Interactive infinite loop (not Clip Studio UI)

| Location | Loop | Risk |
|----------|------|------|
| `interactive_menu.py:128` | `while True` | CLI menu only — hang if stdin broken |

### 2.3 Media sampling loops (bounded by time range)

| Location | Loop | Risk |
|----------|------|------|
| `smart_crop._yolo_detect_trajectory` | `while t < end_time` | ~12 iterations per clip; loads YOLO each call |
| `smart_crop._opencv_detect_trajectory` | `while t < end_time` | ~10 iterations |

**No unbounded queues** (`queue.Queue` not used in pipeline).

---

## 3. Subprocess / FFmpeg execution map

| File | Function | Timeout | Kill on cancel |
|------|----------|---------|----------------|
| `audio_extract.py` | `extract_audio_wav` | **None** | No |
| `export_vertical.py` | `_run_with_fallback` | 300 s | No |
| `ffmpeg_gpu.py` | `_run`, `run_ffmpeg_checked` | 25 s / **7200 s** | No |
| `media_probe.py` | `get_media_duration_seconds` | 120 s | No |
| `ffmpeg_resolve.py` | version probe | default | No |
| `cuda_diagnostics.py` | nvidia-smi | 12 s | No |
| `export_vertical.py` | nvidia-smi headroom | 5 s | No |
| `resolve_panel.py` | resolve_bridge | 30 s | No |
| `smart_crop.py` | ffprobe | implicit | No |

**Gap:** No `try/finally` with `process.kill()` — relies on subprocess completing or timing out.

---

## 4. GPU / Torch / Whisper loading map

| Loader | Cached? | File:line region |
|--------|---------|------------------|
| `WhisperModel` (transcribe) | **No** | `transcription._faster_whisper_run` |
| `WhisperModel` (diarize) | **No** | `speaker_analysis.diarize_audio_file` |
| `SentenceTransformer` | **Yes** global | `semantic_ranking._load_model` |
| `YOLO("yolov8n.pt")` | **No** | `smart_crop._yolo_detect_trajectory` |
| `torch.cuda.empty_cache` | After analyze only | `ui/clip_cards.py` ~965 |

**Whisper does not share weights** between transcribe and diarization in same session.

---

## 5. Missing safeguards (code-verified)

| Safeguard | Status |
|-----------|--------|
| One render job at a time | **Implemented** (`_EXPORT_LOCK`) |
| RAM limits on upload | **Missing** |
| GPU memory limit per phase | Partial (nvidia-smi wait if >8 GB used) |
| FFmpeg process registry | **Missing** |
| Cancel button for analyze/export | **Missing** |
| Duplicate worker prevention | N/A (no workers) |
| Whisper model singleton | **Missing** |
| Max transcript size in session | **Missing** (50k preview truncate only) |
| Export concurrent with transcribe guard | **Missing** (user discipline) |
| `atexit` ffmpeg cleanup | **Missing** |

---

## 6. Exception & timeout handling

| Area | Handling quality |
|------|------------------|
| Export | try/except per clip in UI; raises on ffmpeg failure |
| Transcribe | try/except in UI; logs + user message |
| Analyze | OpenAIRateLimitError special case; partial cache resume |
| GPU prefilter | try/except — **must not kill Streamlit** (`gpu_pipeline` comment) |
| audio_extract | `check=True` — **uncaught subprocess failure crashes spinner** |
| OpenAI | Backoff + JSON repair + fallback model |

---

## 7. Risk register (code-linked)

### CRITICAL

1. **Full-file RAM on upload** — `upload_manifest.compute_upload_fingerprint` uses `bytes(upload.getbuffer())`.
2. **Full WAV RAM on cloud Whisper** — `transcription.transcribe_video` → `read_bytes()`.
3. **Whisper model reload** — VRAM/RAM pressure each transcribe + diarization.

### HIGH

4. **Orphan ffmpeg** — all `subprocess.run` without parent-death handling.
5. **YOLO per export** — smart_crop dynamic mode under batch export.
6. **Streamlit session bloat** — full transcript + clips retained across reruns.
7. **audio_extract no timeout** — hung ffmpeg blocks UI indefinitely.
8. **Overlapping GPU workloads** — no mutex between transcribe, analyze embeddings, export NVENC.

### MEDIUM

9. **300 s export timeout** — long complex filtergraph may fail or appear frozen.
10. **Analysis cache disk growth** — no TTL eviction in code.
11. **Double Streamlit launch** — `run_app.bat` + `launch_ai_clip_studio.ps1` different venvs (.venv vs .venv311).
12. **NVENC probe on cold start** — extra ffmpeg GPU kick.

### LOW

13. **interactive_menu while True** — not shipped UI path.
14. **Resolve bridge 30 s** — user-visible error only.
15. **React scheduler package** — npm dependency name only, not a background job.

---

## 8. BSOD / freeze correlation (software-side)

| Symptom | Most likely software contributor |
|---------|----------------------------------|
| System freeze | RAM exhaustion (upload + session + WAV) |
| App hang | ffmpeg no timeout on extract; analyze blocking main thread |
| Black screen / TDR | Long NVENC + CUDA simultaneous |
| CUDA error dialog | CTranslate2 / cuBLAS missing or OOM |
| BSOD MEMORY_MANAGEMENT | Physical RAM + pagefile exceeded |
| BSOD VIDEO_MEMORY | Rare on 24 GB; possible on 8 GB VRAM GPUs |

---

*Remediation priorities: FIX_PLAN.md*
