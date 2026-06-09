# RT365 AI Clip Studio — Stability Audit

**Scope:** `rt365-ai-editor` Clip Studio only (not landing, Stripe, Railway, webhooks).  
**Hardware context:** RTX 4090 / i9-class CPU / high RAM — risks are software-side (orphan processes, VRAM retention, overlapping jobs, RAM spikes).

## Current pipeline

| Stage | Entry | Engine | Subprocess / GPU |
|-------|--------|--------|------------------|
| Launch | `clip_studio_app.py` → Streamlit :8501 | — | Startup hooks, diagnostics, temp cleanup |
| Upload | `ui/clip_cards.py` → `upload_manifest.save_upload_once` | SHA fingerprint + disk write | None |
| Transcribe | `transcription.transcribe_video` | FFmpeg WAV + faster-whisper / OpenAI | `audio_extract` → `run_subprocess`; GPU via `whisper_runtime` |
| Diarize (opt.) | `speaker_analysis.diarize_audio_file` | Cached faster-whisper `base` | Shared `whisper_runtime` |
| Analyze | `clip_pipeline.run_full_clip_pipeline` | OpenAI + local embeddings | Rate-limit backoff (interruptible) |
| Preview / Export | `export_vertical` | FFmpeg NVENC/x264, ASS burn-in, smart crop | `run_subprocess` + export lock |
| Resolve (opt.) | `ui/resolve_panel.py` | DaVinci XML / paths | File I/O only |

**Process model:** Single Python process (Streamlit). No `multiprocessing`. Background work is synchronous in the request handler with `subprocess` children for FFmpeg.

## Risky files / functions (pre-fix baseline)

| Risk | Location | Impact |
|------|----------|--------|
| Orphan `ffmpeg.exe` | All prior `subprocess.run` without registry | Zombie encodes after crash/cancel |
| Overlapping jobs | UI could start transcribe + export + analyze | CPU/GPU/RAM spikes, driver resets |
| Full upload in RAM | `upload_manifest.compute_upload_fingerprint` (`getbuffer()`) | Multi-GB RAM spike on browser upload |
| Cloud Whisper RAM | `transcription` read entire WAV | OOM on long podcasts |
| Whisper reload | New `WhisperModel` per transcribe/diarize | VRAM churn on RTX 4090 |
| YOLO reload | `smart_crop._yolo_detect_trajectory` per clip | VRAM + latency |
| No global cancel | — | User closes tab; FFmpeg continues |
| `audio_extract` no timeout | `audio_extract.py` | Hung extract blocks UI |
| GPU not released | Only ad-hoc `empty_cache` after analyze | CTranslate2 + Torch VRAM held |

## Crash risks

1. **Unhandled exception mid-FFmpeg** — child keeps running; parent restarts Streamlit → duplicate encodes.
2. **Double-click Transcribe + Export** — two NVENC sessions + Whisper → VRAM pressure / TDR.
3. **Large browser upload** — fingerprint/load entire file into RAM.
4. **OpenAI backoff sleep** — long blocking sleep without cancel check.
5. **Streamlit rerun** — partial state + active subprocess.

## Resource risks

- **VRAM:** faster-whisper, YOLO smart crop, NVENC (driver stack).
- **RAM:** uploads, cloud Whisper WAV, transcript strings in session.
- **Disk:** `outputs/`, `uploads/`, temp ASS/preview files.
- **CPU:** FFmpeg filter graphs, OpenCV smart crop sampling.

## Protections implemented (Phase 2)

| Control | Module |
|---------|--------|
| One job at a time | `clip_engine/job_control.py` + `ui/job_helpers.studio_job` |
| Cancel + kill children | `job_control.request_cancel` + `subprocess_guard.terminate_all_tracked` + sidebar button |
| FFmpeg timeout / tracking | `subprocess_guard.run_subprocess` (used by `audio_extract`, `export_vertical`) |
| Whisper model cache | `whisper_runtime.py` |
| GPU release | `stability.release_gpu_memory()` after transcribe/diarize/analyze |
| Streaming upload | `upload_manifest` fingerprint + write |
| Cloud Whisper size cap | `stability.MAX_CLOUD_WHISPER_WAV_BYTES` |
| YOLO session cache | `smart_crop._get_yolo_model()` |
| Crash log | `logs/crash_report.txt` via `stability.write_crash_report` |
| Startup diagnostics | `logs/startup_diagnostics.txt` (Python, FFmpeg, NVIDIA, CUDA, VRAM, disk, RAM) |
| Temp cleanup (safe) | `stability.cleanup_temp_artifacts` — never deletes `uploads/` source videos |
| Interruptible OpenAI backoff | `openai_resilience._interruptible_sleep` |

## Remaining risks

- Streamlit still runs one thread; a hung **non-tracked** subprocess (third-party) would not be killed.
- Smart crop OpenCV loop is CPU-heavy on very long clips (cancel-aware but not chunked to disk).
- Multiple Streamlit **browser tabs** = multiple Python processes (user workflow).
- CTranslate2 model unload is best-effort; driver may retain some VRAM until process exit.

## Recommended follow-ups (non-blocking)

- Optional `psutil` in requirements for richer diagnostics.
- Post-export `release_gpu_memory()` if NVENC + YOLO used in same session.
- Windows Task Scheduler script to kill orphan `ffmpeg.exe` under `outputs/` path (external safety net).
