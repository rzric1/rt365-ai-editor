# RT365 AI Clip Studio — Stability Fix Report

## What changed

Stability pass focused on **one job at a time**, **tracked FFmpeg with timeout and kill-on-cancel/exit**, **GPU memory release**, **streaming uploads**, **crash logging**, **startup diagnostics (RTX 4090 / CUDA / VRAM)**, and **sidebar cancel**.

## Files modified / added

| Path | Change |
|------|--------|
| `clip_engine/job_control.py` | **New** — job lock, cancel flag, pipeline step |
| `clip_engine/subprocess_guard.py` | **New** — tracked `Popen`, `run_subprocess`, `atexit` cleanup |
| `clip_engine/whisper_runtime.py` | **New** — cached WhisperModel, transcribe helper |
| `clip_engine/stability.py` | **New** — crash log, startup diag, temp cleanup, `release_gpu_memory` |
| `clip_engine/audio_extract.py` | `run_subprocess`, 2h timeout |
| `clip_engine/export_vertical.py` | `run_subprocess`, NVENC headroom 18 GB, ffmpeg path resolve |
| `clip_engine/transcription.py` | whisper runtime, cloud stream, size cap, GPU release |
| `clip_engine/upload_manifest.py` | streaming fingerprint + write, cancel checks |
| `clip_engine/speaker_analysis.py` | shared whisper, cancel in loop, GPU release |
| `clip_engine/smart_crop.py` | cached YOLO, cancel in loop |
| `clip_engine/openai_resilience.py` | interruptible backoff sleep |
| `clip_studio_app.py` | startup hooks + crash report on main failure |
| `ui/job_helpers.py` | **New** — `studio_job` context manager |
| `ui/stability_ui.py` | **New** — cancel button + active job display |
| `ui/sidebar.py` | stability controls block |
| `ui/clip_cards.py` | job guards on upload/transcribe/diarize/analyze/preview/rescore |
| `ui/export_panel.py` | export job lock, cancel between clips |
| `tests/test_stability_controls.py` | **New** — unit tests |
| `STABILITY_AUDIT.md` | **New** — audit document |
| `STABILITY_FIX_REPORT.md` | **New** — this file |

## Risks fixed

- Overlapping transcribe + export + analyze in one session.
- Orphan FFmpeg after cancel/exception (tracked processes + `atexit`).
- FFmpeg extract without timeout.
- Multi-GB RAM spike on browser upload fingerprint.
- Reloading Whisper/YOLO every operation.
- No crash artifact for support/debugging.
- No structured startup check for RTX/CUDA/FFmpeg/disk.

## Remaining risks

See **Remaining risks** in `STABILITY_AUDIT.md`.

## How to test on Windows

1. **Environment**

```powershell
cd C:\dev\rt365-ai-editor
.\.venv311\Scripts\Activate.ps1
python check_environment.py
```

2. **Unit tests**

```powershell
.\.venv311\Scripts\python.exe -m pytest tests\test_stability_controls.py -q
```

3. **Launch app**

```powershell
.\launch_ai_clip_studio.ps1
```

4. **Manual checklist**

| Check | Expected |
|-------|----------|
| Startup | `logs\startup_diagnostics.txt` created/updated with Python, FFmpeg, NVIDIA, CUDA, VRAM lines |
| Import video | Save upload completes; RAM stable in Task Manager |
| Transcribe once | Second transcribe while first runs → busy error OR wait |
| Analyze once | Same job lock behavior |
| Export once | Same; NVENC in Task Manager only during export |
| Cancel | Sidebar **Cancel current job** → FFmpeg tasks end in Task Manager |
| Crash log | Force an error (e.g. bad path) → `logs\crash_report.txt` appends entry |
| Orphans | After cancel/exit, `Get-Process ffmpeg -ErrorAction SilentlyContinue` empty |

5. **Orphan FFmpeg check**

```powershell
Get-Process ffmpeg -ErrorAction SilentlyContinue | Format-Table Id, CPU, StartTime
```

6. **Diagnostics tail**

```powershell
Get-Content .\logs\startup_diagnostics.txt -Tail 40
Get-Content .\logs\crash_report.txt -Tail 40
```

## Exact commands (copy-paste)

```powershell
cd C:\dev\rt365-ai-editor
.\.venv311\Scripts\Activate.ps1
python -m pytest tests\test_stability_controls.py -q
python -c "from clip_engine.stability import run_startup_diagnostics; run_startup_diagnostics(); print('ok')"
.\launch_ai_clip_studio.ps1
```
