# RT365 AI Clip Studio — Stability Report

**Audit date:** 2026-06-01  
**Scope:** Job serialization, cancellation, subprocess/GPU/temp cleanup, exception handling

---

## 1. Verification checklist

| Control | Required | Status | Implementation |
|---------|----------|--------|----------------|
| One active transcription | Yes | **PASS** | `studio_job("transcribe")` + `try_acquire_job` |
| One export at a time | Yes | **PASS** | `studio_job("export")` + `_EXPORT_LOCK` |
| One AI scoring pass at a time | Yes | **PASS** | `studio_job("analyze")` — all analyze buttons share lock name |
| Proper cancellation | Yes | **PASS** | `request_cancel()` → `check_cancelled()` + `terminate_all_tracked()` |
| FFmpeg cleanup (tracked) | Yes | **PASS** | `subprocess_guard.run_subprocess` + `atexit` |
| FFmpeg cleanup (orphan) | Yes | **PASS** (enhanced) | `terminate_orphan_ffmpeg()` startup + cancel + sidebar |
| CUDA cleanup | Partial | **PASS with gaps** | `release_gpu_memory`, `unload_whisper` after transcribe; analyze clears torch cache |
| Temp file cleanup | Yes | **PASS** | `cleanup_temp_artifacts()` — `._tmp.ass`, `.partial.mp4`, stale previews, work WAV |
| Exception handling | Yes | **PASS** | `sys.excepthook` → `crash_report.txt`; per-job `write_crash_report` in `studio_job` |

---

## 2. Job control architecture

```
User action (Streamlit button)
    → ui.job_helpers.studio_job("<name>")
        → try_acquire_job(name)  # raises JobBusyError if another name active
        → pipeline work + check_cancelled() at phase boundaries
        → release_job(name) in finally
```

**Cancel path:**

```
Sidebar "Cancel current job"
    → request_cancel()  # sets Event
    → terminate_all_tracked()
    → terminate_orphan_ffmpeg()
    → release_job(active)
    → st.rerun()
```

**Job names in use:** `upload`, `transcribe`, `diarize`, `analyze`, `preview`, `export`.

**Gap:** Resolve send (`resolve_panel.py`) does not use `studio_job` — can run during idle only; low risk (30 s subprocess).

---

## 3. Subprocess & FFmpeg stability

| Path | Tracked | Cancel-aware | Timeout |
|------|---------|--------------|---------|
| `audio_extract` | Yes | Yes | 7200 s |
| `export_vertical` encode | Yes | Yes | 300 s |
| `nvidia-smi` headroom | Yes | Yes | 5 s |
| `media_probe` | No | No | 120 s |
| `smart_crop` ffprobe | No | No | — |
| `ffmpeg_gpu` probe | No | N/A | 25 s |
| `resolve_bridge` | No | No | 30 s |

**Fix applied:** `_check_gpu_headroom` removed invalid `capture_output=` kwarg that silently disabled VRAM checks.

---

## 4. CUDA / model lifecycle

| Asset | Load | Release |
|-------|------|---------|
| faster-whisper | `whisper_runtime.get_whisper_model` | `unload_whisper()` after `transcribe_video` finally |
| torch (optional) | analyze / embeddings | `release_gpu_memory()` |
| NVENC | FFmpeg child | Process exit on encode complete |

**Risk:** Diarization may load whisper words path separately — monitor VRAM if transcribe + diarize back-to-back without restart.

---

## 5. Temp & crash artifacts

| Artifact | Cleanup |
|----------|---------|
| `outputs/_work/_whisper_input.wav` | Startup `cleanup_temp_artifacts` |
| `._tmp.ass` | Export `finally` + startup sweep |
| `*.partial.mp4` | Startup sweep |
| Old previews (>7 days) | Startup sweep |
| `logs/crash_report.txt` | Append-only forensic log |
| `logs/startup_diagnostics.txt` | Overwritten each app start |

---

## 6. Streamlit-specific stability notes

- **Full script rerun** on every widget change — no background worker queue.
- Long operations block the rerun thread — expected.
- **Session RAM** grows with transcript + clips — not a leak in Python GC sense; session lifetime issue.
- Double-clicking actions before rerun completes can queue UX confusion; job lock prevents overlapping *different* job types.

---

## 7. Test coverage

| Test file | Coverage |
|-----------|----------|
| `tests/test_stability_controls.py` | Job lock, cancel, crash report, startup diag, resource snapshot, orphan finder |

Run in AI venv: `python -m pytest tests/test_stability_controls.py`

---

## 8. Stability verdict

**Core controls are in place** for a single-user workstation deployment. Remaining instability is dominated by **GPU driver TDR**, **optional dependency gaps**, and **untracked short subprocesses** — not missing job locks.

**Post-audit enhancements:** orphan ffmpeg termination, whisper unload after transcribe, resource snapshot logging, GPU headroom fix.
