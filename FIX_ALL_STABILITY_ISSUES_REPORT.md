# Fix All Stability Issues — Final Report

**Date:** 2026-06-01  
**Scope:** `C:\dev\rt365-ai-editor` (RT365 AI Clip Studio only)  
**Hardware target:** RTX 4090, Windows 11, 64 GB RAM, NVMe, CUDA/NVENC

---

## 1. What Reliability Monitor showed

Typical pattern on affected workstations:

- **Faulting application:** `python.exe`
- **Timing:** After installing **Python 3.11.9** alongside other Python versions (including **3.14**)
- **Symptom:** “Stopped working” / silent exit without Python traceback
- **Correlated workloads:** Transcribe (Whisper/CUDA), Analyze (OpenAI + embeddings), Export (FFmpeg NVENC + smart crop)

This pattern matches **wrong interpreter** (system 3.14 vs project 3.11 venv), **missing CUDA DLLs** for CTranslate2, **duplicate Streamlit instances**, and **GPU/RAM not released** between phases—not insufficient RTX 4090 hardware.

---

## 2. Root cause assessment

| Priority | Cause | Mechanism |
|----------|-------|-----------|
| 1 | **Python environment conflict** | Streamlit launched with Python 3.14 or global Python without `faster-whisper` / `ctranslate2` / matching `torch` → native DLL crashes or import failures |
| 2 | **CUDA / Whisper stack mismatch** | `cublas64_12.dll` or wrong `ctranslate2` wheel → GPU probe fails or hard crash in CTranslate2 |
| 3 | **Duplicate app instances** | Second `streamlit` on port 8501 → port/state conflicts, orphan children |
| 4 | **Untracked FFmpeg / ffprobe** | Orphan `ffmpeg.exe` after cancel/crash |
| 5 | **VRAM/RAM retention** | Cached Whisper, YOLO, embeddings + large `st.session_state` transcript |

---

## 3. Files changed

| Area | Files |
|------|-------|
| Environment | `clip_engine/environment_check.py`, `check_environment.py`, `requirements.txt`, `setup_windows.bat` |
| Launch | `launch_ai_clip_studio.ps1`, `launch_ai_clip_studio.bat` |
| Single instance | `clip_engine/app_lock.py`, `clip_studio_app.py` |
| GPU cleanup | `clip_engine/gpu_cleanup.py`, `clip_engine/smart_crop.py`, `clip_engine/semantic_ranking.py`, `ui/clip_cards.py`, `ui/export_panel.py` |
| Subprocess | `clip_engine/subprocess_guard.py`, `clip_engine/media_probe.py`, `clip_engine/smart_crop.py`, `ui/resolve_panel.py` |
| Crash / RAM | `clip_engine/stability.py`, `ui/stability_ui.py`, `ui/session_memory.py` |
| Windows diagnostics | `scripts/collect_windows_diagnostics.ps1` |
| Tests | `tests/test_environment_check.py`, `tests/test_single_instance_lock.py`, `tests/test_subprocess_guard.py`, `tests/test_gpu_cleanup.py` |

---

## 4. Python environment fixed

- **Blocked:** Python **3.14** for Clip Studio (launcher warning + `environment_check` hard fail).
- **Required:** Python **3.11.x** in **`.venv311` only**.
- **Launcher** creates `.venv311` if missing, installs `requirements.txt`, AI upgrades, and CUDA PyTorch cu121.
- **Log:** `logs/environment_check.txt` on every setup/launch.
- **Streamlit gate:** App stops with clear error if venv/deps invalid (`clip_studio_app._ensure_environment_gate`).

---

## 5. CUDA / Whisper fixed

- Dependencies pinned: `faster-whisper`, `ctranslate2`, `numpy`, `opencv-python-headless`.
- **Whisper unload** after transcribe via `cleanup_gpu_after_phase(..., whisper=True)`.
- **CUDA PATH** prepended in launcher (CUDA 12.9 `bin` when present).
- Startup validation reports torch CUDA device and CTranslate2 device count.

---

## 6. FFmpeg cleanup fixed

- `audio_extract`, `export_vertical`, `media_probe`, `smart_crop` ffprobe → **`subprocess_guard.run_subprocess`**
- Resolve bridge → **`run_subprocess_with_input`** (tracked, 30 s timeout)
- Orphan `ffmpeg.exe` termination at startup/cancel (existing + retained)
- All tracked children terminated on `atexit`

---

## 7. Duplicate launch prevention

- **`logs/rt365_app.lock`** with PID + python path
- Launcher **`preflight_single_instance`** before Streamlit
- Streamlit **`acquire_app_lock`** on first startup
- Message: **“RT365 AI Clip Studio is already running.”**
- Stale lock removed when PID dead

---

## 8. Crash logging added / enhanced

- `logs/crash_report.txt` includes: timestamp, traceback, **python executable**, version, **venv prefix**, active job, pipeline step, RAM, GPU snapshot
- `sys.excepthook` + per-job `write_crash_report`
- `logs/gpu_cleanup.log` — VRAM before/after each cleanup phase
- `logs/resource_monitor.log` — session snapshots

---

## 9. Remaining risks

| Risk | Mitigation |
|------|------------|
| User runs `streamlit` from wrong Python | Only use `launch_ai_clip_studio.ps1` |
| Driver TDR (`nvlddmkm`) under heavy GPU | Update Studio driver; avoid parallel Resolve + export |
| Very long podcasts in session RAM | Use **Clear session memory** in sidebar |
| sentence-transformers import failures | `requirements-ai-upgrades.txt` + repair torch stack |
| Reliability Monitor outside repo | Run `scripts\collect_windows_diagnostics.ps1` after incidents |

---

## 10. Exact launch command

```powershell
C:\dev\rt365-ai-editor\launch_ai_clip_studio.ps1
```

Or double-click `launch_ai_clip_studio.bat` (delegates to the script above).

**First-time setup:**

```bat
C:\dev\rt365-ai-editor\setup_windows.bat
```

---

## 11. Stress test steps

1. Run `setup_windows.bat` → confirm `logs/environment_check.txt` ends with PASS.
2. Launch via `launch_ai_clip_studio.ps1` → browser opens :8501, sidebar shows GPU diagnostics OK.
3. Try launching a **second** instance → must show “already running” and exit.
4. Upload 30–60 min video → transcribe (GPU on) → verify `logs/gpu_cleanup.log` has `transcribe` line with VRAM drop.
5. Analyze 20 clips → cancel mid-way → Confirm no orphan ffmpeg in sidebar.
6. Export 5 clips smart_crop + NVENC → check `logs/exports.log`.
7. **Clear session memory** → transcript cleared, video path kept.
8. Run `scripts\collect_windows_diagnostics.ps1` → review `logs/windows_diagnostics.txt`.
9. If crash: correlate `logs/crash_report.txt` timestamp with Event Viewer / Reliability Monitor per `WINDOWS_DIAGNOSTICS.md`.

---

**Verdict:** Workstation hardware is adequate. Stability now depends on **always using the 3.11 venv launcher** and the new gates/cleanup paths above.
