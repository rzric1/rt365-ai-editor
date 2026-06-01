# RT365 AI Clip Studio — Fix Plan (Analysis Only)

**Status:** No code changes applied in this forensic pass. Approve phases before implementation.

---

## CRITICAL

| ID | Fix | Files | Impact |
|----|-----|-------|--------|
| C1 | Stream upload fingerprint — hash 8 MB chunks from disk stream, never `getbuffer()` entire file | `upload_manifest.py` | Prevents multi-GB RAM spike |
| C2 | Singleton `WhisperModel` per session (model size key); explicit unload API | `transcription.py`, `speaker_analysis.py`, `ui/session_state.py` | Stops VRAM re-load churn |
| C3 | Cloud Whisper: refuse or stream-upload if WAV > threshold (e.g. 25 MB); clear error | `transcription.py` | Prevents RAM + API failure |
| C4 | FFmpeg subprocess registry + `atexit` / signal handler to kill children | New `clip_engine/subprocess_guard.py`, all ffmpeg callers | Stops orphan ffmpeg |

---

## HIGH

| ID | Fix | Files | Impact |
|----|-----|-------|--------|
| H1 | **Cancel** button — `threading.Event` checked in export loop and pipeline status callbacks | `export_panel.py`, `clip_cards.py`, `clip_pipeline.py` | User can stop runaway jobs |
| H2 | `audio_extract` timeout (e.g. 3600 s) + kill | `audio_extract.py` | No infinite hang |
| H3 | YOLO model — `@st.cache_resource` or module singleton | `smart_crop.py` | Cuts VRAM spikes on batch export |
| H4 | Phase mutex — block Export while Transcribe/Analyze running (session flags) | `ui/clip_cards.py`, `export_panel.py` | Prevents GPU overlap |
| H5 | Startup + shutdown cleanup — delete stale `_whisper_input.wav`, old previews | `check_environment.py` or `clip_studio_app.py` | Disk + confusion |
| H6 | After transcribe: `del model`, `gc.collect()`, `torch.cuda.empty_cache()` | `transcription.py`, UI hook | VRAM release |

---

## MEDIUM

| ID | Fix | Files | Impact |
|----|-----|-------|--------|
| M1 | Export batch confirmation if N > 10 clips; show disk estimate | `export_panel.py` | UX + disk safety |
| M2 | Analysis cache TTL / max size eviction | `analysis_cache.py` | Disk growth |
| M3 | Cap `cs_formatted` storage — rebuild from segments when needed | `clip_cards.py`, `effective_config.py` | RAM on long podcasts |
| M4 | Make `_GPU_MEMORY_HEADROOM_GB` configurable (default 8, 4090 → 18) | `export_vertical.py`, sidebar | Fewer false waits |
| M5 | Unify launchers — document `.venv311` as canonical; fix `run_app.bat` mismatch | `run_app.bat`, README | Wrong env / missing CUDA |
| M6 | Crash handler — log `nvidia-smi` + traceback to `logs/crash_session.log` | `clip_studio_app.py` | Forensics |
| M7 | `Popen` + communicate for long ffmpeg with progress parse | `export_vertical.py` | Better cancel support |

---

## LOW

| ID | Fix | Files | Impact |
|----|-----|-------|--------|
| L1 | Disable preview renders during batch export | `clip_cards.py` | Less GPU load |
| L2 | `psutil` RAM warning in sidebar before transcribe | `ui/sidebar.py` | User guidance |
| L3 | Document safe workflow in README_QUICKSTART | README | Prevention |
| L4 | Optional `FORCE_CPU_EMBEDDINGS` prominent in UI | `sidebar.py` | Stability on conflicted CUDA stacks |
| L5 | Remove duplicate NVENC probe logging on Streamlit rerun | `ffmpeg_resolve.py` | Log noise |

---

## Suggested implementation order

1. **Week 1:** C1, C4, H2, H5 (memory + process safety, low regression risk)
2. **Week 2:** C2, H3, H6 (GPU lifecycle)
3. **Week 3:** H1, H4, M1 (UX + concurrency guards)
4. **Backlog:** C3, M2–M7, LOW items

---

## Testing checklist (post-fix)

- [ ] Upload 3 GB file — RAM stays < 1 GB spike during fingerprint
- [ ] Transcribe 1 h video twice — VRAM returns to baseline between runs
- [ ] Export 5 clips smart_crop — single YOLO load
- [ ] Kill Streamlit mid-export — no ffmpeg.exe after 10 s
- [ ] Cancel export at clip 3/10 — stops before clip 4
- [ ] Analyze + export overlap blocked with UI message

---

*End of fix plan — awaiting approval before code changes.*
