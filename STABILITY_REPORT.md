# RT365 AI Clip Studio — Stability Report

Verification of lifecycle cleanup and shutdown behavior (static analysis + architectural review).

---

## 1. FFmpeg process termination

| Check | Result | Evidence |
|-------|--------|----------|
| Uses `subprocess.run` with timeout (export) | **Pass** | `export_vertical._FFMPEG_TIMEOUT_SECONDS = 300` |
| Uses timeout (probe) | **Pass** | `media_probe` 120 s |
| Uses timeout (compress utility) | **Pass** | `ffmpeg_gpu.run_ffmpeg_checked` 7200 s |
| Audio extract timeout | **FAIL** | `audio_extract.extract_audio_wav` — no timeout |
| Kill on Streamlit stop | **FAIL** | No signal handlers or `atexit` |
| Kill on user Cancel | **FAIL** | No cancel API |
| Zombie risk after crash | **FAIL** | Windows orphan `ffmpeg.exe` possible |

**Verdict:** FFmpeg **usually** terminates when commands complete or hit timeout; **not guaranteed** on abnormal parent exit.

---

## 2. Temporary file deletion

| Artifact | Deleted? | Mechanism |
|----------|----------|-----------|
| `._tmp.ass` (captions) | **Yes** | `export_vertical` `finally: ass_file.unlink` |
| `_whisper_input.wav` | **No** | Overwritten on next transcribe only |
| Preview MP4s | **No** | `outputs/previews/` |
| Session export dirs | **No** | Intentional user output |
| Analysis cache | **Partial** | `analysis_cache` can clear by key; no auto TTL |
| Diarization JSON cache | **No** | Beside WAV path |
| Streamlit temp uploads | **Streamlit-managed** | OS temp |

**Verdict:** Export subtitle temps are clean; **work directory accumulates** unless user deletes.

---

## 3. Worker shutdown

| Worker type | Present | Clean shutdown |
|-------------|---------|----------------|
| Background threads | Export lock holder only | Dies with process |
| ThreadPoolExecutor (EDL) | Short-lived `with` block | **Pass** |
| Multiprocessing workers | None | N/A |
| Scheduled tasks | None | N/A |

**Verdict:** No worker pool to drain; **Streamlit Ctrl+C** stops Python; children may survive.

---

## 4. GPU memory release

| Action | Released? |
|--------|-----------|
| After Analyze | `torch.cuda.empty_cache()` if torch importable — **Whisper not explicitly freed** |
| After Transcribe | **No** automatic release |
| After Export (NVENC) | FFmpeg process exit releases NVENC context — **Pass** |
| SentenceTransformer | Stays in `_MODEL` until process exit |
| YOLO | Local variable; GC-dependent — **unreliable for CUDA** |

**Verdict:** **Partial** — one explicit cache clear after analysis; Whisper/YOLO VRAM may linger until process end.

---

## 5. Model unload

| Model | Unload call | Verdict |
|-------|-------------|---------|
| faster-whisper | None | **Fail** |
| sentence-transformers | None (singleton) | By design |
| YOLO | None per call | **Fail** |
| OpenAI client | Stateless HTTP | N/A |

---

## 6. Browser / Streamlit session

| Check | Result |
|-------|--------|
| Closing browser tab | Streamlit server **keeps running** |
| Closing terminal / Ctrl+C | Server stops; ffmpeg may orphan |
| Session state persistence | In-memory only — lost on restart |
| `maxUploadSize` 4096 MB | Server accepts large POST |

**Verdict:** User must **stop launcher window** to end server; browser close is insufficient.

---

## 7. DaVinci Resolve connections

| Path | Termination |
|------|-------------|
| `resolve_bridge.py` subprocess | Exits after JSON response or 30 s timeout |
| In-process `resolve_client` (companion) | Connection lifetime = script lifetime |
| Timeline/markers API | No explicit disconnect in bridge |

**Verdict:** **Pass** for subprocess bridge; Resolve app itself stays open (expected).

---

## 8. Logging under failure

| Feature | Status |
|---------|--------|
| Rotating logs | **Pass** — 10 MB × 5 files |
| Exception in UI | Logged via `logger.exception` |
| GPU snapshots | `log_gpu_memory` on export path |
| Crash dump auto | **Not implemented** |
| Windows Event Log integration | **None** |

---

## 9. Stability scorecard (component)

| Component | Score /10 | Notes |
|-----------|-----------|-------|
| Export pipeline | 7 | Lock + timeout + ass cleanup |
| Transcription | 4 | No model reuse, no WAV cleanup |
| Upload path | 3 | RAM spike |
| Analyze pipeline | 6 | Cache + backoff; blocks UI |
| Resolve integration | 8 | Short subprocess |
| Overall process hygiene | 5 | Orphan ffmpeg risk |

---

## 10. Operational stability recommendations (no code)

1. After any crash, run `taskkill /IM ffmpeg.exe /F` and restart Streamlit.
2. Prefer **Local file path** over browser upload for files > 2 GB.
3. Restart Clip Studio between **Transcribe → Analyze → Export** on machines with ≤16 GB RAM.
4. Monitor `logs/gpu.log` and `logs/exports.log` after incidents.
5. Keep one Streamlit instance (check port 8501).

---

*See FIX_PLAN.md for prioritized engineering fixes.*
