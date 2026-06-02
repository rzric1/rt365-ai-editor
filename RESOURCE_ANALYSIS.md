# RT365 AI Clip Studio — Resource Analysis

**Audit date:** 2026-06-01  
**Reference hardware:** RTX 4090 (24 GB VRAM), 64 GB RAM, NVMe SSD, Windows 11

---

## 1. Per-stage resource profile

| Stage | CPU | RAM (process) | VRAM | Disk I/O | Network |
|-------|-----|---------------|------|----------|---------|
| **Streamlit idle** | Low (1 core burst on rerun) | 200–600 MB baseline | Near 0 | Minimal | None |
| **Upload (browser)** | Medium during hash/read | +file size buffered | 0 | Write `uploads/` | None |
| **FFmpeg audio extract** | 1–4 cores | +50–200 MB parent | 0 | Read source video, write WAV | None |
| **faster-whisper CUDA** | Low (GPU bound) | +500 MB–2 GB Python | **2–8 GB** model dependent | Read WAV | None |
| **OpenAI whisper-1** | Low | **Full WAV in RAM** (≤25 MB enforced) | 0 | Read WAV | Upload ~WAV size |
| **Diarize** | Medium | Second whisper pass if not cached | +VRAM if CUDA | Read WAV | None |
| **GPU prefilter / embeddings** | Medium–high burst | +1–3 GB | **1–4 GB** torch | Cache read/write | None |
| **OpenAI analyze** | Low per call | Transcript + JSON in RAM | 0 | Cache JSON | **High** (many sequential HTTPS) |
| **Smart crop export** | High (OpenCV/YOLO) | +500 MB–1.5 GB | **2–6 GB** if YOLO CUDA | Temp ASS + MP4 write | None |
| **NVENC export** | Low parent | Small | **NVENC session** in driver | Sequential large MP4 writes | None |
| **Resolve bridge** | Low | Small | 0 | None | None (local API) |

---

## 2. Peak resource scenarios

### 2.1 Worst-case single session (sequential jobs)

1. Transcribe 2-hour podcast → WAV ~230 MB on disk; GPU Whisper holds model VRAM.
2. Analyze with discovery mode → OpenAI tokens + embedding model + transcript in RAM.
3. Export 20 clips smart_crop + NVENC → repeated GPU (YOLO) + NVENC + disk.

**Peak RAM (parent python):** 8–16 GB realistic with large transcript + embeddings + session state.  
**Peak VRAM:** Whisper + YOLO + torch cache — stay under 20 GB on 4090 if models unloaded between phases.

### 2.2 What 64 GB system RAM provides

- Headroom for OS, browser, Resolve, and Clip Studio concurrently.
- Risk zone: **cloud Whisper** on extracted WAV near limit + large `cs_segments` in `st.session_state` + parallel browser tabs.

### 2.3 Disk

| Path | Growth rate |
|------|-------------|
| `uploads/` | Source videos (user retention) |
| `outputs/clips/session_*` | ~5–50 MB per exported vertical clip |
| `outputs/_work/_whisper_input.wav` | Overwritten per transcribe; cleanup on startup |
| `outputs/cache/analysis/` | JSON per transcript fingerprint |
| `logs/` | Rotating; bounded |

**NVMe benefit:** Reduces blocking when FFmpeg reads high-bitrate source while writing MP4.

---

## 3. Leak & orphan risk matrix

| Risk | Severity | Evidence | Mitigation (current) |
|------|----------|----------|----------------------|
| **VRAM leak — Whisper model** | Medium | Model cached in `whisper_runtime` | `unload_whisper()` after transcribe (added) |
| **VRAM leak — embeddings** | Low | Singleton in `semantic_ranking` | `release_gpu_memory()` after analyze |
| **VRAM leak — torch** | Low | Fragmentation | `torch.cuda.empty_cache()` |
| **RAM leak — session state** | Medium | Streamlit never clears `cs_segments` | User must refresh session / restart app |
| **Thread leak** | Low | Streamlit + job lock only | No thread pool in pipeline |
| **Process leak — tracked FFmpeg** | Low | `subprocess_guard` registry | Cancel + `atexit` terminate_all_tracked |
| **Process leak — orphan FFmpeg** | Medium | `smart_crop` raw subprocess | `find_orphan_ffmpeg_pids` + kill at startup/cancel |
| **Process leak — Resolve bridge** | Low | 30 s timeout | Short-lived |
| **Duplicate workers** | Low | `try_acquire_job` / `JobBusyError` | UI `studio_job` wrapper |
| **Infinite loops** | Low | Pipeline is finite passes | OpenAI retry capped in resilience |
| **Duplicate Whisper loads** | Reduced | Was per-call; now cached + unload | `whisper_runtime.py` |

---

## 4. Network usage

| Operation | Pattern |
|-----------|---------|
| OpenAI chat (analyze) | Many sequential requests; `call_delay_seconds` default 0.75 s |
| OpenAI whisper API | Single large upload per transcribe |
| Rate limits | `OpenAIRateLimitError`; progress resume via `analysis_cache` |

**Offline:** Transcribe can run fully local with faster-whisper + CUDA. Analyze requires API key unless cache hit.

---

## 5. Monitoring hooks

| Mechanism | Output |
|-----------|--------|
| `stability.log_resource_snapshot()` | `logs/resource_monitor.log` + logger |
| `telemetry.log_gpu_memory()` | Session + `logs/gpu.log` |
| `nvidia-smi` in export headroom | Waits if GPU RAM > 18 GB |
| Sidebar **Refresh resource snapshot** | On-demand |
| Startup snapshot | `clip_studio_app._ensure_startup()` |

---

## 6. Recommendations (no redesign)

1. Run Clip Studio from **Python 3.11 AI venv** with faster-whisper + CUDA 12 CTranslate2 wheel.
2. After long analyze sessions, **restart Streamlit** to clear session RAM.
3. Keep **one long job** at a time (UI already enforces).
4. Before batch export, confirm **orphan ffmpeg** sidebar shows none.
5. For 2+ hour sources, prefer **GPU transcribe** to avoid 25 MB cloud WAV cap.
