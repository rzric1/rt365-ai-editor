# RT365 AI Clip Studio — Pipeline Report

**Audit date:** 2026-06-01  
**Target workstation:** Windows 11, RTX 4090 (24 GB VRAM), 64 GB RAM, NVMe SSD, CUDA, FFmpeg, faster-whisper, OpenAI  
**Entry point:** `clip_studio_app.py` (Streamlit on port 8501)

---

## 1. Runtime architecture

| Layer | Component | Process model |
|-------|-----------|---------------|
| Launcher | `launch_ai_clip_studio.ps1`, `run_app.bat` | Parent shell; may prepend CUDA `bin` to PATH |
| UI | Streamlit + Uvicorn | **Single long-lived `python.exe`** |
| Pipeline | `clip_engine/*`, `ui/*` | In-process; cooperative cancel via `job_control` |
| Media children | FFmpeg / ffprobe / nvidia-smi | **Tracked subprocesses** via `subprocess_guard.run_subprocess` |
| ML (optional) | faster-whisper, CTranslate2, PyTorch, sentence-transformers, YOLO | In-process CUDA; Whisper model cached in `whisper_runtime` |
| Cloud | OpenAI HTTP | Sequential calls from Python |
| Resolve | `resolve_bridge.py` | Short-lived **untracked** `subprocess.run` (30 s timeout) |

**Out of scope (not Clip Studio):** Vercel claims SPA, Stripe webhooks, Railway — unchanged per audit charter.

---

## 2. End-to-end pipeline map

```
[Launch] → streamlit run clip_studio_app.py
    │
    ├─ Startup: exception hooks, temp cleanup, orphan ffmpeg sweep, diagnostics
    │
[1 Video import]
    ├─ Local path reference OR browser upload → uploads/ (fingerprint dedupe)
    ├─ media_probe (ffprobe, 120 s) — duration for analysis
    └─ Job: studio_job("upload")
    │
[2 Transcript generation]
    ├─ FFmpeg: video → 16 kHz mono WAV (outputs/_work/_whisper_input.wav)
    │     audio_extract.py → run_subprocess (7200 s max)
    ├─ Whisper processing (prefer_gpu):
    │     ├─ CTranslate2 CUDA probe → faster-whisper on GPU (float16/int8_float16/int8)
    │     ├─ CPU int8/float32 fallback if CUDA fails
    │     └─ OpenAI whisper-1 API if local fails (WAV ≤ 25 MB in RAM)
    ├─ whisper_runtime: singleton WhisperModel per (size, device, compute_type)
    ├─ unload_whisper + release_gpu_memory in finally
    └─ Job: studio_job("transcribe") — global lock name "transcribe"
    │
[2b Diarization] (optional)
    ├─ Gap-based speaker turns from faster-whisper word timestamps
    └─ Job: studio_job("diarize")
    │
[3 Clip detection & ranking] — studio_job("analyze")
    ├─ run_full_clip_pipeline (clip_pipeline.py)
    │   ├─ Analysis cache (disk JSON under outputs/cache/analysis/)
    │   ├─ GPU prefilter: local_candidate_discovery + semantic_ranking embeddings
    │   ├─ OpenAI multipass: collect_candidates_multipass (clip_analysis.py)
    │   ├─ Diversity, overlap dedupe, series split, boundaries, expansion passes
    │   ├─ Virality scoring, quality gate, clip finalizer, duration governor
    │   └─ torch.cuda.empty_cache() after analyze (clip_cards)
    ├─ openai_resilience: backoff, JSON repair, token budget
    └─ One active job lock prevents parallel analyze + transcribe + export
    │
[4 UI / preview]
    ├─ Per-clip preview: export_vertical (preview_mode, 15 s cap)
    └─ Job: studio_job("preview") — blocks other long jobs
    │
[5 Export pipeline]
    ├─ export_vertical_clip_with_captions per clip (sequential)
    │   ├─ _EXPORT_LOCK (threading) — one FFmpeg encode at a time
    │   ├─ nvidia-smi VRAM headroom check (> 18 GB used → wait 5 s)
    │   ├─ captions: ASS generation, optional burn-in (._tmp.ass deleted in finally)
    │   ├─ smart_crop: YOLO/OpenCV trajectory (optional; some ffprobe via raw subprocess)
    │   └─ FFmpeg NVENC h264_nvenc → libx264 fallback (300 s per clip)
    └─ Job: studio_job("export")
    │
[6 Resolve integration]
    ├─ EDL: clip_engine/resolve_export.py (in-thread, 5 s UI timeout)
    └─ Send to Resolve: resolve_bridge.py subprocess (JSON stdin, 30 s) — **not job-locked**
```

---

## 3. Stage reference

### 3.1 Video import

| Item | Detail |
|------|--------|
| Modules | `ui/clip_cards.py`, `clip_engine/upload_manifest.py`, `config.CLIP_STUDIO_MAX_UPLOAD_BYTES` |
| Disk | `uploads/<fingerprint>.<ext>`, duplicate quarantine `uploads/_duplicates/` |
| RAM | Streamlit may buffer full upload in memory during fingerprint |
| FFmpeg | None at upload (path reference only for local file) |

### 3.2 Transcript generation / Whisper

| Item | Detail |
|------|--------|
| Modules | `transcription.py`, `audio_extract.py`, `whisper_runtime.py`, `cuda_diagnostics.py` |
| FFmpeg | PCM WAV extract |
| GPU | CTranslate2 + faster-whisper; model cached until key change or `unload_whisper()` |
| Cloud fallback | OpenAI `whisper-1` verbose_json; 25 MB WAV cap |
| Cancel | `check_cancelled()` in transcribe loop; cancel terminates tracked FFmpeg |

### 3.3 OpenAI calls

| Item | Detail |
|------|--------|
| Clip scoring | `clip_analysis.collect_candidates_multipass` |
| Resilience | `openai_resilience.py` — delays, 429 backoff, JSON repair |
| Tokens | `token_tracking.py`, session telemetry |
| Network | HTTPS; no connection pool limits beyond OpenAI SDK |

### 3.4 Clip detection & ranking

| Item | Detail |
|------|--------|
| Local | `local_candidate_discovery`, `transcript_candidate_scanner`, `clip_discovery` |
| GPU shortlist | `gpu_pipeline.run_gpu_prefilter_pipeline`, `semantic_ranking` |
| AI passes | Multipass GPT with profile-driven models (`ai_profiles`, `effective_config`) |
| Post-process | diversity, split, boundaries, finalizer, quality gate, duration governor |

### 3.5 Caption pipeline

| Item | Detail |
|------|--------|
| Module | `clip_engine/captions.py` |
| Outputs | ASS/SRT sidecars; ASS burn-in for export |
| Temp | `._tmp.ass` adjacent to output; removed in export `finally` |
| Presets | Clean, Bold Viral, Podcast, etc. |

### 3.6 FFmpeg usage

| Call site | Tracked? | Timeout |
|-----------|----------|---------|
| `audio_extract.extract_audio_wav` | Yes | 7200 s |
| `export_vertical._run_with_fallback` | Yes | 300 s / clip |
| `export_vertical._check_gpu_headroom` | Yes (nvidia-smi) | 5 s |
| `media_probe` | No | 120 s |
| `ffmpeg_gpu` NVENC probe | No | 25 s |
| `smart_crop` ffprobe | No | varies |
| `scripts/compress_encode.py` | No | manual |

### 3.7 GPU usage

| Workload | Stack | VRAM notes |
|----------|-------|------------|
| Transcription | faster-whisper / CTranslate2 | Model held in `whisper_runtime` until unload |
| Analyze prefilter | sentence-transformers + torch | Embeddings singleton in `semantic_ranking` |
| Smart crop | ultralytics YOLO (optional) | Per-export inference |
| Export encode | FFmpeg **h264_nvenc** (separate process) | Uses NVENC, not PyTorch VRAM |
| Post-analyze | `release_gpu_memory()` | `torch.cuda.empty_cache()` |

### 3.8 Resolve integration

| Path | Behavior |
|------|----------|
| `ui/resolve_panel.py` | Subprocess to `resolve_bridge.py` |
| `clip_engine/resolve_export.py` | EDL text generation |
| Requirements | Resolve Studio open, scripting enabled |

---

## 4. Concurrency & job names

Global lock (`clip_engine/job_control.py`) — only one of:

| Job name | UI trigger |
|----------|------------|
| `upload` | Save browser upload |
| `transcribe` | Transcribe video |
| `diarize` | Speaker diarization |
| `analyze` | Clip analysis / re-analyze |
| `preview` | Per-clip preview export |
| `export` | Batch vertical export |

**Additional serialization:** `export_vertical._EXPORT_LOCK` ensures one FFmpeg export encode at a time even if lock were bypassed.

**Not job-gated:** Resolve send, environment probes, Streamlit reruns.

---

## 5. Logging & diagnostics

| Artifact | Path |
|----------|------|
| App log | `logs/app.log` (rotating 10 MB × 5) |
| OpenAI | `logs/openai.log` |
| GPU | `logs/gpu.log` |
| Exports | `logs/exports.log` |
| Resource monitor | `logs/resource_monitor.log` |
| Crash | `logs/crash_report.txt` |
| Startup | `logs/startup_diagnostics.txt` |

---

## 6. Workstation fit (RTX 4090 / 64 GB / NVMe)

| Requirement | Status on audited host |
|-------------|------------------------|
| RTX 4090 detected | Yes (nvidia-smi 610.47, 23028 MiB) |
| NVENC listed + probe | Yes |
| faster-whisper / CTranslate2 | **Not installed** in default `python` 3.14 env — GPU Whisper requires venv per `setup_python311_ai_env.ps1` |
| FFmpeg | Yes (WinGet Gyan 8.1.1) |
| 64 GB RAM | Adequate for single-session Clip Studio + large transcripts |
| NVMe | Recommended for `uploads/` and `outputs/clips/session_*` sequential writes |

**Verdict:** Pipeline is architected for this workstation class; production stability requires the **AI venv** (Python 3.11 + faster-whisper CUDA wheel + matching CTranslate2), not system Python 3.14 alone.
