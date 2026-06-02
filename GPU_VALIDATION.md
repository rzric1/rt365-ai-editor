# RT365 AI Clip Studio — GPU Validation (RTX 4090)

**Audit date:** 2026-06-01  
**Host validation run on:** Windows 11, driver 610.47, RTX 4090 23028 MiB

---

## 1. CUDA availability

| Check | Result (audited host) | Module |
|-------|----------------------|--------|
| nvidia-smi | **OK** — NVIDIA GeForce RTX 4090 | `cuda_diagnostics._nvidia_smi_text` |
| Driver CUDA banner | 13.3 (UMD) | `parse_driver_cuda_version` |
| CTranslate2 devices | **0** — `ctranslate2` not installed in active Python | `get_cuda_device_count()` |
| CTranslate2 runtime probe | **FAIL** — import error | `ctranslate2_cuda_runtime_probe()` |
| Torch CUDA | Not installed in active Python | `_torch_cuda_summary` |

**Production requirement:** Use project AI venv (`scripts/setup_python311_ai_env.ps1`) with:

- `faster-whisper`
- `ctranslate2` CUDA 12 wheel
- CUDA Toolkit 12.x `bin` on PATH (for `cublas64_12.dll`) or driver-bundled DLLs

Launcher `launch_ai_clip_studio.ps1` may prepend `CUDA\v12.x\bin`.

---

## 2. NVENC support

| Check | Result (audited host) |
|-------|----------------------|
| FFmpeg lists `h264_nvenc` | **True** |
| NVENC runtime probe (lavfi null encode) | **True** |
| Export path | `export_vertical._build_cmd` → NVENC p4 VBR; libx264 fallback |

**RTX 4090 notes:**

- Ada NVENC supports concurrent sessions; app still serializes exports via `_EXPORT_LOCK`.
- `_GPU_MEMORY_HEADROOM_GB = 18.0` — waits 5 s before export if `nvidia-smi` reports >18 GB used (now functional after `capture_output` fix).

---

## 3. GPU memory release

| Trigger | Action |
|---------|--------|
| After transcription | `unload_whisper()` + `release_gpu_memory("transcription_complete")` |
| After clip analyze (UI) | `release_gpu_memory()` in `clip_cards` |
| Whisper model key change | `_release_model()` inside `whisper_runtime` |
| Manual | Restart Streamlit process (hard reset) |

**Validation procedure:**

1. Open sidebar → **Refresh resource snapshot** — note `vram_allocated_gb`.
2. Run transcribe on a short clip with GPU on (in AI venv).
3. Refresh snapshot — VRAM should drop after job completes.
4. Run analyze with GPU prefilter enabled — spike expected; drop after complete.

---

## 4. Model unloading

| Model | Singleton? | Unload API |
|-------|------------|------------|
| faster-whisper | Yes (`whisper_runtime`) | `unload_whisper()` — **called after each transcribe** |
| sentence-transformers | Yes (`semantic_ranking`) | Process lifetime; no explicit unload |
| YOLO (smart crop) | Per-invocation load pattern | Process exit after export |

---

## 5. Concurrent job limits

| Layer | Limit |
|-------|-------|
| Global job lock | 1 long job (`transcribe`, `analyze`, `export`, etc.) |
| Export encode | `_EXPORT_LOCK` — 1 FFmpeg export |
| Whisper | Single cached model; transcribe job blocks analyze |
| NVENC | Sequential clips + 2 s inter-delay |

**Not limited:** NVENC probe at startup, nvidia-smi queries, Resolve subprocess.

---

## 6. RTX 4090 workstation checklist

- [ ] Install latest Studio/Game Ready driver (≥ 550 series for CUDA 12 user-mode).
- [ ] Install CUDA Toolkit 12.x **or** verify `cublas64_12.dll` load in sidebar diagnostics.
- [ ] Create Python 3.11 venv per `setup_python311_ai_env.ps1`.
- [ ] `pip install faster-whisper` + matching `ctranslate2` CUDA wheel.
- [ ] Launch via `launch_ai_clip_studio.ps1` (PATH + `CUDA_VISIBLE_DEVICES=0`).
- [ ] Sidebar: GPU acceleration ON; verify NVENC probe OK.
- [ ] Run `check_environment.py` and `logs/startup_diagnostics.txt` review.

---

## 7. Expected VRAM budget (4090 24 GB)

| Component | Approximate |
|-----------|-------------|
| Whisper large-v3 CUDA | 4–10 GB |
| Embeddings (MiniLM-class) | 1–2 GB |
| YOLO smart crop | 2–4 GB |
| NVENC (driver) | 0.5–2 GB during encode |
| OS / desktop | 1–2 GB |

**Safe practice:** Do not run Resolve heavy playback + full discovery analyze + batch smart_crop export simultaneously without monitoring `logs/gpu.log`.
