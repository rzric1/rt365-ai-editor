# RT365 AI Clip Studio — Resource Audit

**Legend:** ● High ● Medium ○ Low — estimated at peak load for a 2-hour podcast, RTX 4090, GPU path enabled.

---

## 1. Module resource matrix

| Module | CPU | GPU | VRAM | RAM | Disk I/O | Network | Threads | Child processes |
|--------|-----|-----|------|-----|----------|---------|---------|-----------------|
| `clip_studio_app` / Streamlit | ● | ○ | ○ | ● | ○ | ○ | ● (server thread pool) | 0 |
| `ui/clip_cards` (rerun) | ● | ○ | ○ | ● | ○ | ○ | Same process | 0 |
| `upload_manifest` | ○ | ○ | ○ | **●●** (full file buffer) | ● write | ○ | 1 | 0 |
| `audio_extract` | ● | ○ | ○ | ○ | ● read/write WAV | ○ | 1 | **1× ffmpeg** |
| `transcription` (faster-whisper) | ● | **●●** | **●●** | **●** | ○ | ○ | 1 | 0 |
| `transcription` (OpenAI) | ○ | ○ | ○ | **●** (full WAV bytes) | ○ | **●** | 1 | 0 |
| `speaker_analysis.diarize` | ● | **●** | **●** | **●** | ○ | ○ | 1 | 0 |
| `gpu_pipeline` | ● | **●** | **●** | ● | ○ | ○ | 1 | 0 |
| `semantic_ranking` | ● | **●** | **●** (cached model) | **●** | ○ | ○ | 1 | 0 |
| `clip_analysis` / OpenAI | ● | ○ | ○ | ● (large strings) | ● cache write | **●●** | 1 | 0 |
| `clip_pipeline` | ● | ○–● | ○–● | ● | ● | ● | 1 | 0 |
| `export_vertical` | ● | **●** (NVENC) | **●** | ○ | **●●** | ○ | 1 + lock | **1× ffmpeg/clip** |
| `smart_crop` (dynamic) | **●** | **●** (YOLO) | **●** | ● | ● read video | ○ | 1 | 0 |
| `ffmpeg_gpu` probe | ○ | **●** | ○ | ○ | ○ | ○ | 1 | 1× ffmpeg short |
| `cuda_diagnostics` | ○ | ○ | ○ | ○ | ○ | ○ | 1 | 0–1 nvidia-smi |
| `resolve_bridge` | ○ | ○ | ○ | ○ | ○ | ○ | 1 | **1× python** |
| `analysis_cache` | ○ | ○ | ○ | ○ | ● JSON R/W | ○ | 1 | 0 |
| `telemetry` logs | ○ | ○ | ○ | ○ | ● append | ○ | 1 | 0 |

---

## 2. Peak resource scenarios

### Scenario A — Browser upload 4 GB MP4

- **RAM:** Up to **8+ GB** transient (full `getbuffer()` for fingerprint + chunked write).
- **Disk:** Full file copy to `uploads/`.
- **Risk:** OOM, page file thrashing, UI freeze during save.

### Scenario B — Transcribe 2 h video (GPU)

- **Disk:** WAV ~200–400 MB at `outputs/_work/_whisper_input.wav`.
- **VRAM:** Whisper model (size-dependent: base ~1–2 GB, large-v3 much higher).
- **RAM:** Model weights + decoder buffers; model **not unloaded** after run.
- **Duration:** Minutes to tens of minutes sustained GPU load.

### Scenario C — Analyze (SAFE profile, cache miss)

- **Network:** 20–80+ OpenAI requests (chunked transcript).
- **RAM:** Full `cs_formatted` + candidate dicts in session.
- **GPU:** Optional embeddings batch encode (sentence-transformers).
- **CPU:** JSON parse, dedupe, finalizer loops.

### Scenario D — Export 20 clips (smart_crop + NVENC)

- **GPU:** NVENC sessions sequential; YOLO loaded **per clip** in smart_crop path.
- **Disk:** 20× 1080×1920 MP4 writes (hundreds of MB–GB total).
- **Time:** 300 s timeout × clips worst case; 2 s gap between exports.

---

## 3. Memory leak & handle leak analysis

| Pattern | Leak type | Severity | Location |
|---------|-----------|----------|----------|
| `WhisperModel()` per transcribe/diarize | VRAM/RAM not released | **Critical** | `transcription.py`, `speaker_analysis.py` |
| `SentenceTransformer` global `_MODEL` | Intentional cache; never freed | Medium | `semantic_ranking.py` |
| `YOLO("yolov8n.pt")` per smart_crop export | VRAM + load time | **High** | `smart_crop._yolo_detect_trajectory` |
| Streamlit `session_state` growth | RAM (transcript, clips, telemetry) | **High** | All UI modules |
| OpenCV `VideoCapture` | Handle leak if exception mid-loop | Low | `smart_crop.py` (has `cap.release()` on happy path) |
| FFmpeg `subprocess.run` | Zombie if parent killed | **High** | All ffmpeg call sites |
| Log handlers | Bounded (rotating) | Low | `telemetry.py` |
| Analysis cache on disk | Disk growth, not RAM leak | Medium | `analysis_cache.py` |

**No classic Python reference cycles identified** — primary issue is **large object retention** and **GPU allocator caches**, not unbounded list appends in background threads.

---

## 4. Process leak analysis

| Source | Orphan risk | Why |
|--------|-------------|-----|
| FFmpeg during export | **High** | `subprocess.run` — killing Streamlit does not kill child on Windows without job object |
| FFmpeg during transcribe | Medium | Same; usually completes |
| `resolve_bridge.py` | Low | 30 s timeout |
| Extra Streamlit | Medium | User double-clicks launcher → port conflict / 2× RAM |
| Python multiprocessing | **None** | Not used |

**No worker pool** — thread count stays near Streamlit default (~dozen), not hundreds.

---

## 5. GPU driver reset / BSOD vectors

| Vector | Mechanism | Likelihood on RTX 4090 |
|--------|-----------|------------------------|
| VRAM exhaustion | Whisper + embeddings + NVENC + YOLO overlapping | Medium if user overlaps operations |
| TDR timeout | FFmpeg NVENC + heavy CUDA kernels > 2 s default | Low–Medium on long encodes |
| CUDA OOM in CTranslate2 | Wrong compute type / model too large | Medium on misconfig |
| Missing `cublas64_12.dll` | CPU fallback thrash, not usually BSOD | Low (logged) |
| PSU / thermals | Sustained 4090 + CPU encode | Hardware-dependent |

**BSOD** is rarely caused directly by Python; correlation is usually **nvlddmkm** (NVIDIA), **WHEA**, or **MEMORY_MANAGEMENT** when RAM is exhausted system-wide.

---

## 6. Disk I/O profile

| Operation | Pattern | Size |
|-----------|---------|------|
| Upload save | Sequential write | Up to 4 GB |
| WAV extract | Read video, write WAV | Full duration audio |
| Export | Read video seek per clip, write MP4 | Per clip output |
| Cache | Small JSON random write | KB–MB |
| Logs | Append rotating | ≤50 MB per channel |

**No mmap of full video in Python** — FFmpeg streams from file.

---

## 7. Network profile

| When | Traffic |
|------|---------|
| Cloud Whisper | Upload full WAV |
| Clip analysis | Many HTTPS JSON payloads (MB cumulative) |
| Re-ground on export | Per-clip optional API |
| License check (if enabled) | Small |

**No background network** when idle in UI.

---

## 8. Flags — resource abuse potential

### Can consume excessive RAM

- `upload_manifest.compute_upload_fingerprint` — **entire upload in RAM**
- `transcription` OpenAI path — `wav_path.read_bytes()`
- `st.session_state.cs_formatted` — unbounded transcript text
- Large `cs_clips` list with nested dicts

### Can leak memory (effective)

- Repeated Whisper loads without `del` / `gc` / `empty_cache`
- Global embedding model + Whisper simultaneously

### Can leak handles

- Orphaned ffmpeg.exe
- OpenCV capture on exception paths (rare)

### Can leak processes

- ffmpeg.exe after abnormal Streamlit exit
- Second Streamlit instance

### Can cause GPU driver reset

- Stacked CUDA consumers + NVENC under VRAM pressure
- Very long NVENC session (unlikely alone on 24 GB card)

### Can cause CUDA crashes

- CTranslate2 DLL mismatch (caught, falls back to CPU)
- torch + ctranslate2 version skew

### Can cause VRAM exhaustion

- large-v3 Whisper + embeddings + YOLO + NVENC overlap
- **Most likely on 8 GB cards**; 4090 more headroom but not immune with overlap

### Can cause system freeze

- 4 GB RAM spike + Windows paging
- Disk 100% during parallel upload + export to same drive

---

## 9. Environment variables affecting resources

| Variable | Effect |
|----------|--------|
| `CUDA_VISIBLE_DEVICES` | GPU selection (launcher sets `0`) |
| `FORCE_CPU_WHISPER` | Reduces GPU, increases CPU/RAM time |
| `FORCE_CPU_EMBEDDINGS` | CPU-only embeddings |
| `FORCE_NVENC_EXPORT` | Forces NVENC attempts |
| `OPENAI_API_KEY` | Enables cloud paths |
| `FFMPEG_BINARY` | Custom ffmpeg path |

---

*Cross-reference: CRASH_RISK_REPORT.md for code-level citations; STABILITY_REPORT.md for cleanup verification.*
