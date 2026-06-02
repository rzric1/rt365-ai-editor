# Windows Diagnostics — RT365 AI Clip Studio Crash Correlation

Use this guide after a freeze, crash, BSOD, or GPU reset while running **RT365 AI Clip Studio** on Windows 11 + RTX 4090.

**Companion logs (RT365):** `logs/crash_report.txt`, `logs/app.log`, `logs/gpu.log`, `logs/exports.log`, `logs/resource_monitor.log`, `logs/startup_diagnostics.txt`, launcher `crash_log.txt`.

---

## 1. Before the incident (baseline)

1. Note exact action: Upload / Transcribe / Analyze / Export / Preview / Resolve send.
2. Note video file size, duration, Whisper model size, export clip count, export mode (smart_crop vs full_fit).
3. Save last 200 lines of launcher console or `crash_log.txt`.
4. Copy RT365 logs with timestamps ±5 minutes around the event.
5. Sidebar → **Refresh resource snapshot** if app still responsive.

---

## 2. Reliability Monitor

**Open:** `Win + R` → `perfmon /rel`

| Step | Action |
|------|--------|
| 1 | Find the crash day — stability index drop |
| 2 | Click red **X** entries for `python.exe`, `ffmpeg.exe`, `Streamlit` |
| 3 | **View technical details** — faulting module, bucket ID |
| 4 | Export: right-click → **Save Events As** CSV |

**Correlate with RT365:** Match Reliability timestamp to `logs/crash_report.txt` UTC block.

---

## 3. Event Viewer

**Open:** `Win + R` → `eventvwr.msc`

### 3.1 Application log

**Path:** Windows Logs → Application

**Filter:** Sources `Application Error`, `Windows Error Reporting`, time range around incident.

| Pattern | Meaning |
|---------|---------|
| Faulting `python.exe` + `nvcuda.dll` / `c10.dll` | GPU/CUDA in Python stack |
| Faulting `python.exe` + `ctranslate2.dll` | faster-whisper native crash |
| Faulting `ffmpeg.exe` + `nvenc.dll` | NVENC encode failure |
| Exit `0xc0000005` | Access violation |
| Exit `0xc0000409` | Stack buffer overrun |
| Exit `0xe0434352` | .NET — rare for Clip Studio |

### 3.2 System log

**Path:** Windows Logs → System

| Source | Meaning |
|--------|---------|
| **nvlddmkm** | Display driver TDR — GPU reset; Whisper/NVENC may have hung |
| **WHEA-Logger** | Hardware error (RAM, PCIe) — investigate RAM test if recurring |
| **Kernel-Power** Event 41 | Unexpected shutdown — power loss or hard crash |
| **Disk** | Storage errors during write to `uploads/` or `outputs/` |

### 3.3 LiveKernelEvent

**Path:** Windows Logs → System → filter **LiveKernelEvent**

- Often accompanies **nvlddmkm** TDR on RTX cards.
- Indicates kernel detected hung GPU command buffer.

**RT365 correlation:** If LiveKernelEvent timestamp = end of long NVENC export or CUDA transcribe, reduce concurrent GPU load and update driver.

---

## 4. WHEA (hardware errors)

**Path:** Event Viewer → Applications and Services Logs → Microsoft → Windows → WHEA-Logger → Operational

| Finding | Action |
|---------|--------|
| Corrected memory error | Run MemTest86; check XMP/EXPO |
| PCIe error | Reseat GPU; check PSU rails |
| Repeated WHEA + no driver events | Suspect hardware before blaming RT365 |

---

## 5. nvlddmkm (NVIDIA display driver)

**Symptoms:** Black screen flash, "Display driver stopped responding", app freeze with GPU busy.

**Event Viewer:** System log, Source `nvlddmkm`, Event ID 4101 (common).

**Mitigations for Clip Studio workloads:**

1. Update to latest NVIDIA Studio driver for content creation.
2. Avoid parallel **Resolve playback + Whisper + NVENC export**.
3. Lower Whisper model size for very long files.
4. Disable unnecessary GPU acceleration toggles to isolate (export CPU fallback test).
5. Increase TDR delay (advanced registry) only if Microsoft/docs recommend for your workflow.

---

## 6. Kernel-Power (Event 41)

**Meaning:** System rebooted without clean shutdown.

**Distinguish:**

| Cause | Other evidence |
|-------|----------------|
| PSU overload / spike | Kernel-Power 41, no app log |
| Driver hard reset | nvlddmkm + LiveKernelEvent same second |
| User hard power | No RT365 logs |

---

## 7. Correlation worksheet

| Time (local) | RT365 log | Windows source | Likely cause |
|--------------|-----------|----------------|--------------|
| | `pipeline_step=ffmpeg_export` | nvlddmkm | NVENC TDR |
| | `whisper_cuda_float16` | python + ctranslate2 | CUDA DLL / model |
| | `analyze` + high `process_rss_mb` | no GPU event | RAM pressure |
| | orphan ffmpeg PIDs in resource log | ffmpeg.exe hung | Stuck encode; use Kill orphan |

---

## 8. Post-incident RT365 actions

1. Restart Clip Studio (clears Streamlit session RAM).
2. Sidebar → **Kill orphan ffmpeg** if PIDs listed.
3. Review `logs/crash_report.txt` newest block.
4. Attach `startup_diagnostics.txt` + Reliability export when opening support ticket.

---

## 9. When to escalate

| Signal | Escalate to |
|--------|-------------|
| WHEA errors | Hardware / RAM vendor |
| nvlddmkm only on GPU workloads | NVIDIA driver / thermal |
| python OOM, no WHEA | RT365 — transcript size / session restart |
| ffmpeg-only hang | Disk space / path / corrupt source file |
