# Windows Diagnostics — Correlating RT365 Crashes with System Evidence

Use this guide after a freeze, crash, BSOD, or GPU reset while running **RT365 AI Clip Studio**.

---

## 1. Before the incident (baseline)

1. Note exact action: Upload / Transcribe / Analyze / Export / Preview / Resolve send.
2. Note video file size, duration, Whisper model size, export clip count, export mode (smart_crop vs full_fit).
3. Save last 200 lines of launcher console or `crash_log.txt`.
4. Copy `logs/app.log`, `logs/gpu.log`, `logs/exports.log` timestamps around the event.

---

## 2. Event Viewer

**Open:** `Win + R` → `eventvwr.msc`

### 2.1 Application log

Path: **Windows Logs → Application**

Filter:

- **Sources:** `Application Error`, `Windows Error Reporting`, `Python`, `.NET Runtime` (Streamlit may appear as python.exe)
- **Time:** Custom range around incident

Look for:

- `Faulting application name: python.exe`
- `Faulting module name: nvcuda.dll`, `nvenc.dll`, `ctranslate2.dll`, `c10.dll` (torch), `ffmpeg.exe`
- Exit code `0xc0000005` (access violation), `0xc0000409` (stack buffer), `0xe0434352` (.NET — less common for Python)

### 2.2 System log

Path: **Windows Logs → System**

Look for:

- **nvlddmkm** — Display driver stopped responding and has recovered (TDR)
- **WHEA-Logger** — hardware errors (RAM/PCIe)
- **Disk** — disk errors during heavy write to `uploads/` or `outputs/`

### 2.3 Correlation template

| App log time | System log | Interpretation |
|--------------|------------|----------------|
| python.exe crash + nvcuda.dll | nvlddmkm TDR same minute | GPU timeout / driver reset |
| python.exe OOM | No driver event | RAM exhaustion |
| ffmpeg.exe hang | Disk 153 | I/O bottleneck |
| No app crash, full freeze | High CPU python | Streamlit blocked on subprocess |

---

## 3. Reliability Monitor

**Open:** `Win + R` → `perfmon /rel`

- Daily stability index — drop on crash day
- Click red X entries → **View technical details**
- Match **Source** (python.exe, ffmpeg.exe, Windows)
- **Fault bucket** IDs useful for web search

Export: Right-click column header → **Save Events As** CSV for records.

---

## 4. Windows crash dumps

### 4.1 User-mode dumps (python / ffmpeg)

**Enable (optional):**

```
Win + R → sysdm.cpl → Advanced → Startup and Recovery → Settings
→ Write debugging information: Small memory dump (256 KB) or Automatic
```

For **full** python dumps, use **Windows Error Reporting** local dumps:

Registry (admin):  
`HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\python.exe`  
- `DumpFolder` = `C:\CrashDumps`  
- `DumpType` = `2` (full)

### 4.2 BSOD kernel dumps

Location (if enabled): `C:\Windows\MEMORY.DMP` or `C:\Windows\Minidump\*.dmp`

Analyze with **WinDbg Preview**:

```
!analyze -v
```

Common bugchecks:

| Code | Meaning | RT365 correlation |
|------|---------|-------------------|
| `0x00000116` | VIDEO_TDR_FAILURE | Long NVENC/CUDA |
| `0x0000007E` | SYSTEM_THREAD_EXCEPTION | Driver |
| `0x0000001A` | MEMORY_MANAGEMENT | RAM pressure |
| `0x000000EF` | CRITICAL_PROCESS_DIED | System instability |

---

## 5. NVIDIA driver logs

### 5.1 nvidia-smi (immediate)

```powershell
nvidia-smi
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv -l 1
```

Run during reproduce — watch **memory.used** during Transcribe + Export.

### 5.2 NVIDIA logging

- **NVIDIA Control Panel → Help → Debug Logging** (if available on driver version)
- Driver logs folder (varies):  
  `C:\ProgramData\NVIDIA Corporation\NvTelemetry`  
  `C:\Windows\System32\LogFiles\`

### 5.3 CUDA toolkit alignment

RT365 launcher adds:

`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin`

Verify `cublas64_12.dll` exists — matches `cuda_diagnostics.py` checks in app sidebar.

---

## 6. Application-specific artifacts

| Path | Content |
|------|---------|
| `C:\dev\rt365-ai-editor\logs\app.log` | General pipeline |
| `C:\dev\rt365-ai-editor\logs\gpu.log` | GPU memory snapshots |
| `C:\dev\rt365-ai-editor\logs\exports.log` | Per-export encoder |
| `C:\dev\rt365-ai-editor\logs\openai.log` | API timing |
| `C:\dev\rt365-ai-editor\crash_log.txt` | Launcher stderr capture |

Search logs for last successful phase tag:

- `[PIPELINE TIMING]`
- `faster-whisper transcription done`
- `[GPU MEMORY]`
- `Delivery email` (N/A here)
- `FFmpeg failed`

---

## 7. Process snapshot during hang

**Task Manager → Details:**

| Process | Expected | Problem |
|---------|----------|---------|
| `python.exe` (streamlit) | 1 | 2+ = duplicate launch |
| `ffmpeg.exe` | 0–1 during work | Many = orphan leak |
| GPU engine | python or ffmpeg | Both high = overlap |

**PowerShell:**

```powershell
Get-Process python, ffmpeg -ErrorAction SilentlyContinue | Format-Table Id, CPU, WS, PM -AutoSize
```

`WS` (working set) > 8 GB on python during upload = RAM spike confirmed.

---

## 8. Correlation decision tree

```
Incident during Upload?
  YES → Check python WS > file size → RAM OOM in Event Viewer
  NO ↓
Incident during Transcribe?
  YES → gpu.log + nvidia-smi VRAM → Whisper model size
  NO ↓
Incident during Analyze?
  YES → openai.log rate limits; python CPU 100% (normal); RAM from session
  NO ↓
Incident during Export?
  YES → exports.log encoder; ffmpeg.exe count; nvlddmkm TDR
  NO ↓
BSOD without app log?
  → Minidump + WHEA + RAM test (mdsched.exe)
```

---

## 9. Evidence package for support

Zip these after an incident:

1. `logs/*.log` (last 24 h)
2. Event Viewer export (Application + System, 1 h window)
3. Reliability Monitor screenshot
4. `nvidia-smi -q` text output
5. Streamlit console tail / `crash_log.txt`
6. Note: GPU model, driver version, RAM, file size, settings profile (SAFE/AGGRESSIVE)

---

## 10. RT365-specific timestamps to record

| Field | Example |
|-------|---------|
| `cs_whisper_model` | base / small / large-v3 |
| `cs_gpu_acceleration` | on/off |
| `cs_export_mode_label` | smart_crop vs full_fit |
| Clip count exported | 17 |
| Input mode | Local path vs browser upload |
| AI profile | SAFE |

---

*Hardware BSODs require minidump analysis; software freezes often correlate with python RAM or ffmpeg orphans per STABILITY_REPORT.md.*
