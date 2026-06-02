# Start-Then-Stops Fix Report

**Date:** 2026-06-01

---

## Root cause

`clip_studio_app.main()` calls `acquire_app_lock()`, which called `preflight_single_instance()` including **`_port_in_use()`** (TCP connect to `127.0.0.1:8501`).

After Uvicorn prints `Uvicorn server started on 0.0.0.0:8501`, the **same Streamlit process** already listens on 8501. The port check succeeds → preflight reports **“port in use”** → `acquire_app_lock()` fails → `st.stop()` on the first script run → Streamlit shuts down and the terminal prints **`Stopping...`** (often twice during teardown).

This is a **lifecycle / lock logic bug**, not a CUDA crash and not a PowerShell parse error.

**Exact location:** `clip_engine/app_lock.py` — `acquire_app_lock()` → `preflight_single_instance()` → `_port_in_use()` (removed from acquire path).

---

## Files changed

| File | Change |
|------|--------|
| `clip_engine/app_lock.py` | LISTEN-only port detection for launcher; **no port check** in `acquire_app_lock()` |
| `clip_engine/startup_trace.py` | **New** — `logs/startup_trace.log` |
| `scripts/launcher_trace_event.py` | **New** — PS1 trace helper |
| `clip_studio_app.py` | Startup order: env → startup/diagnostics → lock; trace logging; shutdown atexit |
| `launch_ai_clip_studio.ps1` | Trace events; **removed try/finally around Streamlit**; lock release **only after** Streamlit exits |

---

## Validation

1. PowerShell script parses without errors.
2. Launch: environment PASS → Streamlit stays running on :8501.
3. `logs/startup_trace.log` should include:
   - `launcher started`
   - `environment check passed`
   - `preflight lock check passed`
   - `app imported`
   - `lock acquired`
   - `app rendered first frame`
4. No immediate `Stopping...` until Ctrl+C.

---

## Final launch command

```powershell
powershell -ExecutionPolicy Bypass -File C:\dev\rt365-ai-editor\launch_ai_clip_studio.ps1
```

---

## Notes

- Launcher **preflight** still blocks a second instance when another process is **LISTENING** on 8501 (not TIME_WAIT).
- Lock file is written by the Streamlit process after the server is up; the launcher does not hold the lock during the run.
