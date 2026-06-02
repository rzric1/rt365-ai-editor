# Launcher Fix Report — `launch_ai_clip_studio.ps1`

**Date:** 2026-06-01

---

## Root cause

PowerShell **here-strings** (`@" ... "@`) were used to pass multi-line Python to `python -c`. On this host, the closing `"@` delimiter or embedded Python syntax (`from`, `for`, `if`) was parsed as **PowerShell**, producing errors such as:

- `from clip_engine.environment_check import ...` — invalid PowerShell statement  
- `print('environment_check:', 'PASS' if s.ok else 'FAIL')` — PowerShell misread `if`  
- `for e in s.errors:` — PowerShell `for` loop parse error  

Python must run **inside** `python.exe` arguments or a `.py` file, never as bare lines in a `.ps1` file.

---

## Lines fixed

| Lines (before) | Problem | Fix |
|----------------|---------|-----|
| 93–100 | Multi-line `@" ... "@` environment check | `& $venvPython (Join-Path $ProjectRoot 'check_environment.py')` |
| 108–114 | Multi-line `@" ... "@` preflight | Single-line: `$preflightCode = '...'; & $venvPython -c $preflightCode` |
| 134 | Inline `-c` with semicolons in quotes | `$releaseCode = '...'; & $venvPython -c $releaseCode` |

No Python `import` / `for` / `from` statements remain as top-level PowerShell code.

---

## Verification

1. **PowerShell parse:** Script uses only valid PS1 constructs (no `@"` Python blocks).  
2. **Environment check:** Delegates to existing `check_environment.py` (writes `logs/environment_check.txt`).  
3. **Preflight / release lock:** One-line `-c` strings assigned to variables before invocation.  

**Launch command (unchanged):**

```powershell
powershell -ExecutionPolicy Bypass -File C:\dev\rt365-ai-editor\launch_ai_clip_studio.ps1
```

Or:

```powershell
C:\dev\rt365-ai-editor\launch_ai_clip_studio.ps1
```

**Application behavior:** Unchanged — same venv, CUDA PATH, ffmpeg helper, Streamlit flags, and lock release on exit.

---

## If environment check fails

Run once:

```bat
C:\dev\rt365-ai-editor\setup_windows.bat
```

Then relaunch with the command above.
