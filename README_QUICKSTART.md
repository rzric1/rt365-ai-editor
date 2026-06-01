# RT365 AI Clip Studio — Quick Start

## System Requirements
- Windows 10/11 64-bit
- NVIDIA GPU 8 GB+ VRAM (RTX 3080 or better)
- NVIDIA drivers 520+
- ffmpeg on PATH ([download](https://ffmpeg.org/download.html))
- OpenAI API key ([get one](https://platform.openai.com/api-keys))

## Setup (5 minutes)
1. Double-click `setup_windows.bat`
2. Open `.env`, set `OPENAI_API_KEY=sk-proj-your-key-here`
3. Double-click `launch_ai_clip_studio.bat`

## Workflow
1. Upload video → choose AI Profile → set clip count → Run Analysis
2. Review clips, adjust times if needed
3. Export selected clips (9:16 MP4 + SRT/ASS to `outputs/clips/`)
4. Optional: Send to DaVinci Resolve

## Troubleshooting
| Problem | Fix |
|---|---|
| No clips found | Use SAFE profile; ensure clear speech in video |
| ffmpeg not found | Add ffmpeg/bin to Windows PATH and restart |
| OpenAI 429 error | Wait 60s or switch to SAFE profile |
| GPU not detected | Update NVIDIA drivers and restart |
| Won't start | Re-run setup_windows.bat |
