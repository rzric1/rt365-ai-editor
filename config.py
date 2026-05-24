"""
Application configuration for RT365 AI Editor.

All paths are resolved relative to the project root (folder containing main.py),
so you can run the tool from any working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (project root = directory of this file)
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
TRANSCRIPTS_DIR: Path = PROJECT_ROOT / "transcripts"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"
CLIP_STUDIO_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "clips"
# Must match .streamlit/config.toml [server] maxUploadSize (megabytes)
CLIP_STUDIO_MAX_UPLOAD_MB: int = 4096
CLIP_STUDIO_MAX_UPLOAD_BYTES: int = CLIP_STUDIO_MAX_UPLOAD_MB * 1024 * 1024
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
CLIP_SCORING_PROMPT_PATH: Path = PROMPTS_DIR / "clip_scoring_prompt.txt"

# Default transcript files (used when you run without a path argument)
DEFAULT_SRT_PATH: Path = TRANSCRIPTS_DIR / "input.srt"
DEFAULT_JSON_PATH: Path = TRANSCRIPTS_DIR / "input.json"

# ---------------------------------------------------------------------------
# Environment variables (loaded in main.py via python-dotenv)
# ---------------------------------------------------------------------------

ENV_OPENAI_API_KEY: str = "OPENAI_API_KEY"
ENV_OPENAI_MODEL: str = "OPENAI_MODEL"

# Optional absolute path to ffmpeg (or ffmpeg.exe on Windows). Loaded via .env / process env.
ENV_FFMPEG_BINARY: str = "FFMPEG_BINARY"
DEFAULT_OPENAI_MODEL: str = "gpt-5-mini"

# Frame rate for the last field (FF) in bracket transcripts: [HH:MM:SS:FF - ...]
ENV_TRANSCRIPT_BRACKET_FPS: str = "TRANSCRIPT_BRACKET_FPS"
DEFAULT_TRANSCRIPT_BRACKET_FPS: float = 24.0

# ---------------------------------------------------------------------------
# OpenAI — transcript chunking (tune for long podcasts)
# ---------------------------------------------------------------------------

# Rough character budget per API call (includes timestamps + text).
# Smaller chunks = more API calls but steadier JSON and fewer timeouts.
TRANSCRIPT_CHUNK_MAX_CHARS: int = 12000

# Overlap between consecutive chunks so markers near boundaries are not lost.
TRANSCRIPT_CHUNK_OVERLAP_CHARS: int = 800

# ---------------------------------------------------------------------------
# Marker appearance in Resolve (duration in frames as float; 1.0 is common)
# ---------------------------------------------------------------------------

MARKER_DURATION: float = 1.0

# ---------------------------------------------------------------------------
# Resolve module discovery (Windows install locations)
# ---------------------------------------------------------------------------

# DaVinci Resolve ships the `DaVinciResolveScript` module here. Studio and
# free Resolve use the same scripting folder layout on Windows.
RESOLVE_SCRIPT_MODULE_PATHS: tuple[str, ...] = (
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules",
    r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules",
)


def ensure_directories() -> None:
    """Create transcripts/ and logs/ if they are missing."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_STUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_openai_model() -> str:
    """Model name from environment, with a safe default."""
    return os.environ.get(ENV_OPENAI_MODEL, DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL


def get_transcript_bracket_fps() -> float:
    """
    FPS used when converting the FF field in [HH:MM:SS:FF - ...] to seconds.

    Override with TRANSCRIPT_BRACKET_FPS in .env (e.g. 23.976, 25, 29.97).
    """
    raw = os.environ.get(ENV_TRANSCRIPT_BRACKET_FPS, "").strip()
    if not raw:
        return DEFAULT_TRANSCRIPT_BRACKET_FPS
    try:
        v = float(raw)
        if v <= 0:
            return DEFAULT_TRANSCRIPT_BRACKET_FPS
        return v
    except ValueError:
        return DEFAULT_TRANSCRIPT_BRACKET_FPS
