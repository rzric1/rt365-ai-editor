# -*- coding: utf-8 -*-
"""Context manager for single-job execution in Clip Studio."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st

from clip_engine.job_control import (
    JobBusyError,
    JobCancelledError,
    release_job,
    try_acquire_job,
)
from clip_engine.stability import write_crash_report


@contextmanager
def studio_job(name: str) -> Iterator[None]:
    """Acquire global job lock; release on exit; surface busy/cancel to UI."""
    try:
        try_acquire_job(name)
        yield
    except JobBusyError as exc:
        st.error(str(exc))
        raise
    except JobCancelledError:
        st.warning("Operation cancelled.")
        raise
    except (TimeoutError, OSError) as exc:
        # Surface ffmpeg stall / timeout errors with actionable guidance.
        if name in ("transcribe", "audio_extract"):
            st.error(
                "**Audio extraction failed.** "
                + str(exc)
            )
            st.info(
                "If reading from a USB or network drive, copy the file to local storage (C:) "
                "first and try again."
            )
        try:
            write_crash_report(exc, context=name)
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            write_crash_report(exc, context=name)
        except Exception:
            pass
        raise
    finally:
        release_job(name)
