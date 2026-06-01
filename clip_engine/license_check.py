# -*- coding: utf-8 -*-
"""License validation for RT365 AI Clip Studio."""
from __future__ import annotations

import hashlib
import os
import re
import uuid

LICENSE_ENFORCEMENT_ENABLED: bool = False
TRIAL_EXPORT_LIMIT: int = 3

_KEY_PATTERN = re.compile(r"^RT365-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$")


def get_hardware_fingerprint() -> str:
    """Return a stable 32-char hex fingerprint derived from machine identifiers."""
    try:
        node = uuid.getnode()
    except Exception:
        node = 0
    raw = f"rt365-{node}"
    return hashlib.md5(raw.encode()).hexdigest()


def is_licensed() -> bool:
    """Return True if the app is licensed for use."""
    if not LICENSE_ENFORCEMENT_ENABLED:
        return True
    key = os.environ.get("RT365_LICENSE_KEY", "").strip()
    if not key:
        return False
    return bool(_KEY_PATTERN.match(key))


def get_license_status() -> dict:
    """Return a dict describing the current license state."""
    key = os.environ.get("RT365_LICENSE_KEY", "").strip()
    licensed = is_licensed()
    return {
        "licensed": licensed,
        "trial": not licensed,
        "hardware_fingerprint": get_hardware_fingerprint(),
        "key_present": bool(key),
        "enforcement_enabled": LICENSE_ENFORCEMENT_ENABLED,
        "enforcement_active": LICENSE_ENFORCEMENT_ENABLED,
    }
