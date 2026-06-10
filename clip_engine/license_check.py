# -*- coding: utf-8 -*-
"""
clip_engine/license_check.py
License validation for RT365 AI Clip Studio.

Flow:
  1. If running from the developer workspace (C:\\dev\\rt365-ai-editor) → always valid.
  2. If LICENSE_ENFORCEMENT_ENABLED is False → always valid (dev mode).
  3. If RT365_MASTER_KEY env var is set → always valid (internal override).
  4. If a valid local cache exists (not expired) → valid without network call.
  5. Otherwise → POST to /api/validate-license, cache a successful result for
     CACHE_DAYS days so the user can work offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger("clip_engine.license_check")

_DEV_WORKSPACE_ROOT = Path(r"C:\dev\rt365-ai-editor")


def _running_from_dev_workspace() -> bool:
    """True when this checkout is running from the owner's local dev tree."""
    try:
        root = _DEV_WORKSPACE_ROOT.resolve()
        sources = (Path(__file__).resolve(), Path.cwd().resolve())
        return any(p == root or root in p.parents for p in sources)
    except Exception:
        return False


_LICENSE_ENFORCEMENT_DEFAULT = not _running_from_dev_workspace()
if not _LICENSE_ENFORCEMENT_DEFAULT:
    logger.info("[license] Dev workspace detected — license enforcement disabled")

LICENSE_ENFORCEMENT_ENABLED: bool = _LICENSE_ENFORCEMENT_DEFAULT
CACHE_PATH: Path = Path.home() / ".rt365" / "license.cache"
CACHE_DAYS: int = 30
TRIAL_EXPORT_LIMIT: int = 3
_KEY_PATTERN = re.compile(r"^RT365-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$")


# ── Machine identity ──────────────────────────────────────────────────────────

def get_instance_id() -> str:
    """Return a stable 32-char hex ID derived from MAC address + hostname."""
    try:
        node = uuid.getnode()
    except Exception:
        node = 0
    try:
        host = socket.gethostname()
    except Exception:
        host = "unknown"
    raw = f"{node}-{host}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ── Master / dev overrides ────────────────────────────────────────────────────

def check_master_override() -> bool:
    """Return True if the internal master key env var is set (bypasses normal check)."""
    master = os.getenv("RT365_MASTER_KEY", "").strip()
    return bool(master and len(master) > 8)


# ── Local cache ───────────────────────────────────────────────────────────────

def load_cache() -> dict | None:
    """
    Return the cached license data if the cache exists and has not expired.
    Returns None if missing, malformed, or stale.
    """
    try:
        if not CACHE_PATH.exists():
            return None
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        expiry = datetime.fromisoformat(data["expiry"])
        if datetime.now() < expiry:
            return data
        logger.debug("[license] Cache expired")
    except Exception as exc:
        logger.debug("[license] Cache read failed: %s", exc)
    return None


def save_cache(license_key: str, email: str) -> None:
    """Persist a successful validation result for CACHE_DAYS days."""
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        key_hash = hashlib.sha256(license_key.encode()).hexdigest()
        data = {
            "key_hash": key_hash,
            "email": email,
            "expiry": (datetime.now() + timedelta(days=CACHE_DAYS)).isoformat(),
        }
        CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
        logger.debug("[license] Cache saved, expires %s", data["expiry"])
    except Exception as exc:
        logger.warning("[license] Could not save cache: %s", exc)


def clear_cache() -> None:
    """Delete the local license cache (force re-validation on next launch)."""
    try:
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
    except Exception as exc:
        logger.warning("[license] Could not clear cache: %s", exc)


# ── Remote validation ─────────────────────────────────────────────────────────

def validate_license_key(license_key: str) -> tuple[bool, str]:
    """
    Validate *license_key* and return (valid: bool, message: str).

    The message is the customer email on success, or a user-facing error on failure.
    """
    if not LICENSE_ENFORCEMENT_ENABLED:
        return True, "dev_mode"

    if check_master_override():
        return True, "master_override"

    cached = load_cache()
    if cached:
        return True, cached.get("email", "cached")

    # Basic format check before hitting the network
    normalised = license_key.strip().upper()
    if not _KEY_PATTERN.match(normalised):
        return False, "Invalid license key format. Keys look like RT365-XXXX-XXXX-XXXX-XXXX."

    api_url = os.getenv("RT365_API_URL", "").rstrip("/")
    if not api_url:
        return (
            False,
            "RT365_API_URL is not configured. Add it to your .env file "
            "(e.g. RT365_API_URL=https://your-app.vercel.app).",
        )

    try:
        resp = requests.post(
            f"{api_url}/api/validate-license",
            json={"license_key": normalised, "instance_id": get_instance_id()},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return False, "License validation timed out. Check your internet connection and try again."
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the license server. Check your internet connection."
    except requests.exceptions.HTTPError as exc:
        return False, f"License server returned an error ({exc.response.status_code}). Try again later."
    except Exception as exc:
        return False, f"Validation error: {exc}"

    if data.get("valid"):
        save_cache(normalised, data.get("email", ""))
        return True, data.get("email", "")

    return False, data.get("error", "Invalid license key.")


# ── Legacy compatibility shim ─────────────────────────────────────────────────

# Alias kept for backwards-compatibility with tests and any external callers.
get_hardware_fingerprint = get_instance_id


def is_licensed() -> bool:
    """
    Quick Boolean check used by the legacy gate.
    Prefer validate_license_key() when you need the error message.
    """
    if not LICENSE_ENFORCEMENT_ENABLED:
        return True
    if check_master_override():
        return True
    if load_cache():
        return True
    key = os.environ.get("RT365_LICENSE_KEY", "").strip()
    if not key:
        return False
    valid, _ = validate_license_key(key)
    return valid


def get_license_status() -> dict:
    """Return a status dict for diagnostics / settings UI."""
    key = os.environ.get("RT365_LICENSE_KEY", "").strip()
    cached = load_cache()
    return {
        "licensed": is_licensed(),
        "trial": not is_licensed(),
        "hardware_fingerprint": get_instance_id(),
        "key_present": bool(key),
        "cache_valid": cached is not None,
        "cache_email": cached.get("email") if cached else None,
        "enforcement_enabled": LICENSE_ENFORCEMENT_ENABLED,
        "enforcement_active": LICENSE_ENFORCEMENT_ENABLED,
    }
