# -*- coding: utf-8 -*-
import os
import json
import pytest
from unittest.mock import patch, MagicMock
from clip_engine.license_check import (
    is_licensed,
    get_hardware_fingerprint,
    get_instance_id,
    get_license_status,
    validate_license_key,
    load_cache,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_valid_response():
    """Return a mock requests.Response that looks like a successful validation."""
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"valid": True, "email": "buyer@example.com"}
    return m


def _mock_invalid_response(reason="Invalid license key."):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"valid": False, "error": reason}
    return m


# ── Enforcement disabled ──────────────────────────────────────────────────────

def test_licensed_when_enforcement_disabled():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", False):
        assert is_licensed() is True


# ── No key present ────────────────────────────────────────────────────────────

def test_unlicensed_when_no_key_and_enforcement_on():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch("clip_engine.license_check.load_cache", return_value=None):
            with patch("clip_engine.license_check.check_master_override", return_value=False):
                os.environ.pop("RT365_LICENSE_KEY", None)
                assert is_licensed() is False


# ── Valid key — remote validation succeeds ────────────────────────────────────

def test_valid_key_accepted_via_remote():
    """A well-formed key passes when the remote API returns valid=true."""
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch("clip_engine.license_check.load_cache", return_value=None):
            with patch("clip_engine.license_check.check_master_override", return_value=False):
                with patch("clip_engine.license_check.save_cache"):
                    with patch("requests.post", return_value=_mock_valid_response()):
                        with patch.dict(os.environ, {
                            "RT365_LICENSE_KEY": "RT365-AB12-CD34-EF56-GH78",
                            "RT365_API_URL": "https://example.vercel.app",
                        }):
                            assert is_licensed() is True


# ── Invalid key format ────────────────────────────────────────────────────────

def test_invalid_key_format_rejected():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch("clip_engine.license_check.load_cache", return_value=None):
            with patch("clip_engine.license_check.check_master_override", return_value=False):
                with patch.dict(os.environ, {
                    "RT365_LICENSE_KEY": "INVALID-KEY",
                    "RT365_API_URL": "https://example.vercel.app",
                }):
                    assert is_licensed() is False


# ── validate_license_key unit tests ──────────────────────────────────────────

def test_validate_returns_true_on_valid_remote():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch("clip_engine.license_check.load_cache", return_value=None):
            with patch("clip_engine.license_check.check_master_override", return_value=False):
                with patch("clip_engine.license_check.save_cache"):
                    with patch("requests.post", return_value=_mock_valid_response()):
                        with patch.dict(os.environ, {"RT365_API_URL": "https://example.vercel.app"}):
                            ok, msg = validate_license_key("RT365-AB12-CD34-EF56-GH78")
                            assert ok is True
                            assert msg == "buyer@example.com"


def test_validate_returns_false_on_invalid_remote():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch("clip_engine.license_check.load_cache", return_value=None):
            with patch("clip_engine.license_check.check_master_override", return_value=False):
                with patch("requests.post", return_value=_mock_invalid_response()):
                    with patch.dict(os.environ, {"RT365_API_URL": "https://example.vercel.app"}):
                        ok, msg = validate_license_key("RT365-AB12-CD34-EF56-GH78")
                        assert ok is False
                        assert "Invalid" in msg


def test_validate_returns_false_when_api_url_missing():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch("clip_engine.license_check.load_cache", return_value=None):
            with patch("clip_engine.license_check.check_master_override", return_value=False):
                env = {k: v for k, v in os.environ.items() if k != "RT365_API_URL"}
                with patch.dict(os.environ, env, clear=True):
                    ok, msg = validate_license_key("RT365-AB12-CD34-EF56-GH78")
                    assert ok is False
                    assert "RT365_API_URL" in msg


def test_validate_uses_cache_when_available():
    cached = {"key_hash": "abc", "email": "cached@example.com", "expiry": "2099-01-01T00:00:00"}
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch("clip_engine.license_check.load_cache", return_value=cached):
            ok, msg = validate_license_key("RT365-AB12-CD34-EF56-GH78")
            assert ok is True
            assert msg == "cached@example.com"


def test_validate_dev_mode_bypasses_all():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", False):
        ok, msg = validate_license_key("ANYTHING")
        assert ok is True
        assert msg == "dev_mode"


def test_validate_master_override():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch("clip_engine.license_check.check_master_override", return_value=True):
            ok, msg = validate_license_key("RT365-AB12-CD34-EF56-GH78")
            assert ok is True
            assert msg == "master_override"


# ── Hardware fingerprint ──────────────────────────────────────────────────────

def test_hardware_fingerprint_stable():
    """get_hardware_fingerprint is an alias for get_instance_id — both must be stable."""
    fp1 = get_hardware_fingerprint()
    fp2 = get_hardware_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 32
    # Alias must point to the same function
    assert get_hardware_fingerprint is get_instance_id


# ── get_license_status ────────────────────────────────────────────────────────

def test_license_status_dict_keys():
    status = get_license_status()
    assert "licensed" in status
    assert "trial" in status
    assert "hardware_fingerprint" in status
    assert "cache_valid" in status
    assert "enforcement_enabled" in status
