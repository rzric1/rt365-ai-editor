# -*- coding: utf-8 -*-
import os
import pytest
from unittest.mock import patch
from clip_engine.license_check import is_licensed, get_hardware_fingerprint, get_license_status

def test_licensed_when_enforcement_disabled():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", False):
        assert is_licensed() is True

def test_unlicensed_when_no_key_and_enforcement_on():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RT365_LICENSE_KEY", None)
            assert is_licensed() is False

def test_valid_key_format():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch.dict(os.environ, {"RT365_LICENSE_KEY": "RT365-AB12-CD34-EF56-GH78"}):
            assert is_licensed() is True

def test_invalid_key_format():
    with patch("clip_engine.license_check.LICENSE_ENFORCEMENT_ENABLED", True):
        with patch.dict(os.environ, {"RT365_LICENSE_KEY": "INVALID-KEY"}):
            assert is_licensed() is False

def test_hardware_fingerprint_stable():
    fp1 = get_hardware_fingerprint()
    fp2 = get_hardware_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 32

def test_license_status_dict_keys():
    status = get_license_status()
    assert "licensed" in status
    assert "trial" in status
    assert "hardware_fingerprint" in status
