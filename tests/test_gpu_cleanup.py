# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


def test_gpu_cleanup_log(tmp_path, monkeypatch):
    from clip_engine import gpu_cleanup

    monkeypatch.setattr(gpu_cleanup, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(gpu_cleanup, "GPU_CLEANUP_LOG", tmp_path / "gpu_cleanup.log")
    gpu_cleanup.cleanup_gpu_after_phase("unit_test", whisper=False, yolo=False)
    assert (tmp_path / "gpu_cleanup.log").is_file()
    assert "unit_test" in (tmp_path / "gpu_cleanup.log").read_text(encoding="utf-8")


def test_release_yolo_no_crash():
    from clip_engine.smart_crop import release_yolo_model

    release_yolo_model()


def test_release_embeddings_no_crash():
    from clip_engine.semantic_ranking import release_embedding_model

    release_embedding_model()
