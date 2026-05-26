"""
clip_engine/semantic_ranking.py
GPU-accelerated semantic embeddings for transcript ranking and dedupe.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("clip_engine.semantic_ranking")

_MODEL = None
_MODEL_DEVICE = "cpu"
_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def get_embedding_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def get_gpu_device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _load_model():
    global _MODEL, _MODEL_DEVICE
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers not installed. "
            "Install: pip install -r requirements-ai-upgrades.txt"
        ) from exc

    _MODEL_DEVICE = get_embedding_device()
    _MODEL = SentenceTransformer(_MODEL_NAME, device=_MODEL_DEVICE)
    logger.info(
        "[GPU] Sentence-transformers using CUDA: %s | Device: %s | Model: %s",
        _MODEL_DEVICE == "cuda",
        get_gpu_device_name() if _MODEL_DEVICE == "cuda" else _MODEL_DEVICE,
        _MODEL_NAME,
    )
    return _MODEL


def generate_embeddings(texts: list[str], *, batch_size: int = 64) -> np.ndarray:
    """Encode texts to normalized embedding matrix (n, dim)."""
    if not texts:
        return np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
    model = _load_model()
    clean = [t.strip() or "(empty)" for t in texts]
    emb = model.encode(
        clean,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def semantic_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def cluster_segments(
    embeddings: np.ndarray,
    *,
    similarity_threshold: float = 0.82,
) -> list[list[int]]:
    """
    Greedy clustering by cosine similarity.
    Returns list of index groups.
    """
    n = len(embeddings)
    if n == 0:
        return []
    assigned = [-1] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if assigned[i] >= 0:
            continue
        group = [i]
        assigned[i] = len(clusters)
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            sim = semantic_similarity(embeddings[i], embeddings[j])
            if sim >= similarity_threshold:
                group.append(j)
                assigned[j] = len(clusters)
        clusters.append(group)
    return clusters


def dedupe_by_embedding_similarity(
    candidates: list[dict],
    texts: list[str],
    *,
    similarity_threshold: float = 0.88,
) -> tuple[list[dict], int]:
    """
    Remove near-duplicate candidates using embedding cosine similarity.
    Keeps highest composite_score per cluster.
    """
    if len(candidates) <= 1 or not embeddings_available():
        return candidates, 0
    try:
        emb = generate_embeddings(texts)
    except Exception as exc:
        logger.warning("Embedding dedupe skipped: %s", exc)
        return candidates, 0

    sorted_idx = sorted(
        range(len(candidates)),
        key=lambda i: int(candidates[i].get("composite_score", 0)),
        reverse=True,
    )
    kept: list[dict] = []
    kept_emb: list[np.ndarray] = []
    removed = 0

    for i in sorted_idx:
        vec = emb[i]
        if any(semantic_similarity(vec, k) >= similarity_threshold for k in kept_emb):
            removed += 1
            continue
        kept.append(candidates[i])
        kept_emb.append(vec)

    kept.sort(key=lambda c: float(c.get("start_seconds", 0)))
    return kept, removed


def rank_candidate_segments(
    candidates: list[dict],
    texts: list[str],
    *,
    top_k: int = 60,
    query_text: str | None = None,
) -> list[dict]:
    """Rank candidates by embedding relevance; return top_k."""
    if not candidates:
        return []
    if not embeddings_available():
        return candidates[:top_k]

    try:
        cand_emb = generate_embeddings(texts)
        if query_text:
            q_emb = generate_embeddings([query_text])[0]
            scores = [semantic_similarity(cand_emb[i], q_emb) for i in range(len(candidates))]
        else:
            centroid = cand_emb.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            scores = [semantic_similarity(cand_emb[i], centroid) for i in range(len(candidates))]
    except Exception as exc:
        logger.warning("Embedding rank skipped: %s", exc)
        return candidates[:top_k]

    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda pair: (
            scores[pair[0]]
            + int(pair[1].get("composite_score", 0)) / 200.0
            + float(pair[1].get("local_rank_score", 0)) / 100.0,
        ),
        reverse=True,
    )
    return [c for _, c in indexed[:top_k]]


def candidate_window_text(c: dict, segments: list[dict]) -> str:
    from clip_engine.transcription_utils import extract_transcript_window

    t0 = float(c.get("start_seconds", 0))
    t1 = float(c.get("end_seconds", t0))
    return extract_transcript_window(segments, t0, t1)


def semantic_pipeline_status() -> dict[str, Any]:
    return {
        "embeddings_available": embeddings_available(),
        "cuda_available": cuda_available(),
        "device": get_embedding_device(),
        "gpu_name": get_gpu_device_name(),
        "model": _MODEL_NAME,
        "loaded": _MODEL is not None,
    }
