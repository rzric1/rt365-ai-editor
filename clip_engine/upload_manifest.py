# -*- coding: utf-8 -*-
"""
clip_engine/upload_manifest.py
Deduplicate browser uploads via content fingerprint + manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, UPLOADS_DIR

logger = logging.getLogger("clip_engine.upload_manifest")

CHUNK_HASH_SIZE = 8 * 1024 * 1024
MANIFEST_PATH = UPLOADS_DIR / "upload_manifest.json"
DUPLICATES_DIR = UPLOADS_DIR / "_duplicates"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
MANIFEST_VERSION = 1


def _reset_upload_stream(upload: Any) -> None:
    if hasattr(upload, "seek"):
        try:
            upload.seek(0)
        except Exception:
            pass


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _partial_hashes_from_bytes(data: bytes) -> tuple[str, str]:
    size = len(data)
    first_end = min(size, CHUNK_HASH_SIZE)
    first_hash = _hash_bytes(data[:first_end])
    if size > CHUNK_HASH_SIZE:
        last_hash = _hash_bytes(data[-CHUNK_HASH_SIZE:])
    else:
        last_hash = ""
    return first_hash, last_hash


def _fingerprint_from_parts(
    *,
    original_name: str,
    size_bytes: int,
    first_hash: str,
    last_hash: str,
) -> str:
    parts = hashlib.sha256()
    parts.update(original_name.encode("utf-8"))
    parts.update(b"\0")
    parts.update(str(size_bytes).encode())
    parts.update(b"\0")
    parts.update(first_hash.encode())
    if last_hash:
        parts.update(b"\0")
        parts.update(last_hash.encode())
    return parts.hexdigest()


def compute_upload_fingerprint(upload: Any) -> tuple[str, int, str]:
    """
    Stable fingerprint: original name, size, SHA256 of first/last 8 MB.
    Streams chunks — does not load the entire upload into RAM.
    Returns (fingerprint, size_bytes, original_name).
    """
    _reset_upload_stream(upload)
    original_name = Path(upload.name).name
    size_bytes = 0
    first_buf = bytearray()
    last_buf = bytearray()
    read_chunk = 1024 * 1024

    while True:
        block = upload.read(read_chunk)
        if not block:
            break
        size_bytes += len(block)
        offset = 0
        if len(first_buf) < CHUNK_HASH_SIZE:
            need = CHUNK_HASH_SIZE - len(first_buf)
            take = block[:need]
            first_buf.extend(take)
            offset = len(take)
        remainder = block[offset:]
        if remainder:
            last_buf.extend(remainder)
            if len(last_buf) > CHUNK_HASH_SIZE:
                last_buf = last_buf[-CHUNK_HASH_SIZE:]

    _reset_upload_stream(upload)
    first_hash = _hash_bytes(bytes(first_buf))
    last_hash = _hash_bytes(bytes(last_buf)) if size_bytes > CHUNK_HASH_SIZE else ""

    fp = _fingerprint_from_parts(
        original_name=original_name,
        size_bytes=size_bytes,
        first_hash=first_hash,
        last_hash=last_hash,
    )
    return fp, size_bytes, original_name


def compute_file_content_key(path: Path) -> str:
    """Content key for duplicate grouping (size + partial hashes, no filename)."""
    size = path.stat().st_size
    with path.open("rb") as f:
        first = f.read(CHUNK_HASH_SIZE)
    if size > CHUNK_HASH_SIZE:
        with path.open("rb") as f:
            f.seek(max(0, size - CHUNK_HASH_SIZE))
            last = f.read(CHUNK_HASH_SIZE)
        first_hash = _hash_bytes(first)
        last_hash = _hash_bytes(last)
    else:
        first_hash, last_hash = _partial_hashes_from_bytes(first)
    return _fingerprint_from_parts(
        original_name="",
        size_bytes=size,
        first_hash=first_hash,
        last_hash=last_hash,
    )


def _normalize_saved_path(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


def _resolve_saved_path(saved_path: str) -> Path:
    p = Path(saved_path)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def load_upload_manifest() -> dict[str, Any]:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.is_file():
        return {"version": MANIFEST_VERSION, "entries": []}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read upload manifest: %s", exc)
        return {"version": MANIFEST_VERSION, "entries": []}
    if not isinstance(data, dict):
        return {"version": MANIFEST_VERSION, "entries": []}
    entries = data.get("entries")
    if not isinstance(entries, list):
        data["entries"] = []
    data.setdefault("version", MANIFEST_VERSION)
    return data


def save_upload_manifest(manifest: dict[str, Any]) -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    manifest["version"] = MANIFEST_VERSION
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def find_existing_upload(manifest: dict[str, Any], fingerprint: str) -> dict[str, Any] | None:
    for entry in manifest.get("entries", []):
        if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint:
            return entry
    return None


def _upsert_manifest_entry(
    manifest: dict[str, Any],
    *,
    fingerprint: str,
    original_name: str,
    saved_path: Path,
    size_bytes: int,
) -> None:
    rel = _normalize_saved_path(saved_path)
    now = datetime.now(timezone.utc).isoformat()
    entries = manifest.setdefault("entries", [])
    for entry in entries:
        if entry.get("fingerprint") == fingerprint:
            entry.update(
                {
                    "original_name": original_name,
                    "saved_path": rel,
                    "size_bytes": size_bytes,
                    "fingerprint": fingerprint,
                    "updated_at": now,
                }
            )
            break
    else:
        entries.append(
            {
                "fingerprint": fingerprint,
                "original_name": original_name,
                "saved_path": rel,
                "size_bytes": size_bytes,
                "created_at": now,
            }
        )
    save_upload_manifest(manifest)


def write_upload_to_path(upload: Any, dest: Path, progress_bar: Any) -> None:
    """Write uploaded bytes to dest with optional progress updates (streamed, low RAM)."""
    from clip_engine.job_control import check_cancelled

    _reset_upload_stream(upload)
    chunk = 4 * 1024 * 1024
    written = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while True:
            check_cancelled()
            piece = upload.read(chunk)
            if not piece:
                break
            f.write(piece)
            written += len(piece)
            if progress_bar is not None:
                try:
                    progress_bar.progress(
                        0.0,
                        text=f"Saving: {_format_size(written)}…",
                    )
                except TypeError:
                    progress_bar.progress(0.0)
    if progress_bar is not None:
        try:
            progress_bar.progress(1.0, text=f"Saved {_format_size(written)}.")
        except TypeError:
            progress_bar.progress(1.0)
    _reset_upload_stream(upload)


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / (1024**3):.2f} GB"
    if n >= 1024**2:
        return f"{n / (1024**2):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} bytes"


def save_upload_once(upload: Any, *, progress_bar: Any) -> tuple[Path, bool]:
    """
    Save browser upload once per fingerprint.
    Returns (saved_path, reused_existing).
    """
    _reset_upload_stream(upload)
    fingerprint, size_bytes, original_name = compute_upload_fingerprint(upload)
    manifest = load_upload_manifest()
    entry = find_existing_upload(manifest, fingerprint)

    if entry:
        existing = _resolve_saved_path(str(entry.get("saved_path", "")))
        if existing.is_file():
            logger.info(
                "Reusing upload fingerprint=%s path=%s",
                fingerprint[:12],
                existing,
            )
            return existing, True
        logger.warning(
            "Manifest entry missing file fingerprint=%s path=%s — re-saving",
            fingerprint[:12],
            existing,
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(original_name).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        ext = ".mp4"
    dest = UPLOADS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    write_upload_to_path(upload, dest, progress_bar)
    _upsert_manifest_entry(
        manifest,
        fingerprint=fingerprint,
        original_name=original_name,
        saved_path=dest,
        size_bytes=size_bytes,
    )
    return dest.resolve(), False


def clean_duplicate_uploads() -> dict[str, Any]:
    """
    Move duplicate video files in uploads/ to uploads/_duplicates/.
    Keeps manifest-referenced file when possible, else oldest by mtime.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    DUPLICATES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_upload_manifest()
    manifest_paths = {
        _resolve_saved_path(str(e.get("saved_path", "")))
        for e in manifest.get("entries", [])
        if e.get("saved_path")
    }

    files = [
        p.resolve()
        for p in UPLOADS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    groups: dict[str, list[Path]] = {}
    for path in files:
        try:
            key = compute_file_content_key(path)
        except OSError as exc:
            logger.warning("Skip fingerprint %s: %s", path, exc)
            continue
        groups.setdefault(key, []).append(path)

    moved = 0
    bytes_saved = 0
    groups_affected = 0

    for paths in groups.values():
        if len(paths) < 2:
            continue
        groups_affected += 1
        keep: Path | None = None
        for p in paths:
            if p in manifest_paths:
                keep = p
                break
        if keep is None:
            paths_sorted = sorted(paths, key=lambda p: p.stat().st_mtime)
            keep = paths_sorted[0]
        else:
            paths_sorted = paths

        for dup in paths:
            if dup == keep:
                continue
            dest = DUPLICATES_DIR / dup.name
            if dest.exists():
                dest = DUPLICATES_DIR / f"{dup.stem}_{uuid.uuid4().hex[:6]}{dup.suffix}"
            try:
                size = dup.stat().st_size
                shutil.move(str(dup), str(dest))
                moved += 1
                bytes_saved += size
                logger.info("Moved duplicate upload %s -> %s", dup.name, dest.name)
            except OSError as exc:
                logger.warning("Could not move %s: %s", dup, exc)

    return {
        "moved": moved,
        "bytes_saved": bytes_saved,
        "groups": groups_affected,
    }
