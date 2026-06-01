# -*- coding: utf-8 -*-
"""Generate assets/ai_clip_studio.ico for Windows shortcuts (requires Pillow)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed; skipping icon generation.", file=sys.stderr)
        return 0

    out_dir = _ROOT / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ai_clip_studio.ico"

    w = h = 64
    im = Image.new("RGBA", (w, h), (18, 24, 38, 255))
    dr = ImageDraw.Draw(im)
    margin = 4
    dr.rounded_rectangle(
        [margin, margin, w - margin, h - margin],
        radius=8,
        outline=(120, 200, 255, 255),
        width=2,
    )
    cx, cy = w * 0.52, h * 0.5
    r = w * 0.22
    dr.polygon(
        [
            (cx - r * 0.55, cy - r * 0.85),
            (cx - r * 0.55, cy + r * 0.85),
            (cx + r * 0.95, cy),
        ],
        fill=(255, 255, 255, 230),
    )
    im.save(out, format="ICO", sizes=[(64, 64)])
    print(f"Wrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
