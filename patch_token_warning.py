# -*- coding: utf-8 -*-
"""
patch_token_warning.py
Run from C:\\dev\\rt365-ai-editor with:
    .\.venv311\Scripts\python.exe patch_token_warning.py
"""

import re
import shutil
from pathlib import Path

TARGET = Path("clip_studio_app.py")

if not TARGET.exists():
    raise FileNotFoundError(f"Cannot find {TARGET} — run this from the project root.")

# Backup
backup = TARGET.with_suffix(".py.bak_token_patch")
shutil.copy2(TARGET, backup)
print(f"[OK] Backup written to {backup.name}")

src = TARGET.read_text(encoding="utf-8")
original = src  # keep for diff count

# ── PATCH 1: import line ──────────────────────────────────────────────────────
OLD_IMPORT = (
    "from clip_engine.openai_resilience import "
    "OpenAIRateLimitError, estimate_pipeline_tokens, token_saver_pass_config"
)
NEW_IMPORT = (
    "from clip_engine.openai_resilience import OpenAIRateLimitError, token_saver_pass_config\n"
    "from clip_engine.effective_config import plan_analysis_token_estimate"
)

if OLD_IMPORT in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("[OK] Patch 1: import line updated")
else:
    print("[SKIP] Patch 1: import line not found (may already be patched)")

# ── PATCH 2: internal estimate inside _run_clip_analysis (~line 275) ──────────
OLD_INTERNAL = """\
    n_passes = min(effective.max_gpt_passes, 3)
    max_rounds = 1 if token_saver else 2
    est = estimate_pipeline_tokens(
        formatted,
        target_count=target_count,
        n_passes=n_passes,
        max_pass_rounds=max_rounds,"""

NEW_INTERNAL = """\
    est = plan_analysis_token_estimate(
        formatted,
        effective.profile,
        target_count=target_count,
        clip_style=style_name,"""

if OLD_INTERNAL in src:
    src = src.replace(OLD_INTERNAL, NEW_INTERNAL, 1)
    print("[OK] Patch 2: internal _run_clip_analysis estimate replaced")
else:
    print("[SKIP] Patch 2: internal estimate block not found (may already be patched)")

# ── PATCH 3: Step 3 pre-run warning block (~line 921) ────────────────────────
OLD_WARNING = """\
            _prof = profile_from_ui_label(st.session_state.get("cs_ai_profile_label", ""))
            _n = min(_prof.max_gpt_passes, 3)
            _r = 1 if _prof.token_saver else 2
            _pre = estimate_pipeline_tokens(
                st.session_state.cs_formatted,
                target_count=int(st.session_state.get("cs_target_clips", 20)),
                n_passes=_n,
                max_pass_rounds=_r,
                token_saver_mode=_prof.token_saver,
            )
            _budget = _prof.max_tokens
            if _pre.estimated_total_tokens > _budget:
                st.warning(
                    f"This run may use approximately **{_pre.estimated_total_tokens:,}** tokens. "
                    f"Budget is **{_budget:,}**. Token Saver Mode will be enforced."
                )
            else:
                st.caption(
                    f"Estimated analysis tokens: ~{_pre.estimated_total_tokens:,} "
                    f"(budget {_budget:,}) | ~{_pre.estimated_calls} API calls"
                )"""

NEW_WARNING = """\
            _prof = profile_from_ui_label(st.session_state.get("cs_ai_profile_label", ""))
            _plan = plan_analysis_token_estimate(
                st.session_state.cs_formatted,
                _prof,
                target_count=int(st.session_state.get("cs_target_clips", 20)),
                clip_style=str(st.session_state.get("cs_clip_style", "Balanced")),
            )
            _budget = _prof.max_tokens
            _pruned = _plan.after_prune
            if _pruned > _budget:
                st.warning(
                    f"This run may use approximately **{_pruned:,}** tokens after pruning. "
                    f"Budget is **{_budget:,}**. Token Saver Mode will be enforced."
                )
            else:
                st.caption(
                    f"Estimated tokens (after GPU pruning): ~{_pruned:,} "
                    f"(budget {_budget:,}) | regions: {_plan.effective_regions}, passes: {_plan.effective_passes}"
                )"""

if OLD_WARNING in src:
    src = src.replace(OLD_WARNING, NEW_WARNING, 1)
    print("[OK] Patch 3: Step 3 pre-run warning block replaced")
else:
    print("[SKIP] Patch 3: warning block not found (may already be patched)")

# ── PATCH 4: analyze-button pre-run estimate (~line 1011) ────────────────────
OLD_ANALYZE = """\
                        _prof_run = profile_from_ui_label(st.session_state.get("cs_ai_profile_label", ""))
                        pre_est = estimate_pipeline_tokens(
                            st.session_state.cs_formatted,
                            target_count=int(st.session_state.get("cs_target_clips", 20)),
                            n_passes=min(_prof_run.max_gpt_passes, 3),
                            max_pass_rounds=1 if _prof_run.token_saver else 2,"""

NEW_ANALYZE = """\
                        _prof_run = profile_from_ui_label(st.session_state.get("cs_ai_profile_label", ""))
                        pre_est = plan_analysis_token_estimate(
                            st.session_state.cs_formatted,
                            _prof_run,
                            target_count=int(st.session_state.get("cs_target_clips", 20)),
                            clip_style=str(st.session_state.get("cs_clip_style", "Balanced")),"""

if OLD_ANALYZE in src:
    src = src.replace(OLD_ANALYZE, NEW_ANALYZE, 1)
    print("[OK] Patch 4: analyze-button estimate replaced")
else:
    print("[SKIP] Patch 4: analyze-button block not found (may already be patched)")

# ── Write result ──────────────────────────────────────────────────────────────
if src != original:
    TARGET.write_text(src, encoding="utf-8")
    print("\n[OK] clip_studio_app.py written.")
else:
    print("\n[INFO] No changes made — file may already be patched.")

# ── Verify no leftover references ─────────────────────────────────────────────
remaining = [i + 1 for i, line in enumerate(src.splitlines())
             if "estimate_pipeline_tokens" in line]
if remaining:
    print(f"\n[WARN] estimate_pipeline_tokens still found on lines: {remaining}")
    print("       Check those lines manually.")
else:
    print("[OK] No remaining estimate_pipeline_tokens references.")

# ── Quick compile check ───────────────────────────────────────────────────────
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "py_compile", str(TARGET)],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("[OK] clip_studio_app.py compiles cleanly.")
else:
    print(f"[ERROR] Compile failed:\n{result.stderr}")
    print(f"        Restore from backup: copy {backup.name} clip_studio_app.py")
