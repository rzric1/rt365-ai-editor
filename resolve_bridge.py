from __future__ import annotations
import json, os, sys, re

def setup_resolve_env():
    programdata = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
    api = os.path.join(programdata, "Blackmagic Design", "DaVinci Resolve", "Support", "Developer", "Scripting")
    os.environ.setdefault("RESOLVE_SCRIPT_API", api)
    os.environ.setdefault(
        "RESOLVE_SCRIPT_LIB",
        r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll",
    )
    modules = r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules"
    if modules not in sys.path:
        sys.path.insert(0, modules)

def seconds_to_frames(seconds, fps):
    return int(round(seconds * fps))

def frames_to_tc(frames, fps):
    nominal = round(fps)
    f = frames % nominal
    total_secs = frames // nominal
    s = total_secs % 60
    m = (total_secs // 60) % 60
    h = total_secs // 3600
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

def connect():
    import DaVinciResolveScript as dvr
    resolve = dvr.scriptapp("Resolve")
    if not resolve:
        raise RuntimeError("Could not connect to DaVinci Resolve. Make sure Resolve Studio is open and Preferences > System > General > Enable Fusion page scripting is checked.")
    return resolve

def build_timeline(resolve, payload):
    source_path   = payload["source_path"]
    clips         = payload["clips"]
    fps           = float(payload.get("fps", 30.0))
    handle_secs   = float(payload.get("handle_seconds", 2.0))
    timeline_name = payload.get("timeline_name", "AI Clips")
    project_fps   = payload.get("project_fps", "30")
    color_tag     = payload.get("color_tag", "Blue")
    log = []

    pm      = resolve.GetProjectManager()
    project = pm.GetCurrentProject()
    if not project:
        project = pm.CreateProject("AI Clip Studio")
        log.append("Created new Resolve project: AI Clip Studio")
    else:
        log.append(f"Using existing project: {project.GetName()}")

    project.SetSetting("timelineFrameRate", project_fps)
    project.SetSetting("timelineResolutionWidth",  "1920")
    project.SetSetting("timelineResolutionHeight", "1080")
    log.append(f"Project settings: {project_fps}fps, 1920x1080")

    media_pool = project.GetMediaPool()
    root       = media_pool.GetRootFolder()
    bin_name   = f"AI Clips — {os.path.basename(source_path)}"
    source_bin = media_pool.AddSubFolder(root, bin_name)
    media_pool.SetCurrentFolder(source_bin)
    log.append(f"Created bin: {bin_name}")

    imported = media_pool.ImportMedia([source_path])
    if not imported:
        raise RuntimeError(f"Could not import source video into Resolve Media Pool.\nPath: {source_path}")
    source_clip = imported[0]
    log.append(f"Imported source: {os.path.basename(source_path)}")

    kept = [c for c in clips if c.get("finalizer_action","kept") not in ("rejected","rejected_before_ui") and c.get("start_time") is not None and c.get("end_time") is not None]
    log.append(f"Clips to place: {len(kept)} of {len(clips)} total")
    if not kept:
        raise RuntimeError("No kept clips found in payload.")

    timeline = media_pool.CreateEmptyTimeline(timeline_name)
    if not timeline:
        raise RuntimeError(f"CreateEmptyTimeline returned None. A timeline named '{timeline_name}' may already exist.")
    project.SetCurrentTimeline(timeline)
    log.append(f"Timeline created: '{timeline_name}'")

    marker_frame = 0
    for i, clip in enumerate(kept):
        src_in  = max(0.0, float(clip["start_time"]) - handle_secs)
        src_out = float(clip["end_time"]) + handle_secs
        sf      = seconds_to_frames(src_in,  fps)
        ef      = seconds_to_frames(src_out, fps) - 1
        dur     = ef - sf + 1
        appended = media_pool.AppendToTimeline([{
            "mediaPoolItem": source_clip,
            "startFrame": sf,
            "endFrame": ef,
        }])
        if not appended:
            raise RuntimeError(f"AppendToTimeline failed for clip {i + 1:02d}.")
        log.append(f"  Clip {i+1:02d}: {frames_to_tc(sf,fps)} -> {frames_to_tc(ef,fps)}  ({src_out-src_in:.1f}s)  \"{clip.get('hook_title','')[:50]}\"")
        hook    = clip.get("hook_title") or clip.get("title") or f"Clip {i+1:02d}"
        score   = float(clip.get("score") or clip.get("virality_score") or 0)
        note    = f"Score: {score:.1f} | {clip.get('finalizer_action','kept')}"
        mc      = "Green" if score >= 75 else ("Yellow" if score >= 50 else color_tag)
        tl_frame = seconds_to_frames(3600.0, fps) + marker_frame
        timeline.AddMarker(tl_frame, mc, hook[:30], note, 1, f"ai_clip_{i+1:02d}")
        marker_frame += dur
    log.append(f"Appended {len(kept)} clip(s) to timeline")
    log.append(f"Added {len(kept)} markers (Green>=75, Yellow>=50, Blue=other)")

    timeline.SetStartTimecode("01:00:00:00")

    resolve.OpenPage("cut")
    log.append("Switched to Cut page.")
    pm.SaveProject()
    log.append("Project saved.")
    return {"status": "ok", "timeline_name": timeline_name, "clips_placed": len(kept), "log": log}

if __name__ == "__main__":
    setup_resolve_env()
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Failed to parse payload: {e}"}))
        sys.exit(1)
    try:
        resolve = connect()
        result  = build_timeline(resolve, payload)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
