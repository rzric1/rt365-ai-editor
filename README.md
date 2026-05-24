# RT365 AI Editor (DaVinci Resolve Studio)

Safe **v1** helper for **YouTube podcasts and reaction-style edits**: it reads a transcript, asks OpenAI (Responses API) for editorial marker suggestions, and **only adds timeline markers** in DaVinci Resolve Studio.

It does **not** cut, delete, trim, ripple-delete, overwrite timelines, or modify the media pool.

## What you need

- **Windows 10/11**
- **DaVinci Resolve Studio** with **external scripting** enabled  
  **Resolve → Preferences → General → External scripting using → Local**
- **Python 3.10+** (3.12 recommended)
- An **OpenAI API key** with access to the model you set in `.env` (default: `gpt-5-mini`)

## Quick setup (Windows)

1. **Clone or copy** this folder to your machine, e.g. `c:\dev\rt365-ai-editor`.

2. **Create a virtual environment** (keeps dependencies isolated):

   ```bat
   cd c:\dev\rt365-ai-editor
   py -m venv .venv
   .venv\Scripts\activate
   py -m pip install --upgrade pip
   py -m pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bat
   copy .env.example .env
   ```

   Edit `.env`:

   - `OPENAI_API_KEY=` — your key from OpenAI
   - `OPENAI_MODEL=gpt-5-mini` — change if you use another model
   - `TRANSCRIPT_BRACKET_FPS=24` — used for `[HH:MM:SS:FF - ...]` transcripts (optional)

4. **Start DaVinci Resolve Studio**, open a **project** and **timeline** you want to mark.

5. **Smoke test the Resolve API** (adds one blue marker at ~10 seconds):

   ```bat
   py main.py --test-marker
   ```

6. **Put a transcript** in `transcripts\` as either:

   - `input.srt`, or
   - `input.json` (see format below)

7. **Dry run** (calls OpenAI, prints JSON, **does not** touch Resolve):

   ```bat
   py main.py --dry-run transcripts\input.srt
   ```

8. **Apply markers** to the **current** timeline:

   ```bat
   py main.py transcripts\input.srt
   ```

Logs are written under `logs\` with a timestamped filename.

### Interactive menu (no flags to memorize)

From the project folder:

```bat
py main.py --interactive
```

You get a numbered menu: test Resolve, debug transcript parsing, dry-run AI, apply markers (with a **y/n** safety prompt), show the latest log, or open the `transcripts` / `logs` folders in Explorer (Windows uses `os.startfile`). Transcript paths default to `transcripts\input.srt` when you press Enter.

All non-interactive commands (`--test-marker`, `--dry-run`, transcript path only, etc.) still work the same as before.

## RT365 AI Edit Companion (Streamlit)

The **Edit Companion** is a local, beginner-friendly web UI in your browser. It uses the same transcript loading, Resolve connection, marker placement, and OpenAI model settings as the CLI (`OPENAI_MODEL` from `.env`, default `gpt-5-mini`).

### Launch (Windows)

1. Activate your venv and install dependencies (includes Streamlit):

   ```bat
   cd c:\dev\rt365-ai-editor
   .venv\Scripts\activate
   py -m pip install -r requirements.txt
   ```

2. Start the app:

   ```bat
   streamlit run app.py
   ```

   Or double-click **`run_app.bat`** in the project folder (runs `streamlit run app.py`).

3. Your browser opens to the companion. Keep **DaVinci Resolve** open with the correct **timeline** when you use any **Add markers to Resolve** button.

### Using chat requests

- Set the **transcript path** in the sidebar (default `transcripts\input.srt`) or **upload** an `.srt`, `.txt`, or `.json` file.
- Use **Check Resolve connection** to confirm Resolve sees your project and timeline.
- Type a request in the chat box at the bottom, for example: *Give me chapters for this episode* or *Find good quotes*.
- The app **classifies** your intent (chapters, clip idea, quotes, possible cuts, audio hints, full marker pass, or general advice), shows results in **cards**, and saves each AI response as JSON under `logs\` (`companion_*.json`).
- When you see **Add markers to Resolve** (or **Add START / END markers** for a clip idea), click it only after Resolve is ready — the tool still **only calls AddMarker**; it never cuts, ripple-deletes, or changes the media pool.

## Transcript formats

### SRT

Standard SubRip `.srt` files are supported.

### Bracket timecode (Resolve-style)

If the first non-empty line looks like a range in **frames** (last field `FF`), the file is parsed as bracket blocks instead of classic SRT:

```text
[00:00:49:14 - 00:01:10:04]
Justin, how are you doing?
```

At **24 fps** (default), `49:14` means 49 seconds plus 14 frames, which is about **49.58** seconds. Set **`TRANSCRIPT_BRACKET_FPS`** in `.env` (for example `23.976` or `25`) if your export uses a different rate.

You can use a `.txt` file with the same layout; `.srt` is detected automatically when the content matches this pattern.

**Debug parse only (no OpenAI):** `py main.py --debug-transcript transcripts\input.srt`  
**Module self-test:** `py transcript_loader.py`

### JSON

Either a top-level array of segments or an object with a `segments` array:

```json
{
  "segments": [
    {
      "start_seconds": 0.0,
      "end_seconds": 3.4,
      "text": "Welcome back to the show..."
    }
  ]
}
```

Aliases `start` / `end` are accepted instead of `start_seconds` / `end_seconds`.

## Marker types

The model returns markers with one of these `marker_type` values:

| Type | Typical use |
|------|-------------|
| `HOT_TAKE` | Bold claim / spicy opinion |
| `STRONG_REACTION` | Big laugh, shock, strong emotion |
| `POSSIBLE_CUT` | Tangent / repetition — *hint only*, no auto-cut |
| `SHORT_CLIP` | Potential short vertical clip |
| `GOOD_QUOTE` | Memorable line |
| `CHAPTER` | Chapter / topic boundary |
| `AUDIO_DIP` | Possible level issue — *hint only* |

Each marker includes: `timestamp_seconds`, `marker_type`, `title`, `note`, and `confidence` (0–1).

## Marker alignment (transcript time → timeline frame)

Transcript timestamps are treated as **seconds from the start of the recording**, not from the start of the empty Resolve timeline.

When you place the podcast (or reaction) **after** some leader space, bars, or other clips, markers must land on the **main clip**, not at the timeline origin.

RT365 therefore:

1. Reads **project timeline frame rate** (`timelineFrameRate`) for seconds → frame conversion.
2. Scans **all video and all audio tracks** and finds the **earliest** `GetStart()` among every timeline item (smallest frame where any clip begins).
3. Places markers at:

   **`marker_alignment_frame + round(timestamp_seconds × timeline_fps)`**

   where `marker_alignment_frame` is that earliest clip start, or the **timeline start frame** if no video/audio clips exist (empty timeline).

Logs on each Resolve run include **timeline start frame**, **earliest clip start frame** (if any), and for the first marker the **final computed frame** plus alignment base and FPS.

Only `Timeline.AddMarker` is used — no clip edits.

## Colors in Resolve

Marker colors are chosen in `marker_writer.py` (e.g. `HOT_TAKE` → Red, `CHAPTER` → Blue). Resolve accepts named colors such as `Blue`, `Red`, `Green`, `Yellow`, `Cyan`, `Purple`, `Magenta`.

## Troubleshooting

- **`Could not find DaVinciResolveScript`**  
  Install Resolve Studio and/or adjust `RESOLVE_SCRIPT_MODULE_PATHS` in `config.py` to match your install path.

- **`scriptapp('Resolve') returned None`**  
  Resolve is not running, scripting is disabled, or Resolve is busy — restart Resolve, enable **Local** scripting, try again.

- **Markers appear at the wrong time**  
  Alignment uses the **earliest video/audio clip start** on the current timeline plus `timestamp_seconds` at project FPS. If the wrong clip is earliest (e.g. a short bumper before the episode), move that clip or trim the timeline so the main program clip is the alignment you want — or adjust transcript timestamps. For an empty timeline, the tool falls back to the timeline start frame.

- **Structured output errors from OpenAI**  
  Ensure your `OPENAI_MODEL` supports JSON schema structured outputs on the Responses API; otherwise switch to a current GPT‑5 class model or update the schema per OpenAI docs.

## Safety stance

This project is aimed at **stability first**: the only Resolve write in normal use is `Timeline.AddMarker`. As you extend the repo, keep destructive APIs out of the default path and review diffs carefully.

## License

Use and modify for your own workflow. DaVinci Resolve is a trademark of Blackmagic Design. OpenAI is a separate service with its own terms.
