# RT365 AI Editor — Developer Notes

## Python Environment

Two venvs exist in this project:

| Venv       | Python | Status                        | Use for                        |
|------------|--------|-------------------------------|--------------------------------|
| `.venv311` | 3.11.9 | Fully configured, recommended | Production, GPU transcription  |
| `.venv`    | 3.14.x | May be missing some deps      | Development, testing           |

Always launch with `.venv311\Scripts\python.exe` for production GPU work.

The integrity check in `clip_engine/openai_resilience.py` validates OpenAI call routing.

Run `tests/test_stability_controls.py` after any changes to transcription or OpenAI logic.
