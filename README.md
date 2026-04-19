# ND2 analysis tool

Analyze your time lapse data straight from the microscope!

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone this repo and `cd` into it.
3. `uv run gui`

uv's cache is configured in `pyproject.toml` to sit inside the repo (`.uv-cache/`), so installs stay fast on any filesystem — including external drives — without drive-specific config.

Optional: `python -m nd2_analyzer` or `python src/main.py` after the first run (uses the same env).

For Cellpose + MPS notes, see [MouseLand/cellpose#1063](https://github.com/MouseLand/cellpose/issues/1063).
