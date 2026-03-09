# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local audio/video-to-text transcription tool using faster-whisper. Two-process architecture: a FastAPI backend (`server.py`) handles transcription, and a Streamlit frontend (`app.py`) provides the web UI.

## Commands

```bash
# First-time setup (creates venv, installs deps, requires python3 + ffmpeg)
./setup.sh

# Run the app (starts both API server and Streamlit UI)
./start.sh

# Run components individually
source venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8000    # API on :8000
streamlit run app.py --server.maxUploadSize=2048     # UI on :8501
```

## Architecture

**Backend (`server.py`)** — FastAPI app exposing `/health`, `/models`, and `POST /transcribe`. The transcribe endpoint accepts a file upload + model name query param, saves to a temp file, extracts audio from video if needed, runs whisper, and returns `{text, segments, srt}`.

**Frontend (`app.py`)** — Streamlit app that polls `/health` until the backend is up, then provides file upload UI with model selection. Displays full transcript, timed segments, and download buttons for TXT/SRT.

**`core/` module** — Three utilities used by the server:
- `transcriber.py` — `Transcriber` class that lazy-loads faster-whisper models (cached in `self.models` dict) with `int8` compute type, `beam_size=5`
- `video_utils.py` — `extract_audio()` shells out to `ffmpeg` to convert video to 16kHz mono WAV
- `srt_generator.py` — `generate_srt()` formats timed segments into SRT subtitle format

## Key Details

- Python 3.9+, venv at `./venv/`
- System dependency: `ffmpeg` (required for video file support)
- Uses `faster-whisper` (CTranslate2-based), not OpenAI's `openai-whisper`
- Supported whisper models: tiny, base, small, medium, large-v3
- Downloaded model files (`.pt`) live in `models/` (gitignored)
- Supported upload formats: mp3, wav, m4a (audio); mp4, mov, mkv, avi (video)
- No `core/__init__.py` — imports work via direct module paths
- No test suite currently exists
