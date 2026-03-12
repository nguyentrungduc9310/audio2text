# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local audio/video-to-text transcription tool using faster-whisper, with LLM-powered summarization and translation. Two-process architecture: a FastAPI backend (`server.py`) handles transcription + post-processing, and a Streamlit frontend (`app.py`) provides the web UI.

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

# Pull LLM model for summarize/translate (requires Ollama)
ollama pull qwen3.5:4b    # 8GB RAM
ollama pull qwen3.5:9b    # 16GB+ RAM, better quality
```

## Architecture

**Backend (`server.py`)** — FastAPI app exposing:
- `/health`, `/models` — status and whisper model list
- `POST /transcribe` — file upload transcription
- `/batch/upload`, `/batch/status/{id}`, `/batch/result/{id}`, `/batch/queue` — batch processing
- `GET /llm/status` — check LLM provider availability and models
- `POST /llm/configure` — configure LLM provider (Ollama model or API key/preset)
- `POST /summarize` — summarize text in original language
- `POST /translate` — translate text + segments (context-aware)
- `POST /process` — full pipeline: summarize → translate

**Frontend (`app.py`)** — Streamlit app with sidebar settings for whisper model, VAD filter, post-processing options (meeting context, summarize/translate toggles, target language, LLM provider selection, glossary).

**`core/` module** — Utilities used by the server:
- `transcriber.py` — `Transcriber` class with chunked transcription support
- `model_manager.py` — whisper model loading/caching
- `chunker.py` — `AudioChunker` for splitting long audio files
- `video_utils.py` — `extract_audio()` via ffmpeg
- `subtitle.py` — `generate_srt()` for SRT subtitle format
- `queue_manager.py` — `QueueManager` for batch job tracking
- `llm_provider.py` — `LLMProvider` ABC + `LLMResponse` dataclass
- `ollama_provider.py` — `OllamaProvider` for local LLM via Ollama
- `api_provider.py` — `APIProvider` for cloud APIs (OpenAI/Claude/Gemini compatible)
- `summarizer.py` — `Summarizer` with chunked summarization for long text
- `translator.py` — `Translator` with context-aware translation, glossary support, segment translation

## Key Details

- Python 3.9+, venv at `./venv/`
- System dependency: `ffmpeg` (required for video file support)
- Optional dependency: `ollama` (for local LLM summarize/translate)
- Uses `faster-whisper` (CTranslate2-based), not OpenAI's `openai-whisper`
- Supported whisper models: tiny, base, small, medium, large-v3
- Downloaded model files (`.pt`) live in `models/` (gitignored)
- Supported upload formats: mp3, wav, m4a (audio); mp4, mov, mkv, avi (video)
- `core/__init__.py` exports all core classes
- No test suite currently exists

## Post-Processing Pipeline

```
Audio → Whisper → Text [original language]
                    ↓
              LLM Summarize → Summary [original language]
                    ↓
              LLM Translate → Summary [target language]
                               + Full text [target language]
                               + Segments [target language] → SRT
```

- Meeting context types: general, technical, sales, medical, legal
- Target languages: Vietnamese (default), English, Chinese, Japanese, Korean, French, German, Spanish, Thai
- Glossary support: user-provided term mappings for domain-specific translation
- LLM providers: Ollama (local, default qwen3.5:4b) or API (OpenAI/Claude/Gemini)
