from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import json
import logging
import queue as queue_mod
import threading

logging.basicConfig(level=logging.INFO)

from core.transcriber import Transcriber
from core.video_utils import extract_audio
from core.subtitle import generate_srt
from core.queue_manager import QueueManager, JobStatus
from core.ollama_provider import OllamaProvider
from core.api_provider import APIProvider
from core.summarizer import Summarizer
from core.translator import Translator as LLMTranslator
from core.temp_manager import cleanup_temp_dir, make_temp_file, cleanup_stale_files

app = FastAPI()


def _periodic_cleanup(interval_seconds=600, max_age_seconds=3600):
    """Daemon thread: clean stale temp files every interval_seconds."""
    import time
    while True:
        time.sleep(interval_seconds)
        try:
            cleanup_stale_files(max_age_seconds)
        except Exception:
            pass


@app.on_event("startup")
def startup_cleanup():
    cleanup_temp_dir()
    logging.info("Cleaned up temp directory")
    t = threading.Thread(target=_periodic_cleanup, daemon=True)
    t.start()
    logging.info("Started periodic temp cleanup thread (every 10 min)")

transcriber = Transcriber()

queue = QueueManager()

# LLM providers
ollama_provider = OllamaProvider()
api_provider = APIProvider()


def _get_active_provider(preferred: str = None):
    if preferred == "ollama" and ollama_provider.is_available():
        return ollama_provider
    if preferred == "api" and api_provider.is_available():
        return api_provider
    # Fallback: try both
    if ollama_provider.is_available():
        return ollama_provider
    if api_provider.is_available():
        return api_provider
    return None


# --- Pydantic models ---

class ConfigureRequest(BaseModel):
    provider: str = "ollama"  # "ollama" or "api"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    preset: Optional[str] = None  # "openai", "claude", "gemini"
    ollama_model: Optional[str] = None

class SummarizeRequest(BaseModel):
    text: str
    source_lang: Optional[str] = None
    llm_model: Optional[str] = None
    preferred_provider: Optional[str] = None

class TranslateRequest(BaseModel):
    text: str
    summary: Optional[str] = None
    segments: Optional[List[Dict]] = None
    source_lang: str = "en"
    target_lang: str = "vi"
    meeting_context: str = "general"
    glossary: Optional[Dict[str, str]] = None
    llm_model: Optional[str] = None
    preferred_provider: Optional[str] = None

class ProcessRequest(BaseModel):
    text: str
    segments: Optional[List[Dict]] = None
    source_lang: str = "en"
    target_lang: str = "vi"
    do_summarize: bool = True
    do_translate: bool = True
    meeting_context: str = "general"
    glossary: Optional[Dict[str, str]] = None
    llm_model: Optional[str] = None
    summarize_model: Optional[str] = None
    translate_model: Optional[str] = None
    preferred_provider: Optional[str] = None

SUPPORTED_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3"
]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():

    return {"models": SUPPORTED_MODELS}


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    model: str = "base",
    vad_filter: bool = True
):
    suffix = file.filename.split(".")[-1]
    input_path = make_temp_file(suffix="." + suffix)
    audio_path = None

    try:
        with open(input_path, "wb") as f:
            f.write(file.file.read())

        if suffix.lower() in ["mp4", "mov", "mkv", "avi"]:
            audio_path = make_temp_file(suffix=".wav")
            extract_audio(input_path, audio_path)
        else:
            audio_path = input_path

        text, segments, language = transcriber.transcribe(audio_path, model, vad_filter)
        srt = generate_srt(segments)

        return {
            "text": text,
            "segments": segments,
            "srt": srt,
            "language": language
        }

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if audio_path and audio_path != input_path and os.path.exists(audio_path):
            os.remove(audio_path)


def _process_job(job_id, input_path, audio_path, model, vad_filter):

    try:

        queue.update_status(job_id, JobStatus.PROCESSING, progress=10)

        def on_progress(p):
            pct = 10 + int(p * 70)
            queue.update_status(job_id, JobStatus.PROCESSING, progress=pct)

        def on_segment(seg):
            queue.add_segment(job_id, seg)

        text, segments, language = transcriber.transcribe(audio_path, model, vad_filter, on_progress=on_progress, on_segment=on_segment)

        queue.update_status(job_id, JobStatus.PROCESSING, progress=80)

        srt = generate_srt(segments)

        result = {
            "text": text,
            "segments": segments,
            "srt": srt,
            "language": language
        }

        queue.update_status(job_id, JobStatus.COMPLETED, progress=100, result=result)

    except Exception as e:

        queue.update_status(job_id, JobStatus.FAILED, error=str(e))

    finally:

        if os.path.exists(input_path):
            os.remove(input_path)

        if audio_path != input_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/batch/upload")
def batch_upload(
    files: List[UploadFile] = File(...),
    model: str = "base",
    vad_filter: bool = True,
    background_tasks: BackgroundTasks = None
):
    job_ids = []
    pending_cleanup = []

    try:
        for file in files:
            suffix = file.filename.split(".")[-1]
            input_path = make_temp_file(suffix="." + suffix)
            audio_path = None

            with open(input_path, "wb") as f:
                f.write(file.file.read())

            pending_cleanup.append(input_path)

            if suffix.lower() in ["mp4", "mov", "mkv", "avi"]:
                audio_path = make_temp_file(suffix=".wav")
                pending_cleanup.append(audio_path)
                extract_audio(input_path, audio_path)
            else:
                audio_path = input_path

            job_id = queue.create_job(file.filename, model)

            background_tasks.add_task(
                _process_job, job_id, input_path, audio_path, model, vad_filter
            )

            # Successfully handed off to background task, don't cleanup these files
            pending_cleanup = []

            job_ids.append(job_id)

    except Exception:
        for path in pending_cleanup:
            if os.path.exists(path):
                os.remove(path)
        raise

    return {"job_ids": job_ids}


@app.get("/batch/status/{job_id}")
def batch_status(job_id: str):

    job = queue.get_job(job_id)

    if not job:
        return {"error": "Job not found"}

    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "progress": job["progress"],
        "segments": job.get("segments", []),
        "created_at": job["created_at"]
    }


@app.get("/batch/result/{job_id}")
def batch_result(job_id: str):

    job = queue.get_job(job_id)

    if not job:
        return {"error": "Job not found"}

    if job["status"] != JobStatus.COMPLETED:

        return {
            "status": job["status"],
            "progress": job["progress"],
            "error": job.get("error")
        }

    return {
        "status": job["status"],
        "result": job["result"]
    }


@app.get("/batch/queue")
def batch_queue():

    jobs = queue.list_jobs()

    return {
        "jobs": [
            {
                "id": j["id"],
                "filename": j["filename"],
                "status": j["status"],
                "progress": j["progress"],
                "created_at": j["created_at"]
            }
            for j in jobs
        ]
    }


# ================================
# LLM Endpoints
# ================================

@app.get("/llm/status")
def llm_status():
    ollama_available = ollama_provider.is_available()
    api_available = api_provider.is_available()

    ollama_models = ollama_provider.list_models() if ollama_available else []

    return {
        "ollama": {
            "available": ollama_available,
            "models": ollama_models,
            "default_model": ollama_provider.default_model,
        },
        "api": {
            "available": api_available,
            "model": api_provider.model,
            "base_url": api_provider.base_url,
        },
        "any_available": ollama_available or api_available,
    }


@app.post("/llm/configure")
def llm_configure(req: ConfigureRequest):
    if req.provider == "api":
        api_provider.configure(
            api_key=req.api_key or "",
            base_url=req.base_url or "",
            model=req.model or "",
            preset=req.preset or "",
        )
        return {
            "status": "configured",
            "provider": "api",
            "available": api_provider.is_available(),
            "model": api_provider.model,
        }

    elif req.provider == "ollama":
        if req.ollama_model:
            ollama_provider.default_model = req.ollama_model
        return {
            "status": "configured",
            "provider": "ollama",
            "available": ollama_provider.is_available(),
            "default_model": ollama_provider.default_model,
        }

    return {"error": f"Unknown provider: {req.provider}"}


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    provider = _get_active_provider(req.preferred_provider)
    if provider is None:
        return {"error": "No LLM provider available. Install Ollama or configure an API key."}

    try:
        s = Summarizer(provider)
        result = s.summarize(req.text, model=req.llm_model, source_lang=req.source_lang)
        return {
            "summary": result.text,
            "model": result.model,
            "provider": result.provider,
            "tokens_used": result.tokens_used,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/translate")
def translate(req: TranslateRequest):
    provider = _get_active_provider(req.preferred_provider)
    if provider is None:
        return {"error": "No LLM provider available. Install Ollama or configure an API key."}

    try:
        t = LLMTranslator(provider)

        result = t.translate(
            text=req.text,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            meeting_context=req.meeting_context,
            glossary=req.glossary,
            model=req.llm_model,
        )

        translated_summary = None
        if req.summary:
            summary_resp = t.translate(
                text=req.summary,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                meeting_context=req.meeting_context,
                glossary=req.glossary,
                model=req.llm_model,
            )
            translated_summary = summary_resp.text

        translated_segments = None
        translated_srt = None
        if req.segments:
            translated_segments = t.translate_segments(
                segments=req.segments,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                meeting_context=req.meeting_context,
                glossary=req.glossary,
                model=req.llm_model,
            )
            translated_srt = generate_srt(translated_segments)

        return {
            "translated_text": result.text,
            "translated_summary": translated_summary,
            "translated_segments": translated_segments,
            "translated_srt": translated_srt,
            "model": result.model,
            "provider": result.provider,
            "tokens_used": result.tokens_used,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/process")
def process(req: ProcessRequest):
    provider = _get_active_provider(req.preferred_provider)
    if provider is None:
        return {"error": "No LLM provider available. Install Ollama or configure an API key."}

    summarize_model = req.summarize_model or req.llm_model
    translate_model = req.translate_model or req.llm_model

    result = {}

    try:
        # Step 1: Summarize (in original language)
        if req.do_summarize:
            s = Summarizer(provider)
            summary_resp = s.summarize(req.text, model=summarize_model, source_lang=req.source_lang)
            result["summary"] = summary_resp.text

        # Step 2: Translate
        if req.do_translate:
            t = LLMTranslator(provider)

            # Translate full text
            text_resp = t.translate(
                text=req.text,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                meeting_context=req.meeting_context,
                glossary=req.glossary,
                model=translate_model,
            )
            result["translated_text"] = text_resp.text

            # Translate summary if we have one
            if req.do_summarize and "summary" in result:
                summary_tr = t.translate(
                    text=result["summary"],
                    source_lang=req.source_lang,
                    target_lang=req.target_lang,
                    meeting_context=req.meeting_context,
                    glossary=req.glossary,
                    model=translate_model,
                )
                result["translated_summary"] = summary_tr.text

            # Translate segments
            if req.segments:
                translated_segments = t.translate_segments(
                    segments=req.segments,
                    source_lang=req.source_lang,
                    target_lang=req.target_lang,
                    meeting_context=req.meeting_context,
                    glossary=req.glossary,
                    model=translate_model,
                )
                result["translated_segments"] = translated_segments
                result["translated_srt"] = generate_srt(translated_segments)

        result["summarize_model"] = summarize_model
        result["translate_model"] = translate_model
        result["provider"] = provider.provider_name()

    except Exception as e:
        return {"error": str(e)}

    return result


@app.post("/process/stream")
def process_stream(req: ProcessRequest):
    provider = _get_active_provider(req.preferred_provider)
    if provider is None:
        return {"error": "No LLM provider available. Install Ollama or configure an API key."}

    summarize_model = req.summarize_model or req.llm_model
    translate_model = req.translate_model or req.llm_model

    q = queue_mod.Queue()

    def _run():
        result = {}
        try:
            # Step 1: Summarize
            if req.do_summarize:
                q.put({"type": "step_start", "step": "summarize", "label": "Summarizing...", "model": summarize_model or "default"})
                s = Summarizer(provider)
                summary_resp = s.summarize(req.text, model=summarize_model, on_token=lambda t: q.put({"type": "token", "content": t}), source_lang=req.source_lang)
                result["summary"] = summary_resp.text
                q.put({"type": "step_done", "step": "summarize"})

            # Step 2: Translate full text
            if req.do_translate:
                t = LLMTranslator(provider)

                q.put({"type": "step_start", "step": "translate_text", "label": "Translating full text...", "model": translate_model or "default"})
                text_resp = t.translate(
                    text=req.text,
                    source_lang=req.source_lang,
                    target_lang=req.target_lang,
                    meeting_context=req.meeting_context,
                    glossary=req.glossary,
                    model=translate_model,
                    on_token=lambda t: q.put({"type": "token", "content": t}),
                )
                result["translated_text"] = text_resp.text
                q.put({"type": "step_done", "step": "translate_text"})

                # Translate summary if we have one
                if req.do_summarize and "summary" in result:
                    q.put({"type": "step_start", "step": "translate_summary", "label": "Translating summary...", "model": translate_model or "default"})
                    summary_tr = t.translate(
                        text=result["summary"],
                        source_lang=req.source_lang,
                        target_lang=req.target_lang,
                        meeting_context=req.meeting_context,
                        glossary=req.glossary,
                        model=translate_model,
                        on_token=lambda t: q.put({"type": "token", "content": t}),
                    )
                    result["translated_summary"] = summary_tr.text
                    q.put({"type": "step_done", "step": "translate_summary"})

                # Translate segments
                if req.segments:
                    q.put({"type": "step_start", "step": "translate_segments", "label": "Translating segments...", "model": translate_model or "default"})
                    translated_segments = t.translate_segments(
                        segments=req.segments,
                        source_lang=req.source_lang,
                        target_lang=req.target_lang,
                        meeting_context=req.meeting_context,
                        glossary=req.glossary,
                        model=translate_model,
                    )
                    result["translated_segments"] = translated_segments
                    result["translated_srt"] = generate_srt(translated_segments)
                    q.put({"type": "step_done", "step": "translate_segments"})

            result["summarize_model"] = summarize_model
            result["translate_model"] = translate_model
            result["provider"] = provider.provider_name()

            q.put({"type": "complete", "result": result})

        except Exception as e:
            q.put({"type": "error", "message": str(e)})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    def event_generator():
        while True:
            try:
                event = q.get(timeout=600)
            except queue_mod.Empty:
                yield json.dumps({"type": "error", "message": "Processing timed out"}) + "\n"
                break
            yield json.dumps(event) + "\n"
            if event["type"] in ("complete", "error"):
                break

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


def _stream_from_queue(q):
    """Helper: yield NDJSON events from a queue until complete/error."""
    while True:
        try:
            event = q.get(timeout=600)
        except queue_mod.Empty:
            yield json.dumps({"type": "error", "message": "Processing timed out"}) + "\n"
            break
        yield json.dumps(event) + "\n"
        if event["type"] in ("complete", "error"):
            break


@app.post("/summarize/stream")
def summarize_stream(req: SummarizeRequest):
    provider = _get_active_provider(req.preferred_provider)
    if provider is None:
        return {"error": "No LLM provider available. Install Ollama or configure an API key."}

    model = req.llm_model
    q = queue_mod.Queue()

    def _run():
        try:
            q.put({"type": "step_start", "step": "summarize", "label": "Summarizing...", "model": model or "default"})
            s = Summarizer(provider)
            result = s.summarize(req.text, model=model, on_token=lambda t: q.put({"type": "token", "content": t}), source_lang=req.source_lang)
            q.put({"type": "step_done", "step": "summarize"})
            q.put({"type": "complete", "result": {
                "summary": result.text,
                "model": result.model,
                "provider": result.provider,
            }})
        except Exception as e:
            q.put({"type": "error", "message": str(e)})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return StreamingResponse(_stream_from_queue(q), media_type="application/x-ndjson")


@app.post("/translate/stream")
def translate_stream(req: TranslateRequest):
    provider = _get_active_provider(req.preferred_provider)
    if provider is None:
        return {"error": "No LLM provider available. Install Ollama or configure an API key."}

    model = req.llm_model
    q = queue_mod.Queue()

    def _run():
        try:
            t = LLMTranslator(provider)
            result = {}

            # Sub-step 1: translate full text
            q.put({"type": "step_start", "step": "translate_text", "label": "Translating full text...", "model": model or "default"})
            text_resp = t.translate(
                text=req.text,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                meeting_context=req.meeting_context,
                glossary=req.glossary,
                model=model,
                on_token=lambda tk: q.put({"type": "token", "content": tk}),
            )
            result["translated_text"] = text_resp.text
            q.put({"type": "step_done", "step": "translate_text"})

            # Sub-step 2: translate summary if provided
            if req.summary:
                q.put({"type": "step_start", "step": "translate_summary", "label": "Translating summary...", "model": model or "default"})
                summary_resp = t.translate(
                    text=req.summary,
                    source_lang=req.source_lang,
                    target_lang=req.target_lang,
                    meeting_context=req.meeting_context,
                    glossary=req.glossary,
                    model=model,
                    on_token=lambda tk: q.put({"type": "token", "content": tk}),
                )
                result["translated_summary"] = summary_resp.text
                q.put({"type": "step_done", "step": "translate_summary"})

            # Sub-step 3: translate segments (no token streaming)
            if req.segments:
                q.put({"type": "step_start", "step": "translate_segments", "label": "Translating segments...", "model": model or "default"})
                translated_segments = t.translate_segments(
                    segments=req.segments,
                    source_lang=req.source_lang,
                    target_lang=req.target_lang,
                    meeting_context=req.meeting_context,
                    glossary=req.glossary,
                    model=model,
                )
                result["translated_segments"] = translated_segments
                result["translated_srt"] = generate_srt(translated_segments)
                q.put({"type": "step_done", "step": "translate_segments"})

            result["model"] = model
            result["provider"] = provider.provider_name()

            q.put({"type": "complete", "result": result})

        except Exception as e:
            q.put({"type": "error", "message": str(e)})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return StreamingResponse(_stream_from_queue(q), media_type="application/x-ndjson")
