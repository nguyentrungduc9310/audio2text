from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from typing import List
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)

from core.transcriber import Transcriber
from core.video_utils import extract_audio
from core.subtitle import generate_srt
from core.queue_manager import QueueManager, JobStatus

app = FastAPI()

transcriber = Transcriber()

queue = QueueManager()

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

    with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:

        content = file.file.read()

        tmp.write(content)

        input_path = tmp.name

    audio_path = input_path + ".wav"

    if suffix.lower() in ["mp4", "mov", "mkv", "avi"]:

        extract_audio(input_path, audio_path)

    else:

        audio_path = input_path

    try:

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

        if audio_path != input_path and os.path.exists(audio_path):
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

    for file in files:

        suffix = file.filename.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:

            content = file.file.read()

            tmp.write(content)

            input_path = tmp.name

        audio_path = input_path + ".wav"

        if suffix.lower() in ["mp4", "mov", "mkv", "avi"]:

            extract_audio(input_path, audio_path)

        else:

            audio_path = input_path

        job_id = queue.create_job(file.filename, model)

        background_tasks.add_task(
            _process_job, job_id, input_path, audio_path, model, vad_filter
        )

        job_ids.append(job_id)

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
