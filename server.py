from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from core.transcriber import Transcriber
from core.video_utils import extract_audio
from core.srt_generator import generate_srt

app = FastAPI()

transcriber = Transcriber()

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
async def transcribe(
    file: UploadFile = File(...),
    model: str = "base"
):

    suffix = file.filename.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:

        content = await file.read()

        tmp.write(content)

        input_path = tmp.name

    audio_path = input_path + ".wav"

    if suffix.lower() in ["mp4","mov","mkv","avi"]:

        extract_audio(input_path, audio_path)

    else:

        audio_path = input_path

    text, segments = transcriber.transcribe(audio_path, model)

    srt = generate_srt(segments)

    os.remove(input_path)

    if audio_path != input_path:
        os.remove(audio_path)

    return {
        "text": text,
        "segments": segments,
        "srt": srt
    }