import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
import uvicorn
import whisper
import tempfile
import shutil
import os
import threading
import torch
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large"]

loaded_models = {}
lock = threading.Lock()

server_ready = threading.Event()


def load_models():

    with lock:
        for file in os.listdir(MODELS_DIR):

            if file.endswith(".pt"):

                name = file.replace(".pt", "")

                if name in SUPPORTED_MODELS:

                    logger.info(f"Loading model {name}")

                    loaded_models[name] = whisper.load_model(
                        name,
                        device=device,
                        download_root=MODELS_DIR
                    )

                    logger.info(f"Loaded {name}")

    server_ready.set()


@app.on_event("startup")
def startup():

    load_models()


@app.get("/health")
def health():

    return {"ready": server_ready.is_set()}


@app.get("/models")
def models():

    with lock:
        loaded = list(loaded_models.keys())

    return {
        "loaded": loaded,
        "all": SUPPORTED_MODELS
    }


@app.post("/models/download")
def download(model_name: str):

    if model_name not in SUPPORTED_MODELS:

        raise HTTPException(400, "unsupported model")

    with lock:

        if model_name in loaded_models:

            return {"status": "already_loaded"}

        logger.info(f"Downloading {model_name}")

        model = whisper.load_model(
            model_name,
            device=device,
            download_root=MODELS_DIR
        )

        loaded_models[model_name] = model

    return {"status": "downloaded"}


@app.post("/transcribe")
def transcribe(
        file: UploadFile = File(...),
        model_name: str = "base"
):

    suffix = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:

        shutil.copyfileobj(file.file, tmp)

        input_path = tmp.name

    if suffix in [".mp4", ".mov", ".mkv", ".avi"]:

        audio_path = input_path + ".wav"

        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
            "-y"
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    else:

        audio_path = input_path

    try:

        with lock:

            if model_name not in loaded_models:

                raise HTTPException(400, "model not loaded")

            model = loaded_models[model_name]

        result = model.transcribe(audio_path, language="ja")

        text = result["text"]

    finally:

        os.remove(input_path)

        if audio_path != input_path:

            os.remove(audio_path)

    return {"text": text}


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)