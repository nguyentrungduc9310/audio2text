import streamlit as st
import requests
import time
import io
import os
import math
from pydub import AudioSegment

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Audio → Text", page_icon="🎥")

st.title("🎥 Audio / Video → Text")

session = requests.Session()


with st.spinner("Waiting for server..."):

    while True:

        try:

            r = session.get(f"{API_URL}/health")

            if r.json()["ready"]:
                break

        except:
            pass

        time.sleep(1)


info = session.get(f"{API_URL}/models").json()

available = set(info["loaded"])

all_models = info["all"]

model = st.sidebar.selectbox(
    "Model",
    all_models
)

if model not in available:

    if st.sidebar.button("Download model"):

        session.post(
            f"{API_URL}/models/download",
            params={"model_name": model}
        )

        st.sidebar.success("downloaded")


tab_audio, tab_video = st.tabs(["Audio → Text", "Video → Text"])


with tab_audio:

    file = st.file_uploader(
        "Upload audio",
        type=["mp3", "wav", "m4a"]
    )

    if file:

        if st.button("Transcribe audio"):

            data = file.read()

            audio = AudioSegment.from_file(
                io.BytesIO(data),
                format=os.path.splitext(file.name)[1][1:]
            )

            duration = len(audio)

            chunk = 60000

            parts = math.ceil(duration / chunk)

            progress = st.progress(0)

            text = ""

            for i in range(parts):

                seg = audio[i*chunk:(i+1)*chunk]

                buf = io.BytesIO()

                seg.export(buf, format="wav")

                buf.seek(0)

                resp = session.post(
                    f"{API_URL}/transcribe",
                    params={"model_name": model},
                    files={"file": ("chunk.wav", buf)}
                )

                text += resp.json()["text"]

                progress.progress((i+1)/parts)

            st.text_area("Result", text)

            st.download_button(
                "Download txt",
                text,
                file_name="audio.txt"
            )


with tab_video:

    file = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "mkv"]
    )

    if file:

        if st.button("Transcribe video"):

            resp = session.post(
                f"{API_URL}/transcribe",
                params={"model_name": model},
                files={"file": (file.name, file)}
            )

            text = resp.json()["text"]

            st.text_area("Result", text)

            st.download_button(
                "Download txt",
                text,
                file_name="video.txt"
            )