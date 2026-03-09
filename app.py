import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Audio / Video → Text",
    page_icon="🎙",
    layout="wide"
)

st.title("🎙 Audio / Video → Text Transcription")


# ----------------------------
# Wait for server
# ----------------------------

with st.spinner("Connecting to server..."):

    while True:
        try:
            r = requests.get(f"{API_URL}/health")

            if r.status_code == 200:
                break

        except:
            pass

        time.sleep(1)


st.success("Server connected")


# ----------------------------
# Model selector
# ----------------------------

st.sidebar.title("Settings")

models = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3"
]

model = st.sidebar.selectbox(
    "Whisper Model",
    models,
    index=1
)

st.sidebar.info(f"Selected model: {model}")


# ----------------------------
# Upload
# ----------------------------

file = st.file_uploader(
    "Upload audio or video",
    type=[
        "mp3",
        "wav",
        "m4a",
        "mp4",
        "mov",
        "mkv",
        "avi"
    ]
)

# ----------------------------
# Transcription
# ----------------------------

if file:

    st.write("File:", file.name)

    if st.button("Start Transcription"):

        progress = st.progress(0)

        with st.spinner("Uploading and processing..."):

            r = requests.post(
                f"{API_URL}/transcribe",
                params={"model": model},
                files={"file": (file.name, file)}
            )

            progress.progress(100)

        if r.status_code != 200:

            st.error("Transcription failed")

        else:

            data = r.json()

            text = data["text"]
            segments = data["segments"]
            srt = data["srt"]

            st.success("Transcription complete")


            # ----------------------------
            # Text result
            # ----------------------------

            st.subheader("Full Text")

            st.text_area(
                "Transcript",
                text,
                height=300
            )


            # ----------------------------
            # Segments
            # ----------------------------

            st.subheader("Timeline")

            for seg in segments:

                start = round(seg["start"], 2)
                end = round(seg["end"], 2)

                st.write(
                    f"[{start}s - {end}s] {seg['text']}"
                )


            # ----------------------------
            # Downloads
            # ----------------------------

            st.subheader("Download")

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    "Download TXT",
                    text,
                    file_name="transcript.txt"
                )

            with col2:

                st.download_button(
                    "Download SRT",
                    srt,
                    file_name="subtitle.srt"
                )