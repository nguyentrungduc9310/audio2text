import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Audio / Video → Text PRO++",
    page_icon="🎙",
    layout="wide"
)

st.title("🎙 Audio / Video → Text PRO++")


# ----------------------------
# Wait for server
# ----------------------------

with st.spinner("Connecting to server..."):

    while True:
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)

            if r.status_code == 200:
                break

        except:
            pass

        time.sleep(1)


st.success("Server connected")


# ----------------------------
# Sidebar settings
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

vad_filter = st.sidebar.checkbox(
    "VAD Filter",
    value=True,
    help="Voice Activity Detection — filters out non-speech segments"
)


# ----------------------------
# Mode selection
# ----------------------------

mode = st.radio(
    "Processing Mode",
    ["Single File", "Batch Processing"],
    horizontal=True
)

SUPPORTED_TYPES = [
    "mp3",
    "wav",
    "m4a",
    "mp4",
    "mov",
    "mkv",
    "avi"
]


# ----------------------------
# Single File Mode
# ----------------------------

if mode == "Single File":

    file = st.file_uploader(
        "Upload audio or video",
        type=SUPPORTED_TYPES
    )

    if file:

        st.write("File:", file.name)

        if st.button("Start Transcription"):

            # Upload via batch endpoint (single file)
            upload_r = requests.post(
                f"{API_URL}/batch/upload",
                params={
                    "model": model,
                    "vad_filter": vad_filter
                },
                files=[("files", (file.name, file))],
                timeout=60
            )

            if upload_r.status_code != 200:

                st.error("Upload failed — server returned an error")

            else:

                job_id = upload_r.json()["job_ids"][0]

                progress = st.progress(0)
                status_text = st.empty()
                live_segments = st.container()
                start_time = time.time()
                displayed_count = 0

                while True:

                    elapsed = time.time() - start_time
                    mins, secs = divmod(int(elapsed), 60)

                    status_r = requests.get(
                        f"{API_URL}/batch/status/{job_id}",
                        timeout=10
                    )

                    if status_r.status_code != 200:
                        status_text.text(f"Checking status... ({mins}m {secs}s)")
                        time.sleep(2)
                        continue

                    job = status_r.json()
                    status = job["status"]
                    pct = job["progress"]

                    progress.progress(pct)
                    status_text.text(f"Status: {status} — {pct}% ({mins}m {secs}s)")

                    # Display new segments as they arrive
                    new_segments = job.get("segments", [])
                    if len(new_segments) > displayed_count:
                        with live_segments:
                            for seg in new_segments[displayed_count:]:
                                start_s = round(seg["start"], 2)
                                end_s = round(seg["end"], 2)
                                st.write(f"[{start_s}s - {end_s}s] {seg['text']}")
                        displayed_count = len(new_segments)

                    if status == "completed":
                        break

                    if status == "failed":
                        progress.empty()
                        status_text.empty()
                        error_msg = job.get("error", "Unknown error")
                        st.error(f"Transcription failed: {error_msg}")
                        st.stop()

                    time.sleep(2)

                status_text.empty()
                live_segments.empty()

                # Fetch result
                result_r = requests.get(
                    f"{API_URL}/batch/result/{job_id}",
                    timeout=10
                )

                if result_r.status_code != 200:

                    st.error("Failed to fetch transcription result")

                else:

                    data = result_r.json().get("result", {})

                    text = data.get("text", "")
                    segments = data.get("segments", [])
                    srt = data.get("srt", "")
                    language = data.get("language", "unknown")

                    elapsed = time.time() - start_time
                    mins, secs = divmod(int(elapsed), 60)

                    st.success(f"Transcription complete ({mins}m {secs}s)")

                    st.info(f"Detected language: **{language}**")


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

                    timeline_box = st.container(height=400)

                    with timeline_box:

                        for seg in segments:

                            start_s = round(seg["start"], 2)
                            end_s = round(seg["end"], 2)

                            st.write(
                                f"[{start_s}s - {end_s}s] {seg['text']}"
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


# ----------------------------
# Batch Processing Mode
# ----------------------------

if mode == "Batch Processing":

    files = st.file_uploader(
        "Upload multiple audio or video files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True
    )

    if files:

        st.write(f"Selected {len(files)} file(s)")

        for f in files:
            st.write(f"  - {f.name}")

        if st.button("Start Batch Transcription"):

            with st.spinner("Uploading files..."):

                upload_files = [
                    ("files", (f.name, f))
                    for f in files
                ]

                r = requests.post(
                    f"{API_URL}/batch/upload",
                    params={
                        "model": model,
                        "vad_filter": vad_filter
                    },
                    files=upload_files,
                    timeout=60
                )

            if r.status_code != 200:

                st.error("Batch upload failed")

            else:

                job_ids = r.json()["job_ids"]

                st.session_state["batch_jobs"] = job_ids

                st.success(f"Submitted {len(job_ids)} job(s)")

    # ----------------------------
    # Batch status + results
    # ----------------------------

    if "batch_jobs" in st.session_state:

        st.subheader("Queue Status")

        job_ids = st.session_state["batch_jobs"]

        all_done = True

        for job_id in job_ids:

            status_r = requests.get(f"{API_URL}/batch/status/{job_id}", timeout=10)

            if status_r.status_code != 200:
                continue

            job = status_r.json()

            status = job["status"]
            progress = job["progress"]
            filename = job["filename"]

            if status not in ["completed", "failed"]:
                all_done = False

            icon = "⏳"

            if status == "completed":
                icon = "✅"
            elif status == "failed":
                icon = "❌"
            elif status == "processing":
                icon = "🔄"

            st.write(f"{icon} **{filename}** — {status} ({progress}%)")

            if status == "completed":

                result_r = requests.get(f"{API_URL}/batch/result/{job_id}", timeout=10)

                if result_r.status_code == 200:

                    result = result_r.json().get("result", {})

                    text = result.get("text", "")
                    srt = result.get("srt", "")
                    language = result.get("language", "unknown")

                    with st.expander(f"Results — {filename}"):

                        st.info(f"Detected language: **{language}**")

                        st.text_area(
                            "Transcript",
                            text,
                            height=200,
                            key=f"txt_{job_id}"
                        )

                        col1, col2 = st.columns(2)

                        base_name = filename.rsplit(".", 1)[0]

                        with col1:

                            st.download_button(
                                "Download TXT",
                                text,
                                file_name=f"{base_name}.txt",
                                key=f"dl_txt_{job_id}"
                            )

                        with col2:

                            st.download_button(
                                "Download SRT",
                                srt,
                                file_name=f"{base_name}.srt",
                                key=f"dl_srt_{job_id}"
                            )

            elif status == "failed":

                st.error(f"Error: {job.get('error', 'Unknown error')}")

        if not all_done:

            time.sleep(2)

            st.rerun()
