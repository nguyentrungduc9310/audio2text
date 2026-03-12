import streamlit as st
import requests
import json
import time
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

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
# Sidebar: Post-Processing
# ----------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Post-Processing")

MEETING_CONTEXTS = {
    "general": "General",
    "technical": "Technical (Software/DevOps/AI)",
    "sales": "Sales / Business",
    "medical": "Medical / Healthcare",
    "legal": "Legal / Contracts",
}

TARGET_LANGUAGES = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "th": "Thai",
}

# Check LLM availability
llm_status = None
try:
    llm_r = requests.get(f"{API_URL}/llm/status", timeout=5)
    if llm_r.status_code == 200:
        llm_status = llm_r.json()
except:
    pass

llm_available = llm_status and llm_status.get("any_available", False)

if not llm_available:
    st.sidebar.warning("No LLM available. Install Ollama or configure an API key to enable summarize/translate.")


# ----------------------------
# Sidebar: LLM Provider (before feature toggles)
# ----------------------------

st.sidebar.subheader("LLM Provider")

provider_choice = st.sidebar.selectbox(
    "Provider",
    ["ollama", "api"],
    format_func=lambda x: "Ollama (Local)" if x == "ollama" else "API (Cloud)",
)

ollama_models = []

if provider_choice == "ollama":
    if llm_status and llm_status.get("ollama", {}).get("available"):
        ollama_models = llm_status["ollama"].get("models", [])

    if not ollama_models:
        st.sidebar.text("No Ollama models found")
        st.sidebar.code("ollama pull qwen3.5:4b", language="bash")

elif provider_choice == "api":
    api_preset = st.sidebar.selectbox(
        "API Preset",
        ["openai", "claude", "gemini", "custom"],
        format_func=lambda x: {"openai": "OpenAI", "claude": "Claude", "gemini": "Gemini", "custom": "Custom"}.get(x, x),
    )

    api_key = st.sidebar.text_input("API Key", type="password")

    custom_base_url = ""
    custom_model = ""
    if api_preset == "custom":
        custom_base_url = st.sidebar.text_input("Base URL")
        custom_model = st.sidebar.text_input("Model Name")

    if api_key:
        configure_data = {
            "provider": "api",
            "api_key": api_key,
            "preset": api_preset if api_preset != "custom" else "",
            "base_url": custom_base_url,
            "model": custom_model,
        }
        try:
            requests.post(f"{API_URL}/llm/configure", json=configure_data, timeout=5)
        except:
            pass

st.sidebar.markdown("---")

# ----------------------------
# Sidebar: Feature toggles with inline model selectors
# ----------------------------

meeting_context = st.sidebar.selectbox(
    "Meeting Context",
    list(MEETING_CONTEXTS.keys()),
    format_func=lambda x: MEETING_CONTEXTS[x],
    disabled=not llm_available,
    help="Context helps the LLM translate domain terms correctly"
)

summarize_model = None
do_summarize = st.sidebar.checkbox(
    "Summarize transcript",
    value=False,
    disabled=not llm_available,
    help="Generate a structured summary in the original language"
)
if do_summarize and provider_choice == "ollama" and ollama_models:
    summarize_model = st.sidebar.selectbox(
        "Summarize Model", ollama_models, key="summarize_model_select")

translate_model = None
do_translate = st.sidebar.checkbox(
    "Translate",
    value=False,
    disabled=not llm_available,
    help="Translate transcript and summary to target language"
)
if do_translate and provider_choice == "ollama" and ollama_models:
    translate_model = st.sidebar.selectbox(
        "Translate Model", ollama_models, key="translate_model_select")

target_lang = st.sidebar.selectbox(
    "Target Language",
    list(TARGET_LANGUAGES.keys()),
    format_func=lambda x: TARGET_LANGUAGES[x],
    disabled=not llm_available,
)

# Glossary (optional)
with st.sidebar.expander("Glossary (optional)"):
    glossary_text = st.text_area(
        "Term mappings (one per line: source=target)",
        height=100,
        help="e.g. sprint=sprint\ndeployment=triển khai",
        key="glossary_input",
    )

glossary = None
if glossary_text.strip():
    glossary = {}
    for line in glossary_text.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            glossary[k.strip()] = v.strip()


# ----------------------------
# Helper: post-processing
# ----------------------------

def _stream_ndjson(url, request_data):
    """Make a streaming NDJSON request, yield parsed events."""
    try:
        r = requests.post(url, json=request_data, stream=True, timeout=600)
    except requests.exceptions.Timeout:
        yield {"type": "error", "message": "Request timed out. Try a shorter text or faster model."}
        return
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return

    if r.status_code != 200:
        yield {"type": "error", "message": "Request failed"}
        return

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _render_stream(events_iter, status_widget):
    """Consume NDJSON events, render streaming tokens in status widget. Returns result dict or None."""
    current_text = ""
    text_placeholder = None
    data = None

    for event in events_iter:
        if event["type"] == "step_start":
            current_text = ""
            model_name = event.get("model", "?")
            status_widget.write(f"**{event['label']}** (model: `{model_name}`)")
            text_placeholder = status_widget.empty()
        elif event["type"] == "token":
            current_text += event["content"]
            if text_placeholder:
                text_placeholder.markdown(current_text + " ▌")
        elif event["type"] == "step_done":
            if text_placeholder and current_text:
                text_placeholder.markdown(current_text)
            text_placeholder = None
        elif event["type"] == "complete":
            data = event["result"]
        elif event["type"] == "error":
            status_widget.error(event["message"])
            return None

    return data


def run_summarize_step(text, source_lang, key_prefix):
    """Call /summarize/stream and store result in session_state."""
    request_data = {
        "text": text,
        "source_lang": source_lang,
        "llm_model": summarize_model,
        "preferred_provider": provider_choice,
    }

    with st.status("Summarizing...", expanded=True) as status:
        events = _stream_ndjson(f"{API_URL}/summarize/stream", request_data)
        data = _render_stream(events, status)

        if data is None:
            status.update(label="Summarization failed", state="error")
            return
        if "error" in data:
            st.error(f"Summarization error: {data['error']}")
            status.update(label="Summarization failed", state="error")
            return

        status.update(label="Summarization complete!", state="complete", expanded=False)

    st.session_state[f"pp_summary_{key_prefix}"] = data


def run_translate_step(text, segments, source_lang, summary_text, key_prefix):
    """Call /translate/stream and store result in session_state."""
    request_data = {
        "text": text,
        "summary": summary_text,
        "segments": segments,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "meeting_context": meeting_context,
        "glossary": glossary,
        "llm_model": translate_model,
        "preferred_provider": provider_choice,
    }

    with st.status("Translating...", expanded=True) as status:
        events = _stream_ndjson(f"{API_URL}/translate/stream", request_data)
        data = _render_stream(events, status)

        if data is None:
            status.update(label="Translation failed", state="error")
            return
        if "error" in data:
            st.error(f"Translation error: {data['error']}")
            status.update(label="Translation failed", state="error")
            return

        status.update(label="Translation complete!", state="complete", expanded=False)

    st.session_state[f"pp_translate_{key_prefix}"] = data


def show_post_processing(text, segments, language, key_prefix="single"):
    """Show step-by-step post-processing with separate Summarize and Translate buttons."""

    if not do_summarize and not do_translate:
        st.info("Enable Summarize or Translate in the sidebar to use post-processing.")
        return

    source_lang = language if language != "unknown" else "en"

    # --- Step 1: Summarize ---
    if do_summarize:
        st.subheader("Step 1: Summarize")
        if st.button("Summarize", key=f"btn_summarize_{key_prefix}"):
            run_summarize_step(text, source_lang, key_prefix)

        summary_key = f"pp_summary_{key_prefix}"
        if summary_key in st.session_state:
            summary_data = st.session_state[summary_key]
            st.markdown(summary_data["summary"])
            st.download_button(
                "Download Summary",
                summary_data["summary"],
                file_name="summary.txt",
                key=f"dl_summary_{key_prefix}",
            )

    # --- Step 2: Translate ---
    if do_translate:
        step_label = "Step 2: Translate" if do_summarize else "Translate"
        st.subheader(step_label)
        if st.button("Translate", key=f"btn_translate_{key_prefix}"):
            summary_text = st.session_state.get(f"pp_summary_{key_prefix}", {}).get("summary")
            run_translate_step(text, segments, source_lang, summary_text, key_prefix)

        translate_key = f"pp_translate_{key_prefix}"
        if translate_key in st.session_state:
            tr_data = st.session_state[translate_key]

            # Translated summary
            if tr_data.get("translated_summary"):
                st.markdown(f"**Translated Summary ({TARGET_LANGUAGES.get(target_lang, target_lang)})**")
                st.markdown(tr_data["translated_summary"])
                st.download_button(
                    "Download Translated Summary",
                    tr_data["translated_summary"],
                    file_name="translated_summary.txt",
                    key=f"dl_tr_summary_{key_prefix}",
                )

            # Translated full text
            if tr_data.get("translated_text"):
                st.markdown(f"**Translated Full Text ({TARGET_LANGUAGES.get(target_lang, target_lang)})**")
                st.text_area(
                    "Translated text",
                    tr_data["translated_text"],
                    height=300,
                    key=f"translated_text_{key_prefix}",
                )
                st.download_button(
                    "Download Translated Text",
                    tr_data["translated_text"],
                    file_name="translated_text.txt",
                    key=f"dl_tr_text_{key_prefix}",
                )

            # Translated timeline
            if tr_data.get("translated_segments"):
                st.markdown(f"**Translated Timeline ({TARGET_LANGUAGES.get(target_lang, target_lang)})**")
                timeline_box = st.container(height=400)
                with timeline_box:
                    for seg in tr_data["translated_segments"]:
                        start_s = round(seg["start"], 2)
                        end_s = round(seg["end"], 2)
                        st.write(f"[{start_s}s - {end_s}s] {seg['text']}")

            if tr_data.get("translated_srt"):
                st.download_button(
                    "Download Translated SRT",
                    tr_data["translated_srt"],
                    file_name="translated_subtitle.srt",
                    key=f"dl_tr_srt_{key_prefix}",
                )


# ----------------------------
# Mode selection
# ----------------------------

tab_single, tab_batch, tab_summarize, tab_translate = st.tabs([
    "🎤 Transcribe",
    "📁 Batch",
    "📝 Summarize",
    "🌐 Translate",
])

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

with tab_single:

    file = st.file_uploader(
        "Upload audio or video",
        type=SUPPORTED_TYPES
    )

    if file:

        st.write("File:", file.name)

        if st.button("Start Transcription"):

            # Clear old post-processing results and edited transcript
            for key in list(st.session_state.keys()):
                if key.startswith("pp_summary_single") or key.startswith("pp_translate_single"):
                    del st.session_state[key]
            if "edit_transcript" in st.session_state:
                del st.session_state["edit_transcript"]

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

                    # Store in session state for persistent display
                    st.session_state["last_text"] = data.get("text", "")
                    st.session_state["last_segments"] = data.get("segments", [])
                    st.session_state["last_srt"] = data.get("srt", "")
                    st.session_state["last_language"] = data.get("language", "unknown")

                    st.rerun()

    # Display results from session_state (persists across reruns)
    if "last_text" in st.session_state:
        text = st.session_state["last_text"]
        segments = st.session_state["last_segments"]
        srt = st.session_state["last_srt"]
        language = st.session_state["last_language"]

        st.success("Transcription complete")
        st.info(f"Detected language: **{language}**")

        st.subheader("Full Text")
        edited_text = st.text_area("Transcript", text, height=300, key="edit_transcript")

        st.subheader("Timeline")
        timeline_box = st.container(height=400)
        with timeline_box:
            for seg in segments:
                start_s = round(seg["start"], 2)
                end_s = round(seg["end"], 2)
                st.write(f"[{start_s}s - {end_s}s] {seg['text']}")

        st.subheader("Download")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download TXT", edited_text, file_name="transcript.txt")
        with col2:
            st.download_button("Download SRT", srt, file_name="subtitle.srt")

        if llm_available and (do_summarize or do_translate):
            st.markdown("---")
            st.subheader("Post-Processing")
            show_post_processing(edited_text, segments, language, key_prefix="single")


# ----------------------------
# Batch Processing Mode
# ----------------------------

with tab_batch:

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
                    segments = result.get("segments", [])

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

                        # Batch post-processing per file
                        if llm_available and (do_summarize or do_translate):
                            st.markdown("---")
                            show_post_processing(text, segments, language, key_prefix=f"batch_{job_id}")

            elif status == "failed":

                st.error(f"Error: {job.get('error', 'Unknown error')}")

        if not all_done:

            time.sleep(2)

            st.rerun()


# ----------------------------
# Standalone Summarize Tab
# ----------------------------

with tab_summarize:

    if not llm_available:
        st.warning("No LLM available. Install Ollama or configure an API key in the sidebar.")
    else:
        sum_input = st.text_area(
            "Text to summarize",
            height=300,
            key="standalone_sum_input",
            placeholder="Paste or type the text you want to summarize...",
        )

        if st.button("Summarize", key="btn_standalone_summarize", disabled=not sum_input.strip()):
            for key in list(st.session_state.keys()):
                if key.startswith("pp_summary_standalone_sum"):
                    del st.session_state[key]
            run_summarize_step(sum_input.strip(), "auto", "standalone_sum")

        summary_key = "pp_summary_standalone_sum"
        if summary_key in st.session_state:
            summary_data = st.session_state[summary_key]
            st.subheader("Summary")
            st.markdown(summary_data["summary"])
            st.download_button(
                "Download Summary",
                summary_data["summary"],
                file_name="summary.txt",
                key="dl_standalone_summary",
            )


# ----------------------------
# Standalone Translate Tab
# ----------------------------

with tab_translate:

    if not llm_available:
        st.warning("No LLM available. Install Ollama or configure an API key in the sidebar.")
    else:
        tr_source_lang = st.selectbox(
            "Source Language",
            list(TARGET_LANGUAGES.keys()),
            format_func=lambda x: TARGET_LANGUAGES[x],
            index=1,
            key="standalone_tr_source_lang",
        )

        tr_input = st.text_area(
            "Text to translate",
            height=300,
            key="standalone_tr_input",
            placeholder="Paste or type the text you want to translate...",
        )

        st.info(f"Target language: **{TARGET_LANGUAGES.get(target_lang, target_lang)}** (set in sidebar)")

        if st.button("Translate", key="btn_standalone_translate", disabled=not tr_input.strip()):
            for key in list(st.session_state.keys()):
                if key.startswith("pp_translate_standalone_tr"):
                    del st.session_state[key]
            run_translate_step(
                tr_input.strip(),
                segments=None,
                source_lang=tr_source_lang,
                summary_text=None,
                key_prefix="standalone_tr",
            )

        translate_key = "pp_translate_standalone_tr"
        if translate_key in st.session_state:
            tr_data = st.session_state[translate_key]

            if tr_data.get("translated_text"):
                st.subheader(f"Translation ({TARGET_LANGUAGES.get(target_lang, target_lang)})")
                st.text_area(
                    "Translated text",
                    tr_data["translated_text"],
                    height=300,
                    key="standalone_translated_text",
                )
                st.download_button(
                    "Download Translation",
                    tr_data["translated_text"],
                    file_name="translation.txt",
                    key="dl_standalone_translation",
                )
