# Audio / Video → Text Tool (Whisper Local)

Installation Guide

This guide explains how to install and run the **Audio / Video → Text
transcription tool** locally using Whisper.

------------------------------------------------------------------------

# 1. Requirements

-   macOS / Linux
-   Python 3.9+
-   ffmpeg
-   Homebrew (macOS)

------------------------------------------------------------------------

# 2. Install ffmpeg

ffmpeg is required to extract audio from video files.

``` bash
brew install ffmpeg
```

Verify installation:

``` bash
ffmpeg -version
```

------------------------------------------------------------------------

# 3. Create Project Folder

``` bash
mkdir audio2text
cd audio2text
```

Project structure:

    audio2text/
    │
    ├── server.py
    ├── app.py
    ├── start.sh
    ├── requirements.txt
    │
    ├── models/
    └── venv/

------------------------------------------------------------------------

# 4. Create Python Virtual Environment

``` bash
python3 -m venv venv
```

Activate environment:

``` bash
source venv/bin/activate
```

------------------------------------------------------------------------

# 5. Install Python Dependencies

Create file:

`requirements.txt`

    fastapi
    uvicorn
    openai-whisper
    torch
    pydub
    streamlit
    requests
    python-multipart

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 6. Add Source Code

Create the following files in the project folder:

-   `server.py` -- Whisper API server
-   `app.py` -- Streamlit UI client
-   `start.sh` -- start script

------------------------------------------------------------------------

# 7. Create Start Script

Create file:

`start.sh`

``` bash
#!/bin/bash

source venv/bin/activate

echo "Starting server..."

python server.py &

sleep 5

echo "Starting client..."

streamlit run app.py --server.maxUploadSize=1024
```

Make script executable:

``` bash
chmod +x start.sh
```

------------------------------------------------------------------------

# 8. Run the Application

Start everything with one command:

``` bash
./start.sh
```

The web UI will open at:

    http://localhost:8501

------------------------------------------------------------------------

# 9. Download Whisper Models

Inside the UI you can download models:

-   tiny
-   base
-   small
-   medium
-   large

Models are stored in:

    models/

Example:

    models/
       base.pt
       small.pt
       medium.pt

Once downloaded, they will not download again.

------------------------------------------------------------------------

# 10. Supported Features

✔ Audio → Text\
✔ Video → Text\
✔ MP3 / WAV / M4A support\
✔ MP4 / MOV / MKV video support\
✔ Large file upload\
✔ Local Whisper processing\
✔ Apple Silicon GPU (MPS) support

------------------------------------------------------------------------

# 11. Troubleshooting

### ffmpeg not found

Install again:

``` bash
brew install ffmpeg
```

### Python module missing

Reinstall dependencies:

``` bash
pip install -r requirements.txt
```

### Server not starting

Check logs:

    python server.py

------------------------------------------------------------------------

# 12. Useful Commands

Activate environment

``` bash
source venv/bin/activate
```

Run server manually

``` bash
python server.py
```

Run UI manually

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

# Done

You now have a fully local **Audio / Video → Text transcription tool**
running with Whisper.
