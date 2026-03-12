#!/bin/bash

echo "======================================"
echo " Audio2Text PRO - Setup"
echo "======================================"

# -----------------------------
# Check python
# -----------------------------
echo "Checking Python..."

if ! command -v python3 &> /dev/null
then
    echo "Python3 not found. Please install Python."
    exit
fi

python3 --version


# -----------------------------
# Check ffmpeg
# -----------------------------
echo ""
echo "Checking ffmpeg..."

if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg not found."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installing ffmpeg with brew..."
        brew install ffmpeg
    else
        echo "Please install ffmpeg manually."
        exit
    fi
fi

ffmpeg -version | head -n 1


# -----------------------------
# Create venv
# -----------------------------
echo ""
echo "Creating virtual environment..."

python3 -m venv venv


# -----------------------------
# Activate venv
# -----------------------------
echo "Activating virtual environment..."

source venv/bin/activate


# -----------------------------
# Upgrade pip
# -----------------------------
echo ""
echo "Upgrading pip..."

pip install --upgrade pip


# -----------------------------
# Install requirements
# -----------------------------
echo ""
echo "Installing dependencies..."

pip install -r requirements.txt


# -----------------------------
# Create folders
# -----------------------------
echo ""
echo "Creating project folders..."

mkdir -p models
mkdir -p uploads


# -----------------------------
# Make start script executable
# -----------------------------
chmod +x start.sh


# -----------------------------
# Check Ollama (optional)
# -----------------------------
echo ""
echo "Checking Ollama (optional, for summarize/translate)..."

if ! command -v ollama &> /dev/null; then
    echo "Ollama not found (optional, for summarize/translate)"
    echo "Install: brew install ollama"
    echo "Then run: ollama pull qwen3.5:4b"
else
    echo "Ollama found!"
    echo "Pulling Qwen 3.5 model (4b)..."
    ollama pull qwen3.5:4b || echo "Could not pull model — you can do this later with: ollama pull qwen3.5:4b"
fi


echo ""
echo "======================================"
echo " Setup complete!"
echo ""
echo "Run the app with:"
echo ""
echo "   ./start.sh"
echo ""
echo "Open browser:"
echo "   http://localhost:8501"
echo "======================================"