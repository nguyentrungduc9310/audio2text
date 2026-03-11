#!/bin/bash

source venv/bin/activate

echo "Starting API..."

uvicorn server:app --host 127.0.0.1 --port 8000 &

sleep 3

echo "Starting UI..."

streamlit run app.py --server.maxUploadSize=5120