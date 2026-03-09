#!/bin/bash

source venv/bin/activate

echo "Starting server..."

python server.py &

sleep 5

echo "Starting client..."

streamlit run app.py --server.maxUploadSize=1024