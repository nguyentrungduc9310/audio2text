#!/bin/bash
set -e

if [ -n "$OLLAMA_HOST" ]; then
    python3 -c "
import urllib.request, json, time, os, sys
host = os.environ['OLLAMA_HOST']
print(f'Waiting for Ollama at {host}...')
for i in range(30):
    try:
        r = urllib.request.urlopen(f'{host}/api/tags', timeout=3)
        data = json.loads(r.read())
        models = data.get('models', [])
        if models:
            names = [m.get('name','?') for m in models]
            print(f'Ollama ready with {len(models)} model(s): {names}')
            sys.exit(0)
        else:
            print(f'WARNING: Ollama is running but has no models.')
            print(f'Pull a model on your host: ollama pull qwen3.5:4b')
            print(f'The app will start without LLM features.')
            sys.exit(0)
    except Exception:
        pass
    print(f'  Ollama not reachable yet... ({i+1}/30)')
    time.sleep(2)
print('Warning: Could not connect to Ollama at ' + host)
print('Make sure ollama is running: ollama serve')
print('Starting app without LLM features...')
"
fi

uvicorn server:app --host 0.0.0.0 --port 8000 &
sleep 3
exec streamlit run app.py --server.maxUploadSize=5120
