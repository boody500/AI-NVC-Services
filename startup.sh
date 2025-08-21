#!/bin/bash

# Preload models before starting the app
echo "Preloading models..."
python -c "
from app import ensure_models_loaded
ensure_models_loaded()
print('Models loaded successfully')
"

# Start Gunicorn
gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 1 --preload app:app