#!/bin/bash
# Start Flask app with Gunicorn (Azure-compatible)
exec gunicorn --bind 0.0.0.0:8000 app:app --workers 2 --timeout 600

