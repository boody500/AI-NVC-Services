#!/bin/bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --keep-alive 5 --max-requests 100 app:app