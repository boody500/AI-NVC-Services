#!/bin/bash
# Startup script for Azure
exec gunicorn --workers 4 --bind 0.0.0.0:${PORT:-8000} app:app
