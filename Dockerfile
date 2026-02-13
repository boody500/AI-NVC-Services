# Use lightweight Python base
FROM python:3.10-slim

# Prevent Python from writing pyc files / buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install system dependencies (ffmpeg needed by faster-whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y supervisor

# Copy requirements
COPY requirements.txt .

# Install Python deps (torch + faster-whisper included here)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose the port Azure expects
EXPOSE 8000
ENV PORT=8000

# Run your app (update if you use FastAPI, Flask, etc.)
# Example: FastAPI with Uvicorn
#CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8000", "--workers", "4", "--threads", "4", "--timeout", "1800"]
CMD ["supervisord", "-c", "/app/supervisord.conf"]
