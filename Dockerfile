# Multi-Signal Driver Monitoring & Drowsiness Detection System
# Phase 2H Minimal CPU Deployment Container
FROM python:3.11-slim

# Prevent interactive prompts and set Python unbuffered output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system libraries required by OpenCV, MediaPipe, and video I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configs, weights, and MediaPipe landmark task assets
COPY configs/ ./configs/
COPY src/ ./src/
COPY weights/ ./weights/
COPY scripts/ ./scripts/
COPY face_landmarker.task .

# Default execution: Run inference pipeline in headless mode
ENTRYPOINT ["python", "scripts/run_inference.py", "--headless"]
CMD ["--source", "test_video.mp4", "--backend", "pytorch"]
