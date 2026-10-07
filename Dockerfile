# Multi-Signal Driver Monitoring & Drowsiness Detection System
# AeroDMS Sentinel — Production Web & Container Deployment
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

# Copy source code, application, configs, weights, and assets
COPY configs/ ./configs/
COPY src/ ./src/
COPY weights/ ./weights/
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY results/ ./results/
COPY .streamlit/ ./.streamlit/
COPY face_landmarker.task .
COPY test_video.mp4 .

EXPOSE 8501

# Default execution: Run Streamlit Web Application
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
