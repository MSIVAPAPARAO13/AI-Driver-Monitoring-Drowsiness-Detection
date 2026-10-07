# Production Deployment Guide — AeroDMS Sentinel

This guide provides end-to-end instructions for deploying **AeroDMS Sentinel** (AI Driver Monitoring & Drowsiness Detection System) across local, containerized Docker, and cloud environments (Streamlit Community Cloud, Hugging Face Spaces, Render, AWS/GCP).

---

## 1. System Architecture Overview

```text
Browser Client (HTTPS)
       │
       ▼ (SRTP / UDP Audio & Video)
 WebRTC Stream (aiortc)
       │
       ▼
Streamlit Application Server (app/app.py)
       │
       ├── DriverMonitoringEngine (In-Memory Processing)
       │     ├── YOLOv5nu (weights/phase2f_best.pt, PyTorch CPU)
       │     ├── MediaPipe Face Mesh (face_landmarker.task)
       │     └── Temporal Sensor Fusion & State Machine
       └── ContinualLearningManager (Asynchronous Worker Thread)
```

---

## 2. Resource & Hardware Requirements

| Parameter | Recommended Minimum | Production Ideal | Notes |
|:---|:---:|:---:|:---|
| **CPU** | 2 vCPU (x86_64) | 4 vCPU @ &ge;2.5 GHz | Tested on Intel Core i7-10700 CPU |
| **RAM** | 2.0 GB | 4.0 GB | In-memory frame processing requires ~426 MB peak RSS |
| **GPU** | Not required | Optional CUDA | Primary runtime optimized for CPU inference |
| **Disk Space** | 2.5 GB | 5.0 GB | Model checkpoints total &lt;10 MB; dependencies require ~1.8 GB |
| **Network** | Broadband | Low-latency broadband | WebRTC stream runs at 640x480 @ 30 FPS (~1.5 Mbps) |

---

## 3. WebRTC & HTTPS Security Requirements

> [!IMPORTANT]
> Modern web browsers (Chrome, Firefox, Safari, Edge) strictly enforce the **Secure Contexts specification** (`navigator.mediaDevices.getUserMedia`). Camera access **will be blocked** unless the application is served over **HTTPS** (or `http://localhost`).

### STUN / ICE Server Configuration
The WebRTC video streamer is pre-configured with Google public STUN servers in `app/components/live_monitor.py`:
```python
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)
```
For enterprise networks behind strict symmetric NATs or corporate firewalls, a TURN server relay (e.g., Coturn or Twilio Network Traversal) may be added to `RTC_CONFIG`.

---

## 4. Deployment Option A: Streamlit Community Cloud (Recommended Free Tier)

Streamlit Community Cloud offers zero-configuration deployment directly from GitHub.

1. Fork or push the repository to GitHub:
   `https://github.com/MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection`
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Select repository: `MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection`.
4. Branch: `main`.
5. Main file path: `app/app.py`.
6. App settings / advanced:
   - Python version: `3.11` or `3.12`.
   - Linux system packages: Handled automatically by `packages.txt` (`libgl1`, `libglib2.0-0`, `libgomp1`, `ffmpeg`).
   - Python dependencies: Handled automatically by `requirements.txt`.
7. Click **Deploy**. Streamlit Cloud automatically provisions an HTTPS endpoint with camera permissions.

---

## 5. Deployment Option B: Docker Container

A multi-stage minimal CPU deployment container is provided in `Dockerfile`:

### Build Image
```bash
docker build -t aerodms-sentinel:latest .
```

### Run Container
```bash
docker run -d \
  -p 8501:8501 \
  --name aerodms \
  --restart unless-stopped \
  aerodms-sentinel:latest
```

Access the application at `http://localhost:8501`.

---

## 6. Deployment Option C: Hugging Face Spaces (Docker or Streamlit SDK)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Space SDK: Select **Streamlit** or **Docker**.
3. Set Space visibility to **Public**.
4. Push the repository to the Space git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/aerodms-sentinel
   git push space main
   ```
5. Hugging Face automatically exposes a free, persistent HTTPS endpoint with full WebRTC browser camera support.

---

## 7. Local Production Startup

```bash
# 1. Clone repository
git clone https://github.com/MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection.git
cd AI-Driver-Monitoring-Drowsiness-Detection

# 2. Activate virtual environment
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/macOS

# 3. Verify automated tests
pytest -v

# 4. Launch production Streamlit application
streamlit run app/app.py --server.port 8501 --server.headless true
```

---

## 8. Troubleshooting & Health Checks

| Symptom | Probable Cause | Resolution |
|:---|:---|:---|
| **Camera access denied / Error** | Served over insecure HTTP | Access via `http://localhost` or configure an HTTPS reverse proxy (Caddy / Nginx / Cloudflare). |
| **WebRTC video black screen** | NAT / ICE negotiation failed | Check that UDP ports for WebRTC aren't blocked by host firewall. Ensure STUN servers in `live_monitor.py` are reachable. |
| **`libGL.so.1` missing error** | Missing Linux GUI libraries | Ensure `packages.txt` is present or install `apt-get install -y libgl1 libglib2.0-0`. |
| **High memory consumption** | Multiple sessions accumulating history | `DriverMonitoringEngine` bounds rolling history to `deque(maxlen=150)`. Click "Reset Session Telemetry" or restart worker. |
| **Slow inference (<15 FPS)** | CPU throttling or battery saving mode | Plug workstation into AC power. Set PyTorch thread count via `torch.set_num_threads(4)`. |
