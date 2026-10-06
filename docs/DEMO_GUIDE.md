# System Demonstration & Operational CLI Guide
## AI Driver Monitoring & Drowsiness Detection System

**Document:** Comprehensive Demonstration Guide  
**Location:** `docs/DEMO_GUIDE.md`  
**Target Systems:** Commodity CPU Workstations (Intel x86_64, Windows / Linux / macOS)  
**Verification:** All commands verified against `scripts/run_inference.py` CLI parser.  

---

## 1. Quickstart Prerequisites

Before running demonstrations, ensure your environment is activated and dependencies are installed:

```bash
# Activate your virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Verify Python version (3.10 to 3.13 supported)
python --version

# Verify test suite integrity
pytest -v
```

All 39 automated unit tests must pass before running live inference.

---

## 2. Canonical Demonstration Commands

### Demo 1 — Interactive Video with Graphical HUD Display
Processes an existing driving video session, runs multi-signal fusion, and renders an interactive graphical HUD window with live telemetry and bounding boxes.

```bash
python scripts/run_inference.py \
    --source test_video.mp4 \
    --backend pytorch \
    --display
```

- **Runtime Backend:** PyTorch CPU (Fused FP32)
- **Display:** OpenCV graphical window rendering bounding boxes, facial landmarks, and telemetry dashboard
- **Keyboard Control:** Press `q` or `Esc` in the video window to gracefully exit at any time

---

### Demo 2 — Headless Server / Container Batch Mode
Runs high-throughput inference without opening a GUI window. Ideal for batch processing, automated CI/CD pipelines, Docker containers, and headless cloud instances.

```bash
python scripts/run_inference.py \
    --source test_video.mp4 \
    --backend pytorch \
    --headless
```

- **GUI Window:** Suppressed
- **Throughput:** ~27.5 Wall FPS / 34.8 Loop FPS on Intel Core i7-10700 CPU
- **Output:** Saves annotated video to `runs/detect/output.mp4` by default

---

### Demo 3 — Live Physical Webcam Monitoring
Connects directly to an attached USB or integrated webcam (device index `0`) for real-time in-cabin driver monitoring.

```bash
python scripts/run_inference.py \
    --source 0 \
    --backend pytorch \
    --display
```

- **Camera Device:** Webcam index 0 (use `1` or `2` for external USB cameras)
- **Features Active:**
  - Automatic driver face centroid tracking and selection
  - Real-time EAR, MAR, PERCLOS, and 3D head pose estimation
  - Dynamic state machine (`NORMAL` → `WARNING` → `CRITICAL` → `RECOVERY`)
  - Speech suppression filter (prevents false alerts while talking)
- **Fault Handling:** Graceful recovery if frames drop; enters `FACE_LOST` state after 2.0s without a detected driver face

---

### Demo 4 — Lightweight Portable ONNX Runtime
Executes using the standalone ONNX export (`weights/phase2f_best.onnx`) without loading the PyTorch neural network runtime.

```bash
python scripts/run_inference.py \
    --source test_video.mp4 \
    --backend onnx \
    --headless
```

- **Runtime Backend:** ONNX Runtime CPU (`CPUExecutionProvider`)
- **Throughput:** 22.29 Wall FPS
- **Detection Parity:** Identical to PyTorch (`mAP@0.5 = 0.9877`)
- **Use Case:** Edge devices and embedded Linux systems where PyTorch is not installed

---

## 3. Advanced Operational Configurations

### A. High-Frequency Telemetry Streaming
Stream real-time numerical driver metrics (timestamps, EAR, MAR, PERCLOS, head pose angles, fatigue score, alert state) directly to a CSV log file:

```bash
python scripts/run_inference.py \
    --source test_video.mp4 \
    --backend pytorch \
    --headless \
    --telemetry \
    --telemetry-output logs/driver_telemetry.csv
```

### B. Benchmark & Latency Profiling
Run a fixed-duration benchmark measuring precise wall-clock latency, loop latency, and component throughput:

```bash
python scripts/run_inference.py \
    --source test_video.mp4 \
    --backend pytorch \
    --headless \
    --benchmark \
    --duration 10.0
```

### C. Customizing Inference Parameters

| Argument | Flag | Default | Description |
|:---|:---|:---|:---|
| Source | `--source <path/index>` | `test_video.mp4` | Video file path or webcam index (`0`) |
| Backend | `--backend {pytorch,onnx,openvino}` | `pytorch` | Execution engine backend |
| Confidence | `--conf <float>` | `0.25` | YOLO detection confidence threshold |
| NMS IoU | `--iou <float>` | `0.45` | Non-Maximum Suppression IoU threshold |
| Frame Skip | `--skip-frames <int>` | `2` | Interval between YOLO forward passes (2 = every 3rd frame) |
| Max Frames | `--max-frames <int>` | `None` | Terminate processing after N frames |
| Output Path | `--output <path>` | `None` | Custom path to save rendered video |
| No Save | `--no-save` | `False` | Disable writing output video to disk (saves I/O) |
| Landmarks | `--enable-landmarks` / `--disable-landmarks` | Enabled | Toggle MediaPipe 478-point mesh extraction |

---

## 4. Expected Console Summary Output

At the conclusion of any inference run, the system outputs a standardized verification summary:

```text
==================================================
INFERENCE SUMMARY:
Frames Processed: 1500
Average Loop FPS: 34.79
Wall Clock FPS:   27.48
Total Time (s):   54.58
Output Video:     runs/detect/output.mp4
Detections:       {'eyes_closed': 142, 'eyes_open': 1289, 'yawning': 69}
Detector Stats:   {'backend': 'pytorch', 'device': 'cpu', 'skip_frames': 2}
Physiological:    {'mean_ear': 0.284, 'mean_mar': 0.312, 'p80_perclos': 0.082}
==================================================
```

---

## 5. Troubleshooting & FAQ

1. **Webcam Not Opening:**
   - On Windows, ensure camera permissions are enabled in Windows Settings -> Privacy & Security -> Camera.
   - If device `0` is in use by another application, try `--source 1`.
2. **Slow FPS on CPU Workstations:**
   - Verify that `--skip-frames 2` is enabled. Running YOLO on every single frame drops throughput to ~18 FPS, whereas `skip_frames = 2` achieves 27.5+ FPS while preserving physiological fidelity.
3. **Missing OpenCV GUI on Headless Servers:**
   - Always append `--headless` when running in Docker or SSH terminals to prevent GUI initialization errors.
