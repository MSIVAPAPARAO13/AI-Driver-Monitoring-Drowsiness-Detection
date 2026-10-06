# System Architecture Specification
## AI Driver Monitoring & Safety System

**Project:** AI Driver Monitoring & Drowsiness Detection System  
**Document:** Final End-to-End System Architecture  
**Phase:** Phase 2I — Validation & Architectural Baseline  
**Date:** October 2026  
**Status:** COMPLETE & EMPIRICALLY BENCHMARKED

---

## 1. Architectural Overview

The AI Driver Monitoring System is structured into a modular **4-Tier Pipeline** designed for robust, real-time edge processing on commodity CPUs without requiring specialized accelerator hardware.

```
                           [ INPUT FRAME ]
                                  │
                                  ▼
                     [ DRIVER FACE SELECTION ]
            (Area + Frame Center Proximity + Centroid Smoothing)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       [ TIER 1: YOLO VISUAL ]       [ TIER 2: 3D FACE MESH ]
       (weights/phase2f_best.pt)     (MediaPipe 478 Landmarks)
       - eyes_closed                              │
       - eyes_open               ┌────────────────┼────────────────┐
       - yawning                 ▼                ▼                ▼
           │                   [ EAR & BLINK ]  [ MAR & YAWN ]  [ PERCLOS & POSE ]
           │                   - Euclidean EAR  - Euclidean MAR  - 60s P80 Window
           │                   - Closure Time   - Speech Filter  - 3D Euler Angles
           │                   - Blinks / Min   - True Yawn Time - Head Nod/Turn
           │                             │                │                │
           └─────────────────────────────┼────────────────┼────────────────┘
                                         ▼
                     [ TIER 3: TEMPORAL FUSION ENGINE ]
                     - Composite Fatigue Score (0 - 100)
                     - Finite State Machine:
                       NORMAL ──> WARNING ──> CRITICAL ──> RECOVERY
                                      ▲                      │
                                      └────── FACE_LOST ◄────┘
                                         │
                                         ▼
                         [ TIER 4: ALERT MANAGEMENT ]
                         - Asynchronous Non-Blocking Buzzer
                         - Multi-Threshold Cooldown Guard
                         - Structured Event Logging (CSV)
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  ▼                      ▼                      ▼
         [ VISUAL HUD OVERLAY ]  [ AUDIO ALERT BUZZER ]  [ TELEMETRY CSV ]
```

---

## 2. Tier-by-Tier Specification

### Ingestion & Driver Selection
- **Input Sources:** Video files (`VideoSource`), live webcams (`WebcamSource`), and headless streams (`BaseInputSource`).
- **Driver Selection:** When multiple faces or passengers are in view, the driver face is deterministically selected using a composite heuristic:
  $$\text{Score} = \text{BBox Area} \times 0.50 + (1.0 - \text{Normalized Distance to Center}) \times 0.30 + \text{Temporal Centroid Continuity} \times 0.20$$
  This prevents passenger face drift and maintains lock on the active operator.

### Tier 1: Visual Object Detection (YOLOv5nu)
- **Model Checkpoint:** `weights/phase2f_best.pt` (2.50M parameters, 4.99 MB).
- **Canonical Classes:**
  - Class 0: `eyes_closed`
  - Class 1: `eyes_open`
  - Class 2: `yawning`
- **Execution Policy:** Controlled frame skipping (`skip_frames: 2`) amortizes inference to every 3rd frame (14.14 ms amortized on CPU), while physiological signals run on every frame.

### Tier 2: Facial Landmark Physiological Signals
- **Landmark Engine:** MediaPipe FaceLandmarker extracting 478 3D metric landmarks.
- **Eye Aspect Ratio (EAR):** 6-point Euclidean landmark formula:
  $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
- **Mouth Aspect Ratio (MAR):** 8-point 3D lip Euclidean formula:
  $$\text{MAR} = \frac{\|p_{13} - p_{14}\| + \|p_{78} - p_{308}\|}{2 \|p_{61} - p_{291}\|}$$
- **Blink Dynamics:** Real-time closure duration tracking, separating natural blinks ($0.08 - 0.40\text{ s}$), long blinks ($0.40 - 0.80\text{ s}$), and micro-sleep events ($>0.80\text{ s}$).
- **Sliding-Window PERCLOS:** P80 percentage of frames in which eyes are $\ge 80\%$ closed across a rolling 60-second temporal window.
- **3D Head Pose:** PnP solving 3D Euler angles (Pitch, Yaw, Roll) to detect nodding (drowsiness) or sustained distracted gaze.

### Tier 3: Temporal Multi-Signal Fusion Engine
- **Composite Fatigue Score ($0 - 100$):**
  $$\text{Score} = 35 \cdot S_{\text{eyes}} + 25 \cdot S_{\text{PERCLOS}} + 15 \cdot S_{\text{blink}} + 15 \cdot S_{\text{yawn}} + 10 \cdot S_{\text{pose}}$$
- **Speech vs Yawn Suppression:** Brief mouth openings ($<1.5\text{ s}$) associated with talking or singing are filtered out, requiring $\ge 2.0\text{ s}$ sustained aperture for a confirmed yawn event.
- **State Machine Transitions:**
  - `NORMAL`: Score $<45.0$
  - `WARNING`: Score $\ge 45.0$ sustained for $\ge 1.0\text{ s}$
  - `CRITICAL`: Score $\ge 75.0$ sustained for $\ge 1.5\text{ s}$ or micro-sleep $\ge 1.5\text{ s}$
  - `RECOVERY`: Consecutive clean frames for $\ge 2.0\text{ s}$ required to demote back to `NORMAL`
  - `FACE_LOST`: Grace period of $1.0\text{ s}$, transitioning to `FACE_LOST` if driver face absent $>2.0\text{ s}$

### Tier 4: Alert Management & Output
- **Asynchronous Audio Alert:** Python background daemon thread manages audio beeps/buzzers (1000 Hz warning, 2000 Hz critical) without stalling video frames.
- **Cooldown Timers:** Enforces 2.5-second refractory period to eliminate chattering alert oscillations.
- **Visual HUD Renderer:** Dynamic status banners, eye/mouth metrics, and real-time FPS overlay.
- **Structured Telemetry Logging:** Asynchronous CSV event logger recording 17 discrete telemetry fields.

---

## 3. Multi-Backend Deployment Abstraction

The system encapsulates model execution through [src/detector.py](file:///c:/Users/msiva/Music/Drowsiness-Detection-using-YOLOv5/src/detector.py), exposing identical `DetectionResult` contracts regardless of backend:

| Deployment Backend | Model Checkpoint | Hardware Target | Wall Throughput | Wall Latency | Memory Footprint | Primary Use Case |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PyTorch CPU (Fused)** | `weights/phase2f_best.pt` | Intel / AMD x86_64 | **27.48 FPS** | **36.39 ms** | 482 MB (stable) | Primary standard deployment |
| **ONNX Runtime CPU** | `weights/deployment/phase2f_best.onnx` | Cross-platform CPU | **22.29 FPS** | **44.87 ms** | 527 MB (stable) | Standalone / containerized edge |
| **OpenVINO CPU** | `weights/deployment/phase2f_openvino_model` | Intel CPU / NPU | 14.10 FPS | 70.91 ms | 740 MB | Optional Intel hardware path |
| **Docker (CPU)** | Containerized python:3.11-slim | Cloud / Server | 25+ FPS | <40 ms | Self-contained | Cloud / server headless pipelines |
