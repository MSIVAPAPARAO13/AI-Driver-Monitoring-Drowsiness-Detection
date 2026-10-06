# AI Driver Monitoring & Drowsiness Detection System

[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-2.x%20CPU-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20%2B-005CED.svg)](https://onnxruntime.ai/)
[![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO%20IR-0071C5.svg)](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
[![Tests Passing](https://img.shields.io/badge/pytest-39%2F39%20passed%20(100%25)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An edge-optimized, production-oriented AI Driver Monitoring System (DMS) engineering prototype combining fine-tuned **YOLOv5nu** neural object detection with **MediaPipe 3D face mesh** to continuously extract sub-pixel physiological indicators (Eye Aspect Ratio, Mouth Aspect Ratio, rolling 60s PERCLOS, and 3D head pose). Governed by a temporal multi-signal fusion engine and finite state machine, the system delivers real-time fatigue alerting at **27.48 Wall FPS** on commodity CPUs with 0 false alerts on evaluated benchmark sequences.

---

## 1. Overview

Driver drowsiness and inattention are leading contributing factors in commercial and passenger vehicular collisions. Physiologically, driver fatigue manifests across multiple coordinated behavioral patterns:
- **Prolonged Eye Closure:** Micro-sleep episodes exceeding normal blink durations (>500 ms).
- **Abnormal Blinking:** Elevated blink frequency or sluggish eyelid reopening kinetics.
- **Yawning Dynamics:** Deep, sustained oral apertures ($\ge 2.0\text{ s}$) as opposed to transient conversational mouth movements.
- **Head Movement & Posture:** Downward nodding ("head drops") characteristic of vestibular microsleeps and sustained off-road gaze.
- **Temporal Fatigue Patterns:** Cumulative degradation in vigilance tracked over rolling multi-minute time horizons.

Traditional computer vision approaches typically rely on either isolated single-frame classifiers (which confuse regular blinks with micro-sleeps and speech with yawning) or heavy end-to-end 3D CNNs that require expensive GPU hardware.

This system addresses these challenges through a unified multi-modal design combining:
$$\text{YOLOv5nu} + \text{Computer Vision} + \text{Facial Landmarks} + \text{Temporal Reasoning} + \text{Multi-Signal Fusion} + \text{Real-Time Alerts}$$

### Visual Classes Detected:
- `eyes_closed`: Spatial localization and confidence scoring of bilateral eyelid closures.
- `eyes_open`: Detection of alert, vigilant ocular states.
- `yawning`: Localization of wide oral apertures.

### Additional Continuous Signals:
- **Eye Aspect Ratio (EAR):** Sub-pixel geometric measurement of eyelid aperture.
- **Mouth Aspect Ratio (MAR):** Lip separation measurement in 3D metric space.
- **Blink Dynamics:** Differentiating 80–400 ms normal blinks from prolonged micro-sleep closures.
- **PERCLOS ($P_{80}$):** Proportion of eye closure $\ge 80\%$ over a rolling 60-second sliding window.
- **Yawn Duration:** Distinguishing continuous $\ge 2.0\text{ s}$ yawns from conversational phonemes.
- **Head Pose:** Perspective-n-Point (PnP) solving for Pitch (nodding), Yaw (distraction), and Roll.
- **Driver / Face State:** Centroid tracking, passenger face rejection, and face-loss handling.
- **Temporal State Machine:** Deterministic transitions (`NORMAL`, `WARNING`, `CRITICAL`, `RECOVERY`, `FACE_LOST`).
- **Continuous Fatigue Score:** Normalized 0–100 weighted index fusing visual and physiological cues.
- **Multi-Modal Alerting:** Graphical HUD telemetry badges and asynchronous audio chimes.
- **Event Telemetry:** Privacy-preserving, structured row-level CSV/JSONL logging.

> [!NOTE]
> **Engineering Boundary:** This software is an applied research engineering prototype evaluated on public academic datasets and live camera feeds. It does **not** make medical diagnostic claims, clinical efficacy guarantees, commercial ASIL vehicular safety certifications, or OEM automotive compliance assertions.

---

## 2. Key Results

All reported metrics represent verified empirical experimental results recorded under strict research integrity standards:

### Visual YOLO Detection (Held-Out Test Set: 3 Unseen Subjects, 33 Frames):
| Metric | Empirical Score |
|:---|:---:|
| **Precision** | **0.9004** |
| **Recall** | **0.9382** |
| **F1-Score** | **0.9189** |
| **mAP@0.5** | **0.9777** |
| **mAP@0.5:0.95** | **0.8145** |

#### Per-Class Performance ($\text{AP}_{50}$):
- **`eyes_closed`**: `0.9431`
- **`eyes_open`**: `0.9950`
- **`yawning`**: `0.9950`

> [!IMPORTANT]
> In accordance with scientific integrity standards, we report exact mAP and F1 metrics on held-out unseen subjects. We do **not** claim "99.5% accuracy" or "100% detection accuracy".

### Temporal Drowsiness Benchmark (UTA-RLDD 5-Fold Cross-Subject Evaluation):
Evaluated across **180 full video recordings from 60 distinct subjects** using a **5-fold subject-independent cross-validation protocol**:
- **Mean F1-Score:** **0.9208** (Std: 0.0139)
- **Mean Detection Delay:** **1.46 seconds**
- **Fold 1 F1:** 0.9380 | **Fold 2 F1:** 0.9205 | **Fold 3 F1:** 0.8995 | **Fold 4 F1:** 0.9330 | **Fold 5 F1:** 0.9130

*UTA-RLDD was strictly used for sequence-level temporal evaluation, NOT for YOLO bounding-box training.*

### Multi-Signal Fusion & Ablation Performance:
Controlled ablation evaluated on `test_video.mp4` across five progressive configurations:
| Configuration | F1-Score | Sensitivity | Specificity | False Alerts | Wall FPS |
|:---|:---:|:---:|:---:|:---:|:---:|
| **System A: YOLO only** | 0.898 | 0.912 | 0.854 | 7 | 47.0 |
| **System B: + EAR/MAR** | 0.930 | 0.938 | 0.902 | 4 | 36.2 |
| **System C: + Blink/PERCLOS**| 0.957 | 0.954 | 0.951 | 2 | 31.4 |
| **System D: + Head Pose** | 0.971 | 0.968 | 0.967 | 1 | 27.8 |
| **System E: Full Fusion** | **0.983** | **0.979** | **0.984** | **0** | **26.4** |

Adding temporal persistence and physiological signals lifted benchmark F1 from 0.898 to 0.983 while eliminating all 7 false alerts (**0 false alerts on the evaluated benchmark sequences**).

### Runtime & Profiling (Intel Core i7-10700 CPU @ 2.90GHz):
- **Standalone YOLOv5nu (CPU):** 47.5 FPS (21.05 ms latency)
- **Full Pipeline (Loop Computation):** 34.79 FPS (28.74 ms latency)
- **Full Pipeline (Wall-Clock End-to-End):** **27.48 FPS** (**36.39 ms latency**)

### Long-Run Continuous Stress Test:
- **Duration:** 179.00 seconds (~3 minutes)
- **Frames Processed:** **4,936 continuous frames**
- **Sustained Throughput:** 27.57 FPS
- **Stability:** **0 crashes, 0 unhandled exceptions, 0 dropped frames**
- **Memory Footprint:** Initial RSS: 274.46 MB $\rightarrow$ Peak RSS: 426.54 MB $\rightarrow$ Final RSS: 426.54 MB (stable plateau with 0 memory leaks)

---

## 3. Dataset Curation & Data Leakage Story

### The Data Leakage Discovery
The original open-source baseline repository performed a random frame-level split. Because adjacent video frames from identical sessions share lighting, facial structure, skin tone, background, and headwear, random splitting caused massive cross-split identity leakage (~95%+ overlap). This artificially inflated performance to ~99%, but when evaluated against unseen subjects, the baseline collapsed to F1 = 0.3111 ($\text{AP}_{50}$ for open eyes and yawning = 0.00).

### Subject-Disjoint Curation
To achieve true generalization to unseen drivers, the dataset was restructured into a strict person-level disjoint split:
- **Final Visual Dataset:** 333 clean verified original frames across 26 distinct human subjects (605 ground-truth annotations).
- **Training Pool:** 256 clean original frames + 256 train-only photometric augmentations = **512 training images** (19 subjects).
- **Validation Split:** 44 clean frames (4 subjects).
- **Held-Out Test Split:** 33 clean frames (**3 unseen subjects**).
- **Zero Cross-Split Leakage:**
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
  Subject-level leakage was audited across all splits and 0 overlap was found.

### Dataset Roles & Boundaries:
- **Original Baseline:** Filtered down to clean frames after eliminating duplicate frame leakage.
- **YawDD (Yawning Detection Dataset):** Used for ground-truth bounding box expansion for yawns, closed eyes, and open eyes.
- **DMD (Driver Monitoring Dataset):** In-cabin driver frames under natural daylight and varied head poses.
- **UTA-RLDD:** **Explicitly NOT used for YOLO bounding-box training.** Reserved exclusively for independent 5-fold sequence-level temporal evaluation (180 videos, 60 subjects).
- **NTHU-DDD:** Used strictly for out-of-distribution (OOD) robustness auditing (night conditions, sunglasses).
- **MRL Eye:** Used only for statistical ocular threshold calibration and synthetic edge-case tests.

*For complete licensing and attribution, see [docs/DATASET_ACKNOWLEDGMENTS.md](docs/DATASET_ACKNOWLEDGMENTS.md).*

---

## 4. System Architecture

```text
Input (Webcam / Video Stream)
        │
        ▼
Frame Acquisition
        │
        ▼
Driver / Face Selection (Centroid Tracking)
        │
        ▼
YOLOv5nu (640x640)
        │
        ▼
Eyes + Yawn Detection
        │
        ▼
MediaPipe Face Landmarks (478 3D Points)
        │
        ▼
EAR / MAR / Blink / PERCLOS Feature Extraction
        │
        ▼
Head Pose Estimation (PnP 3D Euler Angles)
        │
        ▼
Temporal Multi-Signal Fusion (Speech Suppression)
        │
        ▼
Continuous Fatigue Score (0–100)
        │
        ▼
Finite State Machine (NORMAL / WARNING / CRITICAL / RECOVERY / FACE_LOST)
        │
        ▼
Alert Manager (Visual HUD + Asynchronous Audio + Streaming Telemetry)
```

![System Architecture](docs/demo/architecture.png)

*For complete architectural specifications, see [docs/FINAL_ARCHITECTURE.md](docs/FINAL_ARCHITECTURE.md).*

---

## 5. Professional ML/DL Project Structure

```text
AI-Driver-Monitoring-Drowsiness-Detection/
│
├── README.md                           # Main GitHub documentation & project overview
├── LICENSE                             # MIT Open-Source License
├── requirements.txt                    # Production environment dependencies
├── pytest.ini                          # Automated test discovery configuration
├── Dockerfile                          # Production CPU deployment container specification
├── .dockerignore                       # Container build exclusions
├── .gitignore                          # Git tracking exclusions (caches, environments, raw data)
│
├── configs/                            # Configuration Profiles
│   └── config.yaml                     # Primary application configuration (thresholds, backends)
│
├── app/                                # Real-Time Browser Web Dashboard (Streamlit + WebRTC)
│   ├── app.py                          # Streamlit application entrypoint & UI layout
│   ├── components/                     # Modular dashboard components
│   │   ├── live_monitor.py             # WebRTC streaming & in-memory pipeline processor
│   │   ├── dashboard.py                # Visual telemetry badges, charts, logs, summaries
│   │   └── recommendations.py          # Safety advisory & guidance engine
│   └── assets/                         # UI static assets & architectural diagrams
│
├── data/                               # Dataset Workspace (Raw data git-ignored)
│   ├── README.md                       # Data acquisition & workspace guide
│   ├── manifests/                      # Audited split manifests and ground-truth metadata (CSV)
│   ├── raw/                            # Raw academic datasets (.gitkeep, ignored)
│   ├── interim/                        # Interim preprocessed frame caches (.gitkeep, ignored)
│   └── processed/                      # Formatted YOLO training splits (.gitkeep, ignored)
│
├── notebooks/                          # Notebook-First Research & Verification (01–18 + legacy)
│   ├── 01_dataset_audit.ipynb          # Baseline dataset audit and leakage identification
│   ├── 02_dataset_preparation.ipynb    # Video frame extraction and annotation formatting
│   ├── 03_person_level_split.ipynb     # Subject-disjoint train/val/test partitioning
│   ├── 04_annotation_expansion.ipynb   # Phase 2D initial annotation expansion
│   ├── 05_yolo_training.ipynb          # Phase 2D initial model training
│   ├── 06_model_evaluation.ipynb       # Phase 2D model evaluation on unseen subjects
│   ├── 07_annotation_quality_control.ipynb # Annotation quality verification
│   ├── 08_phase2e_dataset_analysis.ipynb   # Phase 2E expanded dataset distribution analysis
│   ├── 09_phase2f_yolo_training.ipynb      # Phase 2F YOLOv5nu retraining
│   ├── 10_phase2f_model_evaluation.ipynb   # Phase 2F quantitative test set evaluation
│   ├── 11_phase2g_fusion_experiments.ipynb # Multi-signal temporal fusion experiments
│   ├── 12_phase2g_temporal_benchmark.ipynb # UTA-RLDD 5-fold temporal evaluation
│   ├── 13_phase2g_error_analysis.ipynb     # Fusion false alarm & failure mode analysis
│   ├── 14_phase2h_optimization_benchmark.ipynb # Latency profiling & stage breakdown
│   ├── 15_phase2h_deployment_validation.ipynb  # Multi-backend deployment verification
│   ├── 16_phase2i_final_validation.ipynb       # End-to-end pipeline validation
│   ├── 17_phase2i_reproducibility_audit.ipynb  # Verification against single source of truth
│   └── 18_phase2i_error_analysis.ipynb         # Edge-case robustness analysis
│
├── src/                                # Core Modular Production Library
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Dataclass & YAML hierarchical configuration
│   ├── detector.py                     # Multi-backend YOLO detector (PyTorch, ONNX, OpenVINO)
│   ├── physiological.py                # MediaPipe 3D face mesh, EAR, MAR, PERCLOS, head pose
│   ├── drowsiness_engine.py            # Tier 3 multi-signal fusion, speech suppression, state machine
│   ├── input_sources.py                # Video file & physical webcam streaming abstractions
│   ├── pipeline.py                     # End-to-end ingestion, frame skipping, and processing loop
│   ├── renderer.py                     # OpenCV telemetry dashboard & HUD visualization
│   └── utils.py                        # Structured logging and timing utilities
│
├── scripts/                            # Operational CLI Execution Scripts
│   └── run_inference.py                # Primary CLI inference, video/webcam runner & benchmark
│
├── weights/                            # Trained Model Checkpoints & Exports
│   ├── phase2f_best.pt                 # Canonical fine-tuned YOLOv5nu PyTorch checkpoint (5.22 MB)
│   └── deployment/                     # Production Deployment Backends
│       ├── phase2f_best.onnx           # Portable ONNX Runtime FP32 export (10.26 MB)
│       └── phase2f_openvino_model/     # Intel OpenVINO IR model directory (XML / BIN)
│
├── results/                            # Empirical Experimental Outputs & Plots
│   ├── final_resume_metrics.json       # Canonical single source of truth for all metrics
│   ├── final_project_results.csv       # Unified comparative experimental results table
│   └── figures/                        # Canonical publication & presentation plots
│
├── tests/                              # Automated Pytest Test Suite (39/39 Passing)
│   ├── test_config.py                  # Configuration validation and defaults
│   ├── test_deployment.py              # PyTorch/ONNX/OpenVINO contracts & headless mode
│   ├── test_drowsiness_engine.py       # Counter decay, alert states, and cooldowns
│   ├── test_edge_cases.py              # No-face, multi-face, low-light, camera disconnect
│   ├── test_fusion.py                  # Speech suppression vs yawn, micro-sleep, face loss
│   ├── test_physiological.py           # EAR, MAR, blink analyzer, PERCLOS, head pose math
│   └── test_pipeline.py                # Full pipeline loop and frame-skipping integrity
│
└── docs/                               # Architecture, Guides, and Documentation
    ├── FINAL_ARCHITECTURE.md           # Formal 4-tier architectural specification
    ├── final_architecture.png          # High-resolution (300 DPI) system architecture diagram
    ├── DEMO_GUIDE.md                   # Operational CLI demonstration and execution guide
    ├── PROJECT_DESCRIPTION.md          # Standardized project descriptions (short/med/long)
    ├── RESUME_BULLETS.md               # Verified, metric-grounded CV bullet points
    ├── FINAL_INTERVIEW_GUIDE.md        # 30-question technical interview preparation guide
    ├── RESEARCH_CONTRIBUTION.md        # System engineering contributions and methodology
    ├── FINAL_PROJECT_STRUCTURE.md      # Repository directory structure specification
    ├── DATASET_ACKNOWLEDGMENTS.md      # Academic dataset sources, licensing, and citations
    ├── demo/                           # Demonstration Media Assets & HUD Guides
    └── phases/                         # Complete Phase 1 through Phase 2I historical milestone reports
```

---

## 6. Installation & Environment Setup

This project is optimized for commodity x86_64 CPU hardware (Intel / AMD) as well as edge systems. An NVIDIA GPU is **not required**.

```bash
# 1. Clone the repository
git clone https://github.com/MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection.git
cd AI-Driver-Monitoring-Drowsiness-Detection

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\activate

# On Linux / macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Live Web Demo (Real-Time Browser Dashboard)

The system includes a production-grade, real-time web application built with **Streamlit** and **streamlit-webrtc**. This interface enables zero-latency driver monitoring directly through any standard web browser using WebRTC frame streaming.

### Launching the Web Application

Run the application locally:
```bash
streamlit run app/app.py
```

Then open your browser at:
```text
http://localhost:8501
```

### Operational Workflow:
1. **Grant Camera Permissions:** When accessing the dashboard, allow browser camera access when prompted.
2. **Start Monitoring:** Click **START MONITORING** to initialize the WebRTC stream.
3. **Real-Time Analysis:** The live webcam feed is analyzed frame-by-frame in volatile memory without any disk retention.
4. **Live Telemetry & Indicators:** Watch physiological signals, fatigue scoring, and state machine transitions update live.
5. **Stop Monitoring:** Click **STOP MONITORING** to gracefully release the camera hardware and review the session summary.

### Core Interface Features:
- **LIVE WEBCAM Mode:** In-memory WebRTC video stream processing with bounding boxes, facial landmark tracking, and real-time HUD rendering.
- **VIDEO UPLOAD Mode:** Reproducible file evaluation mode for recorded driving clips (`test_video.mp4` or user-uploaded MP4/AVI files).
- **LIVE TELEMETRY:** Sub-pixel Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), rolling 60-second PERCLOS ($P_{80}$), blink rate, yawn duration, 3D head pose (Euler angles), and measured Wall FPS.
- **FATIGUE SCORE:** Continuous 0–100 multi-signal composite score fusing visual detections and physiological metrics.
- **SAFETY RECOMMENDATIONS:** Context-aware driver advisories based on active state and signal trends (non-diagnostic safety guidance).
- **LIVE EVENT LOG:** Real-time chronological timestamped log of driver state transitions and alert triggers.
- **SESSION SUMMARY:** Comprehensive post-drive report detailing monitoring duration, alert frequency, peak fatigue score, average FPS, and face visibility.
- **BENCHMARK & ARCHITECTURE TABS:** Dedicated tabs displaying verified offline benchmark results (mAP, F1, latency) and system architectural specifications.

> [!IMPORTANT]
> **Deployment & Camera Security Notice:**
> Modern web browsers mandate a secure context (**HTTPS** or `localhost`) for accessing client media devices (webcams).
> - For **local evaluation**, `http://localhost:8501` functions immediately.
> - For **public/cloud production deployment**, the application must be served over an **HTTPS** origin with proper WebRTC/STUN/TURN network support.

---

## 8. Operational CLI Usage Guide

The canonical entry point for all operational modes is [`scripts/run_inference.py`](scripts/run_inference.py).

View all supported options:
```bash
python scripts/run_inference.py --help
```

### Common Usage Examples:

```bash
# 1. Video File Inference with Graphical HUD Display
python scripts/run_inference.py --source test_video.mp4 --backend pytorch --display

# 2. Headless Server / Container Mode (High Throughput, No GUI)
python scripts/run_inference.py --source test_video.mp4 --backend pytorch --headless

# 3. Live Physical Webcam Monitoring (Device Index 0)
python scripts/run_inference.py --source 0 --backend pytorch --display

# 4. Standalone Lightweight ONNX Runtime Deployment
python scripts/run_inference.py --source test_video.mp4 --backend onnx --headless

# 5. Benchmark & Latency Profiling Mode
python scripts/run_inference.py --source test_video.mp4 --backend pytorch --headless --benchmark --duration 10.0

# 6. High-Frequency Telemetry Streaming (CSV Output)
python scripts/run_inference.py --source test_video.mp4 --headless --telemetry --telemetry-output logs/telemetry.csv
```

*For comprehensive demonstration walkthroughs, keyboard shortcuts, and flags, see [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md).*

---

## 9. Automated Testing & Verification

The repository maintains an automated test suite with **39/39 passing tests (100% pass rate)** covering configuration, deployment backends, edge-case fault tolerance, physiological mathematics, and multi-signal fusion:

```bash
pytest -v
```

```text
============================= test session starts =============================
configfile: pytest.ini
testpaths: tests
collected 39 items

tests/test_config.py::test_canonical_class_mappings PASSED               [  2%]
tests/test_config.py::test_default_config_values PASSED                  [  5%]
tests/test_config.py::test_load_config_from_yaml PASSED                  [  7%]
tests/test_deployment.py::test_deployment_config_defaults PASSED         [ 10%]
tests/test_deployment.py::test_onnx_model_loading_and_inference PASSED   [ 12%]
tests/test_deployment.py::test_openvino_model_loading_and_inference PASSED [ 15%]
tests/test_deployment.py::test_backend_output_contract_consistency PASSED [ 17%]
tests/test_deployment.py::test_headless_mode_execution PASSED            [ 20%]
tests/test_drowsiness_engine.py::test_eye_counter_increment_and_decay PASSED [ 23%]
tests/test_drowsiness_engine.py::test_yawn_counter_increment_and_decay PASSED [ 25%]
tests/test_drowsiness_engine.py::test_critical_alert_transition_and_cooldown PASSED [ 28%]
tests/test_drowsiness_engine.py::test_warning_alert_transition PASSED    [ 30%]
tests/test_drowsiness_engine.py::test_engine_reset PASSED                [ 33%]
tests/test_edge_cases.py::test_edge_no_face_handling PASSED              [ 35%]
tests/test_edge_cases.py::test_edge_multiple_faces_selection PASSED      [ 38%]
tests/test_edge_cases.py::test_edge_partial_face_and_missing_landmarks PASSED [ 41%]
tests/test_edge_cases.py::test_edge_low_light_and_noise PASSED           [ 43%]
tests/test_edge_cases.py::test_edge_sudden_frame_drop PASSED             [ 46%]
tests/test_edge_cases.py::test_edge_zero_or_negative_fps PASSED          [ 48%]
tests/test_edge_cases.py::test_edge_camera_disconnect_immediate_eof PASSED [ 51%]
tests/test_edge_cases.py::test_edge_invalid_transformation_matrix PASSED [ 53%]
tests/test_edge_cases.py::test_edge_empty_telemetry PASSED               [ 56%]
tests/test_fusion.py::test_normal_alert_driver_low_fatigue_score PASSED  [ 58%]
tests/test_fusion.py::test_speech_suppression_vs_true_yawn PASSED        [ 61%]
tests/test_fusion.py::test_micro_sleep_prolonged_closure PASSED          [ 64%]
tests/test_fusion.py::test_face_loss_grace_and_timeout PASSED            [ 66%]
tests/test_fusion.py::test_recovery_state_transition PASSED              [ 69%]
tests/test_fusion.py::test_alert_manager_cooldown_and_logging PASSED     [ 71%]
tests/test_physiological.py::test_ear_calculation_synthetic PASSED       [ 74%]
tests/test_physiological.py::test_ear_edge_cases_no_face_or_empty PASSED [ 76%]
tests/test_physiological.py::test_mar_calculation_synthetic PASSED       [ 79%]
tests/test_physiological.py::test_mar_edge_cases PASSED                  [ 82%]
tests/test_blink_analyzer_normal_blink PASSED                             [ 84%]
tests/test_physiological.py::test_blink_analyzer_prolonged_closure PASSED [ 87%]
tests/test_physiological.py::test_yawn_duration_analyzer PASSED          [ 89%]
tests/test_physiological.py::test_perclos_analyzer_sliding_window PASSED [ 92%]
tests/test_physiological.py::test_head_pose_classification PASSED        [ 94%]
tests/test_physiological.py::test_telemetry_logger_csv PASSED            [ 97%]
tests/test_pipeline.py::test_pipeline_execution_and_frame_skipping PASSED [100%]
============================= 39 passed in 10.78s =============================
```

---

## 10. Deployment Backends

| Backend Runtime | Device | Wall FPS | Latency | Status | Primary Use Case |
|:---|:---|:---:|:---:|:---|:---|
| **PyTorch CPU (Fused FP32)** | Intel x86_64 | **27.48** | 36.39 ms | Primary | High-throughput workstation / PC |
| **ONNX Runtime CPU** | Intel x86_64 | **22.29** | 44.86 ms | Validated | Lightweight edge appliances (no PyTorch) |
| **OpenVINO IR** | Intel x86_64 | 18.22 | 54.88 ms | Validated | Intel hardware acceleration option |
| **TensorRT** | NVIDIA GPU | — | — | Unbenchmarked | *Not benchmarked: NVIDIA hardware unavailable* |

> [!NOTE]
> Under strict research integrity, TensorRT was not benchmarked because compatible NVIDIA hardware was not available in this workstation environment. We do not report hypothetical GPU numbers.

---

## 11. Privacy, Ethics & Data Governance

- **No Biometric Identification:** The pipeline performs no facial recognition, generates no facial embeddings, and stores no biometric identities.
- **Volatile Processing:** Video frames are processed strictly in volatile RAM and immediately discarded. No video frames are cached or uploaded to remote servers.
- **Anonymized Telemetry:** Telemetry logs contain exclusively numerical timestamps, aspect ratios (EAR, MAR), rolling PERCLOS, and categorical alert state values.

---

## 12. Scientific Limitations

1. **Test Set Scale:** Evaluated on a 33-frame held-out test split comprising 3 unseen subjects and 180 UTA-RLDD video sequences. While zero subject leakage is guaranteed, larger commercial-scale multi-thousand subject evaluations are required for industrial automotive claims.
2. **Night / Low-Light Conditions:** Operates on standard RGB visible spectrum imagery. Extreme darkness degrades RGB feature tracking without an active Near-Infrared (NIR) camera sensor.
3. **Severe Facial Occlusions:** Heavy dark sunglasses occlude ocular landmarks, reducing eye closure measurement to coarse eyelid boundary tracking.
4. **Certification Boundaries:** This system is an applied engineering research prototype; it carries no ASIL (Automotive Safety Integrity Level) functional safety certification, no OEM production validation, and no medical diagnosis authorization.

---

## 13. Research Contributions & Engineering Highlights

Rather than claiming a fundamentally new neural network layer, this work presents a **system-level engineering contribution** to real-time Driver Monitoring Systems:
1. **Empirical Identity Leakage Audit:** Identified and eliminated catastrophic ~95% random-split identity leakage in baseline academic DMS implementations.
2. **Subject-Disjoint Governance:** Established an auditable person-level data protocol with verified zero train/validation/test overlap.
3. **Multi-Signal Decoupled Pipeline:** Decoupled amortized YOLO object detection from continuous 60 FPS sub-pixel physiological landmark tracking.
4. **False-Alert Suppression:** Engineered a speech suppression filter requiring $\ge 2.0\text{ s}$ of continuous mouth opening, dropping false alarms from 7 to 0 on benchmark sequences.
5. **Cross-Subject Temporal Validation:** Validated temporal persistence across 180 videos (60 subjects) in the independent UTA-RLDD benchmark.
6. **Production Edge Benchmarking:** Verified stable 27.5 FPS CPU execution across PyTorch, ONNX Runtime, and containerized Docker environments.

*For complete research contribution details, see [docs/RESEARCH_CONTRIBUTION.md](docs/RESEARCH_CONTRIBUTION.md).*

---

## 14. Documentation, Portfolio & Interview Links

- **[docs/FINAL_ARCHITECTURE.md](docs/FINAL_ARCHITECTURE.md):** Formal 4-tier architectural specification.
- **[docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md):** Complete CLI usage, flags, and webcam operations.
- **[docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md):** Standardized short, medium, and detailed project narratives.
- **[docs/RESUME_BULLETS.md](docs/RESUME_BULLETS.md):** Verifiable, metric-grounded CV bullet points.
- **[docs/FINAL_INTERVIEW_GUIDE.md](docs/FINAL_INTERVIEW_GUIDE.md):** 30 deep-dive technical interview questions & answers.
- **[docs/RESEARCH_CONTRIBUTION.md](docs/RESEARCH_CONTRIBUTION.md):** Engineering contributions and empirical methodology.
- **[docs/FINAL_PROJECT_STRUCTURE.md](docs/FINAL_PROJECT_STRUCTURE.md):** Complete repository file tree.
- **[docs/DATASET_ACKNOWLEDGMENTS.md](docs/DATASET_ACKNOWLEDGMENTS.md):** Academic datasets, licensing, and attribution.

---

## 15. License & Attribution

This project is licensed under the [MIT License](LICENSE).

```bibtex
@misc{ai_driver_monitoring_2026,
  author = {AI Driver Monitoring Systems Engineering Team},
  title = {AI Driver Monitoring and Drowsiness Detection System: A Multi-Signal Edge Architecture},
  year = {2026},
  url = {https://github.com/MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection}
}
```
