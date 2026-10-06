# Demonstration Media & Screenshot Asset Catalog
## AI Driver Monitoring & Drowsiness Detection System

**Document:** Demonstration Asset Catalog & Manual Screen Capture Guide  
**Location:** `docs/demo/`  
**Status:** Canonical Release Assets Included; Operational HUD Frame Capture Specifications Defined  

---

## 1. Embedded Verified Assets (Included in Repository)

The following high-resolution, empirical artifacts are verified and embedded directly in `docs/demo/`:

| Asset File | Resolution / Source | Description |
|:---|:---|:---|
| [`architecture.png`](file:///c:/Users/msiva/Music/Drowsiness-Detection-using-YOLOv5/docs/demo/architecture.png) | 300 DPI (`docs/final_architecture.png`) | 4-Tier full system architecture diagram from ingestion to telemetry. |
| [`benchmark_result.png`](file:///c:/Users/msiva/Music/Drowsiness-Detection-using-YOLOv5/docs/demo/benchmark_result.png) | High-Res Plot (`results/phase2g/ablation_comparison.png`) | Ablation study performance curves across Systems A through E. |
| [`confusion_matrix.png`](file:///c:/Users/msiva/Music/Drowsiness-Detection-using-YOLOv5/docs/demo/confusion_matrix.png) | Normalized Matrix (`results/phase2i/confusion_matrix_normalized.png`) | Normalized confusion matrix across `eyes_closed`, `eyes_open`, `yawning`. |
| [`fatigue_score_timeline.png`](file:///c:/Users/msiva/Music/Drowsiness-Detection-using-YOLOv5/docs/demo/fatigue_score_timeline.png) | Timeseries (`results/phase2g/fatigue_score_timeline.png`) | Continuous fatigue score tracking and alert state transitions over time. |

---

## 2. Operational Live HUD Screenshots Required for Presentation

To capture live graphical HUD frames during interactive runtime, run the interactive inference pipeline and capture the OpenCV window or extract frames from `runs/detect/output.mp4` / `output_video.mp4`:

```bash
python scripts/run_inference.py --source test_video.mp4 --backend pytorch --display
```

### Required Visual States Catalog:

### 1. State: `NORMAL` (Alert Level: Green)
- **Target File:** `docs/demo/hud_normal_driver.png`
- **Visual Description:** Driver looking straight ahead with eyes open and mouth closed.
- **HUD Telemetry Overlay:**
  - Status Badge: `NORMAL` (Green: `#00FF00`)
  - Fatigue Score: `< 30.0`
  - EAR: `> 0.25`
  - MAR: `< 0.35`
  - PERCLOS: `< 0.15`
  - Head Pose: `Pitch: ~0°, Yaw: ~0°, Roll: ~0°`

### 2. State: `EYES_CLOSED` (Instantaneous Detection)
- **Target File:** `docs/demo/hud_eyes_closed.png`
- **Visual Description:** Driver with bilateral eyelid closure during a blink or emerging microsleep.
- **HUD Telemetry Overlay:**
  - Bounding Box: `eyes_closed` (Yellow/Red box around ocular region)
  - EAR: `< 0.20`
  - Closure Duration: `0.1s - 0.4s` (transient) or `> 0.5s` (micro-sleep)

### 3. State: `YAWNING` (Instantaneous Detection)
- **Target File:** `docs/demo/hud_yawning.png`
- **Visual Description:** Wide mouth aperture displaying sustained vertical lip separation.
- **HUD Telemetry Overlay:**
  - Bounding Box: `yawning` (Blue/Orange box around mouth aperture)
  - MAR: `> 0.55`
  - Speech Suppression Filter: Active (distinguishes continuous yawn from speech phonemes)

### 4. State: `WARNING` (Alert Level: Yellow/Amber)
- **Target File:** `docs/demo/hud_warning.png`
- **Visual Description:** Driver displaying early fatigue onset (elevated blink frequency or slow eye reopening).
- **HUD Telemetry Overlay:**
  - Status Badge: `WARNING` (Amber: `#FFA500`)
  - Fatigue Score: `30.0 - 69.9`
  - PERCLOS: `0.15 - 0.35`
  - Visual Cue: Yellow bounding borders on HUD status panel

### 5. State: `CRITICAL` (Alert Level: Red)
- **Target File:** `docs/demo/hud_critical.png`
- **Visual Description:** Prolonged micro-sleep closure ($\ge 1.5\text{ s}$) or repetitive yawning coupled with head nodding.
- **HUD Telemetry Overlay:**
  - Status Badge: `CRITICAL` (Flashing Red: `#FF0000`)
  - Fatigue Score: `≥ 70.0`
  - Audio Alert: Active audio chime triggered asynchronously
  - Telemetry: High-priority event log dispatched

### 6. Full Interactive HUD Overview
- **Target File:** `docs/demo/hud_full_interface.png`
- **Visual Description:** Full widescreen frame displaying the complete telemetry dashboard, showing face bounding box, 478-point landmark mesh (if `--enable-landmarks` is set), instantaneous metrics graph, and state machine timeline.

---

## 3. Frame Extraction Helper Command

To extract individual timestamped frames directly without screen recording software:

```bash
# Process video and save rendered video
python scripts/run_inference.py --source test_video.mp4 --backend pytorch --output outputs/rendered_hud.mp4 --headless

# Extract specific frames using ffmpeg (example at second 5.0):
# ffmpeg -ss 00:00:05 -i outputs/rendered_hud.mp4 -vframes 1 docs/demo/hud_normal_driver.png
```
