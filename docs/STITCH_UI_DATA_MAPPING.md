# STITCH UI TO RUNTIME DATA & API MAPPING

## System: AeroDMS Sentinel — AI Driver Monitoring & Safety Control Center
**Stitch Project**: `https://stitch.withgoogle.com/projects/5686444226727714666`  
**Application Architecture**: In-Memory Edge Stream (Python 3.13 / Streamlit / WebRTC / PyTorch CPU)

---

## 1. Global Header & Navigation

| UI Component | Data Source | Function / API | Refresh Behavior | Fallback State |
| :--- | :--- | :--- | :--- | :--- |
| **System Brand & Title** | Static Design Specification | `app/app.py` header definition | Static on render | `"AeroDMS Sentinel"` |
| **Inference Active Badge** | WebRTC stream / Pipeline state | `DriverMonitoringEngine.frame_count > 0` | Real-time event | `"STANDBY"` |
| **Continual Learner Status** | Background worker lifecycle | `ContinualLearningManager.worker.status` | Real-time state | `"DISABLED"` |
| **Backend Pill** | Runtime hardware configuration | `SystemConfig.device` (`"PyTorch CPU"`) | Static on load | `"CPU"` |
| **Active Model Version** | Registered checkpoint metadata | `ModelRegistry.get_active_metadata()["model_version"]` | On model reload/promotion | `"v001 [ACTIVE]"` |
| **Pipeline FPS Pill** | Rolling telemetry FPS tracker | `float(np.mean(engine.fps_tracker))` | Per-frame (30 FPS) | `"30.0 FPS"` |
| **Session Uptime Pill** | System clock delta | `time.time() - engine.start_time` | Per-render | `"00:00:00"` |

---

## 2. Screen 1: Live Monitor (Primary Cockpit Viewport)

| UI Component | Data Source | Function / API | Refresh Behavior | Fallback State |
| :--- | :--- | :--- | :--- | :--- |
| **Live Annotated Video Feed** | In-memory frame processing | `engine.process_frame(img_bgr)[0]` via `WebRTCVideoProcessor.recv()` | 30 FPS WebRTC stream | Standby camera guide |
| **YOLO Bounding Boxes** | YOLOv5nu model inferences | `detector.predict(frame).boxes` (`eyes_open`, `eyes_closed`, `yawning`) | Every 3rd frame (amortized) | Boxes hidden if unselected |
| **Driver Face Bounding Box** | 3D Face Mesh centroid box | `physiological.extract_landmarks().bbox` | Every frame (16 ms) | No driver box |
| **In-Video HUD Overlay** | VisualRenderer OpenCV pipeline | `VisualRenderer.render()` (`src/renderer.py`) | In-frame 30 FPS | Clean video frame |
| **Overlay Display Toggles** | Sidebar Streamlit checkboxes | `engine.set_display_options(show_boxes, show_labels, ...)` | Dynamic state change | All enabled (`True`) |
| **DRIVER STATE Card** | Temporal Fusion & State Machine | `telemetry["state"]` (`NORMAL`, `WARNING`, `CRITICAL`, `RECOVERY`, `FACE_LOST`) | Per-frame update | `"NORMAL"` |
| **FATIGUE SCORE Gauge** | Continuous multi-signal score | `telemetry["fatigue_score"]` (0.0 to 100.0) | Per-frame update | `0.0 / 100` |
| **PIPELINE FPS Card** | Smoothed framerate buffer | `telemetry["fps"]` | Per-frame update | `30.0 FPS` |
| **CAMERA STATUS Card** | Streamer connection status | `ctx.state.playing` / `camera_active` | Stream lifecycle | `"Standby"` |
| **CURRENT ALERT Panel** | Structured alert dictionary | `telemetry["current_alert"]` (`severity`, `condition`, `trigger`, `evidence`, `action`) | Event/transition driven | Normal attentive alert |
| **ALERT EVENT HISTORY** | Session transition ring buffer | `engine.get_events_snapshot()` (`deque(maxlen=50)`) | On state change | Empty list notice |
| **Clear History Action** | History buffer reset handler | `engine.clear_events()` | On button click | Empty list |
| **EYE STATE Indicator** | Physiological signal state | `telemetry["eye_closed"]` & closure duration | Per-frame update | `"OPEN"` |
| **EAR Metric Readout** | Eye Aspect Ratio computation | `telemetry["ear"]` (Threshold: `0.21`) | Per-frame update | `"--"` |
| **MAR Metric Readout** | Mouth Aspect Ratio computation | `telemetry["mar"]` (Speech filtered, Threshold: `0.55`) | Per-frame update | `"--"` |
| **PERCLOS Metric Readout** | Sliding 60s window accumulator | `telemetry["perclos"]` (Threshold: `25.0%`) | Per-frame update | `"0.0%"` |
| **HEAD POSE Metric Readout** | 3D PnP Euler angles solve | `telemetry["head_pose_state"]` (`HEAD_FORWARD`, `HEAD_DOWN`, etc.) | Per-frame update | `"FORWARD"` |
| **Rolling Timeseries Chart** | Telemetry rolling history buffer | `engine.get_history_snapshot()` (`deque(maxlen=150)`) | Stream interval | Empty chart |
| **Alert Conditions Help** | System empirical thresholds | `src/config.py` (`AppConfig`) | Static expander | Documented rules |

---

## 3. Screen 2: Offline Video Analysis

| UI Component | Data Source | Function / API | Refresh Behavior | Fallback State |
| :--- | :--- | :--- | :--- | :--- |
| **Source Selector** | User option selection | `st.radio(["Use Canonical test_video.mp4", "Upload Custom Video"])` | User input | Canonical video |
| **File Uploader** | Client filesystem video file | `st.file_uploader(..., type=["mp4", "avi", "mov"])` | File upload | `test_video.mp4` |
| **Evaluation Progress Bar** | Video decoding loop iterator | `processed_count / max_process` | Every 5 frames | `0%` |
| **Processed Video Stage** | In-memory frame renderer | `st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))` | Frame loop | Initial preview |
| **Session Summary Matrix** | Final session aggregation | `engine.get_session_summary()` (`duration_s`, `avg_fps`, `warnings`, `criticals`) | On completion | Awaiting run |
| **Telemetry CSV Export** | Session telemetry data frame | `pd.DataFrame(engine.get_history_snapshot()).to_csv()` | On completion | Disabled |

---

## 4. Screen 3: Continual Learning Workspace

| UI Component | Data Source | Function / API | Refresh Behavior | Fallback State |
| :--- | :--- | :--- | :--- | :--- |
| **Active Model Card** | Model registry database | `cl_manager.registry.get_active_metadata()` | Registry update | Baseline `v001` |
| **3-Tier Buffer Counters** | Continual learning buffer | `cl_manager.buffer.get_counts()` (`verified`, `pseudo_labeled`, `needs_review`) | On observation add | `0` counts |
| **Clear Buffer Button** | Volatile buffer reset | `cl_manager.buffer.clear()` | On button click | Cleared toast |
| **Human Review Sample Card** | Buffer review queue | `cl_manager.buffer.get_review_queue(limit=5)[0]` | Queue update | No samples notice |
| **Review Action Buttons** | Buffer sample verification | `cl_manager.buffer.review_sample(sample_id, "accept"|"correct"|"reject"|"skip")`| On click | Advances queue |
| **Training Controls** | Background learner worker | `cl_manager.worker.trigger_training()`, `.pause()`, `.resume()` | Worker thread | Disabled if idle |
| **Epoch Progress Bar** | Training worker state | `worker_info["current_epoch"] / worker_info["total_epochs"]` | Per epoch | `0%` |
| **Worker Training Logs** | Worker stdout log buffer | `worker_info["training_log"]` | Real-time thread | `"Worker idle."` |
| **Validation Gate Comparison** | Automated candidate evaluation| `worker_info["last_validation_passed"]` & delta metrics | Post-training | No candidate |
| **Promotion Action** | Candidate promotion handler | `cl_manager.promote_candidate(candidate_version)` | On button click | Baseline active |
| **Rollback Action** | Model registry rollback | `cl_manager.rollback()` | On button click | Baseline active |
| **Drift Monitor Metrics** | Environmental drift analyzer | `cl_manager.drift_monitor.get_status()` (`drift_score`, `current_luminance`) | Per observation | `"Stable (0.00)"` |

---

## 5. Screen 4: Model Performance & Benchmarks

| UI Component | Data Source | Function / API | Refresh Behavior | Fallback State |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv5nu Metrics Table** | Phase 2F verified test results | `results/final_resume_metrics.json` | Static verified | Documented table |
| **UTA-RLDD 5-Fold Table** | Phase 2G temporal cross-eval | `results/final_resume_metrics.json` | Static verified | Documented table |
| **Sensor Fusion Ablation Table** | Stepwise ablation experiment | `results/final_resume_metrics.json` | Static verified | Documented table |
| **External Validation Audit**| 7 external driving datasets | `results/external/external_baseline_results.csv` | Static verified | Documented table |

---

## 6. Screen 5: Architecture & Privacy

| UI Component | Data Source | Function / API | Refresh Behavior | Fallback State |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture Diagram** | High-res system schematic | `app/assets/architecture.png` | Static on render | Diagram render |
| **Decoupled Pipeline Spec** | System architectural contracts | `docs/FINAL_ARCHITECTURE.md` | Static on render | Text spec |
| **Zero-Retention Guarantee** | Verified in-memory design | `src/physiological.py` & `src/pipeline.py` | Static on render | Privacy notice |
| **Limitations & Disclaimer** | Operational constraints | Engineering safety report | Static on render | Disclaimer notice |
