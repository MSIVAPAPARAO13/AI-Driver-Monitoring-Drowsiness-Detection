# High-Impact Technical Resume Bullets
## AI Driver Monitoring & Drowsiness Detection System

**Document:** Verified Resume Bullets & Metric Mapping  
**Location:** `docs/RESUME_BULLETS.md`  
**Single Source of Truth:** `results/final_resume_metrics.json` and `results/final_project_results.csv`  
**Standard:** Every numerical claim is strictly verifiable against automated evaluation logs.  

---

## 1. Primary Recommended Resume Bullets (Tailored for ML / Computer Vision / Edge AI)

• **Architected a real-time 4-tier edge driver monitoring system** fusing fine-tuned YOLOv5nu detection with MediaPipe 3D face mesh to extract continuous Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), rolling 60s PERCLOS, and 3D head pose via a temporal finite state machine.

• **Curated a subject-independent multi-source vision dataset** (333 clean verified frames, 26 subjects) from YawDD and DMD, eliminating train/validation/test identity leakage through strict subject-level splitting with zero cross-split subject overlap.

• **Achieved mAP@0.5 = 0.9777 (F1 = 0.9189)** on an unseen 3-subject test split and a **0.9208 mean F1 (±0.0139, 1.46s mean delay)** across 5-fold subject-independent temporal evaluation on 180 videos (60 subjects) in UTA-RLDD.

• **Engineered an edge-optimized CPU pipeline operating at 27.48 Wall FPS (36.39 ms latency)** on an Intel Core i7-10700 processor; deployed across PyTorch CPU, standalone ONNX Runtime, and Docker containerized workflows backed by 39/39 passing automated tests.

---

## 2. Alternative Role-Specific Bullet Variants

### For Edge AI / Embedded Systems Engineering Roles:
• **Optimized and benchmarked multi-backend edge deployment** for an in-cabin computer vision pipeline, achieving 27.48 Wall FPS on PyTorch CPU and 22.29 Wall FPS on ONNX Runtime; implemented an amortized frame-skipping architecture (skip=2) reducing YOLO overhead to 14.14 ms and verified continuous memory plateau over a 4,936-frame stress test.

### For Applied Machine Learning / Deep Learning Research Roles:
• **Resolved catastrophic data leakage in baseline driver drowsiness models** by replacing random frame-level splits with strict subject-disjoint validation; designed an ablation-validated multi-signal fusion engine that reduced false alarms from 7 to 0 (ablation F1 = 0.983) via speech suppression and temporal persistence filtering.

### For Computer Vision / Software Engineering (Full Pipeline) Roles:
• **Engineered an end-to-end driver safety system** with driver face centroid tracking, graceful camera disconnect handling, and asynchronous alerting; verified 100% test coverage across 39 unit/edge-case tests covering video ingestion, headless execution, and state-machine transitions.

---

## 3. Metric Traceability Matrix

| Metric Value Claimed | Source Metric Field | Location in `results/final_resume_metrics.json` |
|:---|:---|:---|
| 333 clean frames, 26 subjects | `dataset_metrics.clean_ground_truth_frames`, `total_subjects` | Lines 7, 11 |
| 0 subject leakage | `dataset_metrics.subject_leakage_*` | Lines 12–14 |
| mAP@0.5 = 0.9777, F1 = 0.9189 | `yolo_detection_metrics.map_50`, `f1_score` | Lines 29, 30 |
| UTA-RLDD Mean F1 = 0.9208 | `temporal_benchmark_uta_rldd.mean_f1` | Line 54 |
| 1.46s detection delay | `temporal_benchmark_uta_rldd.mean_detection_delay_seconds` | Line 59 |
| 27.48 Wall FPS, 36.39 ms latency | `deployment_and_runtime_performance.full_pipeline_wall_fps`, `wall_latency_ms` | Lines 65, 67 |
| 39/39 passed unit tests | `testing_and_verification.pytest_total_tests`, `pytest_passed_tests` | Lines 80, 81 |
| False alerts reduced: 7 to 0 | `multi_signal_fusion_metrics.false_alerts_yolo_only`, `false_alerts_full_fusion` | Lines 43, 44 |
