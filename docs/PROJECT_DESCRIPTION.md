# Project Description & Narrative Summaries
## AI Driver Monitoring & Drowsiness Detection System

**Document:** Canonical Project Descriptions  
**Location:** `docs/PROJECT_DESCRIPTION.md`  
**Purpose:** Standardized project narratives for portfolios, GitHub summaries, CVs, and technical presentations.  
**Constraint:** Grounded strictly in verified empirical facts and system architecture.  

---

### Short Description (2–3 lines)
An edge-optimized AI driver monitoring system that combines YOLOv5nu object detection with MediaPipe facial landmark tracking to extract continuous physiological metrics (EAR, MAR, PERCLOS, 3D head pose). Powered by a temporal multi-signal fusion engine and finite state machine, it delivers real-time fatigue alerting at 27.5 FPS on commodity CPUs with zero false alarms on benchmarked video.

---

### Medium Description (5–6 lines)
The AI Driver Monitoring & Drowsiness Detection System is a modular, production-grade computer vision pipeline designed for real-time vehicular safety on edge hardware. By fusing deep learning bounding-box detection (YOLOv5nu) with sub-pixel physiological indicators (Eye Aspect Ratio, Mouth Aspect Ratio, 60-second rolling PERCLOS, and Perspective-n-Point 3D head pose), the architecture accurately differentiates true micro-sleeps and fatigue from benign speech and regular blinks. Curated with a subject-independent multi-source dataset to eliminate identity leakage, the pipeline achieves an F1-score of 0.9189 on held-out unseen drivers and 0.9208 across 5-fold UTA-RLDD temporal benchmarking, maintaining stable 27.48 Wall FPS on commodity Intel CPUs across PyTorch and ONNX Runtime deployments.

---

### Detailed Description (1–2 paragraphs)
The AI Driver Monitoring & Drowsiness Detection System addresses a core failure of traditional driver monitoring models: severe vulnerability to identity leakage, high false alarm rates caused by transient facial movements, and heavy dependence on power-hungry GPU accelerators. Built on a clean 4-tier modular architecture (Ingestion & Driver Selection, Visual & Physiological Feature Extraction, Temporal Multi-Signal Fusion, and Alert Management), the system simultaneously deploys a fine-tuned YOLOv5nu model alongside a 478-point MediaPipe face mesh. This multi-signal engine extracts high-frequency Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), blink frequency, rolling 60-second P80 PERCLOS, and 3D head pose Euler angles, feeding an integrated fatigue scoring algorithm governed by a robust state machine (`NORMAL`, `WARNING`, `CRITICAL`, `RECOVERY`, `FACE_LOST`). A dedicated speech suppression filter requires sustained mouth opening ($\ge 2.0\text{ s}$) before flagging yawns, preventing normal passenger conversation from triggering false interventions.

To ensure genuine real-world generalization, the system was developed using a rigorous subject-disjoint dataset protocol (333 verified frames across 26 subjects, strictly auditing zero subject overlap between training, validation, and testing). On held-out unseen drivers, the visual detector achieved an mAP@0.5 of 0.9777, while temporal sequence-level validation across 180 multi-condition videos in the UTA-RLDD dataset demonstrated a mean F1-score of 0.9208 (±0.0139) with an average detection latency of 1.46 seconds. Comprehensive CPU profiling and deployment optimization achieved 27.48 Wall FPS (36.39 ms wall latency) on an Intel Core i7-10700 processor, verified across PyTorch CPU, standalone ONNX Runtime, and Docker containerized workflows, backed by a 100% passing test suite across 39 automated unit and edge-case tests.
