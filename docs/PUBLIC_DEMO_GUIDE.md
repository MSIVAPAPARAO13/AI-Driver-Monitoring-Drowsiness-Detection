# Public Demonstration & Reviewer Guide — AeroDMS Sentinel

**Project:** AeroDMS Sentinel — AI Driver Monitoring & Safety Control Center  
**GitHub Repository:** [MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection](https://github.com/MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection)  
**Landing Page:** [https://msivapaparao13.github.io/AI-Driver-Monitoring-Drowsiness-Detection/](https://msivapaparao13.github.io/AI-Driver-Monitoring-Drowsiness-Detection/)

---

## 1. Quick Access for Technical Reviewers & Recruiters

The AeroDMS Sentinel demo can be evaluated locally in under 60 seconds or accessed via public cloud deployment.

### Local Evaluation
```bash
git clone https://github.com/MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection.git
cd AI-Driver-Monitoring-Drowsiness-Detection
.\venv\Scripts\activate
streamlit run app/app.py
```
Open your browser to `http://localhost:8501`.

---

## 2. Interactive Demonstration Walkthroughs

### Workflow 1: Live In-Cabin Webcam Monitoring (Tab 1: `📺 Live Monitor`)
1. **Camera Permission:** Click the **"START"** button on the WebRTC video stage and allow browser camera access.
2. **Detection Overlays:** Verify that the driver's face is framed with the gold/cyan `DRIVER (TRACKED)` region, while sub-regions highlight `eyes_open` (green), `eyes_closed` (red), or `yawning` (orange).
3. **Display Overlay Toggles:** In the left sidebar under **"Display Overlays"**, toggle bounding boxes, labels, confidence, HUD, or FPS badges to inspect real-time rendering flexibility.
4. **Physiological Signal Responsiveness:**
   - **Normal Vigilance:** Look forward with eyes open. The top card will display `🟢 NORMAL — Driver Alert & Attentive`. Fatigue score stays low (&lt;25/100).
   - **Involuntary Blink Test:** Blink normally (100–300 ms). Notice that isolated blinks **do not** trigger a warning, proving temporal filter robustness.
   - **Prolonged Eye Closure / Microsleep:** Close your eyes for &ge;1.2 seconds. The dedicated Current Alert card immediately escalates to `🔴 CRITICAL — HIGH DROWSINESS RISK` accompanied by physiological evidence chips (`EAR ↓`, `PERCLOS ↑`).
   - **Yawn vs. Speech Suppression:** Speak aloud or mouth words quickly. Notice that quick mouth openings **do not** trigger a yawning alert. Yawn widely for &ge;1.5 seconds to register a confirmed yawn.
   - **Head Posture / Distraction:** Turn head left, right, or look down. The Head Orientation card switches to `TURNED_LEFT`, `TURNED_RIGHT`, or `HEAD_DOWN`.
   - **Occlusion / Face Lost:** Cover your face or move outside the camera frame. After a 1.0s grace period, the state updates to `⚪ FACE NOT DETECTED`.

---

### Workflow 2: Offline Driving Video Evaluation (Tab 2: `📁 Video Analysis`)
1. Navigate to the **"📁 Video Analysis"** tab.
2. Select **"Use Canonical test_video.mp4"** (or upload your own driving video).
3. Click **"🚀 Run Pipeline Evaluation"**.
4. Observe the frame-by-frame processing bar with live synchronized OpenCV annotations.
5. Review the final post-run session summary metrics table and export the session telemetry log as `.CSV`.

---

### Workflow 3: MLOps Continual Learning (Tab 3: `🧠 Continual Learning`)
1. In the sidebar, toggle **"Enable Continual Learning (Opt-in)"**.
2. Navigate to **"🧠 Continual Learning"** tab.
3. Review the **3-Tier Label Hierarchy Buffer** (Verified, Pseudo-Labels, Needs Review).
4. In the **Human Review Tool**, review uncertain observations, inspect thumbnail crops, select ground-truth labels, and click **"Accept As-Is"** or **"Save Correction"**.
5. Inspect the **Background Adaptation Worker** to view epoch progress and live training logs without interrupting live inference.
6. Verify the **Validation Gate Evaluation** comparing candidate models against baseline metrics, and test atomic model rollback.

---

### Workflow 4: Empirical Benchmark Audit (Tab 4: `📊 Model Performance`)
1. Navigate to **"📊 Model Performance"**.
2. Inspect the verified held-out test metrics:
   - YOLOv5nu mAP@0.5: `0.9777`, mAP@0.5:0.95: `0.8145`.
   - UTA-RLDD 5-Fold cross-validation Mean F1: `0.9208 ± 0.0139`, Detection delay: `1.46s`.
   - Sensor fusion ablation study (Systems A through E): 0 false alerts.
   - External dataset validation table across 7 independent driving benchmarks.

---

### Workflow 5: Architecture & Privacy Governance (Tab 5: `🏗️ Architecture & Privacy`)
1. Navigate to **"🏗️ Architecture & Privacy"**.
2. Review the 4-tier decoupled pipeline architecture schematic.
3. Confirm privacy governance guarantees:
   - **Zero Facial Recognition:** No biometric identification or identity embeddings.
   - **Volatile RAM Processing:** Video frames are never permanently stored to disk.
   - **Anonymized Telemetry:** System records only numerical aspect ratios and categorical states.
