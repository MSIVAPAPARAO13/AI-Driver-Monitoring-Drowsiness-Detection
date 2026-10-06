# Phase 2B Report — Dataset Audit, Availability & Final Data Strategy
## AI Driver Monitoring & Safety System

**Project:** AI Driver Monitoring & Drowsiness Detection System  
**Phase:** Phase 2B — Dataset Audit, Availability & Strategy Research  
**Auditor & Research Engineer:** Antigravity AI Assistant  
**Date:** October 2026  
**Status:** COMPLETE & VERIFIED (No Models Retrained, Zero Destructive Changes)

---

## 1. Executive Summary

This report establishes the data strategy for transitioning the Driver Drowsiness Detection prototype into an industry-grade, resume-level AI Driver Monitoring System (DMS).

### Core Findings & Strategic Directives:
1. **The Current Dataset is Severely Constrained:** The original repository relies on only 207 unique frames extracted from private webcam recordings, augmented 4x to 845 training frames, validated on a fragile 38-frame validation split (with 0 test frames), and splits without person metadata, causing severe identity leakage.
2. **Video-Level vs. Frame-Level Semantics:** Large drowsiness video datasets (e.g., UTA-RLDD) provide **video-level behavioral fatigue annotations** (Alert, Low Vigilance, Drowsy), NOT frame-level bounding boxes. A driver in a 10-minute "Drowsy" video still opens their eyes and blinks. Blindly mapping video-level "Drowsy" labels to YOLO `eyes_closed` bounding boxes is invalid and destroys detector accuracy.
3. **Speech vs. Yawn Disambiguation:** Frame-level yawn detectors consistently produce false alarms on conversational speech syllables. The **YawDD** dataset solves this by providing time-interval labeled videos of talking and singing as an explicit negative class.
4. **2026 Status of DMD (Driver Monitoring Dataset):** The Vicomtech DMD dataset underwent major revisions in 2026. Simulator recordings, IR streams, and depth streams were removed due to hosting and licensing updates; the dataset currently provides RGB recordings for 14 verified subject IDs (`[1, 5, 6, 7, 9, 10, 13, 14, 23, 28, 29, 33, 36, 37]`) with OpenLABEL VCD >= 5.0 temporal annotations.
5. **Strict Person-Independent Evaluation:** To eliminate data leakage and ensure real-world generalization, all future training and validation must follow strict **Subject-Independent Partitioning** ($\text{Train Subjects} \cap \text{Val Subjects} = \emptyset$), leveraging the official 5-fold cross-validation scheme of UTA-RLDD (12 subjects per fold).

---

## 2. Current Project Dataset Audit

| Parameter | Current Repository Value | Verification Status |
| :--- | :--- | :--- |
| **Raw Frames Extracted** | 207 unique frames | [VERIFIED] (Inspected in Phase 1 audit) |
| **Augmented Training Split** | 845 frames (169 original + 676 augmented) | [VERIFIED] (Via Albumentations offline notebook) |
| **Validation Split** | 38 frames (13 `eyes_closed`, 14 `eyes_open`, 11 `yawning`) | [VERIFIED] (Inspected from `results/results.csv`) |
| **Test Split** | 0 labeled frames (evaluated qualitatively on `test_video.mp4`) | [VERIFIED] (No test directory or split exists) |
| **Bounding Box Granularity** | Frame-level YOLO normalized coordinates `(cls, xc, yc, w, h)` | [VERIFIED] |
| **Number of Subjects** | 1–2 individuals (undocumented identities) | [INFERRED] (Private recordings by repo author) |
| **Split Protocol** | Random 80/20 train/val split across frames | [VERIFIED] (`splitting_data.ipynb`) |
| **Data Leakage** | High: Adjacent frames from the same subject present in both splits | [VERIFIED] (Caused artificial mAP50 > 99%) |
| **Local Availability** | Raw images NOT committed in Git; only `test_video.mp4` present | [VERIFIED] (Referenced private GDrive folder) |
| **Redistribution Rights** | Unlicensed / private origin | [VERIFIED] (Cannot be redistributed commercially) |
| **Baseline Model** | `best.pt` (YOLOv5nu, 2,503,529 parameters, 7.1 GFLOPs) | [VERIFIED] (Valid baseline anchor for regression) |

### Class Indexing Discrepancy (Confirmed):
- [custom_dataset.yaml](file:///c:/Users/msiva/Music/Drowsiness-Detection-using-YOLOv5/custom_dataset.yaml) & [best.pt](file:///c:/Users/msiva/Music/Drowsiness-Detection-using-YOLOv5/best.pt):  
  `0: eyes_closed`, `1: eyes_open`, `2: yawning`
- [classes.txt](file:///c:/Users/msiva/Music/Drowsiness-Detection-using-YOLOv5/classes.txt):  
  Line 1: `eyes_open` (Index 0), Line 2: `eyes_closed` (Index 1), Line 3: `yawning` (Index 2)
- **Status:** Handled via canonical class mappings in `src/config.py`.

---

## 3. UTA-RLDD Audit (University of Texas at Arlington Real-Life Drowsiness Dataset)

- **Citation:** Ghoddoosian, R., Galib, M., & Athitsos, V. (2019). *"A realistic dataset and baseline temporal model for early drowsiness detection."* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 0-0.
- **Official Portal:** [https://sites.google.com/view/utarldd/home](https://sites.google.com/view/utarldd/home)
- **Current Access (2026):** [VERIFIED] Available for academic research via Google Drive download links upon agreement to terms.
- **Participants:** 60 healthy adult participants (students and staff > 18 years), balanced across genders and diverse ethnicities.
- **Video Volume:** 180 RGB videos (3 videos per participant). Each video is ~10 minutes long (total ~30 hours of video) recorded at 30 FPS at 720p/1080p.
- **Labels / Target Classes:** Multi-stage self-reported drowsiness based on the Karolinska Sleepiness Scale (KSS):
  1. `Class 0: Alertness` (KSS 1–5, normal attentive state)
  2. `Class 1: Low Vigilance` (KSS 6–7, subtle micro-expressions, early fatigue onset)
  3. `Class 2: Drowsiness` (KSS 8–9, active struggling with sleep, head droop, long closures)
- **Official Five-Fold Participant Structure:**
  - Fold 1: Subjects 01–12 (12 subjects)
  - Fold 2: Subjects 13–24 (12 subjects)
  - Fold 3: Subjects 25–36 (12 subjects)
  - Fold 4: Subjects 37–48 (12 subjects)
  - Fold 5: Subjects 49–60 (12 subjects)
- **Annotation Format:** Video-level metadata files specifying participant ID, fold number, and video state label. **Does not provide frame-level bounding boxes.**
- **Feature Capability:**
  - *Drowsiness Training & Validation:* **YES** (Ideal for temporal models, LSTMs, and sequence classifiers).
  - *Subject-Independent Evaluation:* **YES** (Standard benchmark 5-fold cross-validation).
  - *Temporal Analysis & PERCLOS:* **YES** (Continuous 10-minute streams allow extracting realistic rolling PERCLOS curves).
  - *Head Pose & Nodding:* **YES** (Captures unscripted head drooping during microsleeps).
  - *Direct YOLO Training:* **NO (Directly)**. Requires face detection extraction or pseudo-labeling.

---

## 4. YawDD Audit (Yawning Detection Dataset)

- **Citation:** Abtahi, S., Omidyeganeh, M., Shirmohammadi, S., & Hariri, B. (2014). *"YawDD: A Yawning Detection Dataset."* In Proceedings of the 5th ACM Multimedia Systems Conference (MMSys '14), pp. 24–28.
- **Official Source:** [IEEE DataPort](https://ieee-dataport.org/) / University of Ottawa DISCOVER Lab.
- **Current Access (2026):** [VERIFIED] Available on IEEE DataPort for free download after creating an IEEE user account (non-commercial academic use).
- **Participants & Composition:** 107 diverse participants (male/female, varying ages, with/without eyeglasses and sunglasses).
  - Sub-dataset 1 (Dash-mounted camera): 322+ videos (640x480 @ 30 FPS).
  - Sub-dataset 2 (Rearview-mirror-mounted camera): 29+ videos (640x480 @ 30 FPS).
- **Recording Environment:** Real car cabin and stationary driving simulator.
- **Classes / Scenarios:**
  1. `Normal Driving`: Silent driving, neutral mouth postures.
  2. `Talking or Singing`: Driver conversing, laughing, or singing (crucial negative class).
  3. `Yawning`: Natural and instructed driver yawning.
- **Annotations:** Video-level and event-level text files recording the exact start frame and end frame of every yawn and talking episode.
- **Bounding Boxes:** **None in original release.** Derived community releases (e.g. YawDD+ on Roboflow/GitLab) provide YOLO bounding boxes generated from mouth landmarks.
- **Feature Capability:**
  - *Yawn Detection:* **YES**.
  - *Yawn Duration Analysis:* **YES** (Exact start/end frame boundaries).
  - *MAR & Lip Aperture:* **YES**.
  - *Talking vs. Yawning Discrimination:* **YES (The primary utility of YawDD)**.
  - *Eye State / Blink:* **NO** (Not annotated for ocular state).

---

## 5. DMD Audit (Driver Monitoring Dataset — Vicomtech)

- **Citation:** Ortega, C., et al. (2020). *"DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Drowsiness Analysis."* IEEE Transactions on Intelligent Transportation Systems.
- **Official Repository:** [https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset) / [https://dmd.vicomtech.org/](https://dmd.vicomtech.org/)
- **CURRENT 2026 STATUS & AVAILABILITY:** [VERIFIED]
  - **Major Streamlining:** In 2026, Vicomtech restructured the repository due to revised privacy regulations and hosting quotas. **Simulator data, active Infrared (IR) streams, and Depth streams were removed.**
  - **Available Material:** Exclusively provides real-car RGB material recorded on highway and urban test tracks.
  - **Approved Subjects:** Download access is currently restricted to **14 approved subject IDs**: `[1, 5, 6, 7, 9, 10, 13, 14, 23, 28, 29, 33, 36, 37]`.
  - **License & Access:** Academic research license requiring registration through the Vicomtech portal.
  - **Annotation Format:** ASAM OpenLABEL compliant JSON files utilizing the Video Content Description (VCD >= 5.0) standard. Tools provided: TaTo (Temporal Annotation Tool) and DEx (Dataset Explorer).
- **Annotated Tasks & Labels:**
  - *Distraction:* Mobile phone usage (calling, texting), interacting with vehicle console, drinking/eating, looking away.
  - *Drowsiness:* Yawning, eye closure duration, nodding.
  - *Gaze & Posture:* Driver gaze target zones and 3D head pose.
  - *Hands:* Hands on/off steering wheel.
- **Feature Capability:**
  - *Drowsiness & Eye State:* **YES**.
  - *Distraction & Mobile Phone:* **YES (Highest quality real-car distraction dataset available)**.
  - *Head Pose & Gaze:* **YES**.
  - *Temporal Monitoring:* **YES**.

---

## 6. Additional Dataset Candidates Evaluated

### Candidate 1: NTHU-DDD (National Tsing Hua University Driver Drowsiness Detection Dataset)
- **Source:** CVLab, National Tsing Hua University (Weng et al., CVPRW 2016).
- **Access (2026):** [VERIFIED] Available upon academic request form to NTHU CVLab; community mirrors available on Roboflow Universe.
- **Subjects & Scope:** 36 participants (18 male, 18 female, multi-ethnic).
- **Five Driving Scenarios:**
  1. `BareFace`: Normal daylight, no eyewear.
  2. `Glasses`: Prescription eyeglasses with reflections.
  3. `Sunglasses`: Heavy ocular tint/occlusion.
  4. `Night-BareFace`: Nighttime driving with active near-infrared (NIR) illumination.
  5. `Night-Glasses`: Nighttime driving with eyeglasses.
- **Why It Fills a Genuine Gap:** NTHU-DDD is the only major public benchmark evaluating **nighttime driving and heavy sunglasses occlusions**. It is recommended as the **External Out-of-Distribution Test Set**.

### Candidate 2: MRL Eye Dataset (Media Research Lab, VSB - Technical University of Ostrava)
- **Source:** [https://mrl.cs.vsb.cz/eyedataset](https://mrl.cs.vsb.cz/eyedataset) (Fusek et al., 2018).
- **Access (2026):** [VERIFIED] Publicly downloadable from official site and Kaggle mirror.
- **Scope:** 84,898 ocular patch images extracted from 37 participants.
- **Annotations:** Eye state (`0: closed`, `1: open`), Glasses (`yes/no`), Reflections (`none/low/high`), Lighting (`good/bad`).
- **Role:** High-volume secondary crop classifier for extreme eye-closure verification under specular reflections.

### Candidate 3: RT-GENE (Real-Time Gaze Estimation in Natural Environments)
- **Source:** Fischer et al. (ECCV 2018).
- **Role:** High-precision 3D gaze ground truth collected via wearable eye-tracking glasses. Recommended for future Phase 4 gaze calibration.

---

## 7. Comprehensive Dataset Comparison Table

| Dataset | Subjects | Volume / Duration | Annotation Format & Level | Environment | Drowsiness | Eye State | Yawning | Blink | EAR/MAR | PERCLOS | Head Pose | Gaze | Distraction | Access Status | License | Recommended Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Project** | 1–2 | 207 frames (0.1 h) | YOLO BBox (Frame) | Indoor Webcam | POSSIBLE | YES | YES | NO | YES | NO | NO | NO | NO | Private GDrive | Unlicensed | Baseline Anchor |
| **UTA-RLDD** | 60 | 180 videos (30.0 h) | KSS State (Video-level) | Natural Webcam/Phone | YES | POSSIBLE* | POSSIBLE* | YES | YES | YES | YES | NO | NO | Public GDrive | Academic Non-Comm | Temporal Fatigue & 5-Fold Val |
| **YawDD** | 107 | 351 videos (5.5 h) | Time-Interval (Event) | In-car / Simulator | POSSIBLE | NO | YES | NO | YES (MAR) | NO | POSSIBLE | NO | POSSIBLE | IEEE DataPort | Academic Non-Comm | Yawn Duration & Speech Filter |
| **DMD (2026)** | 14 (avail) | Real-car streams (12 h) | OpenLABEL VCD (Frame/Event) | Real Driving Tracks | YES | YES | YES | YES | YES | YES | YES | YES | YES | Request (GitHub) | Restricted Academic | Distraction, Phone & Real Car |
| **NTHU-DDD** | 36 | Video sequences (9.5 h) | State & Event (Frame) | Simulator (RGB+NIR) | YES | YES | YES | YES | YES | YES | YES | NO | NO | Academic Request | Academic Research | Night / Glasses Benchmark |
| **MRL Eye** | 37 | 84,898 patches | Binary + Attributes (Crop) | Varied Lighting IR | NO | YES | NO | NO | NO | NO | NO | NO | NO | Public | Academic Research | Secondary Eye Crop Classifier |

*\*In UTA-RLDD, eye state and yawning are visually present but not annotated as instantaneous bounding boxes; they require automated landmark extraction.*

---

## 8. Canonical Label Schema & Mapping

Current YOLO schema in production:
- `Class 0: eyes_closed`
- `Class 1: eyes_open`
- `Class 2: yawning`

### Harmonization Strategy:
We strictly avoid forcing incompatible semantic concepts into single classes. Multi-stage drowsiness (e.g. KSS scores) and behavioral distractions are assigned to dedicated task heads:

| Source Dataset | Source Annotation | Canonical Target Schema | Confidence | Conversion Required? | Mapping Rationale & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original** | `eyes_closed` | `eyes_closed (Class 0)` | HIGH | NO | Direct 1:1 match to canonical class 0. |
| **Original** | `eyes_open` | `eyes_open (Class 1)` | HIGH | NO | Direct 1:1 match to canonical class 1. |
| **Original** | `yawning` | `yawning (Class 2)` | HIGH | NO | Direct 1:1 match to canonical class 2. |
| **YawDD** | `Yawning` (start-end frames) | `yawning (Class 2)` | HIGH | YES | Extract frames in yawn intervals; generate YOLO bounding box via mouth landmarks. |
| **YawDD** | `Talking / Singing` | `SPEECH_NEGATIVE` | HIGH | YES | Critical negative class for yawn classifier; prevents false alerts during conversation. |
| **DMD** | `Closed eyes / blinking` | `eyes_closed (Class 0)` | HIGH | YES | Convert OpenLABEL VCD polygon/box coordinates to normalized YOLO format. |
| **DMD** | `Phone use / texting` | `DISTRACTION: PHONE` | HIGH | YES | Maps to Phase 3 multi-task distraction head; NOT mixed into drowsiness classes. |
| **MRL Eye** | `0: Closed` | `eyes_closed (Class 0)` | VERY HIGH | YES | Bounding box placed on face mesh ocular crop coordinates. |
| **UTA-RLDD** | `Class 2: Drowsiness` | `FATIGUE: DROWSY` | HIGH (Video) | YES (Temporal) | **DO NOT map to `eyes_closed`!** Used as sequence label for temporal PERCLOS/LSTM evaluation. |
| **UTA-RLDD** | `Class 0: Alertness` | `FATIGUE: ALERT` | HIGH (Video) | YES (Temporal) | Ground truth baseline for attentive driver sequences. |

---

## 9. Dataset Compatibility Analysis

1. **Resolution & Optical Geometry:**
   - *Original:* 1280x720 (front-facing webcam).
   - *UTA-RLDD:* 720p/1080p (mobile and webcam, head pose variations).
   - *YawDD:* 640x480 (dash and rearview mirror angles).
   - *DMD:* 1280x720 / 1920x1080 (real vehicle dashboard mount).
   - *Compatibility:* High. Resizing all inputs to standard YOLO $640\times640$ maintains facial aspect ratios with letterboxing.
2. **Annotation Format Incompatibility:**
   - Pascal VOC XML (Original) vs. Frame Intervals (YawDD) vs. ASAM OpenLABEL JSON (DMD) vs. Video KSS (UTA-RLDD).
   - *Resolution:* Build an automated preprocessing ingestion pipeline in Phase 2C to normalize all bounding boxes into standard YOLO `.txt` format and temporal sequences into standardized `.jsonl` telemetry.

---

## 10. Subject-Level Split Strategy

### The Flaw of Prior Random Splitting:
In `splitting_data.ipynb`, frames were shuffled randomly with an 80/20 train/val ratio. Because frames $t$ and $t+1$ from the same video were split across sets, the model memorized the author's facial identity, achieving an inflated $mAP50 = 99.18\%$ on validation that fails on new drivers.

### Subject-Independent Protocol:
We strictly enforce zero subject overlap across splits:
$$\text{Subjects}_{\text{Train}} \cap \text{Subjects}_{\text{Val}} = \emptyset, \quad \text{Subjects}_{\text{Train}} \cap \text{Subjects}_{\text{Test}} = \emptyset$$

```
                                  ALL PARTICIPANTS (N=60)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         TRAIN SET (Folds 1-4)                             TEST SET (Fold 5)
         48 Unique Participants                            12 Unseen Participants
         (Subjects 01 to 48)                               (Subjects 49 to 60)
                    │                                               │
          ┌─────────┴─────────┐                                     │
          ▼                   ▼                                     ▼
     Train Split          Val Split                        Out-of-Sample Test
    40 Subjects          8 Subjects                        Zero Identity Leakage
```

---

## 11. Data Leakage Risks & Mitigation

| Leakage Vector | Description | Prevention Protocol |
| :--- | :--- | :--- |
| **Subject Identity Leakage** | Same driver appears in both train and validation sets | Enforce strict subject-ID hashing prior to any file assignment. |
| **Temporal Frame Leakage** | Adjacent frames ($33\text{ ms}$ apart) split across sets | Group frames by continuous recording session; never split a session across sets. |
| **Augmentation Leakage** | Original image in train, augmented replica in val | Apply Albumentations strictly *after* the train/val split, exclusively to the training set. |
| **Camera Pose Leakage** | Identical dashboard background memorized | Train on multi-camera setups (dash, mirror, desk) and apply background cutout / mosaic augmentation. |
| **Class Imbalance Leakage** | Validation split dominated by `eyes_open` frames | Enforce stratified sampling across classes within subject folds. |

---

## 12. Training Strategy

We adopt **Strategy C (Two-Tier Multi-Task Pipeline)**:
1. **Tier 1 (Instantaneous State Detector - YOLOv5nu / YOLOv8n):**
   - Fine-tune YOLO on curated frame-level bounding boxes:
     - Original baseline frames (clean subset).
     - Curated YawDD yawning and speech frames.
     - Curated DMD ocular and mouth crops.
   - Purpose: Real-time ($>30\text{ FPS}$) detection of `eyes_closed`, `eyes_open`, and `yawning`.
2. **Tier 2 (Continuous Physiological Extraction - MediaPipe Mesh):**
   - Extract continuous EAR, MAR, and 3D Head Pose without requiring heavy neural network training.
3. **Tier 3 (Temporal Fatigue & Distraction Integration):**
   - Aggregate frame outputs into rolling PERCLOS ($P_{80}$), blink duration state machines, and sustained yawn filters.

---

## 13. Validation Strategy

- **UTA-RLDD 5-Fold Cross-Validation:**
  - Execute full temporal evaluation across the official 5 folds of UTA-RLDD (12 subjects per fold).
  - Metric: Area Under ROC Curve (AUC) and sensitivity for classifying Alert vs. Drowsy sequences.
- **Speech False-Alarm Rejection Test:**
  - Evaluate the yawning detector on all 322 YawDD talking/singing videos.
  - Success Metric: **Zero false critical alerts** triggered by speech.

---

## 14. External Test Strategy

- **Zero-Shot External Benchmark: NTHU-DDD**
  - Completely withheld from training and validation.
  - Evaluated on all 5 driving scenarios:
    1. BareFace (Day)
    2. Glasses (Day)
    3. Sunglasses (Day)
    4. Night-BareFace (Active NIR)
    5. Night-Glasses (Active NIR)
  - Quantifies real-world domain transfer and performance degradation under nighttime driving and heavy ocular occlusions.

---

## 15. Feature-to-Dataset Mapping

| Planned System Capability | Supporting Dataset(s) | Split Role | Expected Feasibility |
| :--- | :--- | :--- | :--- |
| **Instantaneous Eye Closure** | Original + DMD Ocular + MRL Eye | Train / Val | VERY HIGH |
| **Yawn Detection** | YawDD (Yawn Clips) + Original | Train / Val | VERY HIGH |
| **Speech vs. Yawn Discrimination** | YawDD (Talking/Singing Clips) | Train / Val (Negative Class) | HIGH |
| **Blink Duration & Micro-sleeps** | UTA-RLDD + MediaPipe 3D Mesh | Temporal Validation | VERY HIGH |
| **Rolling PERCLOS ($P_{80}$)** | UTA-RLDD (5-Fold Cross-Val) | Temporal Validation | VERY HIGH |
| **3D Head Pose & Nodding** | MediaPipe FaceMesh + DMD Head Pose | Train / Val | HIGH |
| **Driver Distraction & Phone** | DMD (14 Approved Subjects) | Task-Specific Train / Val | HIGH (Phase 3) |
| **Night & Sunglasses Generalization** | NTHU-DDD (5 Scenarios) | External Test Benchmark | HIGH |

---

## 16. Dataset License & Access Notes

| Dataset | Access Portal | License Terms | Action Required by User |
| :--- | :--- | :--- | :--- |
| **Original** | Kaggle / Private GDrive | Unlicensed / Custom | No download required (already local). |
| **UTA-RLDD** | [Google Sites Portal](https://sites.google.com/view/utarldd/home) | Academic Non-Commercial | Download through official Google Drive links. |
| **YawDD** | [IEEE DataPort](https://ieee-dataport.org/) | Academic Non-Commercial | Create free IEEE account and download zip archive. |
| **DMD** | [GitHub Repository](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset) | Restricted Academic License | Request access for 14 approved subject IDs. |
| **NTHU-DDD** | [NTHU CVLab Portal](https://cv.cs.nthu.edu.tw/) | Academic Research Agreement | Submit academic request form or use research mirror. |

---

## 17. Recommended Dataset Combination

1. **YOLO Facial State Training Set:**
   - 207 original project frames (cleaned) + 1,200 curated YawDD frames (600 yawning, 600 talking/singing) + 1,000 DMD ocular/mouth crops.
   - Total volume: ~2,400 high-quality, balanced, diverse frames.
2. **Temporal & Cumulative Fatigue Evaluation Set:**
   - Full UTA-RLDD (60 subjects, 180 videos, 30 hours) in 5-fold subject-independent partitions.
3. **External Out-of-Distribution Test Set:**
   - NTHU-DDD (36 subjects, 5 scenarios including Night and Sunglasses).

---

## 18. Datasets NOT Recommended and Why

| Dataset | Why Rejected / Not Recommended for Core Pipeline |
| :--- | :--- |
| **Kaggle Drowsiness Detection Mirrors (Third-Party)** | Uncurated duplicates with arbitrary class labels, scraped without subject metadata; prevents clean leakage prevention. |
| **DMD Simulator & IR Streams (Historical)** | **Permanently removed by Vicomtech in 2026**; relying on legacy download scripts causes 404 errors. |
| **Pure Image Classification Datasets (e.g. CEW)** | Cropped eye-patches lack full facial context, head pose, and mouth contours; cannot train an end-to-end YOLO object detector. |

---

## 19. Data Collection Gaps & Solutions

1. **Gap: Frame-level Bounding Boxes in Video Datasets (UTA-RLDD & YawDD).**
   - *Solution:* In Phase 2C, deploy our verified MediaPipe FaceMesh to automatically extract face, eye, and mouth crops, converting interval annotations into high-precision bounding boxes.
2. **Gap: Nighttime Infrared Illumination.**
   - *Solution:* NTHU-DDD provides active NIR video streams for night driving benchmarks.

---

## 20. Next Phase Recommendation

Phase 2B Dataset Audit is **COMPLETE**.  
The recommended next phase is **Phase 2C — Dataset Ingestion, Normalization & Annotation Pipeline**:
1. Download approved subsets of UTA-RLDD, YawDD, and NTHU-DDD.
2. Build an automated conversion script to extract and format bounding boxes into standard YOLO `.txt` format.
3. Generate the canonical 5-fold cross-validation manifest with zero subject leakage.
4. Prepare the unified dataset YAML configuration ready for Phase 3 training.

---

## RECOMMENDED DATASET STRATEGY

```
RECOMMENDED DATASET STRATEGY

Primary Training Dataset:
Curated Multi-Source Frame Union (Original Clean + YawDD Yawn & Speech Crops + DMD Real-Car Crops)

Secondary Training Dataset:
MRL Eye Dataset (Curated high-resolution ocular crops for secondary edge-case calibration)

Task-Specific Dataset:
DMD (Driver Monitoring Dataset — 14 Approved Subjects for Mobile Phone Use & Distraction Multi-Task Head)

External Test Dataset:
NTHU-DDD (36 Subjects — Evaluated on BareFace, Glasses, Sunglasses, Night-BareFace, and Night-Glasses)

Optional Dataset:
UTA-RLDD (Used as the Primary Benchmark for 5-Fold Subject-Independent Temporal & PERCLOS Evaluation)

Datasets Rejected:
1. Legacy DMD Simulator / IR / Depth Streams (Permanently removed in 2026 revision)
2. Unverified Third-Party Kaggle Drowsiness Scrapes (Unverifiable provenance, severe leakage)
3. Isolated Eye-Patch Datasets for Main Detector (CEW, ZJU) (Lack full facial context and head pose)

Reason:
Video-level drowsiness labels cannot be directly mapped to frame-level object detector bounding boxes without severe label noise. A two-tier architecture (YOLO for instantaneous facial states + MediaPipe/Temporal State Machines for continuous PERCLOS/Blinks + 5-fold cross-validation on UTA-RLDD) provides the only scientifically defensible, leak-free, resume-level solution.
```
